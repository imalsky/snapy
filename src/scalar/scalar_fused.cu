// C/C++
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include <snap/implicit/implicit_dispatch.hpp>

#include "scalar_fused.hpp"

// the WENO5 weights of the default path
#include <snap/recon/interp_impl.cuh>

namespace snap {

namespace {
// every defined tensor on ref's CUDA device with ref's dtype
void check_operands(char const* who, torch::Tensor const& ref,
                    std::initializer_list<torch::Tensor const*> ts) {
  TORCH_CHECK(ref.is_cuda(), who, ": CUDA tensors expected");
  for (auto const* t : ts) {
    if (!t->defined()) continue;
    TORCH_CHECK(
        t->device() == ref.device() && t->scalar_type() == ref.scalar_type(),
        who, ": every tensor on ", ref.device(), " as ", ref.scalar_type());
  }
}
// an extent the kernels hold in an int
int int_extent(int64_t n, char const* who) {
  TORCH_CHECK(n >= 0 && n <= std::numeric_limits<int>::max(), who, ": extent ",
              n, " does not fit an int");
  return static_cast<int>(n);
}
// blocks of a launch of one thread per element
int64_t flat_blocks(int64_t total, int threads, char const* who) {
  int64_t blocks = (total + threads - 1) / threads;
  TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), who,
              ": too many elements for one grid");
  return blocks;
}
// The fused WENO5 kernels write out the uniform-grid tables of
// Weno5InterpImpl (wenol0 ... below): cm for the left face and its mirror cp
// for the right. Any other table is refused. The check compares in the
// table's own dtype (as the kernels' T(...) constants round) and runs once per
// table state (data pointers, version counters and dtype; one copy each to the
// host).
void check_uniform_weno5(torch::Tensor const& cm, torch::Tensor const& cp,
                         char const* who) {
  static std::mutex mu;
  using Key = std::tuple<void const*, int64_t, void const*, int64_t, int>;
  static std::set<Key> seen;
  Key key{cm.data_ptr(), static_cast<int64_t>(cm._version()), cp.data_ptr(),
          static_cast<int64_t>(cp._version()),
          static_cast<int>(cm.scalar_type())};
  std::lock_guard<std::mutex> lock(mu);
  if (seen.count(key)) return;
  static double const table[45] = {
      -1. / 6., 5. / 6., 1. / 3., 0.,  0.,       0.,       1. / 3., 5. / 6.,
      -1. / 6., 0.,      0.,      0.,  11. / 6., -7. / 6., 1. / 3., 1.,
      -2.,      1.,      0.,      0.,  1.,       -4.,      3.,      0.,
      0.,       0.,      1.,      -2., 1.,       0.,       0.,      -1.,
      0.,       1.,      0.,      0.,  0.,       1.,       -2.,     1.,
      0.,       0.,      3.,      -4., 1.};
  auto want =
      torch::from_blob(const_cast<double*>(table), {9, 5}, torch::kFloat64)
          .to(cm.scalar_type());
  TORCH_CHECK(cp.scalar_type() == cm.scalar_type() &&
                  torch::equal(cm.reshape({9, 5}).cpu(), want) &&
                  torch::equal(cp.reshape({9, 5}).cpu(), want.flip({1})),
              who,
              ": the fused path writes out the uniform-grid WENO5 tables; "
              "this reconstruction uses other ones");
  seen.insert(key);
}
}  // namespace

// Separately rounded arithmetic: the libtorch chains this path replaces round
// after every operation, so no product may be contracted into a sum here.
__device__ __forceinline__ double rmul(double a, double b) {
  return __dmul_rn(a, b);
}
__device__ __forceinline__ double radd(double a, double b) {
  return __dadd_rn(a, b);
}
__device__ __forceinline__ double rsub(double a, double b) {
  return __dsub_rn(a, b);
}
__device__ __forceinline__ double rdiv(double a, double b) {
  return __ddiv_rn(a, b);
}
__device__ __forceinline__ float rmul(float a, float b) {
  return __fmul_rn(a, b);
}
__device__ __forceinline__ float radd(float a, float b) {
  return __fadd_rn(a, b);
}
__device__ __forceinline__ float rsub(float a, float b) {
  return __fsub_rn(a, b);
}
__device__ __forceinline__ float rdiv(float a, float b) {
  return __fdiv_rn(a, b);
}

// torch's relu / clamp_min(x, 0): a NaN stays NaN, otherwise ::max(x, 0),
// which gives +0 for -0 as there
template <typename T>
__device__ __forceinline__ T clamp0(T x) {
  return isnan(x) ? x : ::max(x, T(0));
}

// One thread per (scalar, k, j, i), i fastest. As ScalarImpl::forward's
// lateral(): vel * (ui * rl + (1 - ui) * rr) with ui = (vel > 0), plus the
// covariance row of scalar_flux_covariance at x1 cells 1..n1-2:
//   q = 0.5 * (rl * rdl + rr * rdr),  d1(f) = (f[i+1] - f[i-1]) / dx1
//   a * d1(q) [+ nds * d1(b * q) with a centroid shift]
template <typename T>
__global__ void scalar_lateral_flux_kernel(
    T* __restrict__ flux, T const* __restrict__ rl, T const* __restrict__ rr,
    T const* __restrict__ vel, T const* __restrict__ rdl,
    T const* __restrict__ rdr, T const* __restrict__ a, T const* __restrict__ b,
    T const* __restrict__ nds, T const* __restrict__ dx1, int64_t ncell, int n1,
    int64_t total, bool cov, bool centroid) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t c = idx % ncell;  // the cell (k, j, i)
  int i = static_cast<int>(c % n1);

  T v = vel[c];
  T u = v > T(0) ? T(1) : T(0);
  T up = rmul(v, radd(rmul(u, rl[idx]), rmul(rsub(T(1), u), rr[idx])));
  if (!cov) {
    flux[idx] = up;
    return;
  }

  T d = T(0);
  if (i >= 1 && i <= n1 - 2) {
    int m = i - 1;
    T qp = rmul(T(0.5), radd(rmul(rl[idx + 1], rdl[c + 1]),
                             rmul(rr[idx + 1], rdr[c + 1])));
    T qm = rmul(T(0.5), radd(rmul(rl[idx - 1], rdl[c - 1]),
                             rmul(rr[idx - 1], rdr[c - 1])));
    T h = dx1[m];
    T cv = rmul(a[(c / n1) * (n1 - 2) + m], rdiv(rsub(qp, qm), h));
    if (centroid) {
      T bq = rdiv(rsub(rmul(b[c + 1], qp), rmul(b[c - 1], qm)), h);
      d = radd(cv, rmul(nds[m], bq));
    } else {
      d = cv;
    }
  }
  flux[idx] = radd(up, d);
}

void scalar_lateral_flux_cuda(torch::Tensor& flux, torch::Tensor const& rl,
                              torch::Tensor const& rr, torch::Tensor const& vel,
                              std::vector<torch::Tensor> const& factors) {
  at::cuda::CUDAGuard device_guard(rl.device());
  char const* who = "scalar_lateral_flux";
  check_operands(who, rl, {&rr, &flux, &vel});
  TORCH_CHECK(rl.is_contiguous() && rr.is_contiguous() &&
                  flux.is_contiguous() && vel.is_contiguous(),
              who, ": contiguous tensors expected");
  TORCH_CHECK(rl.dim() == 4 && rr.sizes() == rl.sizes() &&
                  flux.sizes() == rl.sizes() &&
                  vel.sizes() == rl.sizes().slice(1),
              who, ": rl, rr, flux (nvar, nc3, nc2, nc1), vel (nc3, nc2, nc1)");
  bool cov = !factors.empty();
  // rdl, rdr, a, b, nds, dx1 (HydroImpl::scalar_flux_covariance_factors)
  bool centroid = cov && factors.size() == 6 && factors[3].defined();
  int64_t n3 = rl.size(1), n2 = rl.size(2), n1l = rl.size(3);
  if (cov) {
    TORCH_CHECK(factors.size() == 6 && n1l >= 3, who,
                ": six covariance factors and at least 3 x1 cells expected");
    TORCH_CHECK(factors[3].defined() == factors[4].defined(), who,
                ": b and nds come together (the centroid shift)");
    auto shaped = [&](int k, std::vector<int64_t> shape) {
      auto const& t = factors[k];
      TORCH_CHECK(t.defined() && t.is_contiguous() &&
                      t.sizes() == torch::IntArrayRef(shape),
                  who, ": covariance factor ", k, " shaped ", shape,
                  " expected");
      check_operands(who, rl, {&t});
    };
    shaped(0, {n3, n2, n1l});
    shaped(1, {n3, n2, n1l});
    shaped(2, {n3, n2, n1l - 2});
    shaped(5, {n1l - 2});
    if (centroid) {
      shaped(3, {n3, n2, n1l});
      shaped(4, {n1l - 2});
    }
  }
  int64_t total = rl.numel();
  if (total == 0) return;
  int64_t ncell = vel.numel();
  int n1 = int_extent(n1l, who);
  int threads = 256;
  int64_t blocks = flat_blocks(total, threads, who);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(rl.scalar_type(), "scalar_lateral_flux_cuda", [&] {
    auto ptr = [&](int n) -> scalar_t const* {
      return cov && factors[n].defined() ? factors[n].data_ptr<scalar_t>()
                                         : nullptr;
    };
    scalar_lateral_flux_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        flux.data_ptr<scalar_t>(), rl.data_ptr<scalar_t>(),
        rr.data_ptr<scalar_t>(), vel.data_ptr<scalar_t>(), ptr(0), ptr(1),
        ptr(2), ptr(3), ptr(4), ptr(5), ncell, n1, total, cov, centroid);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

struct Dims {
  int nc1, nc2, nc3;
};

// flux_positivity_theta: out accumulates x1, then x2, then x3 over the cells
// each dimension's slice covers (all indices of the other two dimensions).
template <typename T>
__global__ void scalar_theta_kernel(
    T* __restrict__ theta, T const* __restrict__ u, T const* __restrict__ f1,
    T const* __restrict__ f2, T const* __restrict__ f3,
    T const* __restrict__ a1, T const* __restrict__ a2,
    T const* __restrict__ a3, T const* __restrict__ vol, T dt, T keep,
    at::native::ScalarBox box, Dims n, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t ncell = static_cast<int64_t>(n.nc3) * n.nc2 * n.nc1;
  int64_t c = idx % ncell;
  int i = static_cast<int>(c % n.nc1);
  int j = static_cast<int>((c / n.nc1) % n.nc2);
  int k = static_cast<int>(c / (static_cast<int64_t>(n.nc1) * n.nc2));

  T out = T(0);
  if (f1 != nullptr && i >= box.il && i <= box.iu) {
    int64_t fa = (static_cast<int64_t>(k) * n.nc2 + j) * (n.nc1 + 1) + i;
    T up = rmul(a1[fa + 1], f1[idx + 1]);
    T lo = rmul(a1[fa], f1[idx]);
    out = radd(out, radd(clamp0(up), clamp0(-lo)));
  }
  if (f2 != nullptr && j >= box.jl && j <= box.ju) {
    int64_t fa = (static_cast<int64_t>(k) * (n.nc2 + 1) + j) * n.nc1 + i;
    T up = rmul(a2[fa + n.nc1], f2[idx + n.nc1]);
    T lo = rmul(a2[fa], f2[idx]);
    out = radd(out, radd(clamp0(up), clamp0(-lo)));
  }
  if (f3 != nullptr && k >= box.kl && k <= box.ku) {
    int64_t plane = static_cast<int64_t>(n.nc2) * n.nc1;
    int64_t fa =
        static_cast<int64_t>(k) * plane + static_cast<int64_t>(j) * n.nc1 + i;
    T up = rmul(a3[fa + plane], f3[idx + plane]);
    T lo = rmul(a3[fa], f3[idx]);
    out = radd(out, radd(clamp0(up), clamp0(-lo)));
  }
  T avail = rmul(rmul(clamp0(u[idx]), vol[c]), keep);
  T drain = rmul(out, dt);
  T th = T(1);
  if (drain > T(0)) {
    T r = rdiv(avail, drain < T(1e-300) ? T(1e-300) : drain);
    th = r > T(1) ? T(1) : r;
  }
  theta[idx] = th;
}

// flux_positivity_scale_ for one flux: faces lo..hi along `stride` (in
// elements: 1 for x1, nc1 for x2, nc1*nc2 for x3), position `pos` of the face
// along its dimension.
template <typename T>
__global__ void scalar_theta_scale_kernel(T const* __restrict__ theta,
                                          T* __restrict__ f, int lo, int hi,
                                          int64_t stride, int64_t extent,
                                          int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int pos = static_cast<int>((idx / stride) % extent);
  if (pos < lo || pos > hi) return;
  T v = f[idx];
  T w = v > T(0) ? theta[idx - stride] : theta[idx];
  f[idx] = rmul(v, w);
}

// CoordinateImpl::divergence, then ds = -dt * div on the interior
template <typename T>
__global__ void scalar_divergence_kernel(
    T* __restrict__ div, T* __restrict__ ds, T const* __restrict__ f1,
    T const* __restrict__ f2, T const* __restrict__ f3,
    T const* __restrict__ a1, T const* __restrict__ a2,
    T const* __restrict__ a3, T const* __restrict__ vol, T mdt,
    at::native::ScalarBox box, Dims n, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t ncell = static_cast<int64_t>(n.nc3) * n.nc2 * n.nc1;
  int64_t c = idx % ncell;
  int i = static_cast<int>(c % n.nc1);
  int j = static_cast<int>((c / n.nc1) % n.nc2);
  int k = static_cast<int>(c / (static_cast<int64_t>(n.nc1) * n.nc2));

  T d = T(0);
  if (f1 != nullptr && i >= box.il && i <= box.iu) {
    int64_t fa = (static_cast<int64_t>(k) * n.nc2 + j) * (n.nc1 + 1) + i;
    d = radd(d, rsub(rmul(a1[fa + 1], f1[idx + 1]), rmul(a1[fa], f1[idx])));
  }
  if (f2 != nullptr && j >= box.jl && j <= box.ju) {
    int64_t fa = (static_cast<int64_t>(k) * (n.nc2 + 1) + j) * n.nc1 + i;
    d = radd(
        d, rsub(rmul(a2[fa + n.nc1], f2[idx + n.nc1]), rmul(a2[fa], f2[idx])));
  }
  if (f3 != nullptr && k >= box.kl && k <= box.ku) {
    int64_t plane = static_cast<int64_t>(n.nc2) * n.nc1;
    int64_t fa =
        static_cast<int64_t>(k) * plane + static_cast<int64_t>(j) * n.nc1 + i;
    d = radd(
        d, rsub(rmul(a3[fa + plane], f3[idx + plane]), rmul(a3[fa], f3[idx])));
  }
  T q = rdiv(d, vol[c]);
  div[idx] = q;
  bool interior = i >= box.il && i <= box.iu && j >= box.jl && j <= box.ju &&
                  k >= box.kl && k <= box.ku;
  ds[idx] = interior ? rmul(q, mdt) : T(0);
}

// The x1 upwind flux of columns longer than 1024 cells (too long for one
// block of scalar_weno5_x1_flux_kernel), one thread per face element (scalar,
// k, j, i), with the default path's arithmetic (bit for bit). Along x1 (size
// n, element stride s, ghosts il = 3, iu = n - 3) the default path stores
// right() of cell p - 1 as the left state of face p and left() of cell p as
// its right state, for faces il..iu+1 and il-1..iu, then copies the edge
// values outward: the cells used are clamp(p - 1, il - 1, iu - 1) and
// clamp(p, il, iu).
template <typename T>
__global__ void scalar_weno5_long_flux_kernel(
    T* __restrict__ out, T const* __restrict__ w, T const* __restrict__ cm,
    T const* __restrict__ cp, bool scale, int64_t stride, int n, int64_t total,
    int64_t ncell, T const* __restrict__ vel) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int p = static_cast<int>((idx / stride) % n);
  int64_t base = idx - static_cast<int64_t>(p) * stride;
  int il = 3, iu = n - 3;
  int cl = min(max(p - 1, il - 1), iu - 1);
  int cr = min(max(p, il), iu);
  T sl[5], sr[5];
#pragma unroll
  for (int q = 0; q < 5; ++q) {
    sl[q] = w[base + static_cast<int64_t>(cl - 2 + q) * stride];
    sr[q] = w[base + static_cast<int64_t>(cr - 2 + q) * stride];
  }
  T wl = clamp0(interp_shared_weno5_coeff_impl(sl, cp, 0, 0, 5, scale));
  T wr = clamp0(interp_shared_weno5_coeff_impl(sr, cm, 0, 0, 5, scale));
  T v = vel[idx % ncell];
  T u = v > T(0) ? T(1) : T(0);
  out[idx] = rmul(v, radd(rmul(u, wl), rmul(rsub(T(1), u), wr)));
}

#define SQR(x) ((x) * (x))  // as interp_impl.cuh, which undefines it
// The rows of Weno5InterpImpl::cm (the uniform-grid WENO5 table) as fma chains
// over their nonzero entries, in index order. _vvdot<5> (interp_impl.cuh)
// builds the same chains (out = 0; out += v1[i] * v2[i], contracted to fma),
// and a zero entry adds exactly zero there, so for finite stencils these are
// bit for bit the table's dot products, without loading the table (the one
// exception: where an intermediate sum is -0 from an underflowed product, a
// skipped zero term no longer turns it into +0, so a zero face of the unscaled
// formula may change its sign).
// Rows 0-2: the candidate polynomials of left(); cp (cm with its columns
// reversed) gives right()'s as wenor0..2. Rows 3-8: the smoothness terms.
template <typename T>
__device__ __forceinline__ T fma0(T a, T c) {
  return fma(a, c, T(0));
}
template <typename T>
__device__ __forceinline__ T wenol0(T const* f) {
  return fma(f[2], T(1. / 3.), fma(f[1], T(5. / 6.), fma0(f[0], T(-1. / 6.))));
}
template <typename T>
__device__ __forceinline__ T wenol1(T const* f) {
  return fma(f[3], T(-1. / 6.), fma(f[2], T(5. / 6.), fma0(f[1], T(1. / 3.))));
}
template <typename T>
__device__ __forceinline__ T wenol2(T const* f) {
  return fma(f[4], T(1. / 3.), fma(f[3], T(-7. / 6.), fma0(f[2], T(11. / 6.))));
}
template <typename T>
__device__ __forceinline__ T wenor0(T const* f) {
  return fma(f[4], T(-1. / 6.), fma(f[3], T(5. / 6.), fma0(f[2], T(1. / 3.))));
}
template <typename T>
__device__ __forceinline__ T wenor1(T const* f) {
  return fma(f[3], T(1. / 3.), fma(f[2], T(5. / 6.), fma0(f[1], T(-1. / 6.))));
}
template <typename T>
__device__ __forceinline__ T wenor2(T const* f) {
  return fma(f[2], T(11. / 6.), fma(f[1], T(-7. / 6.), fma0(f[0], T(1. / 3.))));
}
template <typename T>
__device__ __forceinline__ void weno5_smooth(T const* f, T& b0, T& b1, T& b2) {
  T s3 = fma(f[2], T(1), fma(f[1], T(-2), fma0(f[0], T(1))));
  T s4 = fma(f[2], T(3), fma(f[1], T(-4), fma0(f[0], T(1))));
  T s5 = fma(f[3], T(1), fma(f[2], T(-2), fma0(f[1], T(1))));
  T s6 = fma(f[3], T(1), fma0(f[1], T(-1)));
  T s7 = fma(f[4], T(1), fma(f[3], T(-2), fma0(f[2], T(1))));
  T s8 = fma(f[4], T(1), fma(f[3], T(-4), fma0(f[2], T(3))));
  b0 = 13. / 12. * SQR(s3) + .25 * SQR(s4);
  b1 = 13. / 12. * SQR(s5) + .25 * SQR(s6);
  b2 = 13. / 12. * SQR(s7) + .25 * SQR(s8);
}

// Both WENO5 values of one cell (left(): the state at its lower face; right():
// the state at its upper face) from one stencil: the formula of
// interp_shared_weno5_coeff_impl with the uniform-grid table written out
// (above) and fewer divisions, so the values differ from the default path at
// round-off.
// - The two share vscale, phi and the smoothness indicators: cp's rows 3, 4
//   are cm's rows 7, 8 and the reverse, and cp's row 6 is minus cm's row 6, so
//   right()'s beta0, beta1, beta2 are left()'s beta2, beta1, beta0.
// - phi = src * (1 / vscale), except where 1 / vscale would overflow.
// - The weights alpha_k = c_k / d_k, d_k = (beta_k + 1e-6)^2, are ratios, so
//   they are multiplied through by d0 d1 d2 (with |phi| summing to 5, every
//   d_j d_k lies in [1e-24, 2e9]); one reciprocal of the two weight sums
//   serves both faces (fp64), and the quotient is formed before the scale is
//   applied, so tiny and huge values stay in range (the default formula
//   overflows to Inf on stencils above about 1e290; this one does not).
//   Divisions by variables go from 12 to 2 (besides the mean's / 5).
// - Without scaling (scale false) the products are not bounded: the default
//   formula is used.
template <typename T>
__device__ __forceinline__ void weno5_pair(T const* src, bool scale, T& left,
                                           T& right) {
  T vscale = scale ? (fabs(src[0]) + fabs(src[1]) + fabs(src[2]) +
                      fabs(src[3]) + fabs(src[4])) /
                         5.0
                   : 1.0;
  if (vscale == 0.0) {
    left = 0.0;
    right = 0.0;
    return;
  }
  // 1 / vscale overflows for subnormal vscale: divide there
  bool tiny = vscale < (sizeof(T) == 8 ? 1e-300 : 1e-30);
  T inv = tiny ? T(0) : T(1) / vscale;
  T phi[5];
  for (int k = 0; k < 5; ++k) {
    phi[k] = tiny ? src[k] / vscale : src[k] * inv;
  }
  T beta0, beta1, beta2;
  weno5_smooth(phi, beta0, beta1, beta2);
  T d0 = SQR(beta0 + 1e-6), d1 = SQR(beta1 + 1e-6), d2 = SQR(beta2 + 1e-6);
  if (!scale) {
    T a1 = .6 / d1;
    T p0 = wenol0(phi), p1 = wenol1(phi), p2 = wenol2(phi);
    T a0 = .3 / d0, a2 = .1 / d2;
    left = vscale * (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2);
    p0 = wenor0(phi), p1 = wenor1(phi), p2 = wenor2(phi);
    a0 = .3 / d2, a2 = .1 / d0;
    right = vscale * (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2);
    return;
  }
  T e0 = d1 * d2, e1 = d0 * d2, e2 = d0 * d1;
  T a1 = .6 * e1;
  T nl, dl, nr, dr;
  {
    T p0 = wenol0(phi), p1 = wenol1(phi), p2 = wenol2(phi);
    T a0 = .3 * e0, a2 = .1 * e2;
    nl = a0 * p0 + a1 * p1 + a2 * p2;
    dl = a0 + a1 + a2;
  }
  {
    T p0 = wenor0(phi), p1 = wenor1(phi), p2 = wenor2(phi);
    T a0 = .3 * e2, a2 = .1 * e0;
    nr = a0 * p0 + a1 * p1 + a2 * p2;
    dr = a0 + a1 + a2;
  }
  if (sizeof(T) == 8) {  // dl dr >= 1e-50: an fp64 normal, not an fp32 one
    T q = T(1) / (dl * dr);
    left = vscale * (nl * dr * q);
    right = vscale * (nr * dl * q);
  } else {
    left = vscale * (nl / dl);
    right = vscale * (nr / dr);
  }
}
#undef SQR

// Face states along a line from cell-centred work: the thread of cell p
// (il - 1 <= p <= iu) evaluates its stencil once and writes the right state
// of face p (left(), cells il..iu) and the left state of face p + 1 (right(),
// cells il-1..iu-1), plus the default path's edge copies.
template <typename T>
__global__ void scalar_weno5_faces_kernel(T* __restrict__ out,
                                          T const* __restrict__ w, bool scale,
                                          int64_t stride, int n,
                                          int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int p = static_cast<int>((idx / stride) % n);
  int il = 3, iu = n - 3;
  if (p < il - 1 || p > iu) return;
  int64_t base = idx - static_cast<int64_t>(p) * stride;
  T src[5];
#pragma unroll
  for (int q = 0; q < 5; ++q) {
    src[q] = w[base + static_cast<int64_t>(p - 2 + q) * stride];
  }
  T l, r;
  weno5_pair(src, scale, l, r);
  l = clamp0(l);
  r = clamp0(r);
  T* ilt = out;
  T* irt = out + total;
  if (p >= il) {
    irt[idx] = l;
    if (p == il)
      for (int q = 0; q < il; ++q) irt[base + q * stride] = l;
    if (p == iu)
      for (int q = iu + 1; q < n; ++q) irt[base + q * stride] = l;
  }
  if (p <= iu - 1) {
    ilt[idx + stride] = r;
    if (p == il - 1)
      for (int q = 0; q < il; ++q) ilt[base + q * stride] = r;
    if (p == iu - 1)
      for (int q = iu + 1; q < n; ++q) ilt[base + q * stride] = r;
  }
}

// x1 (contiguous) flux: one block per column; each cell's pair goes to shared
// memory, then each face takes right() of the cell below and left() of the
// cell above (with the edge copies) for the upwind flux.
template <typename T>
__global__ void scalar_weno5_x1_flux_kernel(T* __restrict__ flux,
                                            T const* __restrict__ w, bool scale,
                                            int n, int64_t ncolumn_cell,
                                            T const* __restrict__ vel) {
  extern __shared__ unsigned char smem_raw[];
  T* sl = reinterpret_cast<T*>(smem_raw);
  T* sr = sl + n;
  int64_t col = blockIdx.x;
  int64_t base = col * n;
  int p = threadIdx.x;
  int il = 3, iu = n - 3;
  if (p >= il - 1 && p <= iu) {
    T src[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) src[q] = w[base + p - 2 + q];
    T l, r;
    weno5_pair(src, scale, l, r);
    sl[p] = clamp0(l);
    sr[p] = clamp0(r);
  }
  __syncthreads();
  T wl = sr[min(max(p - 1, il - 1), iu - 1)];
  T wr = sl[min(max(p, il), iu)];
  T v = vel[(col % ncolumn_cell) * n + p];
  T u = v > T(0) ? T(1) : T(0);
  flux[base + p] = rmul(v, radd(rmul(u, wl), rmul(rsub(T(1), u), wr)));
}

void scalar_weno5_cuda(torch::Tensor& faces, torch::Tensor const& w,
                       torch::Tensor const& cm, torch::Tensor const& cp,
                       bool scale, int dim, torch::Tensor const& vel) {
  at::cuda::CUDAGuard device_guard(w.device());
  char const* who = "scalar_weno5";
  check_operands(who, w, {&faces, &vel});
  TORCH_CHECK(w.dim() == 4 && w.is_contiguous() && faces.is_contiguous(), who,
              ": contiguous (nvar, nc3, nc2, nc1) expected");
  TORCH_CHECK(dim >= 1 && dim <= 3, who, ": dim 1, 2 or 3");
  TORCH_CHECK(cm.numel() == 45 && cp.numel() == 45 && cm.is_contiguous() &&
                  cp.is_contiguous() && cm.device() == w.device() &&
                  cp.device() == w.device(),
              who, ": 45 contiguous WENO5 coefficients on ", w.device(),
              " expected");
  TORCH_CHECK(w.size(dim) >= 6, who, ": at least 6 cells along the line");
  int n = int_extent(w.size(dim), who);
  int64_t stride = w.stride(dim);
  int64_t total = w.numel();
  bool upwind = vel.defined();
  if (upwind) {
    TORCH_CHECK(dim == 3 && vel.is_contiguous() &&
                    vel.sizes() == w.sizes().slice(1) &&
                    faces.sizes() == w.sizes(),
                who,
                ": the upwind form is x1 only, vel (nc3, nc2, nc1) and the "
                "flux shaped like w");
  } else {
    std::vector<int64_t> two = {2};
    two.insert(two.end(), w.sizes().begin(), w.sizes().end());
    TORCH_CHECK(faces.sizes() == torch::IntArrayRef(two), who,
                ": faces (2, nvar, nc3, nc2, nc1) expected");
  }
  if (total == 0) return;
  if (!upwind || n <= 1024) check_uniform_weno5(cm, cp, who);  // the fused kernels
  int64_t ncell = total / w.size(0);
  int threads = 256;
  int64_t blocks = flat_blocks(total, threads, who);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "scalar_weno5_cuda", [&] {
    if (upwind && n <= 1024) {
      int64_t columns = total / n;
      TORCH_CHECK(columns <= std::numeric_limits<int>::max(),
                  "scalar_weno5: too many x1 columns for one grid");
      size_t shared = 2 * static_cast<size_t>(n) * sizeof(scalar_t);
      scalar_weno5_x1_flux_kernel<scalar_t><<<columns, n, shared, stream>>>(
          faces.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), scale, n,
          ncell / n, vel.data_ptr<scalar_t>());
    } else if (upwind) {
      auto cmp = cm.to(w.scalar_type());
      auto cpp = cp.to(w.scalar_type());
      scalar_weno5_long_flux_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          faces.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
          cmp.data_ptr<scalar_t>(), cpp.data_ptr<scalar_t>(), scale, stride, n,
          total, ncell, vel.data_ptr<scalar_t>());
    } else {
      scalar_weno5_faces_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          faces.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), scale, stride, n,
          total);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// MeshBlock's implicit x1 tracer transfer before its donor cap: on face i
// (below cell i) P = (M > 0 ? r[i - 1] : r[i]) * M with r[-1] = 0, and the
// tracer each cell holds, held = max(s + ds, 0) * vol
template <typename T>
__global__ void scalar_transfer_faces_kernel(
    T* __restrict__ P, T* __restrict__ held, T const* __restrict__ s,
    T const* __restrict__ ds, T const* __restrict__ r, T const* __restrict__ m,
    T const* __restrict__ vol, int n1, int64_t ncell, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t c = idx % ncell;
  int i = static_cast<int>(c % n1);
  T mi = m[c];
  P[idx] = rmul(mi > T(0) ? (i >= 1 ? r[idx - 1] : T(0)) : r[idx], mi);
  held[idx] = rmul(clamp0(radd(s[idx], ds[idx])), vol[c]);
}

// then, with P capped: ds += (P - P_above) / vol, P_above 0 past the top
template <typename T>
__global__ void scalar_transfer_add_kernel(T* __restrict__ ds,
                                           T const* __restrict__ P,
                                           T const* __restrict__ vol, int n1,
                                           int64_t ncell, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t c = idx % ncell;
  int i = static_cast<int>(c % n1);
  T above = i + 1 < n1 ? P[idx + 1] : T(0);
  ds[idx] = radd(ds[idx], rdiv(rsub(P[idx], above), vol[c]));
}

template <typename T>
__global__ void scalar_average3_kernel(T* __restrict__ out,
                                       T const* __restrict__ u0,
                                       T const* __restrict__ u1,
                                       T const* __restrict__ u2, T w0, T w1,
                                       T w2, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  out[idx] =
      radd(radd(rmul(w0, u0[idx]), rmul(w1, u1[idx])), rmul(w2, u2[idx]));
}

void scalar_implicit_transfer_cuda(torch::Tensor& ds, torch::Tensor const& s,
                                   torch::Tensor const& r,
                                   torch::Tensor const& m,
                                   torch::Tensor const& vol, int il, int iu) {
  at::cuda::CUDAGuard device_guard(ds.device());
  char const* who = "scalar_implicit_transfer";
  check_operands(who, ds, {&s, &r, &m, &vol});
  TORCH_CHECK(ds.is_contiguous() && s.is_contiguous() && r.is_contiguous() &&
                  m.is_contiguous() && vol.is_contiguous() && r.dim() == 4 &&
                  ds.sizes() == r.sizes() && s.sizes() == r.sizes() &&
                  m.sizes() == r.sizes().slice(1) && vol.sizes() == m.sizes(),
              who,
              ": contiguous ds, s, r (nvar, nc3, nc2, nc1) and M, vol (nc3, "
              "nc2, nc1) expected");
  TORCH_CHECK(0 <= il && il <= iu && iu < r.size(3), who, ": x1 interior ", il,
              "..", iu, " outside the ", r.size(3), " cells");
  int64_t total = r.numel();
  if (total == 0) return;
  int n1 = int_extent(r.size(3), who);
  int threads = 256;
  int64_t blocks = flat_blocks(total, threads, who);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto P = torch::empty_like(r);
  auto held = torch::empty_like(r);
  AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "scalar_transfer_faces", [&] {
    scalar_transfer_faces_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        P.data_ptr<scalar_t>(), held.data_ptr<scalar_t>(),
        s.data_ptr<scalar_t>(), ds.data_ptr<scalar_t>(),
        r.data_ptr<scalar_t>(), m.data_ptr<scalar_t>(),
        vol.data_ptr<scalar_t>(), n1, m.numel(), total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // the default path's own column sweep (implicit_dispatch.cu)
  at::native::donor_cap(P.device().type(), P, held, il, iu);
  AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "scalar_transfer_add", [&] {
    scalar_transfer_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        ds.data_ptr<scalar_t>(), P.data_ptr<scalar_t>(),
        vol.data_ptr<scalar_t>(), n1, m.numel(), total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scalar_average3_cuda(torch::Tensor& out, torch::Tensor const& u0,
                          torch::Tensor const& u1, torch::Tensor const& u2,
                          double w0, double w1, double w2) {
  at::cuda::CUDAGuard device_guard(out.device());
  char const* who = "scalar_average3";
  check_operands(who, out, {&u0, &u1, &u2});
  TORCH_CHECK(out.is_contiguous() && u0.is_contiguous() && u1.is_contiguous() &&
                  u2.is_contiguous() && u0.sizes() == out.sizes() &&
                  u1.sizes() == out.sizes() && u2.sizes() == out.sizes(),
              who, ": contiguous tensors of one shape expected");
  int64_t total = out.numel();
  if (total == 0) return;
  int threads = 256;
  int64_t blocks = flat_blocks(total, threads, who);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "scalar_average3_cuda", [&] {
    scalar_average3_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        out.data_ptr<scalar_t>(), u0.data_ptr<scalar_t>(),
        u1.data_ptr<scalar_t>(), u2.data_ptr<scalar_t>(), scalar_t(w0),
        scalar_t(w1), scalar_t(w2), total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// MeshBlockImpl's carry_dry_source, one thread per (scalar, k, j, i): the
// stage average, the libtorch where()s and the add, each operation rounded as
// there. base may alias ds: each thread reads its element before writing it.
template <typename T>
__global__ void scalar_carry_dry_kernel(T* ds, T const* __restrict__ s0,
                                        T const* __restrict__ s, T const* base,
                                        T const* __restrict__ r,
                                        T const* __restrict__ rho_base,
                                        T const* __restrict__ dry, T w0, T w1,
                                        T w2, int64_t ncell, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t c = idx % ncell;
  T sb = radd(radd(rmul(w0, s0[idx]), rmul(w1, s[idx])), rmul(w2, base[idx]));
  T rb = rho_base[c];
  T rr = r[idx];
  T remove = rb == T(0) ? rr : rdiv(sb, rb);
  T d = dry[c];
  T carry = d < T(0) ? remove : rr;
  ds[idx] = radd(ds[idx], rmul(carry, d));
}

void scalar_carry_dry_cuda(torch::Tensor& ds, torch::Tensor const& s0,
                           torch::Tensor const& s, torch::Tensor const& base,
                           torch::Tensor const& r,
                           torch::Tensor const& rho_base,
                           torch::Tensor const& dry, double w0, double w1,
                           double w2) {
  at::cuda::CUDAGuard device_guard(ds.device());
  char const* who = "scalar_carry_dry";
  check_operands(who, ds, {&s0, &s, &base, &r, &rho_base, &dry});
  TORCH_CHECK(ds.dim() == 4, who, ": ds (nvar, nc3, nc2, nc1) expected");
  torch::Tensor const& dsc = ds;
  for (auto const* t : {&dsc, &s0, &s, &base, &r})
    TORCH_CHECK(t->is_contiguous() && t->sizes() == ds.sizes(), who,
                ": contiguous ds, s0, s, base, r of one shape expected");
  for (auto const* t : {&rho_base, &dry})
    TORCH_CHECK(t->is_contiguous() && t->sizes() == ds.sizes().slice(1), who,
                ": contiguous rho_base, dry (nc3, nc2, nc1) expected");
  int64_t total = ds.numel();
  if (total == 0) return;
  int threads = 256;
  int64_t blocks = flat_blocks(total, threads, who);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(ds.scalar_type(), "scalar_carry_dry_cuda", [&] {
    scalar_carry_dry_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        ds.data_ptr<scalar_t>(), s0.data_ptr<scalar_t>(),
        s.data_ptr<scalar_t>(), base.data_ptr<scalar_t>(),
        r.data_ptr<scalar_t>(), rho_base.data_ptr<scalar_t>(),
        dry.data_ptr<scalar_t>(), scalar_t(w0), scalar_t(w1), scalar_t(w2),
        rho_base.numel(), total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
Dims dims_of(torch::Tensor const& u) {
  return Dims{int_extent(u.size(3), "fused scalar"),
              int_extent(u.size(2), "fused scalar"),
              int_extent(u.size(1), "fused scalar")};
}
void check_fused_shapes(torch::Tensor const& u, torch::Tensor const& a1,
                        torch::Tensor const& a2, torch::Tensor const& a3,
                        torch::Tensor const& vol) {
  TORCH_CHECK(u.dim() == 4, "fused scalar: (nvar, nc3, nc2, nc1) expected");
  int64_t n3 = u.size(1), n2 = u.size(2), n1 = u.size(3);
  TORCH_CHECK(
      u.is_contiguous() && vol.is_contiguous() &&
          vol.sizes() == torch::IntArrayRef({n3, n2, n1}),
      "fused scalar: u and the volume must be contiguous, (nc3, nc2, nc1)");
  TORCH_CHECK(
      !a1.defined() || (a1.is_contiguous() &&
                        a1.sizes() == torch::IntArrayRef({n3, n2, n1 + 1})),
      "fused scalar: face_area1 (nc3, nc2, nc1+1) expected");
  TORCH_CHECK(
      !a2.defined() || (a2.is_contiguous() &&
                        a2.sizes() == torch::IntArrayRef({n3, n2 + 1, n1})),
      "fused scalar: face_area2 (nc3, nc2+1, nc1) expected");
  TORCH_CHECK(
      !a3.defined() || (a3.is_contiguous() &&
                        a3.sizes() == torch::IntArrayRef({n3 + 1, n2, n1})),
      "fused scalar: face_area3 (nc3+1, nc2, nc1) expected");
}
template <typename T>
T const* cptr(torch::Tensor const& t) {
  return t.defined() ? t.data_ptr<T>() : nullptr;
}
}  // namespace

// each flux comes with its face area, shaped like u
void check_fluxes(torch::Tensor const& u, torch::Tensor const& f1,
                  torch::Tensor const& f2, torch::Tensor const& f3,
                  torch::Tensor const& a1, torch::Tensor const& a2,
                  torch::Tensor const& a3) {
  check_operands("fused scalar", u, {&f1, &f2, &f3, &a1, &a2, &a3});
  TORCH_CHECK(f1.defined() || f2.defined() || f3.defined(),
              "fused scalar: at least one flux expected");
  std::initializer_list<std::pair<torch::Tensor const*, torch::Tensor const*>>
      pairs = {{&f1, &a1}, {&f2, &a2}, {&f3, &a3}};
  for (auto [f, a] : pairs) {
    if (!f->defined()) continue;
    TORCH_CHECK(f->is_contiguous() && f->sizes() == u.sizes(),
                "fused scalar: fluxes shaped like u expected");
    TORCH_CHECK(a->defined(), "fused scalar: a flux without its face area");
  }
}

// every face the kernels read lies inside the array: lo - 1 >= 0 (donor) and
// hi + 1 < extent (upper face of the last cell)
void check_box(at::native::ScalarBox box, Dims n, torch::Tensor const& f1,
               torch::Tensor const& f2, torch::Tensor const& f3) {
  auto ok = [](int lo, int hi, int extent) {
    return 1 <= lo && lo <= hi && hi + 1 < extent;
  };
  TORCH_CHECK(!f1.defined() || ok(box.il, box.iu, n.nc1),
              "fused scalar: x1 interior must leave a ghost cell on each side");
  TORCH_CHECK(!f2.defined() || ok(box.jl, box.ju, n.nc2),
              "fused scalar: x2 interior must leave a ghost cell on each side");
  TORCH_CHECK(!f3.defined() || ok(box.kl, box.ku, n.nc3),
              "fused scalar: x3 interior must leave a ghost cell on each side");
}

void scalar_theta_cuda(torch::Tensor& theta, torch::Tensor const& u,
                       torch::Tensor const& f1, torch::Tensor const& f2,
                       torch::Tensor const& f3, torch::Tensor const& a1,
                       torch::Tensor const& a2, torch::Tensor const& a3,
                       torch::Tensor const& vol, double dt, double keep,
                       at::native::ScalarBox box) {
  at::cuda::CUDAGuard device_guard(u.device());
  check_operands("scalar_theta", u, {&theta, &vol});
  check_fused_shapes(u, a1, a2, a3, vol);
  TORCH_CHECK(theta.is_contiguous() && theta.sizes() == u.sizes(),
              "scalar_theta: theta shaped like u expected");
  check_fluxes(u, f1, f2, f3, a1, a2, a3);
  check_box(box, dims_of(u), f1, f2, f3);
  int64_t total = u.numel();
  if (total == 0) return;
  int threads = 256;
  int64_t blocks = flat_blocks(total, threads, "scalar_theta");
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(u.scalar_type(), "scalar_theta_cuda", [&] {
    scalar_theta_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        theta.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), cptr<scalar_t>(f1),
        cptr<scalar_t>(f2), cptr<scalar_t>(f3), cptr<scalar_t>(a1),
        cptr<scalar_t>(a2), cptr<scalar_t>(a3), vol.data_ptr<scalar_t>(),
        scalar_t(dt), scalar_t(keep), box, dims_of(u), total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scalar_theta_scale_cuda(torch::Tensor const& theta, torch::Tensor& f1,
                             torch::Tensor& f2, torch::Tensor& f3,
                             at::native::ScalarBox box) {
  at::cuda::CUDAGuard device_guard(theta.device());
  check_operands("scalar_theta_scale", theta, {&f1, &f2, &f3});
  TORCH_CHECK(theta.dim() == 4 && theta.is_contiguous(),
              "scalar_theta_scale: contiguous (nvar, nc3, nc2, nc1) theta");
  auto n = dims_of(theta);
  check_box(box, n, f1, f2, f3);
  int64_t total = theta.numel();
  if (total == 0) return;
  int threads = 256;
  int64_t blocks = flat_blocks(total, threads, "scalar_theta_scale");
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      theta.scalar_type(), "scalar_theta_scale_cuda", [&] {
        auto run = [&](torch::Tensor& f, int lo, int hi, int64_t stride,
                       int64_t extent) {
          if (!f.defined()) return;
          TORCH_CHECK(f.is_contiguous() && f.sizes() == theta.sizes(),
                      "scalar_theta_scale: fluxes shaped like theta expected");
          scalar_theta_scale_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              theta.data_ptr<scalar_t>(), f.data_ptr<scalar_t>(), lo, hi,
              stride, extent, total);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        };
        run(f1, box.il, box.iu + 1, 1, n.nc1);
        run(f2, box.jl, box.ju + 1, n.nc1, n.nc2);
        run(f3, box.kl, box.ku + 1, static_cast<int64_t>(n.nc1) * n.nc2, n.nc3);
      });
}

void scalar_divergence_cuda(torch::Tensor& div, torch::Tensor& ds,
                            torch::Tensor const& f1, torch::Tensor const& f2,
                            torch::Tensor const& f3, torch::Tensor const& a1,
                            torch::Tensor const& a2, torch::Tensor const& a3,
                            torch::Tensor const& vol, double dt,
                            at::native::ScalarBox box) {
  at::cuda::CUDAGuard device_guard(div.device());
  check_operands("scalar_divergence", div, {&ds, &vol});
  check_fused_shapes(div, a1, a2, a3, vol);
  TORCH_CHECK(ds.is_contiguous() && ds.sizes() == div.sizes(),
              "scalar_divergence: ds shaped like div expected");
  check_fluxes(div, f1, f2, f3, a1, a2, a3);
  check_box(box, dims_of(div), f1, f2, f3);
  int64_t total = div.numel();
  if (total == 0) return;
  int threads = 256;
  int64_t blocks = flat_blocks(total, threads, "scalar_divergence");
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(div.scalar_type(), "scalar_divergence_cuda", [&] {
    scalar_divergence_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        div.data_ptr<scalar_t>(), ds.data_ptr<scalar_t>(), cptr<scalar_t>(f1),
        cptr<scalar_t>(f2), cptr<scalar_t>(f3), cptr<scalar_t>(a1),
        cptr<scalar_t>(a2), cptr<scalar_t>(a3), vol.data_ptr<scalar_t>(),
        scalar_t(-dt), box, dims_of(div), total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_scalar_lateral_flux,
                       &snap::scalar_lateral_flux_cuda);
REGISTER_CUDA_DISPATCH(call_scalar_weno5, &snap::scalar_weno5_cuda);
REGISTER_CUDA_DISPATCH(call_scalar_implicit_transfer,
                       &snap::scalar_implicit_transfer_cuda);
REGISTER_CUDA_DISPATCH(call_scalar_average3, &snap::scalar_average3_cuda);
REGISTER_CUDA_DISPATCH(call_scalar_carry_dry, &snap::scalar_carry_dry_cuda);
REGISTER_CUDA_DISPATCH(call_scalar_theta, &snap::scalar_theta_cuda);
REGISTER_CUDA_DISPATCH(call_scalar_theta_scale, &snap::scalar_theta_scale_cuda);
REGISTER_CUDA_DISPATCH(call_scalar_divergence, &snap::scalar_divergence_cuda);

}  // namespace at::native
