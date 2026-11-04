// C/C++
#include <limits>

// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include "scalar_fused.hpp"

// the WENO5 weights of the default path
#include <snap/recon/interp_impl.cuh>

namespace snap {

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
  TORCH_CHECK(rl.is_contiguous() && rr.is_contiguous() &&
                  flux.is_contiguous() && vel.is_contiguous(),
              "scalar_lateral_flux: contiguous tensors expected");
  TORCH_CHECK(
      rl.dim() == 4 && vel.dim() == 3 && vel.numel() * rl.size(0) == rl.numel(),
      "scalar_lateral_flux: shapes (nvar, nc3, nc2, nc1) and "
      "(nc3, nc2, nc1) expected");
  bool cov = !factors.empty();
  bool centroid = cov && factors[3].defined();
  int64_t total = rl.numel();
  int64_t ncell = vel.numel();
  int n1 = static_cast<int>(rl.size(3));
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
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

// One thread per face element (scalar, k, j, i). Along the reconstruction
// dimension (size n, element stride s, ghosts il = 3, iu = n - 3) the default
// path stores right() of cell p - 1 as the left state of face p and left() of
// cell p as its right state, for faces il..iu+1 and il-1..iu, then copies the
// edge values outward: the cells used are clamp(p - 1, il - 1, iu - 1) and
// clamp(p, il, iu). Used only for the x1 upwind flux of columns longer than
// 1024 cells, with the default path's arithmetic (bit for bit).
template <typename T, bool UPWIND>
__global__ void scalar_weno5_kernel(T* __restrict__ out,
                                    T const* __restrict__ w,
                                    T const* __restrict__ cm,
                                    T const* __restrict__ cp, bool scale,
                                    int64_t stride, int n, int64_t total,
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
  if (UPWIND) {
    T v = vel[idx % ncell];
    T u = v > T(0) ? T(1) : T(0);
    out[idx] = rmul(v, radd(rmul(u, wl), rmul(rsub(T(1), u), wr)));
  } else {
    out[idx] = wl;          // ILT
    out[total + idx] = wr;  // IRT
  }
}

// Both WENO5 values of one cell (left(): cm, the state at its lower face;
// right(): cp = cm with its columns reversed, the state at its upper face)
// from one stencil: the formula of interp_shared_weno5_coeff_impl with fewer
// divisions, so the values differ from the default path at round-off.
// - The two share vscale, phi and the smoothness indicators: cp's rows 3, 4
//   are cm's rows 7, 8 and the reverse, and cp's row 6 is minus cm's row 6, so
//   right()'s beta0, beta1, beta2 are left()'s beta2, beta1, beta0.
// - phi = src * (1 / vscale), except where 1 / vscale would overflow.
// - The weights alpha_k = c_k / d_k, d_k = (beta_k + 1e-6)^2, are ratios, so
//   they are multiplied through by d0 d1 d2 (with |phi| summing to 5, every
//   d_j d_k lies in [1e-24, 2e9]); one reciprocal of the two weight sums
//   serves both faces (fp64), and the quotient is formed before the scale is
//   applied, so tiny and huge values stay in range. Divisions by variables
//   go from 12 to 2 (besides the mean's / 5).
// - Without scaling (scale false) the products are not bounded: the default
//   formula is used.
#define SQR(x) ((x) * (x))  // as interp_impl.cuh, which undefines it
template <typename T>
__device__ __forceinline__ void weno5_pair(T const* src, T const* cm,
                                           T const* cp, bool scale, T& left,
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
  T beta0 = 13. / 12. * SQR(_vvdot<5>(phi, cm + 15)) +
            .25 * SQR(_vvdot<5>(phi, cm + 20));
  T beta1 = 13. / 12. * SQR(_vvdot<5>(phi, cm + 25)) +
            .25 * SQR(_vvdot<5>(phi, cm + 30));
  T beta2 = 13. / 12. * SQR(_vvdot<5>(phi, cm + 35)) +
            .25 * SQR(_vvdot<5>(phi, cm + 40));
  T d0 = SQR(beta0 + 1e-6), d1 = SQR(beta1 + 1e-6), d2 = SQR(beta2 + 1e-6);
  if (!scale) {
    T a1 = .6 / d1;
    T p0 = _vvdot<5>(phi, cm), p1 = _vvdot<5>(phi, cm + 5),
      p2 = _vvdot<5>(phi, cm + 10);
    T a0 = .3 / d0, a2 = .1 / d2;
    left = vscale * (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2);
    p0 = _vvdot<5>(phi, cp), p1 = _vvdot<5>(phi, cp + 5),
    p2 = _vvdot<5>(phi, cp + 10);
    a0 = .3 / d2, a2 = .1 / d0;
    right = vscale * (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2);
    return;
  }
  T e0 = d1 * d2, e1 = d0 * d2, e2 = d0 * d1;
  T a1 = .6 * e1;
  T nl, dl, nr, dr;
  {
    T p0 = _vvdot<5>(phi, cm), p1 = _vvdot<5>(phi, cm + 5),
      p2 = _vvdot<5>(phi, cm + 10);
    T a0 = .3 * e0, a2 = .1 * e2;
    nl = a0 * p0 + a1 * p1 + a2 * p2;
    dl = a0 + a1 + a2;
  }
  {
    T p0 = _vvdot<5>(phi, cp), p1 = _vvdot<5>(phi, cp + 5),
      p2 = _vvdot<5>(phi, cp + 10);
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
                                          T const* __restrict__ w,
                                          T const* __restrict__ cm,
                                          T const* __restrict__ cp, bool scale,
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
  weno5_pair(src, cm, cp, scale, l, r);
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
__global__ void scalar_weno5_x1_flux_kernel(
    T* __restrict__ flux, T const* __restrict__ w, T const* __restrict__ cm,
    T const* __restrict__ cp, bool scale, int n, int64_t ncolumn_cell,
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
    weno5_pair(src, cm, cp, scale, l, r);
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
  TORCH_CHECK(w.dim() == 4 && w.is_contiguous() && faces.is_contiguous(),
              "scalar_weno5: contiguous (nvar, nc3, nc2, nc1) expected");
  TORCH_CHECK(dim >= 1 && dim <= 3, "scalar_weno5: dim 1, 2 or 3");
  TORCH_CHECK(cm.numel() == 45 && cp.numel() == 45 && cm.is_contiguous() &&
                  cp.is_contiguous(),
              "scalar_weno5: 45 contiguous WENO5 coefficients expected");
  int n = static_cast<int>(w.size(dim));
  TORCH_CHECK(n >= 6, "scalar_weno5: at least 6 cells along the line");
  int64_t stride = w.stride(dim);
  int64_t total = w.numel();
  int64_t ncell = total / w.size(0);
  bool upwind = vel.defined();
  if (upwind) {
    TORCH_CHECK(dim == 3 && vel.is_contiguous() && vel.numel() == ncell &&
                    faces.sizes() == w.sizes(),
                "scalar_weno5: the upwind form is x1 only");
  } else {
    TORCH_CHECK(faces.numel() == 2 * total, "scalar_weno5: (2, ...) faces");
  }
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "scalar_weno5_cuda", [&] {
    auto cmp = cm.to(w.scalar_type());
    auto cpp = cp.to(w.scalar_type());
    if (upwind && n <= 1024) {
      int64_t columns = total / n;
      TORCH_CHECK(columns <= std::numeric_limits<int>::max(),
                  "scalar_weno5: too many x1 columns for one grid");
      size_t shared = 2 * static_cast<size_t>(n) * sizeof(scalar_t);
      scalar_weno5_x1_flux_kernel<scalar_t><<<columns, n, shared, stream>>>(
          faces.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
          cmp.data_ptr<scalar_t>(), cpp.data_ptr<scalar_t>(), scale, n,
          ncell / n, vel.data_ptr<scalar_t>());
    } else if (upwind) {
      scalar_weno5_kernel<scalar_t, true><<<blocks, threads, 0, stream>>>(
          faces.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
          cmp.data_ptr<scalar_t>(), cpp.data_ptr<scalar_t>(), scale, stride, n,
          total, ncell, vel.data_ptr<scalar_t>());
    } else {
      scalar_weno5_faces_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          faces.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
          cmp.data_ptr<scalar_t>(), cpp.data_ptr<scalar_t>(), scale, stride, n,
          total);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
__global__ void scalar_implicit_transfer_kernel(
    T* __restrict__ ds, T const* __restrict__ r, T const* __restrict__ m,
    T const* __restrict__ vol, int n1, int64_t ncell, int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  int64_t c = idx % ncell;
  int i = static_cast<int>(c % n1);
  // P at face i and at face i + 1 (0 past the top)
  T mi = m[c];
  T p0 = rmul(mi > T(0) ? (i >= 1 ? r[idx - 1] : T(0)) : r[idx], mi);
  T p1 = T(0);
  if (i + 1 < n1) {
    T mj = m[c + 1];
    p1 = rmul(mj > T(0) ? r[idx] : r[idx + 1], mj);
  }
  ds[idx] = radd(ds[idx], rdiv(rsub(p0, p1), vol[c]));
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

void scalar_implicit_transfer_cuda(torch::Tensor& ds, torch::Tensor const& r,
                                   torch::Tensor const& m,
                                   torch::Tensor const& vol) {
  at::cuda::CUDAGuard device_guard(ds.device());
  TORCH_CHECK(ds.is_contiguous() && r.is_contiguous() && m.is_contiguous() &&
                  vol.is_contiguous() && ds.sizes() == r.sizes() &&
                  m.numel() * r.size(0) == r.numel() &&
                  vol.numel() == m.numel(),
              "scalar_implicit_transfer: contiguous (nvar, nc3, nc2, nc1) and "
              "(nc3, nc2, nc1) expected");
  int64_t total = r.numel();
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      r.scalar_type(), "scalar_implicit_transfer_cuda", [&] {
        scalar_implicit_transfer_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                ds.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(),
                m.data_ptr<scalar_t>(), vol.data_ptr<scalar_t>(),
                static_cast<int>(r.size(3)), m.numel(), total);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scalar_average3_cuda(torch::Tensor& out, torch::Tensor const& u0,
                          torch::Tensor const& u1, torch::Tensor const& u2,
                          double w0, double w1, double w2) {
  at::cuda::CUDAGuard device_guard(out.device());
  TORCH_CHECK(out.is_contiguous() && u0.is_contiguous() && u1.is_contiguous() &&
                  u2.is_contiguous() && u0.sizes() == out.sizes() &&
                  u1.sizes() == out.sizes() && u2.sizes() == out.sizes(),
              "scalar_average3: contiguous tensors of one shape expected");
  int64_t total = out.numel();
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(out.scalar_type(), "scalar_average3_cuda", [&] {
    scalar_average3_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        out.data_ptr<scalar_t>(), u0.data_ptr<scalar_t>(),
        u1.data_ptr<scalar_t>(), u2.data_ptr<scalar_t>(), scalar_t(w0),
        scalar_t(w1), scalar_t(w2), total);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
Dims dims_of(torch::Tensor const& u) {
  return Dims{static_cast<int>(u.size(3)), static_cast<int>(u.size(2)),
              static_cast<int>(u.size(1))};
}
void check_fused_shapes(torch::Tensor const& u, torch::Tensor const& a1,
                        torch::Tensor const& a2, torch::Tensor const& a3,
                        torch::Tensor const& vol) {
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
  check_fused_shapes(u, a1, a2, a3, vol);
  TORCH_CHECK(theta.is_contiguous() && theta.sizes() == u.sizes());
  check_box(box, dims_of(u), f1, f2, f3);
  for (auto const* f : {&f1, &f2, &f3})
    TORCH_CHECK(
        !f->defined() || (f->is_contiguous() && f->sizes() == u.sizes()),
        "fused scalar: fluxes shaped like u expected");
  int64_t total = u.numel();
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
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
  TORCH_CHECK(theta.is_contiguous());
  auto n = dims_of(theta);
  check_box(box, n, f1, f2, f3);
  int64_t total = theta.numel();
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      theta.scalar_type(), "scalar_theta_scale_cuda", [&] {
        auto run = [&](torch::Tensor& f, int lo, int hi, int64_t stride,
                       int64_t extent) {
          if (!f.defined()) return;
          TORCH_CHECK(f.is_contiguous() && f.sizes() == theta.sizes());
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
  check_fused_shapes(div, a1, a2, a3, vol);
  TORCH_CHECK(ds.is_contiguous() && ds.sizes() == div.sizes());
  TORCH_CHECK(f1.defined() || f2.defined() || f3.defined(),
              "At least one flux tensor must be defined");
  check_box(box, dims_of(div), f1, f2, f3);
  for (auto const* f : {&f1, &f2, &f3})
    TORCH_CHECK(
        !f->defined() || (f->is_contiguous() && f->sizes() == div.sizes()),
        "fused scalar: fluxes shaped like the divergence expected");
  int64_t total = div.numel();
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
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
REGISTER_CUDA_DISPATCH(call_scalar_theta, &snap::scalar_theta_cuda);
REGISTER_CUDA_DISPATCH(call_scalar_theta_scale, &snap::scalar_theta_scale_cuda);
REGISTER_CUDA_DISPATCH(call_scalar_divergence, &snap::scalar_divergence_cuda);

}  // namespace at::native
