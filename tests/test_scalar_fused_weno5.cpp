// The fused scalar WENO5 kernel (call_scalar_weno5), whose weights take fewer
// divisions than the default arithmetic, against the default reconstruction
// of a scalar (Reconstruct without a hydro: WENO5, edge copies, clamp at 0) on
// lines that stress that arithmetic: values over many decades, constants down
// to the subnormals and up to 1e300, zero, fronts of large contrast, a huge
// spike, nearly
// constant lines (smoothness indicators near the 1e-6 epsilon), oscillation,
// negative values, NaN and Inf. Along x1, x2 and x3, as face states, on lines
// up to 1024 cells and longer; fp64 with and without scaling, and fp32 with
// subnormal values. Each finite value must agree to round-off relative to the
// largest finite input within 4 cells along the line (so a low plateau beside
// a front is judged on its own scale), and NaN and Inf must appear where the
// default has them, except on the 1e300 plateau, where the default formula
// overflows and the kernel must return the plateau.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <snap/recon/interpolation.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/scalar/scalar_fused.hpp>
#include <string>
#include <vector>

#include "cuda_test_gate.hpp"

using namespace snap;

namespace {

constexpr int NVAR = 17;
constexpr int HUGE_VAR = 16;  // a constant 1e300 plateau

torch::Tensor make_lines(int n, int dim) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  std::vector<int64_t> shape = {NVAR, 3, 2, 2};
  shape[dim] = n;
  auto w = torch::empty(shape, opts);
  std::vector<int64_t> vshape = {1, 1, 1};
  vshape[dim - 1] = n;
  auto x = torch::arange(n, opts).view(vshape).expand(w[0].sizes());
  auto one = torch::ones_like(x);
  auto mid = static_cast<double>(n / 2);
  w[0] = 1.5 + torch::sin(x / 3.0);
  w[1] = 1e-20 * (1.5 + torch::sin(x / 2.0));
  w[2] = one;
  w[3] = 1e-200 * one;
  w[4] = 1e-305 * one;
  w[5] = 1e-310 * one;  // subnormal
  w[6] = 1e-320 * one;  // subnormal
  w[7] = 0. * one;
  w[8] = torch::where(x < mid, one, 1e-30 * one);
  w[9] = torch::where(x < mid, 1e-2 * one, 1e-300 * one);
  w[10] = torch::where(x == mid, 1e303 * one, 0. * one);
  w[11] = 1. + 1e-3 * torch::sin(x);  // beta near the epsilon
  w[12] = 1. + 3e-4 * torch::cos(2.3 * x);
  w[13] = torch::where(torch::remainder(x, 2.) == 0., one, 1e-8 * one);
  w[14] = torch::sin(x / 1.7);  // negative values
  w[15] = torch::where(
      x == 3., std::numeric_limits<double>::quiet_NaN() * one,
      torch::where(x == n - 5., std::numeric_limits<double>::infinity() * one,
                   one));
  w[HUGE_VAR] = 1e300 * one;
  return w;
}

// the largest finite |w| within r cells along dim (cut at the line's ends)
torch::Tensor local_max(torch::Tensor w, int dim, int r) {
  auto m = torch::where(torch::isfinite(w), w.abs(), torch::zeros_like(w));
  auto out = m.clone();
  int64_t n = w.size(dim);
  for (int s = 1; s <= r && s < n; ++s) {
    auto lo = out.narrow(dim, 0, n - s);
    lo.copy_(torch::maximum(lo, m.narrow(dim, s, n - s)));
    auto hi = out.narrow(dim, s, n - s);
    hi.copy_(torch::maximum(hi, m.narrow(dim, 0, n - s)));
  }
  return out;
}

// b against the reference a, variable by variable: |b - a| <= tol * scale;
// variable `skip` is not compared
void expect_close(torch::Tensor a, torch::Tensor b, torch::Tensor scale,
                  double tol, std::string const& what, int skip = -1) {
  a = a.cpu().to(torch::kFloat64);
  b = b.cpu().to(torch::kFloat64);
  scale = scale.cpu().to(torch::kFloat64).expand_as(a);
  ASSERT_EQ(a.sizes(), b.sizes()) << what;
  int vdim = a.dim() == 5 ? 1 : 0;
  for (int v = 0; v < NVAR; ++v) {
    if (v == skip) continue;
    auto av = a.select(vdim, v), bv = b.select(vdim, v),
         sv = scale.select(vdim, v);
    EXPECT_TRUE(torch::equal(torch::isnan(av), torch::isnan(bv)))
        << what << " variable " << v
        << ": NaN where the default has none, or the reverse";
    EXPECT_TRUE(torch::equal(torch::isinf(av), torch::isinf(bv)))
        << what << " variable " << v << ": Inf pattern differs";
    auto fin = torch::isfinite(av) & torch::isfinite(bv);
    if (!fin.any().item<bool>()) continue;
    auto err = (bv - av).abs().masked_select(fin) /
               sv.masked_select(fin).clamp_min(1e-320);
    double worst = err.max().item<double>();
    EXPECT_LE(worst, tol) << what << " variable " << v << ": difference "
                          << worst << " of the local largest value";
  }
}

void check(Reconstruct rec, torch::Tensor w, int dim, bool scale, double tol,
           std::string const& tag) {
  auto weno5 = std::dynamic_pointer_cast<Weno5InterpImpl>(rec->pinterp1);
  ASSERT_TRUE(weno5) << "the reconstruction is not WENO5";
  auto dev = w.device();
  auto lmax = local_max(w, dim, 4);
  auto ref = rec->forward(w, dim);
  auto faces = torch::empty_like(ref);
  at::native::call_scalar_weno5(dev.type(), faces, w, weno5->cm, weno5->cp,
                                scale, dim);
  // with scaling the default formula overflows on the 1e300 plateau (the
  // scale times the unnormalised weighted sum); the kernel forms the quotient
  // first, so its faces must be the plateau itself
  int skip = scale && w.scalar_type() == torch::kFloat64 ? HUGE_VAR : -1;
  expect_close(ref, faces, lmax.unsqueeze(0), tol, tag + " faces", skip);
  if (skip >= 0) {
    auto huge = faces.select(1, HUGE_VAR).cpu();
    EXPECT_TRUE(torch::isfinite(huge).all().item<bool>()) << tag;
    EXPECT_LE(((huge - 1e300).abs() / 1e300).max().item<double>(), tol) << tag;
  }
}

Reconstruct make_rec(bool scale, torch::Device dev, torch::Dtype dtype) {
  auto op = ReconstructOptionsImpl::create();
  op->interp()->type("weno5").scale(scale);
  Reconstruct rec(op);
  rec->to(dev, dtype);
  return rec;
}

}  // namespace

TEST(scalar_fused, weno5_matches_default_reconstruction) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "no CUDA device";
  }
  torch::Device dev(torch::kCUDA, 0);
  for (bool scale : {true, false}) {
    auto rec = make_rec(scale, dev, torch::kFloat64);
    for (int dim : {3, 2, 1}) {
      std::vector<int> ns = {20};
      if (dim == 3) ns = {20, 1024, 1025, 1030};
      if (dim == 2) ns = {20, 1030};
      for (int n : ns) {
        auto w = make_lines(n, dim).to(dev);
        check(rec, w, dim, scale, 1e-12,
              "fp64 scale " + std::to_string(scale) + " dim " +
                  std::to_string(dim) + " n " + std::to_string(n));
      }
    }
  }
}

TEST(scalar_fused, weno5_fp32_subnormal_scale) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "no CUDA device";
  }
  torch::Device dev(torch::kCUDA, 0);
  auto rec = make_rec(true, dev, torch::kFloat32);
  for (int dim : {3, 2}) {
    auto w = make_lines(20, dim).to(torch::kFloat32);
    w[3].fill_(1e-40);  // fp32 subnormals: 1 / mean would overflow
    w[4].fill_(3e-39);
    w[5].fill_(1e-38);
    w[6].fill_(2e-30);
    check(rec, w.to(dev), dim, true, 1e-5, "fp32 dim " + std::to_string(dim));
  }
}

// The kernels write out the uniform-grid tables, so they must refuse a
// reconstruction whose left (cm) or right (cp) table is any other: the
// default path reads both, and a changed cp moved its result while the fused
// result stayed put.
TEST(scalar_fused, weno5_refuses_other_tables) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "no CUDA device";
  }
  torch::Device dev(torch::kCUDA, 0);
  auto rec = make_rec(true, dev, torch::kFloat64);
  auto weno5 = std::dynamic_pointer_cast<Weno5InterpImpl>(rec->pinterp1);
  ASSERT_TRUE(weno5);
  auto w = make_lines(20, 3).to(dev);
  for (bool right : {false, true}) {
    auto cm = weno5->cm.clone();
    auto cp = weno5->cp.clone();
    (right ? cp : cm)[0][2] += 0.125;
    auto faces = torch::empty({2, NVAR, 3, 2, 20}, w.options());
    EXPECT_THROW(
        at::native::call_scalar_weno5(dev.type(), faces, w, cm, cp, true, 3),
        c10::Error)
        << (right ? "cp" : "cm") << " changed, face states";
  }
}
