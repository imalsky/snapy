// The fused implicit x1 tracer transfer (call_scalar_implicit_transfer)
// against the libtorch body of MeshBlockImpl::advance_local's transfer with
// its donor cap (at::native::donor_cap), bit for bit, in fp64, on columns
// where the cap binds (counted), with empty donors, a held amount clamped at
// 0, NaN and signed zeros.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <limits>
#include <snap/implicit/implicit_dispatch.hpp>
#include <snap/scalar/scalar_fused.hpp>

#include "cuda_test_gate.hpp"

namespace {

// the default path: advance_local's transfer; P is returned capped
torch::Tensor reference(torch::Tensor ds, torch::Tensor s, torch::Tensor r,
                        torch::Tensor M, torch::Tensor vol, int il, int iu,
                        torch::Tensor* P_out) {
  int nc1 = r.size(-1);
  auto r_below = torch::zeros_like(r);
  r_below.slice(-1, 1, nc1) = r.slice(-1, 0, nc1 - 1);
  auto P = torch::where(M > 0., r_below, r) * M;
  auto held = (s + ds).clamp_min(0.) * vol;
  at::native::donor_cap(P.device().type(), P, held, il, iu);
  auto P_above = torch::zeros_like(P);
  P_above.slice(-1, 0, nc1 - 1) = P.slice(-1, 1, nc1);
  *P_out = P;
  return ds.clone().add_((P - P_above) / vol);
}

bool same_bits(torch::Tensor a, torch::Tensor b) {
  return torch::equal(a.cpu().contiguous().view(torch::kInt64),
                      b.cpu().contiguous().view(torch::kInt64));
}

}  // namespace

TEST(scalar_fused, implicit_transfer_matches_libtorch_bitwise) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "no CUDA device";
  }
  torch::Device dev(torch::kCUDA, 0);
  double const nan = std::numeric_limits<double>::quiet_NaN();
  int64_t nvar = 4, n3 = 2, n2 = 3, n1 = 14;
  int il = 3, iu = n1 - 4;
  torch::manual_seed(11);
  auto rnd = [&](std::vector<int64_t> shape) {
    return torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat64))
        .to(dev);
  };
  auto vol = rnd({n3, n2, n1}).abs() + 0.5;
  auto s = rnd({nvar, n3, n2, n1}).abs();
  auto r = s / (rnd({n3, n2, n1}).abs() + 1.);
  // stage increments up to the cell's content: some cells hold nothing
  auto ds = 0.3 * rnd({nvar, n3, n2, n1}) * s;
  ds[0][0][0][5] = -2. * s[0][0][0][5].item<double>();
  // transfers of a few cell masses, both signs, so donors overdraw
  auto M = 3. * rnd({n3, n2, n1}) * vol;
  // empty donors, signed zeros, NaN
  r[1][0][1].slice(0, 4, 8).zero_();
  s[1][0][1].slice(0, 4, 8).zero_();
  r[2][1][2][6] = -0.;
  M[0][2][7] = 0.;
  M[1][1][8] = -0.;
  r[3][1][0][9] = nan;

  torch::Tensor P;
  auto want = reference(ds, s, r, M, vol, il, iu, &P);
  auto got = ds.clone();
  at::native::call_scalar_implicit_transfer(
      dev.type(), got, s, r, M.contiguous(), vol.contiguous(), il, iu);
  EXPECT_TRUE(same_bits(want, got));

  // the cap must have bound on some faces, or this compares nothing new
  int nc1 = r.size(-1);
  auto r_below = torch::zeros_like(r);
  r_below.slice(-1, 1, nc1) = r.slice(-1, 0, nc1 - 1);
  auto P_raw = torch::where(M > 0., r_below, r) * M;
  auto bound = (P_raw != P) & P_raw.isfinite();
  int64_t nbound = bound.sum().item<int64_t>();
  EXPECT_GT(nbound, 20) << "faces where the donor cap binds";
}
