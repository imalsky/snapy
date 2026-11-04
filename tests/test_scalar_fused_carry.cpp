// The fused carry of a dry-density source into the scalars
// (call_scalar_carry_dry) against the libtorch expressions of
// MeshBlockImpl::advance_local's carry_dry_source, bit for bit, on values
// that reach every arm: negative, positive, +0, -0 and NaN dry increments,
// zero (both signs), NaN and ordinary stage densities, and NaN, Inf and signed
// zeros in the scalars; with the base passed separately and as ds itself
// (the native call site), in fp64.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <limits>
#include <snap/scalar/scalar_fused.hpp>

#include "cuda_test_gate.hpp"

namespace {

// the default path: carry_dry_source's libtorch body
torch::Tensor reference(torch::Tensor ds, torch::Tensor s0, torch::Tensor s,
                        torch::Tensor base, torch::Tensor r,
                        torch::Tensor rho_base, torch::Tensor dry, double w0,
                        double w1, double w2) {
  auto s_base = w0 * s0 + w1 * s + w2 * base;
  auto remove_r = torch::where(rho_base.unsqueeze(0) == 0., r,
                               s_base / rho_base.unsqueeze(0));
  auto carry_r = torch::where(dry.unsqueeze(0) < 0., remove_r, r);
  return ds.clone().add_(carry_r * dry.unsqueeze(0));
}

bool same_bits(torch::Tensor a, torch::Tensor b) {
  return torch::equal(a.cpu().contiguous().view(torch::kInt64),
                      b.cpu().contiguous().view(torch::kInt64));
}

}  // namespace

TEST(scalar_fused, carry_dry_matches_libtorch_bitwise) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "no CUDA device";
  }
  torch::Device dev(torch::kCUDA, 0);
  auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(dev);
  double const nan = std::numeric_limits<double>::quiet_NaN();
  double const inf = std::numeric_limits<double>::infinity();
  int64_t nvar = 5, n3 = 2, n2 = 3, n1 = 8;
  torch::manual_seed(3);
  auto rnd = [&](std::vector<int64_t> shape) {
    return torch::randn(shape, torch::TensorOptions().dtype(torch::kFloat64))
        .to(dev);
  };
  auto ds = rnd({nvar, n3, n2, n1}), s0 = rnd({nvar, n3, n2, n1}).abs(),
       s = rnd({nvar, n3, n2, n1}).abs(), base = rnd({nvar, n3, n2, n1}),
       r = rnd({nvar, n3, n2, n1}).abs();
  // special values in the scalars
  s[0][0][0][1] = nan;
  s[1][0][1][2] = inf;
  s0[2][1][2][3] = -0.;
  r[3][1][0][4] = nan;
  r[4][0][2][5] = -0.;
  ds[0][1][1][6] = -0.;
  // the dry increment along x1: < 0, > 0, +0, -0, NaN, < 0, > 0, < 0
  auto dry = torch::empty({n3, n2, n1}, opts);
  double d[8] = {-0.3, 0.2, 0., -0., nan, -1e-300, 1e300, -2.};
  for (int i = 0; i < n1; ++i) dry.select(2, i).fill_(d[i]);
  // the stage density: ordinary, with +0, -0 and NaN entries where dry < 0
  auto rho_base = rnd({n3, n2, n1}).abs() + 0.5;
  rho_base[0][0][0] = 0.;
  rho_base[1][2][0] = -0.;
  rho_base[0][1][7] = nan;
  rho_base[1][1][5] = 0.;
  double w0 = 0.75, w1 = 0.25, w2 = 0.25;

  // base passed separately (the stage-forcing call site)
  auto want = reference(ds, s0, s, base, r, rho_base, dry, w0, w1, w2);
  auto got = ds.clone();
  at::native::call_scalar_carry_dry(dev.type(), got, s0, s, base, r, rho_base,
                                    dry, w0, w1, w2);
  EXPECT_TRUE(same_bits(want, got)) << "separate base";

  // base aliasing ds (the native call site)
  auto want2 = reference(ds, s0, s, ds, r, rho_base, dry, w0, w1, w2);
  auto got2 = ds.clone();
  at::native::call_scalar_carry_dry(dev.type(), got2, s0, s, got2, r, rho_base,
                                    dry, w0, w1, w2);
  EXPECT_TRUE(same_bits(want2, got2)) << "base aliasing ds";

  // the RK-stage weights of the first stage (w2 = 1) as well
  auto want3 = reference(ds, s0, s, base, r, rho_base, dry, 0., 1., 1.);
  auto got3 = ds.clone();
  at::native::call_scalar_carry_dry(dev.type(), got3, s0, s, base, r, rho_base,
                                    dry, 0., 1., 1.);
  EXPECT_TRUE(same_bits(want3, got3)) << "weights 0, 1, 1";
}
