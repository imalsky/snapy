#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>

#include <cmath>
#include <snap/recon/interpolation.hpp>
#include <snap/recon/x1_coeffs.hpp>
#include <vector>
using namespace snap;

namespace {
enum { DIM3 = 1, DIM2 = 2, DIM1 = 3 };
constexpr int kNvar = 2, kN = 14;
torch::Tensor faces(double q) {
  auto f = torch::empty({kN + 1}, torch::kFloat64);
  auto a = f.accessor<double, 1>();
  double x = 0.3, h = 0.11;
  for (int k = 0; k <= kN; ++k) {
    a[k] = x;
    x += h;
    h *= q;
  }
  return f;
}
torch::Tensor state() {
  auto w = torch::empty({kNvar, kN, kN, kN}, torch::kFloat64);
  auto flat = w.view({-1});
  for (int64_t i = 0; i < flat.numel(); ++i) {
    flat[i] = 1.5 + 0.1 * std::sin(0.37 * static_cast<double>(i));
  }
  return w;
}
torch::Tensor out_for(torch::Tensor const& w, int dim, int stencils) {
  auto vec = w.sizes().vec();
  vec[dim] -= stencils - 1;
  return torch::empty(vec, w.options());
}
std::vector<Interp> installed(torch::Tensor const& x1f) {
  Weno5Interp weno;
  Center5Interp cp5;
  weno->set_x1_coeffs(x1f);
  cp5->set_x1_coeffs(x1f);
  EXPECT_TRUE(weno->has_x1_coeffs());
  EXPECT_TRUE(cp5->has_x1_coeffs());
  return {weno.ptr(), cp5.ptr()};
}
}  // namespace

TEST(X1DimGuard, a_square_block_defeats_the_row_count_guard) {
  auto x1f = faces(1.07);
  auto w = state();
  for (auto const& p : installed(x1f)) {
    auto out = out_for(w, DIM2, p->stencils());
    EXPECT_EQ(out.size(DIM2), x1f.numel() - 5)
        << p->options->type()
        << ": this test is only meaningful while the x2 output and the table "
           "have the same length";
  }
}

TEST(X1DimGuard, left_refuses_x2_on_a_square_block) {
  auto x1f = faces(1.07);
  auto w = state();
  for (auto const& p : installed(x1f)) {
    auto out = out_for(w, DIM2, p->stencils());
    EXPECT_THROW(p->left(w, DIM2, out), c10::Error)
        << p->options->type()
        << ": x1 coefficients were applied along x2 without complaint";
  }
}

TEST(X1DimGuard, right_refuses_x2_on_a_square_block) {
  auto x1f = faces(1.07);
  auto w = state();
  for (auto const& p : installed(x1f)) {
    auto out = out_for(w, DIM2, p->stencils());
    EXPECT_THROW(p->right(w, DIM2, out), c10::Error)
        << p->options->type()
        << ": x1 coefficients were applied along x2 without complaint";
  }
}

TEST(X1DimGuard, forward_refuses_x2_and_x3) {
  auto x1f = faces(1.07);
  auto w = state();
  for (auto const& p : installed(x1f)) {
    EXPECT_THROW(p->forward(w, DIM2), c10::Error) << p->options->type();
    EXPECT_THROW(p->forward(w, DIM3), c10::Error) << p->options->type();
  }
}

TEST(X1DimGuard, x1_still_runs) {
  auto x1f = faces(1.07);
  auto w = state();
  for (auto const& p : installed(x1f)) {
    auto lr = p->forward(w, DIM1);
    EXPECT_EQ(lr.first.size(DIM1), kN - 4) << p->options->type();
    EXPECT_TRUE(torch::isfinite(lr.first).all().item<bool>())
        << p->options->type();
    EXPECT_TRUE(torch::isfinite(lr.second).all().item<bool>())
        << p->options->type();
  }
}

TEST(X1DimGuard, an_uninstalled_interpolator_still_serves_x2) {
  auto w = state();
  Weno5Interp weno;
  Center5Interp cp5;
  std::vector<Interp> bare = {weno.ptr(), cp5.ptr()};
  for (auto const& p : bare) {
    ASSERT_FALSE(p->has_x1_coeffs());
    auto lr = p->forward(w, DIM2);
    EXPECT_EQ(lr.first.size(DIM2), kN - 4) << p->options->type();
    EXPECT_TRUE(torch::isfinite(lr.first).all().item<bool>())
        << p->options->type();
  }
}
