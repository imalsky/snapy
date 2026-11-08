#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <snap/recon/interpolation.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/recon/x1_coeffs.hpp>
#include <vector>

#include "device_testing.hpp"
using namespace snap;

namespace {
constexpr double kEps = 2.220446049250313e-16;
void expect_row_sum(torch::Tensor const& row, char const* what) {
  double s = row.sum().item<double>();
  double scale = row.abs().sum().item<double>();
  EXPECT_LE(std::abs(s - 1.), 8. * kEps * scale)
      << what << ": sum = " << s << ", 8*eps*sum|c| = " << 8. * kEps * scale;
}
torch::Tensor faces_uniform(int n, double h = 1.) {
  return torch::arange(n, torch::kFloat64) * h;
}
torch::Tensor faces_geometric(int n, double q, double h0 = 1.) {
  std::vector<double> y(n);
  y[0] = 0.;
  double h = h0;
  for (int i = 1; i < n; ++i) {
    y[i] = y[i - 1] + h;
    h *= q;
  }
  return torch::tensor(y, torch::kFloat64);
}
torch::Tensor faces_irregular(int n, double h0 = 1., int seed = 7) {
  std::vector<double> y(n);
  y[0] = 0.;
  uint64_t st = static_cast<uint64_t>(seed) + 0x9e3779b97f4a7c15ULL;
  for (int i = 1; i < n; ++i) {
    st = st * 6364136223846793005ULL + 1442695040888963407ULL;
    double r = static_cast<double>((st >> 11) & 0xfffff) / 1048576.;  // [0,1)
    y[i] = y[i - 1] + h0 * (0.55 + 0.9 * r);
  }
  return torch::tensor(y, torch::kFloat64);
}
torch::Tensor cell_avg_poly(torch::Tensor const& x1f,
                            std::vector<double> const& c) {
  auto f = x1f.accessor<double, 1>();
  int64_t n = x1f.numel() - 1;
  std::vector<double> u(n);
  const double g[3] = {-std::sqrt(3. / 5.), 0., std::sqrt(3. / 5.)};
  const double wg[3] = {5. / 18., 8. / 18., 5. / 18.};
  for (int64_t k = 0; k < n; ++k) {
    double a = f[k], b = f[k + 1], acc = 0.;
    for (int i = 0; i < 3; ++i) {
      double x = .5 * (a + b) + .5 * (b - a) * g[i];
      double p = 0.;
      for (int d = static_cast<int>(c.size()) - 1; d >= 0; --d)
        p = p * x + c[d];
      acc += wg[i] * p;
    }
    u[k] = acc;
  }
  return torch::tensor(u, torch::kFloat64);
}
double poly_at(std::vector<double> const& c, double x) {
  double p = 0.;
  for (int d = static_cast<int>(c.size()) - 1; d >= 0; --d) p = p * x + c[d];
  return p;
}
std::vector<double> primitive_row5(std::vector<double> const& z, double xf) {
  const int n = 5;
  std::vector<double> lp(n + 1, 0.);
  for (int j = 0; j <= n; ++j) {
    double s = 0.;
    for (int i = 0; i <= n; ++i) {
      if (i == j) continue;
      double term = 1. / (z[j] - z[i]);
      for (int l = 0; l <= n; ++l) {
        if (l == j || l == i) continue;
        term *= (xf - z[l]) / (z[j] - z[l]);
      }
      s += term;
    }
    lp[j] = s;
  }
  std::vector<double> w(n, 0.);
  for (int k = 0; k < n; ++k) {
    double acc = 0.;
    for (int j = k + 1; j <= n; ++j) acc += lp[j];
    w[k] = (z[k + 1] - z[k]) * acc;
  }
  return w;
}
}  // namespace

TEST(X1Coeffs, uniform_matches_shipped_rows) {
  Weno5Interp weno;
  Center5Interp cp5;
  for (double h : {1., 1.e-3, 7.5e3}) {
    auto x1f = faces_uniform(12, h);
    auto w = weno5_x1_coeffs(x1f);
    auto p = cp5_x1_coeffs(x1f);
    auto shipped_m = weno->cm.reshape({45});
    auto shipped_p = weno->cp.reshape({45});
    for (int64_t m = 0; m < w.first.size(0); ++m) {
      auto cm45 = w.first[m].slice(0, 0, 45);
      auto cp45 = w.second[m].slice(0, 0, 45);
      EXPECT_LT((cm45 - shipped_m).abs().max().item<double>(), 1.e-14)
          << "h = " << h << ", row " << m;
      EXPECT_LT((cp45 - shipped_p).abs().max().item<double>(), 1.e-14)
          << "h = " << h << ", row " << m;
      auto d = w.first[m].slice(0, 45, 48);
      EXPECT_NEAR(d[0].item<double>(), .3, 1.e-15);
      EXPECT_NEAR(d[1].item<double>(), .6, 1.e-15);
      EXPECT_NEAR(d[2].item<double>(), .1, 1.e-15);
      EXPECT_LT((p.first[m] - cp5->cm).abs().max().item<double>(), 1.e-15);
      EXPECT_LT((p.second[m] - cp5->cp).abs().max().item<double>(), 1.e-15);
    }
  }
}

TEST(X1Coeffs, row_sums) {
  std::vector<std::pair<char const*, torch::Tensor>> grids = {
      {"uniform", faces_uniform(16)},
      {"geometric q=1.05", faces_geometric(16, 1.05)},
      {"geometric q=0.94080", faces_geometric(16, 0.94080)},
      {"geometric q=1.2", faces_geometric(16, 1.2)},
  };
  {
    std::vector<double> y{0.};
    for (int i = 0; i < 8; ++i) y.push_back(y.back() + 1.);
    for (int i = 0; i < 7; ++i) y.push_back(y.back() + 4.);
    grids.emplace_back("jump 1:4", torch::tensor(y, torch::kFloat64));
  }
  for (auto const& [name, x1f] : grids) {
    auto w = weno5_x1_coeffs(x1f);
    auto p = cp5_x1_coeffs(x1f);
    for (int64_t m = 0; m < w.first.size(0); ++m) {
      for (auto const& t : {w.first, w.second}) {
        for (int r = 0; r < 3; ++r) {
          expect_row_sum(t[m].slice(0, 5 * r, 5 * r + 5), name);
        }
        auto d = t[m].slice(0, 45, 48);
        EXPECT_GT(d.min().item<double>(), 0.) << name << " row " << m;
        expect_row_sum(d, name);
      }
      expect_row_sum(p.first[m], name);
      expect_row_sum(p.second[m], name);
    }
  }
}

TEST(X1Coeffs, cp5_row_matches_an_independent_derivation) {
  double worst = 0.;
  for (double q : {1.0, 1.05, 0.94080, 1.2, 0.8, 1.6}) {
    auto x1f = faces_geometric(16, q);
    auto p = cp5_x1_coeffs(x1f);
    auto f = x1f.accessor<double, 1>();
    for (int64_t m = 0; m < p.first.size(0); ++m) {
      std::vector<double> y(6);
      for (int k = 0; k < 6; ++k) y[k] = f[m + k];
      double h2 = y[3] - y[2];
      std::vector<double> z(6);
      for (int k = 0; k < 6; ++k) z[k] = (y[k] - y[3]) / h2;
      auto ref_p = primitive_row5(z, 0.);
      std::vector<double> ym(6), zm(6);
      for (int k = 0; k < 6; ++k) ym[k] = -y[5 - k];
      double h2m = ym[3] - ym[2];
      for (int k = 0; k < 6; ++k) zm[k] = (ym[k] - ym[3]) / h2m;
      auto t = primitive_row5(zm, 0.);
      std::vector<double> ref_m(5);
      for (int n = 0; n < 5; ++n) ref_m[n] = t[4 - n];
      for (int n = 0; n < 5; ++n) {
        worst =
            std::max(worst, std::abs(p.first[m][n].item<double>() - ref_m[n]));
        worst =
            std::max(worst, std::abs(p.second[m][n].item<double>() - ref_p[n]));
        EXPECT_NEAR(p.first[m][n].item<double>(), ref_m[n], 1.e-13)
            << "q = " << q << ", row " << m << ", cm col " << n;
        EXPECT_NEAR(p.second[m][n].item<double>(), ref_p[n], 1.e-13)
            << "q = " << q << ", row " << m << ", cp col " << n;
      }
    }
  }
  std::cout << "  max|sum_r d_r p_r - 6-node primitive row| = " << worst
            << std::endl;
}

TEST(X1Coeffs, polynomial_exactness_and_face_index) {
  auto x1f = faces_geometric(18, 1.1, 0.37);
  auto w = weno5_x1_coeffs(x1f);
  auto p = cp5_x1_coeffs(x1f);
  auto f = x1f.accessor<double, 1>();
  for (int deg = 0; deg <= 4; ++deg) {
    std::vector<double> c(deg + 1, 0.);
    c[deg] = 1.;
    if (deg > 0) c[0] = -0.3;  // a non-trivial constant part
    auto u = cell_avg_poly(x1f, c);
    double umax = u.abs().max().item<double>();
    for (int64_t m = 0; m < w.first.size(0); ++m) {
      auto win = u.slice(0, m, m + 5);
      const double tgt[2] = {poly_at(c, f[m + 2]), poly_at(c, f[m + 3])};
      for (int side = 0; side < 2; ++side) {
        auto t = side == 0 ? w.first[m] : w.second[m];
        auto opt = side == 0 ? p.first[m] : p.second[m];
        if (deg <= 2) {
          for (int r = 0; r < 3; ++r) {
            double v =
                (t.slice(0, 5 * r, 5 * r + 5) * win).sum().item<double>();
            double tol =
                64. * kEps * umax *
                t.slice(0, 5 * r, 5 * r + 5).abs().sum().item<double>();
            EXPECT_LE(std::abs(v - tgt[side]), std::max(tol, 1.e-13 * umax))
                << "deg " << deg << " row " << m << " side " << side << " r "
                << r;
          }
        }
        double vo = (opt * win).sum().item<double>();
        EXPECT_LE(std::abs(vo - tgt[side]), 1.e-12 * std::max(umax, 1.))
            << "cp5 deg " << deg << " row " << m << " side " << side;
      }
    }
  }
}

TEST(X1Coeffs, beta_is_scale_free) {
  auto x1f = faces_geometric(16, 1.15, 2.5);
  auto w = weno5_x1_coeffs(x1f);
  auto f = x1f.accessor<double, 1>();
  const double slope = 3.25;
  auto u = cell_avg_poly(x1f, {-1.5, slope});
  for (int64_t m = 0; m < w.first.size(0); ++m) {
    auto win = u.slice(0, m, m + 5);
    double h2 = f[m + 3] - f[m + 2];
    double expect = slope * h2 * (slope * h2);
    for (int side = 0; side < 2; ++side) {
      auto t = side == 0 ? w.first[m] : w.second[m];
      for (int r = 0; r < 3; ++r) {
        double a =
            (t.slice(0, 15 + 10 * r, 20 + 10 * r) * win).sum().item<double>();
        double b =
            (t.slice(0, 20 + 10 * r, 25 + 10 * r) * win).sum().item<double>();
        double beta = 13. / 12. * a * a + .25 * b * b;
        EXPECT_NEAR(beta, expect, 1.e-10 * expect)
            << "row " << m << " side " << side << " r " << r;
      }
    }
  }
}

TEST(X1Coeffs, refusals) {
  EXPECT_THROW(weno5_x1_coeffs(torch::zeros({5}, torch::kFloat64)),
               c10::Error);  // too few faces
  EXPECT_THROW(weno5_x1_coeffs(torch::zeros({3, 7}, torch::kFloat64)),
               c10::Error);  // not 1-D
  auto bad = faces_uniform(10).clone();
  bad[4] = bad[5];
  EXPECT_THROW(weno5_x1_coeffs(bad), c10::Error);  // not strictly increasing
}

TEST(X1Coeffs, reconstruct_seam_and_buffer_names) {
  auto opts = ReconstructOptionsImpl::create();
  opts->interp()->type("weno5");
  Reconstruct recon(opts);
  auto x1f = faces_geometric(21, 1.03);
  recon->set_x1_faces(x1f);
  auto named = recon->named_buffers();
  ASSERT_TRUE(named.contains("interp1.cm_x1"));
  ASSERT_TRUE(named.contains("interp1.cp_x1"));
  ASSERT_TRUE(named.contains("interp2.cm_x1"));
  ASSERT_TRUE(named.contains("interp2.cp_x1"));
  EXPECT_EQ(named["interp1.cm_x1"].size(0), x1f.numel() - 5);
  EXPECT_EQ(named["interp1.cm_x1"].size(1), 48);
  EXPECT_EQ(named["interp2.cm_x1"].size(1), 5);
  auto live = named["interp1.cm_x1"];
  live[0][45] = 0.5;
  EXPECT_EQ(std::dynamic_pointer_cast<Weno5InterpImpl>(recon->pinterp1)
                ->cm_x1[0][45]
                .item<double>(),
            0.5);
}

TEST(X1Coeffs, uncovered_types_refuse_faces) {
  for (char const* type : {"dc", "plm", "ppm", "cp3", "weno3"}) {
    auto opts = ReconstructOptionsImpl::create();
    opts->interp()->type(type);
    Reconstruct recon(opts);
    EXPECT_THROW(recon->set_x1_faces(faces_geometric(21, 1.03)), c10::Error)
        << type;
    EXPECT_FALSE(recon->pinterp1->has_x1_coeffs()) << type;
  }
}
TEST(X1Coeffs, ghost_faces_needed_by_the_stencils) {
  for (int nghost : {3, 4}) {
    for (int nx1 : {16, 64}) {
      const int nc1 = nx1 + 2 * nghost;
      const int rec_ng = 5 / 2 + 1;  // == 3, independent of the deck's nghost
      const int il = rec_ng;
      const int iu = nc1 - rec_ng;
      const int nout = iu - il + 2;
      auto x1f = faces_geometric(nc1 + 1, 1.01);
      auto w = weno5_x1_coeffs(x1f);
      ASSERT_EQ(w.first.size(0), nout);
      const int f_first = nghost;
      const int f_last = nc1 - nghost;
      int lo = nc1, hi = 0;
      for (int f : {f_first, f_last}) {
        for (int off : {2, 3}) {
          int m = f - off;
          ASSERT_GE(m, 0) << "face " << f << " off " << off;
          ASSERT_LT(m, nout) << "face " << f << " off " << off;
          lo = std::min(lo, m);
          hi = std::max(hi, m + 5);
        }
      }
      int g_lo = lo - nghost;
      int g_hi = hi - nghost;
      std::cout << "  nghost = " << nghost << " nx1 = " << nx1
                << ": stencils for the first and last interior face span "
                << "global faces " << g_lo << " .. " << g_hi
                << " (interior is 0 .. " << nx1 << "), so they need " << -g_lo
                << " ghost faces below and " << g_hi - nx1
                << " above; extra = max(0, 3 - nghost) = "
                << std::max(0, 3 - nghost) << std::endl;
      EXPECT_EQ(g_lo, -3);
      EXPECT_EQ(g_hi, nx1 + 3);
      EXPECT_GE(lo, 0);
      EXPECT_LE(hi, x1f.numel() - 1);
    }
  }
}

TEST_P(DeviceTest, module_nu_path) {
  const int nc1 = 24;
  const int nvar = 3;
  for (double q : {1.0, 1.08}) {
    auto x1f = faces_geometric(nc1 + 1, q, 0.6);
    auto f = x1f.accessor<double, 1>();
    std::vector<std::vector<double>> cs;
    for (int j = 0; j < nvar; ++j) {
      cs.push_back({1.3 + j, 0.21, -0.04, 0.006, 0.0011});
    }
    auto u = torch::zeros({nvar, 1, 1, nc1}, torch::kFloat64);
    for (int j = 0; j < nvar; ++j) u[j][0][0] = cell_avg_poly(x1f, cs[j]);
    auto w = u.to(device, dtype);
    Weno5Interp weno;
    Center5Interp cp5;
    weno->to(device, dtype);
    cp5->to(device, dtype);
    auto shape = w.sizes().vec();
    shape[3] -= 4;
    ASSERT_EQ(shape[3], x1f.numel() - 5);
    auto ml = torch::empty(shape, w.options());
    auto mr = torch::empty(shape, w.options());
    ASSERT_FALSE(weno->has_x1_coeffs());
    weno->left(w, 3, ml);
    weno->right(w, 3, mr);
    weno->set_x1_coeffs(x1f);
    cp5->set_x1_coeffs(x1f);
    ASSERT_TRUE(weno->has_x1_coeffs());
    EXPECT_EQ(weno->cm_x1.size(0), shape[3]);
    EXPECT_EQ(weno->cm_x1.size(1), kWeno5NuCoeff);
    EXPECT_EQ(cp5->cm_x1.size(1), kPoly5NuCoeff);
    auto nl = torch::empty(shape, w.options());
    auto nr = torch::empty(shape, w.options());
    weno->left(w, 3, nl);
    weno->right(w, 3, nr);
    auto pl = torch::empty(shape, w.options());
    auto pr = torch::empty(shape, w.options());
    cp5->left(w, 3, pl);
    cp5->right(w, 3, pr);
    double tol = dtype == torch::kFloat64 ? 1.e-10 : 2.e-4;
    double umax = u.abs().max().item<double>();
    if (q == 1.0) {
      double dl = (nl - ml).abs().max().item<double>();
      double dr = (nr - mr).abs().max().item<double>();
      std::cout << "  " << torch::Device(device).str() << " "
                << torch::toString(dtype)
                << " uniform: max|_nu - shipped| = " << dl << " / " << dr
                << std::endl;
      EXPECT_LT(dl, tol * std::max(umax, 1.));
      EXPECT_LT(dr, tol * std::max(umax, 1.));
    }
    auto plc = pl.to(torch::kCPU, torch::kFloat64);
    auto prc = pr.to(torch::kCPU, torch::kFloat64);
    for (int j = 0; j < nvar; ++j) {
      for (int64_t m = 0; m < shape[3]; ++m) {
        EXPECT_NEAR(plc[j][0][0][m].item<double>(), poly_at(cs[j], f[m + 2]),
                    tol * std::max(umax, 1.))
            << "q " << q << " cp5 left, row " << m;
        EXPECT_NEAR(prc[j][0][0][m].item<double>(), poly_at(cs[j], f[m + 3]),
                    tol * std::max(umax, 1.))
            << "q " << q << " cp5 right, row " << m;
      }
    }
    auto nlc = nl.to(torch::kCPU, torch::kFloat64);
    for (int64_t m = 0; m < shape[3]; ++m) {
      double got = nlc[0][0][0][m].item<double>();
      double own = std::abs(got - poly_at(cs[0], f[m + 2]));
      double other = std::abs(got - poly_at(cs[0], f[m + 3]));
      EXPECT_LT(own, other) << "q " << q << " weno5 left, row " << m;
    }
  }
}

TEST(X1Coeffs, a_shifted_row_to_face_mapping_is_detected) {
  const int nc1 = 40;
  auto x1f = faces_irregular(nc1 + 1, 0.5, 11);
  auto f = x1f.accessor<double, 1>();
  std::vector<double> c{0.7, 0.33, -0.051, 0.0071, 0.00042};
  auto u = torch::zeros({1, 1, 1, nc1}, torch::kFloat64);
  u[0][0][0] = cell_avg_poly(x1f, c);
  auto shape = u.sizes().vec();
  shape[3] -= 4;
  const int64_t nout = shape[3];
  Center5Interp uni;
  auto bad_uniform = torch::empty(shape, u.options());
  uni->left(u, 3, bad_uniform);
  Center5Interp cp5;
  cp5->set_x1_coeffs(x1f);
  auto good = torch::empty(shape, u.options());
  cp5->left(u, 3, good);
  double worst_good = 0., worst_uniform = 0.;
  for (int64_t m = 0; m < nout; ++m) {
    double want = poly_at(c, f[m + 2]);
    worst_good =
        std::max(worst_good, std::abs(good[0][0][0][m].item<double>() - want));
    worst_uniform = std::max(
        worst_uniform, std::abs(bad_uniform[0][0][0][m].item<double>() - want));
  }
  const double umax = u.abs().max().item<double>();
  EXPECT_LT(worst_good, 1.e-13 * umax);
  EXPECT_GT(worst_uniform, 1.e-6 * umax)
      << "the uniform rows happen to be right on this grid; pick a sharper q";
  ASSERT_FALSE(torch::equal(cp5->cm_x1, cp5->cp_x1));
  cp5->cm_x1.copy_(cp5->cp_x1);
  auto swapped = torch::empty(shape, u.options());
  cp5->left(u, 3, swapped);
  double worst_swap_own = 0., worst_swap_next = 0.;
  for (int64_t m = 0; m < nout; ++m) {
    double got = swapped[0][0][0][m].item<double>();
    worst_swap_own =
        std::max(worst_swap_own, std::abs(got - poly_at(c, f[m + 2])));
    worst_swap_next =
        std::max(worst_swap_next, std::abs(got - poly_at(c, f[m + 3])));
  }
  EXPECT_LT(worst_swap_next, 1.e-13 * umax);
  EXPECT_GT(worst_swap_own, 1.e-3 * umax);
  auto rows = cp5_x1_coeffs(x1f).first;
  ASSERT_EQ(rows.size(0), nout);
  auto shifted = rows.roll(-1, 0).contiguous();
  ASSERT_FALSE(torch::equal(shifted, rows));
  cp5->cm_x1.copy_(shifted);
  ASSERT_TRUE(torch::equal(cp5->cm_x1, shifted))
      << "the test could not install a modified table";
  auto wrong = torch::empty(shape, u.options());
  cp5->left(u, 3, wrong);
  double worst_shift = 0.;
  for (int64_t m = 0; m + 1 < nout; ++m) {
    worst_shift = std::max(
        worst_shift,
        std::abs(wrong[0][0][0][m].item<double>() - poly_at(c, f[m + 2])));
  }
  std::cout << "  worst |cp5 - exact at its own face|: correct rows "
            << worst_good << ", shipped uniform rows " << worst_uniform
            << ", sides swapped " << worst_swap_own << ", rows shifted by one "
            << worst_shift << std::endl;
  EXPECT_GT(worst_shift, 1.e-3 * umax);
}

TEST(X1Coeffs, cpu_face_index_under_threading_and_reordering) {
  const int nvar = 4, nc3 = 6, nc2 = 13, nc1 = 211;
  auto x1f = faces_irregular(nc1 + 1, 0.43, 23);
  auto f = x1f.accessor<double, 1>();
  std::vector<std::vector<double>> cs;
  for (int j = 0; j < nvar; ++j) {
    cs.push_back({0.9 + 0.3 * j, 0.17, -0.031, 0.0042, 0.00071});
  }
  auto u = torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64);
  for (int j = 0; j < nvar; ++j) {
    auto cell = cell_avg_poly(x1f, cs[j]);
    for (int k = 0; k < nc3; ++k) {
      for (int i = 0; i < nc2; ++i) u[j][k][i] = cell * (1. + 0.05 * i);
    }
  }
  Center5Interp cp5;
  cp5->set_x1_coeffs(x1f);
  auto shape = u.sizes().vec();
  shape[3] -= 4;
  auto worst_of = [&](torch::Tensor const& l, torch::Tensor const& r) {
    double worst = 0.;
    for (int j = 0; j < nvar; ++j) {
      for (int k = 0; k < nc3; ++k) {
        for (int i = 0; i < nc2; ++i) {
          double s = 1. + 0.05 * i;
          for (int64_t m = 0; m < shape[3]; ++m) {
            worst = std::max(worst, std::abs(l[j][k][i][m].item<double>() -
                                             s * poly_at(cs[j], f[m + 2])));
            worst = std::max(worst, std::abs(r[j][k][i][m].item<double>() -
                                             s * poly_at(cs[j], f[m + 3])));
          }
        }
      }
    }
    return worst;
  };
  double umax = u.abs().max().item<double>();
  const int saved = at::get_num_threads();
  for (int nthreads : {1, 2, saved > 4 ? saved : 4}) {
    at::set_num_threads(nthreads);
    auto l = torch::empty(shape, u.options());
    auto r = torch::empty(shape, u.options());
    cp5->left(u, 3, l);
    cp5->right(u, 3, r);
    double worst = worst_of(l, r);
    auto perm = u.permute({0, 3, 1, 2}).contiguous().permute({0, 2, 3, 1});
    ASSERT_NE(perm.stride(3), 1);
    auto pshape = shape;
    auto pl = torch::empty(pshape, u.options());
    auto pr = torch::empty(pshape, u.options());
    cp5->left(perm, 3, pl);
    cp5->right(perm, 3, pr);
    double worst_perm = worst_of(pl, pr);
    std::cout << "  nthreads = " << nthreads
              << "  worst |cp5 - exact| = " << worst << " (contiguous) / "
              << worst_perm << " (x1 strided)"
              << "  = " << worst / (kEps * umax) << " / "
              << worst_perm / (kEps * umax) << " eps*max|u|" << std::endl;
    EXPECT_LT(worst, 256. * kEps * umax) << "nthreads = " << nthreads;
    EXPECT_LT(worst_perm, 256. * kEps * umax) << "nthreads = " << nthreads;
  }
  at::set_num_threads(saved);
}

TEST(X1Coeffs, reconstruct_forward_face_index_on_a_stretched_grid) {
  const int nghost = 3, nx1 = 32;
  const int nc1 = nx1 + 2 * nghost;
  auto x1f = faces_irregular(nc1 + 1, 0.31, 5);
  auto f = x1f.accessor<double, 1>();
  std::vector<double> c{1.1, 0.27, -0.038, 0.0051, 0.00033};
  auto opts = ReconstructOptionsImpl::create();
  opts->interp()->type("cp5");
  Reconstruct recon(opts);
  recon->set_x1_faces(x1f);
  auto u = torch::zeros({1, 1, 1, nc1}, torch::kFloat64);
  u[0][0][0] = cell_avg_poly(x1f, c);
  auto wlr = recon->forward(u, 3, /*floor=*/false);
  const int il = 3, iu = nc1 - 3;
  double worst_rt = 0., worst_lt = 0.;
  for (int i = il; i <= iu; ++i) {
    double want = poly_at(c, f[i]);
    worst_rt = std::max(worst_rt,
                        std::abs(wlr[IRT][0][0][0][i].item<double>() - want));
    worst_lt = std::max(worst_lt,
                        std::abs(wlr[ILT][0][0][0][i].item<double>() - want));
  }
  std::cout
      << "  Reconstruct::forward, irregular widths: worst |IRT - exact| = "
      << worst_rt << ", worst |ILT - exact| = " << worst_lt << std::endl;
  double umax = u.abs().max().item<double>();
  EXPECT_LT(worst_rt, 1.e-13 * umax);
  EXPECT_LT(worst_lt, 1.e-13 * umax);
}

TEST(X1Coeffs, a_non_x1_dimension_is_refused) {
  auto x1f = faces_geometric(21, 1.03);
  Weno5Interp weno;
  Center5Interp cp5;
  weno->set_x1_coeffs(x1f);
  cp5->set_x1_coeffs(x1f);
  auto u = torch::rand({2, 1, 9, 20}, torch::kFloat64) + 1.;
  auto out = torch::empty({2, 1, 5, 20}, torch::kFloat64);
  EXPECT_THROW(weno->left(u, 2, out), c10::Error);
  EXPECT_THROW(cp5->right(u, 2, out), c10::Error);
}

TEST(X1Coeffs, cuda_matches_cpu_on_both_launch_paths) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  for (int nc1 : {64, 1100, 2600}) {  // 64 -> block path, > 1024 -> tiled path
    const int nvar = 5, nc3 = 2, nc2 = 3;
    auto x1f = faces_irregular(nc1 + 1, 0.5, 31 + nc1);
    auto u = torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64);
    auto xv = 0.5 * (x1f.slice(0, 0, nc1) + x1f.slice(0, 1, nc1 + 1));
    for (int j = 0; j < nvar; ++j) {
      auto line = 2.0 + 0.3 * j + torch::sin(0.07 * xv + 0.4 * j) +
                  0.2 * torch::cos(0.013 * xv);
      for (int k = 0; k < nc3; ++k) {
        for (int i = 0; i < nc2; ++i) u[j][k][i] = line * (1. + 0.05 * i);
      }
    }
    auto shape = u.sizes().vec();
    shape[3] -= 4;
    Weno5Interp wc;
    Center5Interp pc;
    wc->set_x1_coeffs(x1f);
    pc->set_x1_coeffs(x1f);
    auto wl = torch::empty(shape, u.options());
    auto wr = torch::empty(shape, u.options());
    auto pl = torch::empty(shape, u.options());
    auto pr = torch::empty(shape, u.options());
    wc->left(u, 3, wl);
    wc->right(u, 3, wr);
    pc->left(u, 3, pl);
    pc->right(u, 3, pr);
    Weno5Interp wg;
    Center5Interp pg;
    wg->to(torch::kCUDA, torch::kFloat64);
    pg->to(torch::kCUDA, torch::kFloat64);
    wg->set_x1_coeffs(x1f);
    pg->set_x1_coeffs(x1f);
    ASSERT_TRUE(wg->cm_x1.device().is_cuda());
    auto g = u.to(torch::kCUDA);
    auto gwl = torch::empty(shape, g.options());
    auto gwr = torch::empty(shape, g.options());
    auto gpl = torch::empty(shape, g.options());
    auto gpr = torch::empty(shape, g.options());
    wg->left(g, 3, gwl);
    wg->right(g, 3, gwr);
    pg->left(g, 3, gpl);
    pg->right(g, 3, gpr);
    double dwl = (gwl.cpu() - wl).abs().max().item<double>();
    double dwr = (gwr.cpu() - wr).abs().max().item<double>();
    double dpl = (gpl.cpu() - pl).abs().max().item<double>();
    double dpr = (gpr.cpu() - pr).abs().max().item<double>();
    int64_t nbit = (gwl.cpu() == wl).sum().item<int64_t>() +
                   (gwr.cpu() == wr).sum().item<int64_t>() +
                   (gpl.cpu() == pl).sum().item<int64_t>() +
                   (gpr.cpu() == pr).sum().item<int64_t>();
    int64_t ntot = 4 * gwl.numel();
    std::cout << "  nc1 = " << nc1 << (nc1 > 1024 ? " (tiled)" : " (block)")
              << "  max|cuda - cpu|  weno5 " << dwl << " / " << dwr << "  cp5 "
              << dpl << " / " << dpr << "   bitwise equal " << nbit << " / "
              << ntot << std::endl;
    EXPECT_LT(dwl, 1.e-12) << "nc1 = " << nc1;
    EXPECT_LT(dwr, 1.e-12) << "nc1 = " << nc1;
    EXPECT_LT(dpl, 1.e-12) << "nc1 = " << nc1;
    EXPECT_LT(dpr, 1.e-12) << "nc1 = " << nc1;
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
