#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>

#include <cmath>
#include <snap/hydro/hydro_dispatch.hpp>
#include <vector>

namespace {

constexpr int kNc1 = 20;
constexpr int kIu = 16;  // il = nc1 - 1 - iu = 3, owned cells [3, 16]
constexpr int kIl = kNc1 - 1 - kIu;
constexpr double kA = 1.0, kB = -0.03;
double rop_at(double j) { return kA + kB * j; }
double rop_of(int j) { return rop_at(j); }
constexpr double kP0 = 100.0;  // constant pressure; rho carries the profile
constexpr double kGrav = 1.0;

std::vector<double> widths(double q, double h0 = 0.08) {
  std::vector<double> h(kNc1);
  h[0] = h0;
  for (int i = 1; i < kNc1; ++i) h[i] = h[i - 1] * q;
  return h;
}

std::vector<double> centres(std::vector<double> const& h) {
  std::vector<double> x1v(kNc1);
  double face = 0.;
  for (int i = 0; i < kNc1; ++i) {
    x1v[i] = face + 0.5 * h[i];
    face += h[i];
  }
  return x1v;
}

double face_of(std::vector<double> const& h, int i) {
  double f = 0.;
  for (int k = 0; k < i; ++k) f += h[k];
  return f;
}

struct Out {
  torch::Tensor psf_lo, psf_hi, pref, dsf, dref;
};
Out run(std::vector<double> const& h) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto w = torch::zeros({snap::IPR + 1, 1, 1, kNc1}, opts);
  for (int j = 0; j < kNc1; ++j) {
    w[snap::IPR][0][0][j] = kP0;
    w[snap::IDN][0][0][j] = kP0 * rop_of(j);
  }
  auto dx1f =
      torch::from_blob(const_cast<double*>(h.data()), {kNc1}, opts).clone();
  auto sizes = w.sizes().slice(1).vec();
  Out o{torch::empty(sizes, opts), torch::empty(sizes, opts),
        torch::empty(sizes, opts), torch::empty(sizes, opts),
        torch::empty(sizes, opts)};
  auto rows = snap::hydro_ref_x1_stretched_rows(dx1f, kIu, /*phys_in=*/true,
                                                /*phys_out=*/true,
                                                /*wall_clamp=*/true);
  at::native::call_hydro_ref_x1(torch::kCPU, w, dx1f, torch::Tensor(), o.psf_lo,
                                o.psf_hi, o.pref, o.dsf, o.dref, kIu, kGrav,
                                snap::kX1RefStretched, /*phys_in=*/true,
                                /*phys_out=*/true, /*wall_clamp=*/true, rows);
  return o;
}

double at1(torch::Tensor const& t, int i) { return t[0][0][i].item<double>(); }

double steps_to_ghost(std::vector<double> const& h, int jw, int dir, int k) {
  double d = 0.5 * h[jw];  // wall cell centre to the wall itself
  for (int m = 1; m <= k; ++m) {
    double hm = h[jw - dir * (m - 1)];  // the mirrored width of ghost m
    d += (m < k) ? hm : 0.5 * hm;       // whole cells, then half of ghost k
  }
  return d / (0.5 * (h[jw] + h[jw - dir]));
}

}  // namespace

TEST(HydroRefX1RfWall, face_rho_over_p_is_interpolated_to_the_face_position) {
  auto h = widths(1.2);
  auto x1v = centres(h);
  auto o = run(h);
  int tested = 0;
  for (int i = kIl + 3; i <= kIu - 2; ++i) {
    double xf = face_of(h, i);
    double t = (xf - x1v[i - 1]) / (x1v[i] - x1v[i - 1]);
    ASSERT_GT(t, 0.) << "i = " << i;
    ASSERT_LT(t, 1.) << "i = " << i;

    double want = rop_of(i - 1) + t * (rop_of(i) - rop_of(i - 1));
    double got = at1(o.dsf, i) / at1(o.psf_lo, i);
    EXPECT_NEAR(got, want, 1.e-12 * std::abs(want))
        << "face " << i << ": t = " << t;

    double midpoint = 0.5 * (rop_of(i - 1) + rop_of(i));
    EXPECT_GT(std::abs(want - midpoint), 1.e-6)
        << "face " << i << " does not separate the two rules";
    ++tested;
  }
  EXPECT_GE(tested, 8);
  auto hu = widths(1.0);
  auto xu = centres(hu);
  auto ou = run(hu);
  for (int i = kIl + 3; i <= kIu - 2; ++i) {
    double xf = face_of(hu, i);
    double t = (xf - xu[i - 1]) / (xu[i] - xu[i - 1]);
    EXPECT_NEAR(t, 0.5, 1.e-14) << "i = " << i;
    double want = 0.5 * (rop_of(i - 1) + rop_of(i));
    EXPECT_NEAR(at1(ou.dsf, i) / at1(ou.psf_lo, i), want,
                1.e-12 * std::abs(want))
        << "uniform face " << i;
  }
}

TEST(HydroRefX1RfWall, the_lower_wall_cell_is_continued_by_distance) {
  auto h = widths(1.2);
  auto o = run(h);
  ASSERT_LT(rop_of(kIl + 1), rop_of(kIl));
  double kk1 = steps_to_ghost(h, kIl, -1, 1);
  double kk2 = steps_to_ghost(h, kIl, -1, 2);
  EXPECT_GT(kk1, 0.);
  EXPECT_GT(kk2, kk1);

  double want = (rop_at(kIl - kk2) + 4. * rop_at(kIl - kk1) + 6. * rop_of(kIl) +
                 4. * rop_of(kIl + 1) + rop_of(kIl + 2)) /
                16.;
  double got = at1(o.dref, kIl) / at1(o.pref, kIl);
  EXPECT_NEAR(got, want, 1.e-12 * std::abs(want));

  double shipped = rop_of(kIl);
  EXPECT_GT(std::abs(want - shipped), 1.e-6)
      << "the lower wall cell does not separate the two rules";
}

TEST(HydroRefX1RfWall, the_upper_wall_cell_is_continued_by_distance) {
  auto h = widths(1.2);
  auto o = run(h);

  double r0 = rop_of(kIu), r1 = rop_of(kIu - 1);
  ASSERT_GT(r1, r0);
  double kk1 = steps_to_ghost(h, kIu, +1, 1);
  double kk2 = steps_to_ghost(h, kIu, +1, 2);
  EXPECT_GT(kk1, 0.);
  EXPECT_GT(kk2, kk1);
  auto ghost = [&](double kk) { return r0 * std::pow(r0 / r1, kk); };
  double want = (rop_of(kIu - 2) + 4. * rop_of(kIu - 1) + 6. * r0 +
                 4. * ghost(kk1) + ghost(kk2)) /
                16.;
  double got = at1(o.dref, kIu) / at1(o.pref, kIu);
  EXPECT_NEAR(got, want, 1.e-12 * std::abs(want));

  double shipped = (rop_of(kIu - 2) + 4. * rop_of(kIu - 1) + 6. * r0 +
                    4. * ghost(1.) + ghost(2.)) /
                   16.;
  EXPECT_GT(std::abs(want - shipped), 1.e-6)
      << "the upper wall cell does not separate the two rules";
}

TEST(HydroRefX1RfWall, on_a_uniform_column_the_step_count_is_the_ghost_index) {
  auto h = widths(1.0);
  for (int k : {1, 2, 3}) {
    EXPECT_NEAR(steps_to_ghost(h, kIl, -1, k), k, 1.e-14) << "lower k = " << k;
    EXPECT_NEAR(steps_to_ghost(h, kIu, +1, k), k, 1.e-14) << "upper k = " << k;
  }
}
