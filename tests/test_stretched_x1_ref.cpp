#include <gtest/gtest.h>
#include <snap/snap.h>
#include <unistd.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <snap/hydro/balance_column.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>
#include <string>
#include <vector>

#include "../src/hydro/hydro_dispatch.hpp"
using namespace snap;

namespace {
std::string card(bool stretched, std::string const& forcing = "}",
                 std::string const& inner = "reflecting",
                 std::string const& outer = "reflecting",
                 std::string const& geometry = "cartesian",
                 std::string const& recon = "weno5") {
  return std::string(
             "reference-state: {Tref: 300., Pref: 1.e5}\n"
             "species:\n"
             "  - name: dry\n"
             "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
             "    cv_R: 2.5\n"
             "geometry:\n"
             "  type: ") +
         geometry +
         "\n"
         "  bounds: {x1min: 1000., x1max: 3000., x2min: 0., x2max: 1., "
         "x3min: 0., x3max: 1.}\n"
         "  cells: {nx1: 24, nx2: 1, nx3: 1, nghost: 3" +
         (stretched ? ", x1rat: 1.058" : "") +
         "}\n"
         "dynamics:\n"
         "  equation-of-state: {type: ideal-gas}\n"
         "  reconstruct:\n"
         "    vertical: {type: " +
         recon +
         ", scale: false, shock: false}\n"
         "    horizontal: {type: weno5, scale: false, shock: false}\n"
         "  riemann-solver: {type: lmars}\n"
         "forcing:\n"
         "  const-gravity: {grav1: -9.8" +
         forcing +
         "\n"
         "boundary-condition:\n"
         "  external:\n"
         "    x1-inner: " +
         inner +
         "\n"
         "    x1-outer: " +
         outer + "\n";
}
std::string write_temp(std::string const& text) {
  char fname[] = "/tmp/test-stretched-x1-ref-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << text;
  return fname;
}
void build_and_step(std::string const& text) {
  auto fname = write_temp(text);
  std::shared_ptr<MeshBlockImpl> block;
  try {
    auto options = MeshBlockOptionsImpl::from_yaml(fname);
    std::remove(fname.c_str());
    block = std::make_shared<MeshBlockImpl>(options);
  } catch (...) {
    std::remove(fname.c_str());
    throw;
  }
  block->to(torch::kCPU, torch::kFloat64);
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  auto z = coord->x1v - 1000.;
  w[IDN].copy_(torch::exp(-z / 8.e3));
  w[IPR].copy_(8.e3 * 9.8 * torch::exp(-z / 8.e3));
  auto u = block->phydro->peos->compute("W->U", {w});
  Variables vars;
  vars["hydro_w"] = torch::empty_like(w);
  block->phydro->forward(1.e-3, u, vars);
}
void refuses(std::string const& message, std::string const& forcing = "}",
             std::string const& inner = "reflecting",
             std::string const& outer = "reflecting",
             std::string const& geometry = "cartesian",
             std::string const& recon = "weno5") {
  bool threw = false;
  try {
    build_and_step(card(true, forcing, inner, outer, geometry, recon));
  } catch (c10::Error const& e) {
    threw = true;
    EXPECT_NE(std::string(e.what()).find(message), std::string::npos)
        << "expected \"" << message << "\", got:\n"
        << e.what();
  }
  EXPECT_TRUE(threw) << "a stretched x1 did not refuse: " << message;
  if (geometry == "cartesian") {
    EXPECT_NO_THROW(
        build_and_step(card(false, forcing, inner, outer, geometry, recon)));
  } else {  // the uniform twin need not be a runnable deck; it must not refuse
    try {
      build_and_step(card(false, forcing, inner, outer, geometry));
    } catch (c10::Error const& e) {
      EXPECT_EQ(std::string(e.what()).find(message), std::string::npos)
          << e.what();
    }
  }
}
}  // namespace

TEST(StretchedX1Ref, uniform_and_stretched_reflecting_decks_run) {
  EXPECT_NO_THROW(build_and_step(card(false)));
  EXPECT_NO_THROW(build_and_step(card(true)));
}

TEST(StretchedX1Ref, refuses_curvature_correction) {
  refuses(
      "the dx^2/12 x1 curvature correction of cp3/cp5/weno5 is not "
      "supported on a stretched x1",
      ", gravity-work: face-wallc}");
}

TEST(StretchedX1Ref, refuses_curvature_correction_cp5) {
  refuses(
      "the dx^2/12 x1 curvature correction of cp3/cp5/weno5 is not "
      "supported on a stretched x1",
      ", gravity-work: face-wallc}", "reflecting", "reflecting", "cartesian",
      "cp5");
}

TEST(StretchedX1Ref, refuses_a_stretched_cp3_deck_outright) {
  refuses("[cp3] has no stretched-x1 reconstruction",
          ", gravity-work: face-wallc}", "reflecting", "reflecting",
          "cartesian", "cp3");
}

TEST(StretchedX1Ref, refuses_curvature_with_radial_exact_off) {
  if (HydroImpl::gravity_work_radial_exact())
    GTEST_SKIP() << "SNAP_GRAVITY_WORK_RADIAL_EXACT is on";
  refuses(
      "the dx^2/12 x1 curvature correction of cp3/cp5/weno5 is not "
      "supported on a stretched x1");
}

TEST(StretchedX1Ref, refuses_diffusion_kappa_iso_only) {
  refuses("[Diffusion] not supported on a stretched x1",
          "}\n  diffusion: {kappa_iso: 1.e-3}");
}

TEST(StretchedX1Ref, refuses_diffusion) {
  refuses("[Diffusion] not supported on a stretched x1",
          "}\n  diffusion: {nu_iso: 1.e-3}");
}

TEST(StretchedX1Ref, refuses_bot_heat) {
  refuses("[BotHeat] not supported on a stretched x1",
          "}\n  bot-heat: {flux: 1., depth: 1}");
}

TEST(StretchedX1Ref, refuses_top_cool) {
  refuses("[TopCool] not supported on a stretched x1",
          "}\n  top-cool: {flux: -1., depth: 1}");
}

TEST(StretchedX1Ref, refuses_periodic_x1) {
  refuses("needs reflecting x1 boundaries", "}", "periodic", "periodic");
}

namespace {
std::string cubed_card(bool stretched, int axis) {
  auto text = "distribute: {layout: cubed}\n" +
              (axis == 1 ? card(stretched, "}", "periodic", "periodic")
                         : card(stretched));
  if (axis > 1) {
    auto n = std::string("nx") + char('0' + axis);
    text.replace(text.find(n + ": 1"), n.size() + 3, n + ": 4");
    text += "    x" + std::to_string(axis) + "-inner: periodic\n    x" +
            std::to_string(axis) + "-outer: periodic\n";
  }
  return text;
}
std::array<bool, 3> periodic_xyz(std::string const& text) {
  auto fname = write_temp(text);
  auto options = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname.c_str());
  auto layout = options->layout();
  return {layout->periodic_x(), layout->periodic_y(), layout->periodic_z()};
}
}  // namespace

TEST(StretchedX1Ref, refuses_layout_periodic_x1) {
  auto flags = periodic_xyz(cubed_card(true, 1));
  EXPECT_FALSE(flags[0]);
  EXPECT_FALSE(flags[1]);
  EXPECT_TRUE(flags[2]);
  bool threw = false;
  try {
    build_and_step(cubed_card(true, 1));
  } catch (c10::Error const& e) {
    threw = true;
    EXPECT_NE(std::string(e.what()).find("needs reflecting x1 boundaries"),
              std::string::npos)
        << e.what();
  }
  EXPECT_TRUE(threw) << "a stretched x1 under the layout's x1 wrap ran";
  EXPECT_NO_THROW(build_and_step(cubed_card(false, 1)));
}

TEST(StretchedX1Ref, layout_periodic_x2_x3_pass_the_x1_check) {
  for (int axis : {2, 3}) {
    auto flags = periodic_xyz(cubed_card(true, axis));
    EXPECT_EQ(flags[0], axis == 2) << "x" << axis;
    EXPECT_EQ(flags[1], axis == 3) << "x" << axis;
    EXPECT_FALSE(flags[2]) << "x" << axis;
    EXPECT_NO_THROW(build_and_step(cubed_card(true, axis))) << "x" << axis;
  }
}

TEST(StretchedX1Ref, refuses_outflow_x1_outer) {
  refuses("the x1-outer boundary is not reflecting", "}", "reflecting",
          "outflow");
}

TEST(StretchedX1Ref, refuses_outflow_x1_inner) {
  refuses("the x1-inner boundary is not reflecting", "}", "outflow",
          "reflecting");
}

TEST(StretchedX1Ref, refuses_spherical_polar) {
  refuses("is supported only on a cartesian geometry", "}", "reflecting",
          "reflecting", "spherical-polar");
}

TEST(StretchedX1Ref, refuses_gnomonic_equiangle) {
  refuses("is supported only on a cartesian geometry", "}", "reflecting",
          "reflecting", "gnomonic-equiangle");
}

TEST(StretchedX1Ref, refuses_wb_ref4_switch) {
  if (!HydroImpl::wb_ref4()) GTEST_SKIP() << "SNAP_WB_REF4 is not set";
  refuses(
      "SNAP_WB_REF4 (or SNAP_X1_CENTROID_EXACT, which implies it) is not "
      "supported on a stretched x1 (geometry/x1f or cells/x1rat, #280)");
}

TEST(StretchedX1Ref, refuses_x1_mass_covariance_switch) {
  if (!HydroImpl::x1_mass_covariance())
    GTEST_SKIP() << "SNAP_X1_MASS_COVARIANCE is not set";
  refuses(
      "SNAP_X1_MASS_COVARIANCE is not supported on a stretched x1 "
      "(geometry/x1f or cells/x1rat, #280)");
  EXPECT_NO_THROW(build_and_step(card(false)))
      << "a uniform x1 with SNAP_X1_MASS_COVARIANCE set must still run";
}

namespace {
std::pair<torch::Tensor, torch::Tensor> column(int n, double q, double height) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto k = torch::arange(n + 1, opt);
  auto faces =
      height * torch::expm1(k * std::log(q)) / std::expm1(n * std::log(q));
  auto dx1f = faces.slice(0, 1) - faces.slice(0, 0, n);
  auto zc = 0.5 * (faces.slice(0, 1) + faces.slice(0, 0, n));
  auto w = torch::zeros({IPR + 1, 1, 1, n}, opt);
  double H = 1.07e5, g = 22.;
  w[IDN][0][0].copy_(1.e7 / (g * H) * torch::exp(-zc / H));
  w[IPR][0][0].copy_(1.e7 * torch::exp(-zc / H));
  return {w, dx1f};
}
}  // namespace

TEST(StretchedX1Ref, keyed_column_balances_to_1e13) {
  auto [w, dx1f] = column(128, 1.02, 3.e5);
  auto [wb, err, sweeps] =
      balance_column(w, dx1f, 22., /*wall_clamp=*/true, /*rtol=*/1.e-13,
                     /*max_iter=*/400, "cartesian", /*stretched=*/true);
  std::printf("keyed nx1 = 128: residual %.3e after %d sweeps\n", err, sweeps);
  EXPECT_LT(err, 1.e-13);
  EXPECT_THROW(balance_column(w, dx1f, 22., true, 1.e-13, 400, "cartesian",
                              /*stretched=*/false),
               c10::Error);
}

TEST(StretchedX1Ref, balance_column_refuses_wb_ref4) {
  if (!HydroImpl::wb_ref4()) GTEST_SKIP() << "SNAP_WB_REF4 is not set";
  std::string message =
      "balance_column: SNAP_WB_REF4 (or SNAP_X1_CENTROID_EXACT, which implies "
      "it) is not supported on a stretched x1 (#280)";
  auto [w, dx1f] = column(32, 1.058, 3.e5);
  bool threw = false;
  try {
    balance_column(w, dx1f, 22., true, 1.e-10, 120, "cartesian",
                   /*stretched=*/true);
  } catch (c10::Error const& e) {
    threw = true;
    EXPECT_NE(std::string(e.what()).find(message), std::string::npos)
        << e.what();
  }
  EXPECT_TRUE(threw) << "balance_column did not refuse REF4 on a stretched x1";
  auto [wu, dxu] = column(32, 1.0 + 1.e-12, 3.e5);  // uniform to round-off
  dxu.fill_(3.e5 / 32.);
  EXPECT_NO_THROW(balance_column(wu, dxu, 22., true, 1.e-10, 120, "cartesian",
                                 /*stretched=*/false));
}

namespace {
constexpr double kG = 22., kR = 3936.0434, kTs = 600., kPs = 1.e7;
double adiabat_p(double z) {
  return kPs * std::pow(1. - kG / (3.5 * kR) * z / kTs, 3.5);
}
double adiabat_rho(double z) {
  return adiabat_p(z) / (kR * (kTs - kG / (3.5 * kR) * z));
}
template <typename F>
double cell_average(F f, double a, double b) {
  static double const x[4] = {0.1834346424956498, 0.5255324099163290,
                              0.7966664774136267, 0.9602898564975363};
  static double const g[4] = {0.3626837833783620, 0.3137066458778873,
                              0.2223810344533745, 0.1012285362903763};
  double c = 0.5 * (a + b), h = 0.5 * (b - a), sum = 0.;
  for (int k = 0; k < 4; ++k) sum += g[k] * (f(c - h * x[k]) + f(c + h * x[k]));
  return 0.5 * sum;
}
std::vector<double> pref_error(std::vector<double> const& f, bool stretched,
                               bool phys) {
  int n = static_cast<int>(f.size()) - 1;
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto w = torch::zeros({IPR + 1, 1, 1, n}, opt);
  auto dx1f = torch::empty({n}, opt);
  for (int i = 0; i < n; ++i) {
    w[IDN][0][0][i] = cell_average(adiabat_rho, f[i], f[i + 1]);
    w[IPR][0][0][i] = cell_average(adiabat_p, f[i], f[i + 1]);
    dx1f[i] = f[i + 1] - f[i];
  }
  auto anchor = torch::full({1, 1, 1}, adiabat_p(f[n]), opt);
  auto lo = torch::empty({1, 1, n}, opt), hi = torch::empty_like(lo);
  auto pref = torch::empty_like(lo), dsf = torch::empty_like(lo),
       dref = torch::empty_like(lo);
  auto rows = stretched
                  ? hydro_ref_x1_stretched_rows(dx1f, n - 1, phys, phys, true)
                  : torch::Tensor();
  at::native::call_hydro_ref_x1(
      torch::kCPU, w, dx1f, anchor, lo, hi, pref, dsf, dref, n - 1, kG,
      stretched ? kX1RefStretched : kX1RefUniform, phys, phys, true, rows);
  std::vector<double> e(n);
  for (int i = 0; i < n; ++i) {
    double pbar = w[IPR][0][0][i].item<double>();
    e[i] = std::abs(pref[0][0][i].item<double>() - pbar) / pbar;
  }
  return e;
}
std::vector<double> uniform_faces(double x0, double h, int n) {
  std::vector<double> f(n + 1);
  for (int k = 0; k <= n; ++k) f[k] = x0 + k * h;
  return f;
}
}  // namespace

TEST(StretchedX1Ref, iv_adiabat_cells_within_2x_uniform) {
  int n = 32;
  double q = 1. / 1.058, L = 3.e5;
  std::vector<double> f(n + 1);
  for (int k = 0; k <= n; ++k)
    f[k] = L * std::expm1(k * std::log(q)) / std::expm1(n * std::log(q));
  f[n] = L;
  auto es = pref_error(f, true, true);
  double worst = 0.;
  for (int i = 0; i < n; ++i) {
    double h = f[i + 1] - f[i], eu;
    if (i < 2) {  // bottom wall rows
      eu = pref_error(uniform_faces(0., h, 8), false, true)[i];
    } else if (i >= n - 2) {  // top wall rows
      eu = pref_error(uniform_faces(L - 8 * h, h, 8), false, true)[8 - n + i];
    } else {  // interior: the six-face row of a uniform patch
      eu = pref_error(uniform_faces(f[i] - 4 * h, h, 9), false, false)[4];
    }
    double ratio = es[i] / std::max(eu, 1.e-14);
    worst = std::max(worst, ratio);
    std::printf("cell %2d h %8.1f stretched %.3e uniform %.3e ratio %.3f\n", i,
                h, es[i], eu, ratio);
    EXPECT_LE(ratio, 2.) << "cell " << i;
  }
  std::printf("worst ratio %.3f\n", worst);
}

TEST(StretchedX1Ref, iv_rows_on_uniform_faces_are_the_uniform_weights) {
  int n = 12;
  auto rows = hydro_ref_x1_stretched_rows(
      torch::full({n}, 2.5e3, torch::kFloat64), n - 1, true, true, true);
  double const w6[6] = {11. / 1440., -31. / 480., 401. / 720.,
                        401. / 720., -31. / 480., 11. / 1440.};
  double const w6e[2][6] = {{95. / 288., 1427. / 1440., -133. / 240.,
                             241. / 720., -173. / 1440., 3. / 160.},
                            {-3. / 160., 637. / 1440., 511. / 720., -43. / 240.,
                             77. / 1440., -11. / 1440.}};
  double dw = 0.;
  for (int i = 0; i < n; ++i) {
    int start = i < 2 ? 0 : (i >= n - 2 ? n - 5 : i - 2);
    EXPECT_EQ(rows[i][0].item<double>(), start) << "cell " << i;
    for (int m = 0; m < 6; ++m) {
      double expect = i < 2        ? w6e[i][m]
                      : i >= n - 2 ? w6e[n - 1 - i][5 - m]
                                   : w6[m];
      dw = std::max(dw, std::abs(rows[i][1 + m].item<double>() - expect));
    }
  }
  std::printf("max |dw| against the uniform weights: %.3e\n", dw);
  EXPECT_EQ(dw, 0.);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
