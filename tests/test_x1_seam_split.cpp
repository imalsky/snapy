// One process, a spherical-polar column as one block and as two blocks split
// in x1, stepped together: the split column must keep the one-block state.
//
// SNAP_X1_CENTROID_EXACT reads past the seam twice: the r^2 -> plain-mean
// conversion of the ghost cells (five-cell window) and the pressure source
// (six-face window, two faces past the seam). Both take the neighbour's
// values (hydro.cpp _x1_ghost_rows), so after 20 steps of a seeded column
// the two states agree to round-off, with the full pressure force and in
// hydrostatic-split mode. RED before that exchange (one-sided windows at
// every block edge).
//
// SNAP_GRAVITY_WORK_RADIAL_EXACT exchanges the stage's density change across
// the seam and takes the centred slope there, one-sided only at the walls
// (#303), so the split column keeps the one-block state; with that switch set
// in the environment (1, or 0 for the control) the second test checks a zero
// gap at nz 32/64/128 and, on, E + P (the one-block P) on the split column
// (docs/derivations/curved-gravity-work-weight.md sec 7). An implicit scheme
// refuses an x1 split, so the VIC row has no seam (vic_refuses_x1_split).
//
// SNAP_WB_REF4 alone (ctest test_x1_seam_split_wb_ref4): each block computes
// the resolution flag from its own scan pressures, ghosts included, and only
// (pref, dref) are exchanged; on a column cold enough that the flag switches
// on one cell above the seam, the split column must still keep the one-block
// state (docs/derivations/wb-ref4.md sec 7).
//
// Each ctest entry (tests/CMakeLists.txt) runs one test with its switch set,
// and passes only if that test ran and passed: a missing switch is a failure.
// The _cuda tests are the same checks on the GPU; their entries report a skip
// when CUDA is not available.

// external
#include <gtest/gtest.h>

// C/C++
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// torch
#include <torch/torch.h>
#include <unistd.h>

#include "cuda_test_gate.hpp"

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/hydro/wb_ref4.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

constexpr double kR0 = 5.0, kLz = 2.0, kGamma = 1.4, kSeed = 0.05;

// implicit-scheme of the columns: 0 explicit, 9 the full VIC solve
int g_scheme = 0;

std::string column_yaml(int nx1, double nh, char const* gw, int ng = 3) {
  char buf[2048];
  std::snprintf(buf, sizeof(buf), R"(
geometry:
  type: spherical-polar
  bounds: {x1min: %g, x1max: %g, x2min: 1.5207963267948966, x2max: 1.6207963267948966, x3min: 0., x3max: 0.1}
  cells: {nx1: %d, nx2: 1, nx3: 1, nghost: %d}
dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    weight: 8.31446
    density-floor: 1.e-12
    pressure-floor: 1.e-12
    temperature-floor: 1.e-12
    limiter: true
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver:
    type: lmars
integration:
  type: rk3
  cfl: 0.4
  implicit-scheme: %d
forcing:
  const-gravity: {grav1: -1., non-hydrostatic: %g, gravity-work: %s}
boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: periodic
    x3-outer: periodic
)",
                kR0, kR0 + kLz, nx1, ng, g_scheme, nh, gw);
  return buf;
}

Mesh make_column(int nb1, int nx1, double nh, char const* gw, int ng = 3,
                 torch::Device device = torch::kCPU, double g1 = -1.) {
  char fname[] = "/tmp/x1-seam-split-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream out(fname);
  out << column_yaml(nx1, nh, gw, ng);
  out.close();
  auto block_opts = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  block_opts->hydro()->grav()->grav1(g1);
  if (g1 == 0. || ng < 3) {  // donor cell: weno5 needs nghost >= 3 (S184)
    block_opts->hydro()->recon1()->interp()->type("dc");
    block_opts->hydro()->recon23()->interp()->type("dc");
  }
  if (nb1 > 1) {
    block_opts->layout()->type() = "cubed";
    block_opts->layout()->pz(nb1);
  }
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(nb1);
  auto mesh = Mesh(mesh_opts);
  mesh->to(device, torch::kFloat64);
  return mesh;
}

// isothermal column (g = 1, H = 1) with a density bump on the mid seam and a
// sin(pi z / Lz) radial wind, ghosts included; cold: T = H = 0.25 e^{1.1 - z}
// instead, so dz/H = 0.5 at z = 1.1 when nx1 = 16 (hydrostatic in plane
// parallel: ln p = -int dz / T = -4 (e^{z - 1.1} - e^{-1.1}))
void fill_column(Mesh mesh, MeshVariables& vars, bool cold = false) {
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto coord = mesh->blocks[b]->pcoord;
    int nc1 = coord->options->nc1();
    auto z = coord->x1v - kR0;
    auto temp = cold ? 0.25 * torch::exp(1.1 - z) : torch::ones_like(z);
    auto p = cold ? torch::exp(-4. * (torch::exp(z - 1.1) - std::exp(-1.1)))
                  : torch::exp(-z);
    auto rho =
        p / temp * (1. + 0.02 * torch::exp(-((z - 0.5 * kLz) / 0.2).square()));
    auto in = torch::logical_and(z > 0., z < kLz);
    auto v1 =
        torch::where(in, kSeed * std::sqrt(kGamma) * torch::sin(M_PI * z / kLz),
                     torch::zeros_like(z));
    auto w = torch::zeros({mesh->blocks[b]->phydro->peos->nvar(),
                           coord->options->nc3(), coord->options->nc2(), nc1},
                          coord->x1v.options());
    w[IDN].copy_(rho.view({1, 1, nc1}));
    w[IPR].copy_(p.view({1, 1, nc1}));
    w[IVX].copy_(v1.view({1, 1, nc1}));
    vars[b]["hydro_w"] = w;
  }
}

void step(Mesh mesh, MeshVariables& vars, double dt) {
  int nstage = mesh->blocks.front()->pintg->stages.size();
  for (int stage = 0; stage < nstage; ++stage) mesh->forward(vars, dt, stage);
}

std::vector<size_t> by_x1(Mesh mesh) {
  std::vector<size_t> order(mesh->blocks.size());
  for (size_t b = 0; b < order.size(); ++b) order[b] = b;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return mesh->blocks[a]->pcoord->options->x1min() <
           mesh->blocks[b]->pcoord->options->x1min();
  });
  return order;
}

torch::Tensor column_state(Mesh mesh, MeshVariables const& vars) {
  std::vector<torch::Tensor> parts;
  for (auto b : by_x1(mesh)) {
    auto coord = mesh->blocks[b]->pcoord;
    parts.push_back(
        vars[b].at("hydro_u").slice(-1, coord->il(), coord->iu() + 1));
  }
  return torch::cat(parts, -1);
}

// sum V [E + rho phi(x1v) - g1 <(x1 - x1v)^2> s[rho]] over the blocks: the
// E + P that SNAP_GRAVITY_WORK_RADIAL_EXACT conserves (meshblock.cpp pe=)
double energy_p(Mesh mesh, MeshVariables const& vars) {
  double total = 0.;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto coord = mesh->blocks[b]->pcoord;
    int is = coord->il(), ie = coord->iu() + 1;
    auto u = vars[b].at("hydro_u");
    auto rho = u[IDN];
    auto pe = rho * coord->x1v;
    // across the seam the ghost rho is the neighbour's (exchanged after every
    // stage), so this is the P of one block (#303)
    auto [below, above] = mesh->blocks[b]->phydro->x1_neighbors();
    pe.slice(-1, is, ie) -=
        corrected_pe_work_seam(rho, coord->x1f, coord->x1v, is, ie, -1.,
                               X1Measure::radial, below >= 0, above >= 0);
    auto vol = coord->cell_volume();
    total += ((u[IPR] + pe) * vol).slice(-1, is, ie).sum().item<double>();
  }
  return total;
}

// max over rho, rho v1 and E of |split - one| / max|one| after nstep steps
double split_gap(int nx1, double nh, char const* gw, int nstep,
                 double* ep_drift = nullptr, bool cold = false,
                 torch::Device device = torch::kCPU) {
  auto one = make_column(1, nx1, nh, gw, 3, device);
  auto two = make_column(2, nx1, nh, gw, 3, device);
  EXPECT_EQ(one->blocks.size(), 1u);
  EXPECT_EQ(two->blocks.size(), 2u);
  MeshVariables v1(1), v2(2);
  fill_column(one, v1, cold);
  fill_column(two, v2, cold);
  one->initialize(v1);
  two->initialize(v2);
  double dt = 0.3 * (kLz / nx1) / std::sqrt(kGamma);
  double ep0 = energy_p(two, v2), ep_max = 0.;
  for (int n = 0; n < nstep; ++n) {
    step(one, v1, dt);
    step(two, v2, dt);
    ep_max = std::max(ep_max, std::abs(energy_p(two, v2) - ep0) / ep0);
  }
  if (ep_drift) *ep_drift = ep_max;
  auto a = column_state(one, v1), c = column_state(two, v2);
  EXPECT_TRUE(a.sizes() == c.sizes());
  double gap = 0.;
  for (int var : {IDN, IVX, IPR}) {
    double d = (a[var] - c[var]).abs().max().item<double>();
    gap = std::max(gap, d / a[var].abs().max().item<double>());
  }
  return gap;
}

void centroid_exact_split_matches_one_block(torch::Device device) {
  ASSERT_TRUE(std::getenv("SNAP_X1_CENTROID_EXACT"));
  ASSERT_FALSE(std::getenv("SNAP_GRAVITY_WORK_RADIAL_EXACT") ||
               std::getenv("SNAP_WB_REF4"));
  torch::set_num_threads(1);
  ASSERT_TRUE(x1_centroid_exact_enabled());
  for (double nh : {1., 0.}) {
    double gap = split_gap(32, nh, "cell", 20, nullptr, false, device);
    std::printf("non-hydrostatic %g: 2 blocks vs 1, max rel gap %.3e\n", nh,
                gap);
    // round-off: 20 steps of a 32-cell column
    EXPECT_LE(gap, 1e-13) << "non-hydrostatic " << nh;
  }
}

void radial_exact_split_gap(torch::Device device) {
  ASSERT_TRUE(std::getenv("SNAP_GRAVITY_WORK_RADIAL_EXACT"));  // 1, or 0
  bool on = HydroImpl::gravity_work_radial_exact();
  torch::set_num_threads(1);
  for (int nx1 : {32, 64, 128}) {
    double drift = 0.;
    double gap =
        split_gap(nx1, 1., "face", 20 * nx1 / 32, &drift, false, device);
    std::printf(
        "switch %d, nz %d: 2 blocks vs 1, max rel gap %.3e; split E+P "
        "drift %.3e\n",
        on, nx1, gap, drift);
    if (on) EXPECT_LE(drift, 1e-13) << "nz " << nx1;
    // switch on: the seam cells take one block's slope (#303), no gap; switch
    // off: no corrected-PE work, the gap is round-off only (1.9e-14 at nz 128
    // on CUDA), bounded as in the tests above
    if (on)
      EXPECT_EQ(gap, 0.) << "nz " << nx1;
    else
      EXPECT_LE(gap, 1e-13) << "nz " << nx1;
  }
}

void wb_ref4_flag_at_the_seam_split_matches_one_block(torch::Device device) {
  ASSERT_TRUE(std::getenv("SNAP_WB_REF4"));
  ASSERT_FALSE(std::getenv("SNAP_X1_CENTROID_EXACT"));
  ASSERT_TRUE(wb_ref4_enabled());
  ASSERT_FALSE(x1_centroid_exact_enabled());
  torch::set_num_threads(1);
  for (double nh : {1., 0.}) {
    double gap = split_gap(16, nh, "cell", 20, nullptr, /*cold=*/true, device);
    std::printf(
        "SNAP_WB_REF4, cold, non-hydrostatic %g: 2 blocks vs 1, max rel gap "
        "%.3e\n",
        nh, gap);
    EXPECT_LE(gap, 1e-13) << "non-hydrostatic " << nh;
  }
  // the flag reads scan pressures three cells away: fewer ghosts is an error
  for (int ng : {1, 2}) try {
      make_column(2, 16, 1., "cell", ng, device);
      ADD_FAILURE() << "nghost " << ng << " accepted";
    } catch (c10::Error const& e) {
      EXPECT_NE(std::string(e.what()).find("needs nghost >= 3"),
                std::string::npos)
          << e.what();
    }
}

}  // namespace

TEST(X1SeamSplit, switches_off_nghost_1_sets_up) {
  ASSERT_FALSE(std::getenv("SNAP_X1_CENTROID_EXACT") ||
               std::getenv("SNAP_GRAVITY_WORK_RADIAL_EXACT") ||
               std::getenv("SNAP_WB_REF4"));
  // the nghost >= 3 check is the switches' own: the default nghost is 1
  EXPECT_NO_THROW(make_column(1, 16, 1., "cell", 1));
  EXPECT_NO_THROW(make_column(2, 16, 1., "cell", 1));
}

TEST(X1SeamSplit, centroid_exact_split_matches_one_block) {
  centroid_exact_split_matches_one_block(torch::kCPU);
}

TEST(X1SeamSplit, radial_exact_split_gap) {
  radial_exact_split_gap(torch::kCPU);
}

// the VIC row's slope is per block (implicit_hydro.cpp); that is the one-block
// slope only because an implicit scheme refuses an x1 split: pin the refusal
TEST(X1SeamSplit, vic_refuses_x1_split) {
  g_scheme = 9;
  auto one = make_column(1, 32, 1., "face");
  auto two = make_column(2, 32, 1., "face");
  g_scheme = 0;
  MeshVariables v1(1), v2(2);
  fill_column(one, v1, false);
  fill_column(two, v2, false);
  one->initialize(v1);
  two->initialize(v2);
  double dt = 0.3 * (kLz / 32) / std::sqrt(kGamma);
  EXPECT_NO_THROW(step(one, v1, dt));
  try {
    step(two, v2, dt);
    ADD_FAILURE() << "an implicit scheme stepped a column split in x1";
  } catch (std::exception const& e) {
    EXPECT_NE(std::string(e.what()).find("requires nb1 = 1"), std::string::npos)
        << e.what();
  }
}

TEST(X1SeamSplit, wb_ref4_flag_at_the_seam_split_matches_one_block) {
  wb_ref4_flag_at_the_seam_split_matches_one_block(torch::kCPU);
}

TEST(X1SeamSplit, centroid_exact_split_matches_one_block_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  centroid_exact_split_matches_one_block(torch::Device(torch::kCUDA, 0));
}

TEST(X1SeamSplit, radial_exact_split_gap_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  radial_exact_split_gap(torch::Device(torch::kCUDA, 0));
}

TEST(X1SeamSplit, wb_ref4_flag_at_the_seam_split_matches_one_block_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  wb_ref4_flag_at_the_seam_split_matches_one_block(
      torch::Device(torch::kCUDA, 0));
}

// SNAP_X1_MASS_COVARIANCE: the one-sided wall rho_1 spans three cells, so a
// block with two would read the ghost it replaces; setup refuses it. On
// nghost 3 the coordinate setup already refuses two cells, so use nghost 2.
TEST(X1SeamSplit, x1_mass_covariance_needs_3_x1_cells) {
  ASSERT_TRUE(std::getenv("SNAP_X1_MASS_COVARIANCE"));
  ASSERT_TRUE(HydroImpl::x1_mass_covariance());
  try {
    make_column(1, 2, 1., "cell", 2);
    ADD_FAILURE() << "nx1 2 accepted";
  } catch (c10::Error const& e) {
    EXPECT_NE(std::string(e.what()).find("needs at least 3 x1 cells"),
              std::string::npos)
        << e.what();
  }
  EXPECT_NO_THROW(make_column(1, 3, 1., "cell", 2));
}

// SNAP_WB_REF4 with grav1 = 0: no reference is built, so the nghost >= 3
// check stays off and a donor-cell column on nghost 1 must set up and step
TEST(X1SeamSplit, wb_ref4_gravity_0_nghost_1_steps) {
  ASSERT_TRUE(std::getenv("SNAP_WB_REF4"));
  ASSERT_TRUE(wb_ref4_enabled());
  torch::set_num_threads(1);
  for (int nb1 : {1, 2}) {
    Mesh mesh = nullptr;
    ASSERT_NO_THROW(mesh =
                        make_column(nb1, 16, 1., "cell", 1, torch::kCPU, 0.));
    MeshVariables vars(nb1);
    fill_column(mesh, vars);
    mesh->initialize(vars);
    double dt = 0.3 * (kLz / 16) / std::sqrt(kGamma);
    for (int n = 0; n < 5; ++n) ASSERT_NO_THROW(step(mesh, vars, dt));
    EXPECT_TRUE(torch::isfinite(column_state(mesh, vars)).all().item<bool>())
        << "nb1 " << nb1;
  }
}
