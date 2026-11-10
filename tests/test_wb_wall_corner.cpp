// RED on main 5eeb9b6. A resting polytrope that is uniform in x2, with
// reflecting x1 walls, periodic x2 and isotropic viscosity, grows u2 in the
// wall-adjacent row at the x2 block edge. Nothing in the state varies in x2, so
// u2 must stay zero. One block is enough: its periodic x2 edge is a seam too.
// Viscosity alone does it. The same column behind a user wall (the stock
// reflecting condition registered under another name) must stay at rest too.

// C/C++
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// POSIX
#include <unistd.h>

// gtest
#include <gtest/gtest.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/balance_column.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

// the stock reflecting wall under a name of its own: nothing may key on the
// name "reflecting" to find a physical wall
BC_FUNCTION(user_wall_inner, var, dim, op) {
  get_bc_func()["reflecting_inner"](var, dim, op);
}

BC_FUNCTION(user_wall_outer, var, dim, op) {
  get_bc_func()["reflecting_outer"](var, dim, op);
}

namespace {

// Anders & Brown (2017) polytrope, eps = 1e-2, n_rho = 3, in units Rd = 1
// (weight = Rgas); 32 x 8 cells
const char* kConfig = R"(
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 6.488906699059797, x2min: 0., x2max: 1.6222266747649493, x3min: 0., x3max: 1.}
  cells: {nx1: 32, nx2: 8, nx3: 1, nghost: 3}

distribute:
  layout: slab
  nb2: 1
  nb3: 1
  blocks_per_process: 1

dynamics:
  wb-wall-clamp: true
  equation-of-state:
    type: ideal-gas
    gammad: 1.6666666666666667
    weight: 8.31446
    density-floor: 1.e-12
    pressure-floor: 1.e-12
    temperature-floor: 1.e-12
  reconstruct:
    vertical: {type: weno5, scale: true, shock: false}
    horizontal: {type: weno5, scale: true, shock: false}
  riemann-solver:
    type: lmars

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: periodic
    x2-outer: periodic
    x3-inner: periodic
    x3-outer: periodic

integration:
  type: rk3
  cfl: 0.4
  implicit-scheme: 0

forcing:
  const-gravity:
    grav1: -2.49
  diffusion:
    nu_iso: 0.0234
)";

Mesh make_mesh(torch::Device device, std::string const& wall, int nb2 = 1,
               bool tracer = false) {
  std::string config = kConfig;
  if (tracer) {
    config += R"(
scalar:
  nvar: 1
  names: [tracer]
  reconstruct: {type: weno5, shock: false}
)";
  }
  for (std::string face : {"x1-inner: ", "x1-outer: "}) {
    config.replace(config.find(face + "reflecting") + face.size(), 10, wall);
  }
  config.replace(config.find("nb2: 1"), 6, "nb2: " + std::to_string(nb2));
  config.replace(config.find("blocks_per_process: 1"), 21,
                 "blocks_per_process: " + std::to_string(nb2));
  char fname[] = "/tmp/wb-wall-corner-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream out(fname);
  out << config;
  out.close();
  auto mesh = Mesh(MeshOptionsImpl::from_yaml(fname));
  std::remove(fname);
  mesh->to(device, torch::kFloat64);
  return mesh;
}

}  // namespace

// T = 1 + Lz - z, rho = T^m, p = rho T, projected onto the scheme's discrete
// hydrostatic balance, at rest: uniform in x2. Ten steps. Unprojected, the
// polytrope alone grows u1 to 1.9e-7 c_s in the top rows with no diffusion.
void x2_uniform_rest_keeps_u2_zero(torch::Device device,
                                   std::string const& wall) {
  torch::set_num_threads(1);
  auto mesh = make_mesh(device, wall);
  ASSERT_EQ(mesh->blocks.size(), 1u);
  auto block = mesh->blocks[0];
  auto coord = block->pcoord;
  constexpr double Lz = 6.488906699059797, m = 1.49;

  MeshVariables vars(1);
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::kFloat64);
  auto temp = (1. + Lz - coord->x1v.cpu()).view({1, 1, -1});
  w[IDN] = temp.pow(m).expand_as(w[IDN]);
  w[IPR] = w[IDN] * temp;
  auto cells = block->part({0, 0, 0}, PartOptions().exterior(false));
  auto dx = coord->dx1f.cpu().narrow(0, coord->il(), coord->options->nx1());
  auto [balanced, residual, sweeps] =
      balance_column(w.index(cells).contiguous(), dx.contiguous(), 2.49,
                     /*wall_clamp=*/true, /*rtol=*/5.e-14, /*max_iter=*/400);
  w.index_put_(cells, balanced);
  vars[0]["hydro_w"] = w.to(device);
  mesh->initialize(vars);

  int nstage = block->pintg->stages.size();
  for (int step = 0; step < 10; ++step) {
    auto dt = mesh->max_time_step(vars);
    for (int stage = 0; stage < nstage; ++stage) mesh->forward(vars, dt, stage);
  }

  auto in = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto wf = vars[0].at("hydro_w");
  auto cs = (5. / 3. * wf[IPR].index(in) / wf[IDN].index(in)).sqrt();
  auto mach2 = (wf[IVY].index(in).abs() / cs).max().item<double>();
  auto mach = (wf.narrow(0, IVX, 3).square().sum(0).index(in).sqrt() / cs)
                  .max()
                  .item<double>();
  // RED: on 5eeb9b6 u2 is not zero in the wall row at the x2 edge
  EXPECT_LE(mach2, 1e-12) << "max |u2|/c_s " << mach2;
  EXPECT_LE(mach, 1e-12) << "max |v|/c_s " << mach;
}

TEST(WallCorner, x2_uniform_rest_keeps_u2_zero) {
  x2_uniform_rest_keeps_u2_zero(torch::kCPU, "reflecting");
}

TEST(WallCorner, x2_uniform_rest_keeps_u2_zero_cuda) {
#ifndef USE_CUDA
  GTEST_SKIP() << "CUDA support is disabled in this build";
#endif
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is not available";
  x2_uniform_rest_keeps_u2_zero(torch::Device(torch::kCUDA, 0), "reflecting");
}

TEST(WallCorner, user_wall_keeps_u2_zero) {
  x2_uniform_rest_keeps_u2_zero(torch::kCPU, "user_wall");
}

TEST(WallCorner, user_wall_keeps_u2_zero_cuda) {
#ifndef USE_CUDA
  GTEST_SKIP() << "CUDA support is disabled in this build";
#endif
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is not available";
  x2_uniform_rest_keeps_u2_zero(torch::Device(torch::kCUDA, 0), "user_wall");
}

// Compare every interior primitive and conserved field after ten identical
// steps. The x2 split introduces extra wall corners but no physical boundary.
namespace {
std::vector<torch::Tensor> wall_corner_fields(int nb2, torch::Device device,
                                              bool tracer = false) {
  auto mesh = make_mesh(device, "reflecting", nb2, tracer);
  EXPECT_EQ(mesh->blocks.size(), static_cast<size_t>(nb2));
  MeshVariables vars(mesh->blocks.size());
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto coord = block->pcoord;
    constexpr double Lz = 6.488906699059797, m = 1.49;

    auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                           coord->options->nc2(), coord->options->nc1()},
                          torch::kFloat64);
    auto temp = (1. + Lz - coord->x1v.cpu()).view({1, 1, -1});
    w[IDN] = temp.pow(m).expand_as(w[IDN]);
    w[IPR] = w[IDN] * temp;
    auto cells = block->part({0, 0, 0}, PartOptions().exterior(false));
    auto dx = coord->dx1f.cpu().narrow(0, coord->il(), coord->options->nx1());
    auto [balanced, residual, sweeps] =
        balance_column(w.index(cells).contiguous(), dx.contiguous(), 2.49,
                       /*wall_clamp=*/true, /*rtol=*/5.e-14, /*max_iter=*/400);
    w.index_put_(cells, balanced);
    // A smooth tangential velocity makes each corner depend on its exchanged
    // column; copying the adjacent edge column is wrong even on a balanced
    // background. Coordinates and the perturbation are global for both grids.
    auto phase = 2. * std::acos(-1.) * coord->x2v.cpu() / 1.6222266747649493;
    w[IVY] = (1.e-3 * phase.sin()).view({1, -1, 1}).expand_as(w[IVY]);
    vars[b]["hydro_w"] = w.to(device);
    if (tracer) {
      // Vary along both axes, including the wall/seam corner patches.
      auto x1 = coord->x1v.cpu().view({1, 1, -1}) / Lz;
      auto x2 = phase.view({1, -1, 1});
      auto r = (0.25 + 0.03 * x1 + 0.02 * x2.sin() + 0.01 * x1 * x2.cos())
                   .expand_as(w[IDN])
                   .unsqueeze(0)
                   .clone();
      vars[b]["scalar_r"] = r.to(device);
      for (int i : {-1, 1}) {
        for (int j : {-1, 1}) {
          auto corner = r.index(block->part({0, j, i}, PartOptions()));
          EXPECT_GT((corner.select(3, 0) - corner.select(3, 1))
                        .abs()
                        .max()
                        .item<double>(),
                    0.);
          EXPECT_GT((corner.select(2, 0) - corner.select(2, 1))
                        .abs()
                        .max()
                        .item<double>(),
                    0.);
        }
      }
    }
  }
  mesh->initialize(vars);
  for (int step = 0; step < 10; ++step) {
    auto dt = mesh->max_time_step(vars);
    for (int stage = 0; stage < mesh->blocks[0]->pintg->stages.size(); ++stage)
      mesh->forward(vars, dt, stage);
  }
  if (tracer) {
    // Exchange refreshes conserved wall corners. The cached primitive must
    // describe those same cells when Mesh::forward returns.
    for (size_t b = 0; b < mesh->blocks.size(); ++b) {
      auto const& v = vars[b];
      auto expected = v.at("scalar_s") / v.at("hydro_u")[IDN].unsqueeze(0);
      for (int i : {-1, 1}) {
        for (int j : {-1, 1}) {
          auto corner = mesh->blocks[b]->part({0, j, i}, PartOptions());
          auto error = (v.at("scalar_r").index(corner) - expected.index(corner))
                           .abs()
                           .max()
                           .item<double>();
          EXPECT_EQ(error, 0.) << "block " << b << " corner " << i << "," << j;
        }
      }
    }
  }
  std::vector<size_t> order;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) order.push_back(b);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return mesh->blocks[a]->pcoord->options->x2min() <
           mesh->blocks[b]->pcoord->options->x2min();
  });
  std::vector<torch::Tensor> result;
  for (auto name : {"hydro_w", "hydro_u"}) {
    std::vector<torch::Tensor> parts;
    for (auto b : order) {
      auto cells =
          mesh->blocks[b]->part({0, 0, 0}, PartOptions().exterior(false));
      parts.push_back(vars[b].at(name).index(cells).clone());
    }
    result.push_back(torch::cat(parts, 2));
  }
  return result;
}
}  // namespace

void x2_split_matches_one_block_exactly(torch::Device device) {
  torch::set_num_threads(1);
  auto one = wall_corner_fields(1, device);
  auto two = wall_corner_fields(2, device);
  for (size_t field = 0; field < one.size(); ++field) {
    ASSERT_EQ(one[field].sizes(), two[field].sizes());
    double error = (one[field] - two[field]).abs().max().item<double>();
    std::printf("wall corner %s %s max abs error = %.17g\n",
                device.is_cuda() ? "cuda" : "cpu",
                field == 0 ? "hydro_w" : "hydro_u", error);
    EXPECT_EQ(error, 0.) << (field == 0 ? "hydro_w" : "hydro_u");
  }
}

TEST(WallCorner, x2_split_matches_one_block_exactly) {
  x2_split_matches_one_block_exactly(torch::kCPU);
}

TEST(WallCorner, x2_split_matches_one_block_exactly_cuda) {
#ifndef USE_CUDA
  GTEST_SKIP() << "CUDA support is disabled in this build";
#endif
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is not available";
  x2_split_matches_one_block_exactly(torch::Device(torch::kCUDA, 0));
}

void scalar_corner_primitive_matches_conserved(torch::Device device) {
  torch::set_num_threads(1);
  for (int nb2 : {1, 2}) wall_corner_fields(nb2, device, true);
}

TEST(WallCorner, scalar_corner_primitive_matches_conserved) {
  scalar_corner_primitive_matches_conserved(torch::kCPU);
}

TEST(WallCorner, scalar_corner_primitive_matches_conserved_cuda) {
#ifndef USE_CUDA
  GTEST_SKIP() << "CUDA support is disabled in this build";
#endif
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is not available";
  scalar_corner_primitive_matches_conserved(torch::Device(torch::kCUDA, 0));
}
