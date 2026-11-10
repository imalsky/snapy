// C/C++
#include <unistd.h>

#include <algorithm>
#include <fstream>

// external
#include <gtest/gtest.h>

// snap
#include <snap/mesh/meshblock.hpp>

// tests
#include "device_testing.hpp"

using namespace snap;

namespace {

const char* scalar_config = R"(
reference-state:
  Tref: 300.
  Pref: 1.e5

species:
  - name: dry
    composition: {O: 0.42, N: 1.56, Ar: 0.01}
    cv_R: 2.5

dynamics:
  equation-of-state:
    type: ideal-gas
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver:
    type: lmars

scalar:
  nvar: 2
  names: [tr0, tr1]
  reconstruct: {type: weno5, shock: false}

geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 1., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: 16, nx2: 8, nx3: 4, nghost: 3}

boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: reflecting
    x2-outer: reflecting
    x3-inner: reflecting
    x3-outer: reflecting
)";

std::string write_temp_config() {
  char fname[] = "/tmp/snapy_scalar_test.XXXXXX";
  int fd = mkstemp(fname);
  close(fd);
  std::ofstream out(fname);
  out << scalar_config;
  out.close();
  return fname;
}

}  // namespace

TEST_P(DeviceTest, initialize_and_transport_scalar) {
  auto filename = write_temp_config();
  auto opts = MeshBlockOptionsImpl::from_yaml(filename);
  auto block = MeshBlock(opts);
  block->to(device, dtype);

  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] =
      torch::zeros({5, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  vars["hydro_w"][IDN].fill_(2.0);
  vars["hydro_w"][IPR].fill_(3.0);
  vars["scalar_r"] =
      torch::zeros({2, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  vars["scalar_r"][0].fill_(0.25);
  vars["scalar_r"][1].fill_(0.5);

  block->initialize(vars);

  ASSERT_TRUE(vars.count("scalar_s"));
  ASSERT_EQ(vars.at("scalar_s").sizes(),
            std::vector<int64_t>({2, nc3, nc2, nc1}));
  EXPECT_TRUE(torch::allclose(
      vars.at("scalar_s"),
      vars.at("hydro_w")[IDN].unsqueeze(0) * vars.at("scalar_r")));

  auto ds = block->pscalar->forward(1.0e-3, vars.at("scalar_s"), vars);
  EXPECT_EQ(ds.sizes(), vars.at("scalar_s").sizes());
  EXPECT_EQ(ds.device(), vars.at("scalar_s").device());
  EXPECT_EQ(ds.scalar_type(), vars.at("scalar_s").scalar_type());
  EXPECT_TRUE(torch::isfinite(ds).all().item<bool>());

  std::remove(filename.c_str());
}

// The upper bound is the positivity of the complement b*rho - s, so it must
// hold both sides: r never above b, and s never below zero. A cell already at
// the bound is the case that separates blending toward b*F_mass from blending
// toward zero -- the latter bounds r too, but freezes the plateau instead of
// advecting it.
TEST_P(DeviceTest, scalar_upper_bound_holds_both_sides) {
  auto filename = write_temp_config();
  auto opts = MeshBlockOptionsImpl::from_yaml(filename);
  opts->scalar()->upper_bound() = 1.0;
  opts->hydro()->eos()->limiter() =
      true;  // the upper bound requires its counterpart
  auto block = MeshBlock(opts);
  block->to(device, dtype);

  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] =
      torch::zeros({5, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  vars["hydro_w"][IDN].fill_(2.0);
  vars["hydro_w"][IPR].fill_(3.0);
  vars["hydro_w"][IVX].fill_(1.0);
  vars["scalar_r"] =
      torch::zeros({2, nc3, nc2, nc1}, torch::device(device).dtype(dtype));
  vars["scalar_r"][0].narrow(2, nc1 / 3, nc1 / 4).fill_(1.0);  // at the bound
  vars["scalar_r"][1].fill_(0.5);

  block->initialize(vars);
  auto s = vars.at("scalar_s");
  auto rho = vars.at("hydro_w")[IDN].unsqueeze(0);
  auto s0 = s.clone();

  // The hydro is never advanced here, so give the tracer a mass flux to ride
  // on; a uniform one has zero divergence, which keeps rho consistent with
  // hydro_w.
  block->phydro->flux1()[IDN].fill_(2.0);
  BoundaryFuncOptions bops;
  bops.nghost(block->pcoord->options->nghost());
  bops.type(kScalar);
  for (int n = 0; n < 12; ++n) {
    s = s + block->pscalar->forward(2.0e-2, s, vars);
    // a real step refills ghosts before the next one; without it theta's ghost
    // values go stale and the donor factors at the interior edge are wrong
    for (int i = 0; i < block->options->bfuncs().size(); ++i) {
      if (block->options->bfuncs()[i] == nullptr) continue;
      block->options->bfuncs()[i](s, 3 - i / 2, bops);
    }
  }
  auto r = s / rho;
  double tol = (dtype == torch::kFloat32) ? 1e-5 : 1e-12;

  EXPECT_LE(r.max().template item<double>(), 1.0 + tol);
  EXPECT_GE(s.min().template item<double>(), -tol);

  // The two bounds above are satisfied by a limiter that does nothing at all:
  // with theta identically zero every face flux collapses to b*F_mass, whose
  // divergence vanishes on this uniform mass flux, so the field never moves and
  // both assertions pass. Pin the content, not just the flag.
  int ng = block->pcoord->options->nghost();
  auto in = torch::indexing::Slice(ng, nc1 - ng);
  auto hat = s.index({0, "...", in});
  auto hat0 = s0.index({0, "...", in});

  // (a) the top-hat advected at all -- this is what kills theta == 0.
  EXPECT_GT((hat - hat0).abs().max().template item<double>(), tol);

  // (b) and it went DOWNSTREAM, which diffusion alone would not do. The wind is
  // +x1 at 1.0 for 12 steps of 2e-2.
  auto x = torch::arange(hat.size(-1), hat.options());
  auto centroid = [&](torch::Tensor f) {
    return (f * x).sum().template item<double>() /
           std::max(f.sum().template item<double>(), 1e-30);
  };
  EXPECT_GT(centroid(hat), centroid(hat0) + 1.0);

  // (c) the second channel is uniform, so a correct scheme leaves it alone.
  EXPECT_NEAR(s.index({1, "...", in}).min().template item<double>(),
              s.index({1, "...", in}).max().template item<double>(), tol);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
