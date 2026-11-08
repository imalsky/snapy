
#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/recon/interpolation.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/recon/x1_coeffs.hpp>
#include <string>
#include <vector>

using namespace snap;

namespace {

constexpr int kNx1 = 16, kNghost = 3;
constexpr double kX1min = 0., kX1max = 16.;
enum { DIM1 = 3 };

std::vector<std::string> const kRefused = {"dc", "plm", "ppm", "cp3", "weno3"};
std::vector<std::string> const kCovered = {"weno5", "cp5"};

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

std::string deck(std::string const& type, std::string const& cells_extra) {
  char buf[2048];
  std::snprintf(buf, sizeof(buf), R"(
geometry:
  type: cartesian
  bounds: {x1min: %g, x1max: %g, x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: %d, nx2: 1, nx3: 1, nghost: %d%s}
dynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    weight: 8.31446
  reconstruct:
    vertical: {type: %s, scale: false, shock: false}
    horizontal: {type: %s, scale: false, shock: false}
  riemann-solver:
    type: lmars
integration:
  type: rk3
  cfl: 0.4
boundary-condition:
  external:
    x1-inner: reflecting
    x1-outer: reflecting
    x2-inner: periodic
    x2-outer: periodic
    x3-inner: periodic
    x3-outer: periodic
)",
                kX1min, kX1max, kNx1, kNghost, cells_extra.c_str(),
                type.c_str(), type.c_str());
  return buf;
}

Mesh make_mesh(std::string const& config) {
  char fname[] = "/tmp/test-x1-stretched-refusal-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << config;
  auto block_opts = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  auto mesh_opts = MeshOptionsImpl::create();
  mesh_opts->block(block_opts);
  mesh_opts->blocks_per_process(1);
  return Mesh(mesh_opts);
}

std::string const kStretched = ", x1rat: 1.1";
std::string const kUniform = "";

std::string refusal_of(std::function<void()> const& f) {
  try {
    f();
  } catch (c10::Error const& e) {
    return e.what();
  }
  return "";
}

::testing::AssertionResult refused(std::string const& what,
                                   std::string const& type,
                                   std::string const& msg) {
  if (msg.empty()) {
    return ::testing::AssertionFailure()
           << what << " with x1 reconstruction '" << type
           << "' was accepted: it would run uniform-grid arithmetic on a "
              "stretched x1 and say nothing";
  }
  if (msg.find(type) == std::string::npos ||
      msg.find("weno5") == std::string::npos) {
    return ::testing::AssertionFailure()
           << what << " with '" << type
           << "' was refused, but not by route A's own check (the message "
              "must name the type and the covered pair): "
           << msg;
  }
  return ::testing::AssertionSuccess();
}

}  // namespace

TEST(StretchedX1Refusal, the_five_uncovered_types_refuse_a_face_vector) {
  auto x1f = faces_geometric(21, 1.03);
  for (auto const& type : kRefused) {
    auto opts = ReconstructOptionsImpl::create();
    opts->interp()->type(type);
    Reconstruct recon(opts);
    EXPECT_TRUE(refused("a face vector", type,
                        refusal_of([&] { recon->set_x1_faces(x1f); })));
    EXPECT_FALSE(recon->pinterp1->has_x1_coeffs()) << type;
  }
}

TEST(StretchedX1Refusal, weno5_and_cp5_still_take_a_stretched_x1) {
  auto x1f = faces_geometric(21, 1.03);
  for (auto const& type : kCovered) {
    auto opts = ReconstructOptionsImpl::create();
    opts->interp()->type(type);
    Reconstruct recon(opts);
    ASSERT_NO_THROW(recon->set_x1_faces(x1f)) << type;
    EXPECT_TRUE(recon->pinterp1->has_x1_coeffs()) << type;
    EXPECT_TRUE(recon->pinterp2->has_x1_coeffs()) << type;
    auto w = torch::rand({5, 1, 1, 20}, torch::kFloat64) + 1.;
    auto wlr = recon->forward(w, DIM1);
    EXPECT_TRUE(torch::isfinite(wlr).all().item<bool>()) << type;
  }
}

TEST(StretchedX1Refusal, a_stretched_deck_with_an_uncovered_type_is_refused) {
  for (auto const& type : kRefused) {
    EXPECT_TRUE(refused("a stretched x1rat deck", type, refusal_of([&] {
                          make_mesh(deck(type, kStretched));
                        })));
  }
}

TEST(StretchedX1Refusal, a_stretched_deck_with_a_covered_type_still_builds) {
  for (auto const& type : kCovered) {
    Mesh mesh = nullptr;
    ASSERT_NO_THROW(mesh = make_mesh(deck(type, kStretched))) << type;
    auto pmb = mesh->blocks.front();
    EXPECT_TRUE(pmb->phydro->precon1->pinterp1->has_x1_coeffs()) << type;
    EXPECT_TRUE(pmb->phydro->precon1->pinterp2->has_x1_coeffs()) << type;
    EXPECT_FALSE(pmb->phydro->precon23->pinterp1->has_x1_coeffs()) << type;
  }
}

TEST(StretchedX1Refusal, every_uniform_deck_is_unaffected) {
  std::vector<std::string> all = kRefused;
  all.insert(all.end(), kCovered.begin(), kCovered.end());
  for (auto const& type : all) {
    Mesh mesh = nullptr;
    ASSERT_NO_THROW(mesh = make_mesh(deck(type, kUniform)))
        << "uniform " << type << " deck no longer builds";
    auto pmb = mesh->blocks.front();
    EXPECT_FALSE(pmb->phydro->precon1->pinterp1->has_x1_coeffs()) << type;
    EXPECT_FALSE(pmb->phydro->precon1->pinterp2->has_x1_coeffs()) << type;
    EXPECT_FALSE(pmb->phydro->precon23->pinterp1->has_x1_coeffs()) << type;
  }
}
