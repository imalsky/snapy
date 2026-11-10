#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/recon/interpolation.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/recon/x1_coeffs.hpp>
#include <snap/scalar/scalar.hpp>
#include <string>
#include <utility>
#include <vector>
using namespace snap;

namespace {
constexpr int kNx1 = 16, kNghost = 3;
constexpr double kX1min = 0., kX1max = 16.;
enum { DIM1 = 3, DIM2 = 2 };
std::string deck(std::string const& cells_extra,
                 std::string const& geom_extra = "") {
  char buf[2048];
  std::snprintf(buf, sizeof(buf), R"(
geometry:
  type: cartesian
  bounds: {x1min: %g, x1max: %g, x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: %d, nx2: 8, nx3: 1, nghost: %d%s}
%sdynamics:
  equation-of-state:
    type: ideal-gas
    gammad: 1.4
    weight: 8.31446
  reconstruct:
    vertical: {type: weno5, scale: false, shock: false}
    horizontal: {type: weno5, scale: false, shock: false}
  riemann-solver:
    type: lmars
scalar:
  nvar: 1
  names: [tracer]
  reconstruct: {type: weno5, shock: false}
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
                geom_extra.c_str());
  return buf;
}
Mesh make_mesh(std::string const& config) {
  char fname[] = "/tmp/test-deck-x1-wiring-XXXXXX";
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
std::string face_list(double q) {
  double lq = std::log(q), den = std::expm1(kNx1 * lq);
  std::string s = "  x1f: [";
  char b[32];
  for (int k = 0; k < kNx1; ++k) {
    std::snprintf(b, sizeof(b), "%.17g",
                  kX1min + (kX1max - kX1min) * std::expm1(k * lq) / den);
    s += (k ? ", " : "") + std::string(b);
  }
  std::snprintf(b, sizeof(b), "%.17g", kX1max);
  return s + ", " + b + "]\n";
}
using Tables = std::pair<torch::Tensor, torch::Tensor>;
Tables installed(Interp const& p) {
  if (auto w = std::dynamic_pointer_cast<Weno5InterpImpl>(p)) {
    return {w->cm_x1, w->cp_x1};
  }
  auto c = std::dynamic_pointer_cast<Center5InterpImpl>(p);
  EXPECT_NE(c, nullptr) << "a weno5 deck holds weno5 + cp5";
  return c ? Tables{c->cm_x1, c->cp_x1} : Tables{};
}
Tables from_faces(Interp const& p, torch::Tensor const& x1f) {
  return std::dynamic_pointer_cast<Weno5InterpImpl>(p) ? weno5_x1_coeffs(x1f)
                                                       : cp5_x1_coeffs(x1f);
}
void expect_faces_installed(ReconstructImpl const& recon,
                            torch::Tensor const& x1f) {
  for (auto const& p : {recon.pinterp1, recon.pinterp2}) {
    EXPECT_TRUE(p->has_x1_coeffs())
        << p->options->type()
        << " on recon1 has no x1 coefficients: the deck's stretched x1 never "
           "reached the reconstructor";
    auto got = installed(p), want = from_faces(p, x1f);
    ASSERT_EQ(got.first.numel(), want.first.numel()) << p->options->type();
    EXPECT_TRUE(torch::equal(got.first, want.first))
        << "installed cm rows are not the ones this block's x1f gives";
    EXPECT_TRUE(torch::equal(got.second, want.second))
        << "installed cp rows are not the ones this block's x1f gives";
  }
}
void expect_no_faces(ReconstructImpl const& recon, char const* what) {
  for (auto const& p : {recon.pinterp1, recon.pinterp2}) {
    EXPECT_FALSE(p->has_x1_coeffs()) << what << " must hold no x1 table";
    auto got = installed(p);
    EXPECT_EQ(got.first.numel(), 0) << what;
    EXPECT_EQ(got.second.numel(), 0) << what;
  }
}
std::shared_ptr<ReconstructImpl> scalar_x1(Scalar const& ps) {
  for (auto const& item : ps->named_children()) {
    if (item.key() == "recon1")
      return std::dynamic_pointer_cast<ReconstructImpl>(item.value());
  }
  return ps->precon.ptr();
}
void expect_forward_runs(MeshBlock pmb) {
  auto coord = pmb->pcoord;
  int nc1 = coord->options->nc1(), nc2 = coord->options->nc2(),
      nc3 = coord->options->nc3();
  auto w = torch::ones({pmb->phydro->peos->nvar(), nc3, nc2, nc1},
                       coord->x1v.options());
  auto wlr = pmb->phydro->precon1->forward(w, DIM1);
  EXPECT_EQ(wlr.size(DIM1 + 1), nc1);
  auto r =
      torch::ones({pmb->pscalar->nvar(), nc3, nc2, nc1}, coord->x1v.options());
  EXPECT_EQ(scalar_x1(pmb->pscalar)->forward(r, DIM1).size(DIM1 + 1), nc1);
  EXPECT_EQ(pmb->pscalar->precon->forward(r, DIM2).size(DIM2 + 1), nc2);
}
void check_stretched(std::string const& config) {
  auto mesh = make_mesh(config);
  auto pmb = mesh->blocks.front();
  ASSERT_TRUE(pmb->pcoord->options->stretched_x1())
      << "the deck's grid key did not reach the coordinate";
  expect_faces_installed(*pmb->phydro->precon1, pmb->pcoord->x1f);
  expect_faces_installed(*scalar_x1(pmb->pscalar), pmb->pcoord->x1f);
  expect_no_faces(*pmb->phydro->precon23, "recon23");
  expect_no_faces(*pmb->pscalar->precon, "scalar x2/x3");
  expect_forward_runs(pmb);
}
}  // namespace

TEST(DeckX1Wiring, x1rat_deck_installs_the_block_faces_on_recon1) {
  check_stretched(deck(", x1rat: 1.05"));
}

TEST(DeckX1Wiring, face_list_deck_installs_the_block_faces_on_recon1) {
  check_stretched(deck("", face_list(1.05)));
}

TEST(DeckX1Wiring, uniform_deck_installs_nothing) {
  for (char const* cells : {"", ", x1rat: 1."}) {
    auto mesh = make_mesh(deck(cells));
    auto pmb = mesh->blocks.front();
    EXPECT_FALSE(pmb->pcoord->options->stretched_x1())
        << "a uniform deck must not look stretched";
    expect_no_faces(*pmb->phydro->precon1, "recon1 on a uniform deck");
    expect_no_faces(*scalar_x1(pmb->pscalar), "scalar x1 on a uniform deck");
    expect_no_faces(*pmb->phydro->precon23, "recon23 on a uniform deck");
    expect_no_faces(*pmb->pscalar->precon, "scalar x2/x3 on a uniform deck");
    expect_forward_runs(pmb);
  }
}
