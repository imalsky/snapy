#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

// cp5 at the default nghost 1 reads 3 ghosts: refused at setup (S184); before
// that it gave 4 on the first x1 face of w = 2 + x1, where 2 is exact
TEST(Cp5Nghost, refuses_or_reconstructs_the_first_x1_face) {
  std::ofstream("cp5_nghost.yaml") << R"(
reference-state: {Tref: 300., Pref: 1.e5}
species: [{name: dry, composition: {O: 0.42, N: 1.56, Ar: 0.01}, cv_R: 2.5}]
geometry: {type: cartesian, cells: {nx1: 16}, bounds: {x1min: 0., x1max: 16.}}
dynamics: {equation-of-state: {type: ideal-gas}, reconstruct: {vertical: {type: cp5, scale: false, shock: false}}}
boundary-condition: {external: {x1-inner: reflecting, x1-outer: reflecting}}
)";
  std::shared_ptr<snap::MeshBlockImpl> b;
  try {
    b = std::make_shared<snap::MeshBlockImpl>(
        snap::MeshBlockOptionsImpl::from_yaml("cp5_nghost.yaml"));
  } catch (c10::Error const&) {
  }
  std::remove("cp5_nghost.yaml");
  if (!b) return;
  auto w = (2. + b->pcoord->x1v).view({1, 1, 1, -1}).repeat({5, 1, 1, 1});
  auto wlr = b->phydro->precon1->forward(w, 3);
  int ng = b->pcoord->options->nghost();
  EXPECT_DOUBLE_EQ(wlr[snap::ILT][snap::IDN][0][0][ng].item<double>(), 2.);
}
