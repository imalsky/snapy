#include <gtest/gtest.h>
#include <snap/snap.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/scalar/scalar.hpp>
#include <string>
#include <vector>
#if defined(__has_include)
#if __has_include(<snap/recon/x1_coeffs.hpp>)
#define SNAPY_HAS_ROUTE_A 1
#endif
#endif
using namespace snap;

namespace {
std::string deck(char const* interp, int ntracer, int nx1,
                 std::string const& cells_extra = "") {
  std::string names = "[";
  for (int i = 0; i < ntracer; ++i) {
    names += (i ? ", tracer" : "tracer") + std::to_string(i);
  }
  names += "]";
  char buf[4096];
  std::snprintf(buf, sizeof(buf), R"(
geometry:
  type: cartesian
  bounds: {x1min: 0., x1max: 16., x2min: 0., x2max: 1., x3min: 0., x3max: 1.}
  cells: {nx1: %d, nx2: 8, nx3: 1, nghost: 3%s}
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
scalar:
  nvar: %d
  names: %s
  reconstruct: {type: %s, scale: false, shock: false}
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
                nx1, cells_extra.c_str(), interp, interp, ntracer,
                names.c_str(), interp);
  return buf;
}
std::shared_ptr<MeshBlockImpl> make_block(std::string const& config) {
  char fname[] = "/tmp/test-uniform-module-tree-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << config;
  auto options = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  return std::make_shared<MeshBlockImpl>(options);
}
std::vector<std::string> buffer_names(MeshBlockImpl const& block) {
  std::vector<std::string> names;
  for (auto const& item : block.named_buffers(/*recurse=*/true)) {
    names.push_back(item.key());
  }
  std::sort(names.begin(), names.end());
  return names;
}
std::filesystem::path key_reporting_forcing() {
  auto path = std::filesystem::temp_directory_path() / "snapy_280_keys.pt";
  torch::jit::Module module("snapy_280_keys");
  module.define(
      "def forward(self, variables: Dict[str, Tensor], dt: float, stage: int) "
      "-> Dict[str, Tensor]:\n"
      "  msg = \"FORCINGKEYS<\" + \"|\".join(sorted(list(variables.keys()))) "
      "+ \">\"\n"
      "  assert dt < -1.0, msg\n"
      "  return {\"hydro_du\": torch.zeros_like(variables[\"hydro_u\"])}\n");
  module.save(path.string());
  return path;
}
std::vector<std::string> forcing_keys(std::shared_ptr<MeshBlockImpl> block) {
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::dtype(torch::kFloat64));
  vars["hydro_w"][IDN].fill_(1.);
  vars["hydro_w"][IPR].fill_(3.);
  if (block->pscalar->nvar() > 0) {
    vars["scalar_r"] = torch::full({block->pscalar->nvar(), nc3, nc2, nc1},
                                   1.e-3, torch::dtype(torch::kFloat64));
  }
  block->initialize(vars);
  auto forcing = key_reporting_forcing();
  block->set_user_stage_forcings({forcing.string()});
  std::string msg;
  try {
    block->advance_local(vars, 0., 0);
  } catch (std::exception const& exc) {
    msg = exc.what();
  }
  std::filesystem::remove(forcing);
  auto lo = msg.rfind("FORCINGKEYS<");
  EXPECT_NE(lo, std::string::npos)
      << "the stage forcing never ran, so the input dictionary was never "
         "built; message was: "
      << msg;
  if (lo == std::string::npos) return {};
  lo += std::string("FORCINGKEYS<").size();
  auto hi = msg.find('>', lo);
  EXPECT_NE(hi, std::string::npos);
  std::vector<std::string> keys;
  std::string field = msg.substr(lo, hi - lo);
  size_t at = 0;
  while (at <= field.size()) {
    auto bar = field.find('|', at);
    if (bar == std::string::npos) bar = field.size();
    if (bar > at) keys.push_back(field.substr(at, bar - at));
    at = bar + 1;
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}
bool ends_with(std::string const& s, char const* tail) {
  std::string t(tail);
  return s.size() >= t.size() &&
         s.compare(s.size() - t.size(), t.size(), t) == 0;
}
std::vector<std::string> route_a_buffers(
    std::vector<std::string> const& names) {
  std::vector<std::string> hits;
  for (auto const& n : names) {
    if (ends_with(n, "cm_x1") || ends_with(n, "cp_x1")) hits.push_back(n);
  }
  return hits;
}
std::string joined(std::vector<std::string> const& v) {
  std::string s;
  for (auto const& e : v) s += (s.empty() ? "" : ", ") + e;
  return s;
}
bool has_child(torch::nn::Module& m, char const* name) {
  for (auto const& item : m.named_children()) {
    if (item.key() == name) return true;
  }
  return false;
}
struct Case {
  char const* tag;
  char const* interp;
  int ntracer;
  int nx1;
};
constexpr Case kUniform[] = {{"uniform_weno5", "weno5", 1, 16},
                             {"uniform_cp5", "cp5", 2, 20}};
}  // namespace

TEST(UniformModuleTree, a_uniform_deck_gains_no_route_a_buffer) {
  for (auto const& c : kUniform) {
    auto block = make_block(deck(c.interp, c.ntracer, c.nx1));
#ifdef SNAPY_HAS_ROUTE_A
    ASSERT_FALSE(block->pcoord->options->stretched_x1())
        << c.tag << " is supposed to be a uniform deck";
#endif
    auto buffers = buffer_names(*block);
    auto keys = forcing_keys(block);
    EXPECT_TRUE(
        std::includes(keys.begin(), keys.end(), buffers.begin(), buffers.end()))
        << c.tag << ": the forcing dictionary does not contain every buffer";
    EXPECT_TRUE(route_a_buffers(buffers).empty())
        << c.tag << ": a uniform deck carries route A buffers: "
        << joined(route_a_buffers(buffers));
    EXPECT_TRUE(route_a_buffers(keys).empty())
        << c.tag
        << ": a user forcing on a uniform deck is handed route A "
           "keys: "
        << joined(route_a_buffers(keys));
  }
}

TEST(UniformModuleTree, a_uniform_deck_gains_no_scalar_recon1) {
  for (auto const& c : kUniform) {
    auto block = make_block(deck(c.interp, c.ntracer, c.nx1));
    ASSERT_GT(block->pscalar->nvar(), 0) << c.tag;
    EXPECT_FALSE(has_child(*block->pscalar, "recon1"))
        << c.tag
        << ": the scalar module carries a second Reconstruct that a uniform "
           "deck never needs";
  }
}
#ifdef SNAPY_HAS_ROUTE_A

TEST(UniformModuleTree, a_stretched_deck_still_installs_everything) {
  auto block = make_block(deck("weno5", 1, 16, ", x1rat: 1.05"));
  ASSERT_TRUE(block->pcoord->options->stretched_x1());
  auto buffers = buffer_names(*block);
  auto hits = route_a_buffers(buffers);
  EXPECT_EQ(hits.size(), 8u) << "installed route A buffers: " << joined(hits);
  EXPECT_TRUE(has_child(*block->pscalar, "recon1"))
      << "a stretched deck's tracers have no x1 reconstruction of their own";
}

TEST(UniformModuleTree, clone_keeps_the_installed_tables) {
  auto block = make_block(deck("weno5", 1, 16, ", x1rat: 1.05"));
  auto copy = std::dynamic_pointer_cast<MeshBlockImpl>(block->clone());
  ASSERT_NE(copy, nullptr);
  auto before = route_a_buffers(buffer_names(*block));
  auto after = route_a_buffers(buffer_names(*copy));
  EXPECT_EQ(before, after);
  for (auto const& p :
       {copy->phydro->precon1->pinterp1, copy->phydro->precon1->pinterp2}) {
    EXPECT_TRUE(p->has_x1_coeffs())
        << p->options->type() << " lost its table in the clone";
  }
}
#endif  // SNAPY_HAS_ROUTE_A

TEST(UniformModuleTree, clone_of_a_uniform_deck_stays_empty) {
  auto block = make_block(deck("weno5", 1, 16));
  auto copy = std::dynamic_pointer_cast<MeshBlockImpl>(block->clone());
  ASSERT_NE(copy, nullptr);
  EXPECT_TRUE(route_a_buffers(buffer_names(*copy)).empty());
  EXPECT_EQ(buffer_names(*block), buffer_names(*copy));
}
