#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <snap/coord/coordinate.hpp>
#include <string>
#include <vector>
using namespace snap;

namespace {
std::string card(std::string const& cells, std::string const& extra = "",
                 std::string const& bounds = "{}") {
  return "geometry:\n"
         "  type: cartesian\n"
         "  bounds: " +
         bounds +
         "\n"
         "  cells: " +
         cells + "\n" + extra;
}
std::string face_list(std::vector<double> const& f) {
  std::string s = "  x1f: [";
  char buf[32];
  for (size_t k = 0; k < f.size(); ++k) {
    std::snprintf(buf, sizeof(buf), "%.17g", f[k]);
    s += (k ? ", " : "") + std::string(buf);
  }
  return s + "]\n";
}
CoordinateOptions load(std::string const& config) {
  char fname[] = "/tmp/test-x1-qmax-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << config;
  CoordinateOptions op;
  try {
    op = CoordinateOptionsImpl::from_yaml(fname);
  } catch (...) {
    std::remove(fname);
    throw;
  }
  std::remove(fname);
  return op;
}
std::vector<double> faces_of(std::vector<double> const& w) {
  std::vector<double> f{0.};
  for (double h : w) f.push_back(f.back() + h);
  return f;
}
std::vector<double> junction(int n, double s) {
  std::vector<double> w;
  for (int i = 0; i < n / 2; ++i) w.push_back(1.);
  for (int i = 0; i < n / 2; ++i) w.push_back(s);
  return faces_of(w);
}
double worst_ratio(Cartesian const& coord) {
  auto d = coord->dx1f;
  double worst = 1.;
  for (int64_t i = 0; i + 1 < d.size(0); ++i) {
    double a = d[i].item<double>(), b = d[i + 1].item<double>();
    worst = std::max(worst, std::max(a / b, b / a));
  }
  return worst;
}
}  // namespace

TEST(X1QmaxGuard, a_junction_deck_is_refused_naming_index_ratio_and_bound) {
  for (double s : {2., 4., 16.}) {
    const int n = 16;
    auto text = card("{nx1: " + std::to_string(n) + ", nghost: 3}",
                     face_list(junction(n, s)));
    try {
      load(text);
      ADD_FAILURE() << "an s = " << s << " junction deck was accepted";
    } catch (c10::Error const& e) {
      std::string msg = e.what();
      EXPECT_NE(msg.find("face " + std::to_string(n / 2)), std::string::npos)
          << "s = " << s << ": " << msg;
      EXPECT_NE(msg.find("1.5"), std::string::npos)
          << "s = " << s << ": " << msg;
      char want[32];
      std::snprintf(want, sizeof(want), "%g", s);
      EXPECT_NE(msg.find(want), std::string::npos)
          << "s = " << s << ": " << msg;
    }
  }
}

TEST(X1QmaxGuard, a_downward_junction_is_refused_as_well) {
  std::vector<double> w;
  for (int i = 0; i < 8; ++i) w.push_back(2.);
  for (int i = 0; i < 8; ++i) w.push_back(1.);
  EXPECT_THROW(load(card("{nx1: 16, nghost: 3}", face_list(faces_of(w)))),
               c10::Error);
}

TEST(X1QmaxGuard, every_contract_x1rat_deck_still_constructs) {
  for (double q : {1.058, 1. / 1.058, 1.2, 1. / 1.2, 0.95, 1.05, 1.1, 1.01}) {
    for (int n : {16, 32, 64, 128}) {
      char b[32];
      std::snprintf(b, sizeof(b), "%.17g", q);
      std::string cells =
          "{nx1: " + std::to_string(n) + ", nghost: 3, x1rat: " + b + "}";
      CoordinateOptions op;
      ASSERT_NO_THROW(op = load(card(cells, "", "{x1min: 0., x1max: 1.}")))
          << cells;
      ASSERT_TRUE(op->stretched_x1()) << cells;
      EXPECT_NO_THROW(Cartesian{op}) << cells;
    }
  }
}

TEST(X1QmaxGuard, the_largest_measured_contract_ratio_is_accepted) {
  auto op = load(card("{nx1: 128, nghost: 3, x1rat: 0.83333333333333337}", "",
                      "{x1min: 0., x1max: 1.}"));
  Cartesian coord(op);
  double worst = worst_ratio(coord);
  EXPECT_GT(worst, 1.19) << "this deck is meant to be the 1.2 stress case";
  EXPECT_LT(worst, 1.5) << "measured worst adjacent ratio " << worst;
}

TEST(X1QmaxGuard, a_non_self_similar_list_inside_the_bound_is_accepted) {
  std::vector<double> w{1.6, 2.2, 1.6, 2.2, 1.7, 2.3, 1.8, 2.6};
  auto op = load(card("{nx1: 8, nghost: 3}", face_list(faces_of(w))));
  ASSERT_TRUE(op->stretched_x1());
  Cartesian coord(op);
  EXPECT_LT(worst_ratio(coord), 1.5);
}

TEST(X1QmaxGuard, the_uniform_path_is_untouched) {
  for (char const* cells :
       {"{nx1: 16, nghost: 3}", "{nx1: 16, nghost: 3, x1rat: 1.}"}) {
    auto op = load(card(cells, "", "{x1min: 0., x1max: 16.}"));
    EXPECT_FALSE(op->stretched_x1()) << cells;
    EXPECT_NO_THROW(Cartesian{op}) << cells;
    Cartesian coord(op);
    auto d = coord->dx1f;
    EXPECT_TRUE(torch::equal(d, torch::full_like(d, d[0].item<double>())))
        << cells;
  }
}
