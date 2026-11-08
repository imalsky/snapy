#include <gtest/gtest.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <snap/coord/coordinate.hpp>
#include <snap/layout/layout.hpp>
#include <string>
#include <vector>

using namespace snap;

namespace {

std::string card(std::string const& cells, std::string const& extra = "",
                 std::string const& bounds = "{x1min: 0., x1max: 16.}") {
  return "geometry:\n"
         "  type: cartesian\n"
         "  bounds: " +
         bounds +
         "\n"
         "  cells: " +
         cells + "\n" + extra;
}

std::string write_temp_config(std::string const& config) {
  char fname[] = "/tmp/test-stretched-x1-XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_NE(fd, -1);
  if (fd != -1) close(fd);
  std::ofstream(fname) << config;
  return fname;
}

CoordinateOptions load(std::string const& config) {
  auto fname = write_temp_config(config);
  CoordinateOptions op;
  try {
    op = CoordinateOptionsImpl::from_yaml(fname);
  } catch (...) {
    std::remove(fname.c_str());
    throw;
  }
  std::remove(fname.c_str());
  return op;
}

std::vector<double> mirrored(std::vector<double> const& f, int ng) {
  int n = static_cast<int>(f.size()) - 1;
  std::vector<double> g;
  for (int m = ng; m >= 1; --m) g.push_back(2. * f[0] - f[m]);
  g.insert(g.end(), f.begin(), f.end());
  for (int m = 1; m <= ng; ++m) g.push_back(2. * f[n] - f[n - m]);
  return g;
}

torch::Tensor as_tensor(std::vector<double> const& v) {
  return torch::tensor(v, torch::kFloat64);
}

void expect_metrics(Cartesian const& coord, std::vector<double> const& g,
                    int i0) {
  int nc = coord->options->nc1();
  auto ge = as_tensor(g).narrow(0, i0, nc + 3);
  auto ve = 0.5 * (ge.slice(0, 0, nc + 2) + ge.slice(0, 1, nc + 3));
  auto x1f = ge.slice(0, 1, nc + 2);
  EXPECT_TRUE(torch::equal(coord->x1f, x1f)) << coord->x1f << x1f;
  EXPECT_TRUE(
      torch::equal(coord->dx1f, x1f.slice(0, 1, nc + 1) - x1f.slice(0, 0, nc)));
  EXPECT_TRUE(torch::equal(coord->x1v, ve.slice(0, 1, nc + 1)));
  EXPECT_TRUE(
      torch::equal(coord->dx1v, ve.slice(0, 1, nc + 1) - ve.slice(0, 0, nc)))
      << coord->dx1v;
}

const std::vector<double> kFaces = {0.,  1.25, 3.,  5.25, 7.,
                                    9.5, 11.5, 14., 16.};

std::string face_list(std::vector<double> const& f) {
  std::string s = "  x1f: [";
  char buf[32];
  for (size_t k = 0; k < f.size(); ++k) {
    std::snprintf(buf, sizeof(buf), "%.17g", f[k]);
    s += (k ? ", " : "") + std::string(buf);
  }
  return s + "]\n";
}

}  // namespace

TEST(StretchedX1, face_list_gives_mirrored_ghosts_and_metrics) {
  auto op = load(card("{nx1: 8, nghost: 3}", face_list(kFaces)));
  Cartesian coord(op);
  expect_metrics(coord, mirrored(kFaces, 4), 0);
  int ng = 3, n = 8;
  for (int m = 0; m < ng; ++m) {
    EXPECT_EQ(coord->dx1f[ng - 1 - m].item<double>(),
              coord->dx1f[ng + m].item<double>());
    EXPECT_EQ(coord->dx1f[ng + n + m].item<double>(),
              coord->dx1f[ng + n - 1 - m].item<double>());
  }
  EXPECT_EQ(coord->x1f[ng].item<double>(), 0.);
  EXPECT_EQ(coord->x1f[ng + n].item<double>(), 16.);
}

TEST(StretchedX1, face_list_may_omit_the_x1_bounds) {
  auto op = load(card("{nx1: 8, nghost: 3}", face_list(kFaces), "{}"));
  Cartesian coord(op);
  expect_metrics(coord, mirrored(kFaces, 4), 0);
}

TEST(StretchedX1, x1rat_gives_a_geometric_grid) {
  for (std::string q : {"0.94517958412098302", "1.058"}) {
    double qq = std::stod(q);
    Cartesian coord(load(card("{nx1: 16, nghost: 3, x1rat: " + q + "}")));
    int ng = 3, n = 16;
    auto f = coord->x1f.slice(0, ng, ng + n + 1);
    EXPECT_EQ(f[0].item<double>(), 0.);
    EXPECT_EQ(f[n].item<double>(), 16.);
    auto h = f.slice(0, 1, n + 1) - f.slice(0, 0, n);
    auto ratio = h.slice(0, 1, n) / h.slice(0, 0, n - 1);
    EXPECT_LT((ratio - qq).abs().max().item<double>(), 1e-12) << ratio;
    double h0 = 16. * (qq - 1.) / (std::pow(qq, n) - 1.);
    EXPECT_NEAR(h[0].item<double>(), h0, 1e-13 * 16.);

    std::vector<double> faces(f.data_ptr<double>(),
                              f.data_ptr<double>() + n + 1);
    expect_metrics(coord, mirrored(faces, ng + 1), 0);
  }
}

TEST(StretchedX1, x1rat_one_is_the_uniform_grid) {
  Cartesian none(load(card("{nx1: 12, nghost: 3}")));
  Cartesian one(load(card("{nx1: 12, nghost: 3, x1rat: 1.}")));
  for (auto name : {"x1f", "dx1f", "x1v", "dx1v"}) {
    EXPECT_TRUE(
        torch::equal(none->named_buffers()[name], one->named_buffers()[name]))
        << name;
  }
  double dx = 16. / 12;
  auto f = torch::linspace(-3 * dx, 16. + 3 * dx, 19, torch::kFloat64);
  EXPECT_TRUE(torch::equal(none->x1f, f));
  EXPECT_TRUE(torch::equal(none->dx1f, torch::ones(18, torch::kFloat64) * dx));
  EXPECT_TRUE(torch::equal(none->dx1v, torch::ones(18, torch::kFloat64) * dx));
}

TEST(StretchedX1, blocks_slice_the_global_face_vector) {
  std::vector<double> faces;
  double q = 0.9;
  for (int k = 0; k <= 12; ++k)
    faces.push_back(100. * (1. - std::pow(q, k)) / (1. - std::pow(q, 12)));
  faces.back() = 100.;
  auto global = load(card("{nx1: 12, nghost: 2}", face_list(faces),
                          "{x1min: 0., x1max: 100.}"));
  auto parsed = faces;
  auto g = mirrored(parsed, 3);
  for (int l = 0; l < 3; ++l) {
    auto op = global->clone();
    op->x1min(parsed[4 * l]).x1max(parsed[4 * l + 4]).nx1(4);
    ASSERT_EQ(op->ix1(), 4 * l);
    Cartesian coord(op);
    expect_metrics(coord, g, 4 * l);
  }

  auto op = global->clone();
  op->x1min(parsed[0]).x1max(0.5 * (parsed[4] + parsed[5])).nx1(4);
  EXPECT_THROW(Cartesian{op}, c10::Error);
}

TEST(StretchedX1, refuses_bad_keys) {
  auto refused = [](std::string const& config) {
    EXPECT_THROW(load(config), c10::Error) << config;
  };
  refused(card("{nx1: 8, nghost: 3, x1rat: 0.95}", face_list(kFaces)));
  refused(card("{nx1: 8, nghost: 3, x1rat: 0.}"));
  refused(card("{nx1: 8, nghost: 3, x1rat: -1.05}"));
  auto bad = kFaces;
  bad[4] = bad[3];
  refused(card("{nx1: 8, nghost: 3}", face_list(bad)));
  refused(card("{nx1: 7, nghost: 3}", face_list(kFaces)));
  refused(card("{nx1: 8, nghost: 3}", face_list(kFaces),
               "{x1min: 0., x1max: 15.}"));
  refused(card("{nx1: 3, nghost: 3, x1rat: 0.95}"));
  refused("geometry:\n  type: cartesian\n" + face_list(kFaces));
}

TEST(StretchedX1, global_faces_do_not_depend_on_nghost) {
  int const G = 4;
  for (int ng : {1, 2, 3}) {
    auto cells = "{nx1: 8, nghost: " + std::to_string(ng);
    auto op = load(card(cells + "}", face_list(kFaces)));
    auto g = op->global_x1f();
    ASSERT_EQ(g.size(0), 8 + 2 * G + 1) << "nghost " << ng;
    EXPECT_TRUE(torch::equal(g, as_tensor(mirrored(kFaces, G))))
        << "nghost " << ng;
    Cartesian st(op);
    EXPECT_TRUE(torch::equal(st->x1f, g.narrow(0, G - ng, 8 + 2 * ng + 1)));
    expect_metrics(st, mirrored(kFaces, G), G - ng - 1);
    auto uop = load(card(cells + "}"));
    auto ug = uop->global_x1f();
    ASSERT_EQ(ug.size(0), 8 + 2 * G + 1) << "nghost " << ng;
    Cartesian un(uop);
    EXPECT_TRUE(torch::equal(un->x1f, ug.narrow(0, G - ng, 8 + 2 * ng + 1)));
    double dx = 2.;
    EXPECT_TRUE(torch::equal(
        un->x1f, torch::linspace(-ng * dx, 16. + ng * dx, 8 + 2 * ng + 1,
                                 torch::kFloat64)));
  }
}

TEST(StretchedX1, repartition_refuses_an_uneven_x1_split) {
  auto split = [](CoordinateOptions const& op, int nb1, int rank) {
    auto layout = LayoutOptionsImpl::create();
    layout->type("cubed").pz(nb1).rank(rank);
    op->clone()->repartition(layout);
  };
  auto op = load(card("{nx1: 10, nghost: 3, x1rat: 0.95}"));
  ASSERT_TRUE(op->stretched_x1());
  for (int r = 0; r < 2; ++r) EXPECT_NO_THROW(split(op, 2, r));
  for (int r = 0; r < 3; ++r) {
    try {
      split(op, 3, r);
      ADD_FAILURE() << "nx1 = 10 split into nb1 = 3 x1 blocks did not throw"
                    << " (rank " << r << ")";
    } catch (c10::Error const& e) {
      std::string msg = e.what();
      EXPECT_NE(msg.find("global_nx1 = 10"), std::string::npos) << msg;
      EXPECT_NE(msg.find("nb1 = 3"), std::string::npos) << msg;
    }
  }
}
