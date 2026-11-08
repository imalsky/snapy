// C/C++
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/input/check_keys.hpp>
#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>

#include "coordinate.hpp"
#include "gnomonic_equiangle.hpp"
#include "spherical_polar.hpp"

namespace snap {

torch::Tensor derivative(torch::Tensor value, torch::Tensor distance, int dim) {
  auto grad = torch::zeros_like(value);
  int n = value.size(dim);
  if (n <= 1) return grad;

  auto full = torch::indexing::Slice();
  std::vector<torch::indexing::TensorIndex> dst(value.dim(), full);
  std::vector<torch::indexing::TensorIndex> lo(value.dim(), full);
  std::vector<torch::indexing::TensorIndex> hi(value.dim(), full);

  dst[dim] = 0;
  lo[dim] = 0;
  hi[dim] = 1;
  grad.index_put_(dst, (value.index(hi) - value.index(lo)) /
                           distance.index(hi).clamp_min(1.e-30));

  if (n > 2) {
    dst[dim] = torch::indexing::Slice(1, n - 1);
    lo[dim] = torch::indexing::Slice(0, n - 2);
    hi[dim] = torch::indexing::Slice(2, n);

    std::vector<torch::indexing::TensorIndex> d_lo(distance.dim(), full);
    std::vector<torch::indexing::TensorIndex> d_hi(distance.dim(), full);
    d_lo[dim] = torch::indexing::Slice(1, n - 1);
    d_hi[dim] = torch::indexing::Slice(2, n);
    auto denom = distance.index(d_lo) + distance.index(d_hi);
    grad.index_put_(
        dst, (value.index(hi) - value.index(lo)) / denom.clamp_min(1.e-30));
  }

  dst[dim] = n - 1;
  lo[dim] = n - 2;
  hi[dim] = n - 1;
  grad.index_put_(dst, (value.index(hi) - value.index(lo)) /
                           distance.index(hi).clamp_min(1.e-30));

  return grad;
}

CoordinateOptions CoordinateOptionsImpl::from_yaml(
    std::string const& filename) {
  auto op = CoordinateOptionsImpl::create();

  auto config = YAML::LoadFile(filename);
  auto node = config["geometry"];
  if (!node) return op;  // return default options

  check_keys(node, "geometry", {"type", "bounds", "cells", "x1f"});
  op->type(node["type"].as<std::string>("cartesian"));

  double x1min = 0, x2min = 0, x3min = 0, x1max = 1, x2max = 1, x3max = 1;

  if (node["bounds"]) {
    check_keys(node["bounds"], "geometry/bounds",
               {"x1min", "x1max", "x2min", "x2max", "x3min", "x3max",
                "x2min_pi", "x2max_pi", "x3min_pi", "x3max_pi"});
    x1min = node["bounds"]["x1min"].as<double>(0.0);

    if (node["bounds"]["x2min_pi"]) {
      x2min = node["bounds"]["x2min_pi"].as<double>() * M_PI;
    } else {
      x2min = node["bounds"]["x2min"].as<double>(0.0);
    }

    if (node["bounds"]["x3min_pi"]) {
      x3min = node["bounds"]["x3min_pi"].as<double>() * M_PI;
    } else {
      x3min = node["bounds"]["x3min"].as<double>(0.0);
    }

    x1max = node["bounds"]["x1max"].as<double>(1.0);

    if (node["bounds"]["x2max_pi"]) {
      x2max = node["bounds"]["x2max_pi"].as<double>() * M_PI;
    } else {
      x2max = node["bounds"]["x2max"].as<double>(1.0);
    }

    if (node["bounds"]["x3max_pi"]) {
      x3max = node["bounds"]["x3max_pi"].as<double>() * M_PI;
    } else {
      x3max = node["bounds"]["x3max"].as<double>(1.0);
    }
  }

  op->global_x1min() = x1min;
  op->global_x2min() = x2min;
  op->global_x3min() = x3min;
  op->global_x1max() = x1max;
  op->global_x2max() = x2max;
  op->global_x3max() = x3max;

  // `bounds:` without `cells:` IS a declared grid: ONE cell per axis spanning
  // it. The block must be given those bounds too, or ixN() offsets it out of
  // its own one-cell face array.
  if (!node["cells"]) {
    TORCH_CHECK(!node["x1f"], "geometry/x1f needs geometry/cells/nx1");
    op->x1min() = x1min, op->x1max() = x1max, op->global_nx1() = 1;
    op->x2min() = x2min, op->x2max() = x2max, op->global_nx2() = 1;
    op->x3min() = x3min, op->x3max() = x3max, op->global_nx3() = 1;
    return op;
  }

  check_keys(node["cells"], "geometry/cells",
             {"nx1", "nx2", "nx3", "nghost", "interp_order", "x1rat"});
  op->global_nx1() = node["cells"]["nx1"].as<int>(1);
  op->global_nx2() = node["cells"]["nx2"].as<int>(1);
  op->global_nx3() = node["cells"]["nx3"].as<int>(1);

  auto x1rat = node["cells"]["x1rat"];
  auto x1f = node["x1f"];
  TORCH_CHECK(!(x1rat && x1f),
              "geometry: give cells/x1rat or x1f, not both (#280)");
  int gnx1 = op->global_nx1();
  std::vector<double> faces;
  if (x1f) {
    faces = x1f.as<std::vector<double>>();
    TORCH_CHECK(static_cast<int>(faces.size()) == gnx1 + 1, "geometry/x1f has ",
                faces.size(), " faces; cells/nx1 = ", gnx1, " needs ",
                gnx1 + 1);
    auto bounds = node["bounds"];
    TORCH_CHECK(!(bounds && bounds["x1min"]) || faces.front() == x1min,
                "geometry/x1f starts at ", faces.front(),
                " but bounds/x1min = ", x1min);
    TORCH_CHECK(!(bounds && bounds["x1max"]) || faces.back() == x1max,
                "geometry/x1f ends at ", faces.back(),
                " but bounds/x1max = ", x1max);
  } else if (x1rat) {
    double q = x1rat.as<double>();
    TORCH_CHECK(q > 0., "geometry/cells/x1rat must be > 0, got ", q);
    if (q != 1.) {  // q = 1 is the uniform grid: keep today's path
      double lq = std::log(q), den = std::expm1(gnx1 * lq);
      for (int k = 0; k < gnx1; ++k)
        faces.push_back(x1min + (x1max - x1min) * std::expm1(k * lq) / den);
      faces.push_back(x1max);
    }
  }
  if (!faces.empty()) {
    for (size_t k = 1; k < faces.size(); ++k)
      TORCH_CHECK(faces[k] > faces[k - 1],
                  "stretched x1 faces must increase strictly; face ", k, " = ",
                  faces[k], " <= ", faces[k - 1]);
    for (size_t k = 1; k + 1 < faces.size(); ++k) {
      double below = faces[k] - faces[k - 1], above = faces[k + 1] - faces[k];
      double ratio = std::max(above / below, below / above);
      TORCH_CHECK(ratio <= CoordinateOptionsImpl::kX1MaxWidthRatio,
                  "stretched x1: the cell widths across face ", k,
                  " differ by a factor ", ratio, " (", below, " and ", above,
                  "), outside the stage-1 bound [1/",
                  CoordinateOptionsImpl::kX1MaxWidthRatio, ", ",
                  CoordinateOptionsImpl::kX1MaxWidthRatio, "] (#280)");
    }
    op->x1f_interior(faces);
    op->global_x1min() = faces.front();
    op->global_x1max() = faces.back();
  }

  auto layout = LayoutOptionsImpl::from_yaml(filename);
  auto playout =
      LayoutImpl::create(std::make_shared<LayoutOptionsImpl>(*layout));
  int rank = playout->options->rank();
  auto [lx2, lx3, lx1] = playout->loc_of(rank);

  if (playout->options->type() == "cubed-sphere") lx1 = 0;

  int nb1 = playout->options->pz();
  int nb2 = playout->options->px();
  int nb3 = playout->options->py();

  if (op->global_nx1() % nb1 != 0) {
    TORCH_CHECK(
        false,
        "Number of total x1 grids must be divisible by the number of mesh "
        "blocks in x1 direction");
  }

  if (op->global_nx2() % nb2 != 0) {
    TORCH_CHECK(
        false,
        "Number of total x2 grids must be divisible by the number of mesh "
        "blocks in x2 direction");
  }

  if (op->global_nx3() % nb3 != 0) {
    TORCH_CHECK(
        false,
        "Number of totla x3 grids must be divisible by the number of mesh "
        "blocks in x3 direction");
  }

  op->nghost() = node["cells"]["nghost"].as<int>(1);
  op->interp_order() = node["cells"]["interp_order"].as<int>(2);
  op->repartition(layout);

  TORCH_CHECK(
      !op->stretched_x1() || (gnx1 >= op->x1f_ghosts() && op->nx1() > 1),
      "a stretched x1 needs nx1 >= ", op->x1f_ghosts(),
      " (max(nghost, 3) + 1) and more than one x1 cell per block; got nx1 = ",
      gnx1, ", nghost = ", op->nghost(), ", block nx1 = ", op->nx1());

  // the per-block cell counts are known only after the repartition; an axis
  // is collapsed by its total count, not by one block's share of it
  if (op->global_nx1() > 1 && op->nx1() < op->nghost()) {
    TORCH_CHECK(false,
                "Number of x1 grids must be greater than the ghost zone size");
  }

  if (op->global_nx2() > 1 && op->nx2() < op->nghost()) {
    TORCH_CHECK(false,
                "Number of x2 grids must be greater than the ghost zone size");
  }

  if (op->global_nx3() > 1 && op->nx3() < op->nghost()) {
    TORCH_CHECK(false,
                "Number of x3 grids must be greater than the ghost zone size");
  }
  return op;
}

void CoordinateOptionsImpl::_resolve_axis(char const* ax, double lo, double hi,
                                          int nx, double& glo, double& ghi,
                                          int& gnx) {
  if (gnx == 0) {
    // A block built from block-local options alone IS the whole domain.
    TORCH_CHECK(nx > 0 && hi > lo, "[CoordinateOptions] cannot adopt this ",
                "block as the global ", ax, " grid: need n", ax, " > 0 and ",
                ax, "max > ", ax, "min, got n", ax, " = ", nx, ", ", ax,
                "min = ", lo, ", ", ax, "max = ", hi);
    glo = lo, ghi = hi, gnx = nx;
    return;
  }

  // a few ULP: `repartition` reaches the last upper bound by accumulation
  double tol = 16 * std::numeric_limits<double>::epsilon() *
               std::max({std::abs(glo), std::abs(ghi), std::abs(ghi - glo)});
  TORCH_CHECK(nx > 0 && hi > lo, "[CoordinateOptions] block ", ax,
              " is not a slice of the declared global grid: need n", ax,
              " > 0 and ", ax, "max > ", ax, "min, got n", ax, " = ", nx, ", ",
              ax, "min = ", lo, ", ", ax, "max = ", hi);
  TORCH_CHECK(lo >= glo - tol && hi <= ghi + tol, "[CoordinateOptions] block ",
              ax, " = [", lo, ", ", hi, "] lies outside the declared global ",
              ax, " grid [", glo, ", ", ghi, "]");

  // block_faces_ slices by the rounded start and the local nx. A span that
  // merely lies inside the global interval can still name a different set of
  // cells than [lo, hi]: [0, 0.3] with nx 2 on a 10-cell [0, 1] grid is
  // accepted by the check above and then cut back to [0, 0.2].
  double dx = (ghi - glo) / static_cast<double>(gnx);
  int i0 = _offset(lo - glo, dx);
  int i1 = _offset(hi - glo, dx);
  bool on_lo = std::abs(lo - (glo + static_cast<double>(i0) * dx)) <= tol;
  bool on_hi = std::abs(hi - (glo + static_cast<double>(i1) * dx)) <= tol;
  TORCH_CHECK(i0 >= 0 && i1 <= gnx && i1 - i0 == nx && on_lo && on_hi,
              "[CoordinateOptions] block ", ax, " = [", lo, ", ", hi,
              "] with n", ax, " = ", nx, " is not exactly ", nx,
              " cells of the declared global ", ax, " grid [", glo, ", ", ghi,
              "] (n = ", gnx, ")");
}

int CoordinateOptionsImpl::_face_index(double x) const {
  auto const& f = x1f_interior();
  auto it = std::lower_bound(f.begin(), f.end(), x);
  if (it == f.end()) return static_cast<int>(f.size()) - 1;
  if (it != f.begin() && x - *(it - 1) < *it - x) --it;
  return static_cast<int>(it - f.begin());
}

void CoordinateOptionsImpl::_resolve_x1_faces() {
  auto const& f = x1f_interior();
  int n = static_cast<int>(f.size()) - 1;
  TORCH_CHECK(global_nx1() == 0 || global_nx1() == n,
              "[CoordinateOptions] global_nx1 = ", global_nx1(), " but ",
              "x1f_interior has ", n + 1, " faces");
  global_nx1() = n, global_x1min() = f.front(), global_x1max() = f.back();

  int i0 = _face_index(x1min()), i1 = _face_index(x1max());
  TORCH_CHECK(f[i0] == x1min() && f[i1] == x1max() && i1 - i0 == nx1(),
              "[CoordinateOptions] block x1 = [", x1min(), ", ", x1max(),
              "] with nx1 = ", nx1(), " is not exactly ", nx1(),
              " cells of the stretched global x1 grid");
}

void CoordinateOptionsImpl::resolve_global_grid() {
  if (stretched_x1()) {
    _resolve_x1_faces();
  } else {
    _resolve_axis("x1", x1min(), x1max(), nx1(), global_x1min(), global_x1max(),
                  global_nx1());
  }
  _resolve_axis("x2", x2min(), x2max(), nx2(), global_x2min(), global_x2max(),
                global_nx2());
  _resolve_axis("x3", x3min(), x3max(), nx3(), global_x3min(), global_x3max(),
                global_nx3());
}

void CoordinateOptionsImpl::repartition(LayoutOptions const& layout) {
  // The local bounds and counts are dead here. resolve reads them, and
  // without that call 14 cards abort. Present the whole declared grid,
  // including its count: the resolve rejects a stale local nx that does
  // not cover that grid. The rank's own slice is written just below.
  if (global_nx1() > 0) {
    x1min(global_x1min());
    x1max(global_x1max());
    nx1(global_nx1());
  }
  if (global_nx2() > 0) {
    x2min(global_x2min());
    x2max(global_x2max());
    nx2(global_nx2());
  }
  if (global_nx3() > 0) {
    x3min(global_x3min());
    x3max(global_x3max());
    nx3(global_nx3());
  }

  resolve_global_grid();

  auto playout =
      LayoutImpl::create(std::make_shared<LayoutOptionsImpl>(*layout));
  int rank = layout->rank();
  auto [lx2, lx3, lx1] = playout->loc_of(rank);

  if (layout->type() == "cubed-sphere") lx1 = 0;

  int nb1 = layout->pz();
  int nb2 = layout->px();
  int nb3 = layout->py();

  if (stretched_x1()) {  // equal cell counts, bounds read off the faces
    TORCH_CHECK(nb1 > 0 && global_nx1() % nb1 == 0,
                "[CoordinateOptions] a stretched x1 splits into equal cell "
                "counts per x1 block: global_nx1 = ",
                global_nx1(), " is not divisible by nb1 = ", nb1);
    int n = global_nx1() / nb1;
    x1min(x1f_interior()[lx1 * n]);
    x1max(x1f_interior()[(lx1 + 1) * n]);
  } else {
    x1min(global_x1min() + lx1 * (global_x1max() - global_x1min()) / nb1);
    x1max(x1min() + (global_x1max() - global_x1min()) / nb1);
  }

  x2min(global_x2min() + lx2 * (global_x2max() - global_x2min()) / nb2);
  x2max(x2min() + (global_x2max() - global_x2min()) / nb2);

  x3min(global_x3min() + lx3 * (global_x3max() - global_x3min()) / nb3);
  x3max(x3min() + (global_x3max() - global_x3min()) / nb3);

  nx1(global_nx1() / nb1);
  nx2(global_nx2() / nb2);
  nx3(global_nx3() / nb3);
}

CoordinateImpl::CoordinateImpl(const CoordinateOptions& options_,
                               torch::nn::Module* p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const*>(p);

  auto const& op = options;
  op->resolve_global_grid();

  // Slice ONE global face array per axis rather than building a per-block one.
  // Building the faces from the block's own [xmin, xmax] makes the SAME global
  // face come out of different arithmetic in different decompositions, and even
  // out of two different blocks that share it: at C64 nb2=4, 15 of 71 x2 faces
  // were internally inconsistent and 26 differed from the nb2=2 grid, each by
  // exactly 1 ULP. Every metric term is a function of these coordinates, and
  // the geometric source coefficients amplify that ULP by ~7 orders. Sliced
  // from a global array, a face has one value whoever owns it.
  if (op->stretched_x1()) {
    TORCH_CHECK(op->type() == "cartesian",
                "a stretched x1 (geometry/x1f or cells/x1rat) is supported "
                "only on a cartesian geometry in stage 1 (#280); got '",
                op->type(), "'");
    x1f = op->global_x1f()
              .narrow(0, op->ix1() - op->nghost() + op->x1f_ghosts(),
                      op->nx1() + 2 * op->nghost() + 1)
              .clone();
  } else {
    x1f = block_faces_(op->global_x1min(), op->global_x1max(), op->global_nx1(),
                       op->nx1(), op->ix1(), op->nghost());
  }
  x2f = block_faces_(op->global_x2min(), op->global_x2max(), op->global_nx2(),
                     op->nx2(), op->ix2(), op->nghost());
  x3f = block_faces_(op->global_x3min(), op->global_x3max(), op->global_nx3(),
                     op->nx3(), op->ix3(), op->nghost());
}

torch::Tensor CoordinateOptionsImpl::global_x1f() const {
  int gnx = global_nx1(), ng = x1f_ghosts();
  TORCH_CHECK(gnx > 0, "global_x1f() before resolve_global_grid()");
  if (!stretched_x1()) {
    double gmin = global_x1min(), gmax = global_x1max();
    auto dx = (gmax - gmin) / gnx;
    int n0 = nghost(), more = ng - n0;
    auto core = torch::linspace(gmin - n0 * dx, gmax + n0 * dx,
                                gnx + 2 * n0 + 1, torch::kFloat64);
    auto k = torch::arange(1, more + 1, torch::kFloat64) * dx;
    return torch::cat({core[0] - k.flip(0), core, core[-1] + k});
  }

  TORCH_CHECK(gnx >= ng, "ghost rule (c) mirrors ", ng, " widths per end, ",
              "but x1 has only ", gnx, " cells");
  auto f = torch::tensor(x1f_interior(), torch::kFloat64);
  auto lo = 2. * f[0] - f.slice(0, 1, ng + 1).flip(0);
  auto hi = 2. * f[gnx] - f.slice(0, gnx - ng, gnx).flip(0);
  return torch::cat({lo, f, hi});
}

torch::Tensor CoordinateImpl::block_faces_(double gmin, double gmax, int gnx,
                                           int nx, int ix, int nghost) {
  auto dx = (gmax - gmin) / gnx;

  if (nx <= 1) {  // degenerate axis: one cell, no ghost layers
    // sliced from the global grid like every other axis: gmin + gnx*dx need
    // not round to gmax
    return torch::linspace(gmin, gmax, gnx + 1, torch::kFloat64)
        .narrow(0, ix, 2)
        .clone();
  }

  // Global face g (g = -nghost .. gnx + nghost) sits at array index g + nghost;
  // this block's first face is global face ix - nghost.
  auto all = torch::linspace(gmin - nghost * dx, gmax + nghost * dx,
                             gnx + 2 * nghost + 1, torch::kFloat64);
  return all.narrow(0, ix, nx + 2 * nghost + 1).clone();
}

void CoordinateImpl::reset_coordinates(std::array<MeshGenerator, 3> meshgens) {
  auto const& op = options;

  if (meshgens[0] != nullptr) {
    int nx1f = x1f.size(0);
    auto rx = compute_logical_position(
        torch::linspace(0, nx1f, nx1f, torch::kFloat64), nx1f, true);
    x1f.copy_(meshgens[0](rx, op->x1min(), op->x1max()));
  }

  if (meshgens[1] != nullptr) {
    int nx2f = x2f.size(0);
    auto rx = compute_logical_position(
        torch::linspace(0, nx2f, nx2f, torch::kFloat64), nx2f, true);
    x2f.copy_(meshgens[1](rx, op->x2min(), op->x2max()));
  }

  if (meshgens[2] != nullptr) {
    int nx3f = x3f.size(0);
    auto rx = compute_logical_position(
        torch::linspace(0, nx3f, nx3f, torch::kFloat64), nx3f, true);
    x3f.copy_(meshgens[2](rx, op->x3min(), op->x3max()));
  }
}

void CoordinateImpl::print(std::ostream& stream) const {
  stream << "x1f = [";
  for (int i = 0; i < x1f.size(0); ++i) {
    stream << x1f[i].item<float>();
    if (i < x1f.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl << "x1v = [";
  for (int i = 0; i < x1v.size(0); ++i) {
    stream << x1v[i].item<float>();
    if (i < x1v.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl;

  stream << "x2f = [";
  for (int i = 0; i < x2f.size(0); ++i) {
    stream << x2f[i].item<float>();
    if (i < x2f.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl << "x2v = [";
  for (int i = 0; i < x2v.size(0); ++i) {
    stream << x2v[i].item<float>();
    if (i < x2v.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl;

  stream << "x3f = [";
  for (int i = 0; i < x3f.size(0); ++i) {
    stream << x3f[i].item<float>();
    if (i < x3f.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl << "x3v = [";
  for (int i = 0; i < x3v.size(0); ++i) {
    stream << x3v[i].item<float>();
    if (i < x3v.size(0) - 1) {
      stream << ", ";
    }
  }
  stream << "]" << std::endl;
}

torch::Tensor CoordinateImpl::center_width1() const {
  return dx1f.unsqueeze(0).unsqueeze(1);
}

torch::Tensor CoordinateImpl::center_width2() const {
  return dx2f.unsqueeze(0).unsqueeze(2);
}

torch::Tensor CoordinateImpl::center_width3() const {
  return dx3f.unsqueeze(1).unsqueeze(2);
}

torch::Tensor CoordinateImpl::center_distance1() const {
  return dx1v.unsqueeze(0).unsqueeze(1);
}

torch::Tensor CoordinateImpl::center_distance2() const {
  return dx2v.unsqueeze(0).unsqueeze(2);
}

torch::Tensor CoordinateImpl::center_distance3() const {
  return dx3v.unsqueeze(1).unsqueeze(2);
}

// grid-only geometry is built once and rebuilt only when its inputs change
torch::Tensor CoordinateImpl::cached_geometry_(
    GeomCache& c, std::function<torch::Tensor()> const& build) const {
  auto version = [](torch::Tensor const& t) {
    return t.is_inference() ? int64_t(0) : int64_t(t._version());
  };
  // same dx1f/dx2f/dx3f objects, device, dtype and versions; value unedited
  if (!c.value.defined() || c.device != dx1f.device() ||
      c.dtype != dx1f.scalar_type() || !c.dx1.is_same(dx1f) ||
      !c.dx2.is_same(dx2f) || !c.dx3.is_same(dx3f) || c.v1 != version(dx1f) ||
      c.v2 != version(dx2f) || c.v3 != version(dx3f) ||
      c.vv != version(c.value)) {
    c.value = build();
    c.device = dx1f.device();
    c.dtype = dx1f.scalar_type();
    c.dx1 = dx1f;
    c.dx2 = dx2f;
    c.dx3 = dx3f;
    c.v1 = version(dx1f);
    c.v2 = version(dx2f);
    c.v3 = version(dx3f);
    c.vv = version(c.value);
  }
  return c.value;
}

torch::Tensor CoordinateImpl::face_area1() const {
  return cached_geometry_(_area1_cache, [&] {
    return dx3f.outer(dx2f).unsqueeze(2).expand({-1, -1, x1f.size(0)});
  });
}

torch::Tensor CoordinateImpl::face_area2() const {
  return cached_geometry_(_area2_cache, [&] {
    return dx3f.outer(dx1f).unsqueeze(1).expand({-1, x2f.size(0), -1});
  });
}

torch::Tensor CoordinateImpl::face_area3() const {
  return cached_geometry_(_area3_cache, [&] {
    return dx2f.outer(dx1f).unsqueeze(0).expand({x3f.size(0), -1, -1});
  });
}

torch::Tensor CoordinateImpl::cell_volume() const {
  return cached_geometry_(_vol_cache, [&] {
    return torch::einsum("km,mji->kji",
                         {dx3f.unsqueeze(1), dx2f.outer(dx1f).unsqueeze(0)});
  });
}

torch::Tensor CoordinateImpl::face_moment2_x1() const {
  return dx1f.square() / 12.;
}

torch::Tensor CoordinateImpl::radial_face_moment2_(torch::Tensor const& x1f,
                                                   int nc1) {
  auto rm = x1f.slice(0, 0, nc1);
  auto rp = x1f.slice(0, 1, nc1 + 1);
  auto h = rp - rm;
  auto rbar = 0.5 * (rm + rp);
  // rbar <= h/2 is a degenerate face (an x1 ghost that reaches r <= 0): the
  // r-weighted moment is not defined there, and neither is the rest of the
  // metric. Drop the correction rather than divide by a vanishing radius.
  auto frac = torch::where(
      rbar > 0.5 * h, h.square() / (12. * rbar.square().clamp_min(1.e-300)),
      torch::zeros_like(h));
  return h.square() / 12. * (1. - frac);
}

torch::Tensor CoordinateImpl::face_centroid_shift_x1() const {
  // Cartesian: the cell measure and the x2/x3 face measure are the same
  // uniform weight, so the two centroids coincide and the shift is exactly 0.
  return torch::zeros_like(dx1f);
}

torch::Tensor CoordinateImpl::radial_face_centroid_shift_(
    torch::Tensor const& x1f, int nc1) {
  auto rm = x1f.slice(0, 0, nc1);
  auto rp = x1f.slice(0, 1, nc1 + 1);
  auto h = rp - rm;
  auto rbar = 0.5 * (rm + rp);
  // r_v - r_c: the VOLUME centroid (weight r^2 -- where a cell average lives)
  // minus the AREA centroid (weight r -- where the face average lives):
  //   r_v = (3/4)(rp^4-rm^4)/(rp^3-rm^3)
  //       = 3 rbar (4 rbar^2 + h^2)/(12 rbar^2 + h^2)
  //   r_c = (2/3)(rp^3-rm^3)/(rp^2-rm^2) = (12 rbar^2 + h^2)/(12 rbar)
  // Both are ~rbar, so subtracting them directly would cancel. Doing the
  // algebra first gives an exact, cancellation-free form:
  //   r_v - r_c = h^2 (12 rbar^2 - h^2) / (12 rbar (12 rbar^2 + h^2))
  //             = h^2/(12 rbar) + O(h^4/rbar^3)
  auto h2 = h.square();
  auto twelve_r2 = 12. * rbar.square();
  auto shift = h2 * (twelve_r2 - h2) /
               (12. * rbar * (twelve_r2 + h2)).clamp_min(1.e-300);
  // degenerate face (an x1 ghost reaching r <= 0): no meaningful centroid
  return torch::where(rbar > 0.5 * h, shift, torch::zeros_like(h));
}

torch::Tensor CoordinateImpl::find_cell_index(
    torch::Tensor const& coords) const {
  torch::Tensor index = torch::zeros_like(coords, torch::dtype(torch::kInt64));

  // x1dir
  index.slice(1, 0, 1) = torch::searchsorted(x1f, coords.slice(1, 0, 1));

  // x2dir
  if (coords.size(1) > 1) {
    index.slice(1, 1, 2) = torch::searchsorted(x2f, coords.slice(1, 1, 2));
  }

  // x3dir
  if (coords.size(1) > 2) {
    index.slice(1, 2, 3) = torch::searchsorted(x3f, coords.slice(1, 2, 3));
  }
  return index;
}

torch::Tensor CoordinateImpl::divergence(torch::Tensor flux1,
                                         torch::Tensor flux2,
                                         torch::Tensor flux3) const {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1, DIMC = 0 };

  auto vol = cell_volume();

  torch::Tensor dflx;
  if (flux1.defined()) {
    dflx = torch::zeros_like(flux1);
  } else if (flux2.defined()) {
    dflx = torch::zeros_like(flux2);
  } else if (flux3.defined()) {
    dflx = torch::zeros_like(flux3);
  } else {
    TORCH_CHECK(false, "At least one flux tensor must be defined");
  }

  int si = il();
  int ei = iu() + 1;
  int sj = jl();
  int ej = ju() + 1;
  int sk = kl();
  int ek = ku() + 1;

  if (flux1.defined() > 0) {
    dflx.slice(DIM1, si, ei) +=
        face_area1(si + 1, ei + 1) * flux1.slice(DIM1, si + 1, ei + 1) -
        face_area1(si, ei) * flux1.slice(DIM1, si, ei);
  }

  if (flux2.defined() > 0) {
    dflx.slice(DIM2, sj, ej) +=
        face_area2(sj + 1, ej + 1) * flux2.slice(DIM2, sj + 1, ej + 1) -
        face_area2(sj, ej) * flux2.slice(DIM2, sj, ej);
  }

  if (flux3.defined() > 0) {
    dflx.slice(DIM3, sk, ek) +=
        face_area3(sk + 1, ek + 1) * flux3.slice(DIM3, sk + 1, ek + 1) -
        face_area3(sk, ek) * flux3.slice(DIM3, sk, ek);
  }

  return dflx / vol;
}

torch::Tensor CoordinateImpl::curl(torch::Tensor velocity) const {
  TORCH_CHECK(velocity.dim() == 4 && velocity.size(0) >= 3,
              "CoordinateImpl::curl expects a vector tensor with shape "
              "[3,nx3,nx2,nx1]");

  enum { DIM1 = 2, DIM2 = 1, DIM3 = 0 };

  auto v1 = velocity[VEL1];
  auto v2 = velocity[VEL2];
  auto v3 = velocity[VEL3];

  auto dtype = velocity.scalar_type();
  auto device = velocity.device();
  auto dx1 = center_distance1().to(device).to(dtype);
  auto dx2 = center_distance2().to(device).to(dtype);
  auto dx3 = center_distance3().to(device).to(dtype);
  auto dl1 = center_width1().to(device).to(dtype);
  auto dl2 = center_width2().to(device).to(dtype);
  auto dl3 = center_width3().to(device).to(dtype);

  auto curl1 = derivative(v3 * dl3, dx2, DIM2) / dl3.clamp_min(1.e-30) -
               derivative(v2 * dl2, dx3, DIM3) / dl2.clamp_min(1.e-30);
  auto curl2 = derivative(v1 * dl1, dx3, DIM3) / dl1.clamp_min(1.e-30) -
               derivative(v3 * dl3, dx1, DIM1) / dl3.clamp_min(1.e-30);
  auto curl3 = derivative(v2 * dl2, dx1, DIM1) / dl2.clamp_min(1.e-30) -
               derivative(v1 * dl1, dx2, DIM2) / dl1.clamp_min(1.e-30);

  return torch::stack({curl1, curl2, curl3});
}

torch::Tensor CoordinateImpl::forward(torch::Tensor prim, torch::Tensor flux1,
                                      torch::Tensor flux2, torch::Tensor flux3,
                                      torch::Tensor face_pressure1) {
  (void)prim;
  (void)face_pressure1;
  return divergence(flux1, flux2, flux3);
}

Coordinate CoordinateImpl::create(CoordinateOptions const& opts,
                                  torch::nn::Module* p,
                                  std::string const& name) {
  TORCH_CHECK(p, "[Coordinate] Parent module is null");
  TORCH_CHECK(opts, "[Coordinate] Options pointer is null");

  if (opts->type() == "cartesian") {
    return p->register_module(name, Cartesian(opts, p));
  } else if (opts->type() == "cylindrical") {
    return p->register_module(name, Cylindrical(opts, p));
  } else if (opts->type() == "spherical-polar") {
    return p->register_module(name, SphericalPolar(opts, p));
  } else if (opts->type() == "gnomonic-equiangle") {
    return p->register_module(name, GnomonicEquiangle(opts, p));
  } else {
    TORCH_CHECK(false, "Unknown coordinate type: ", opts->type());
  }
}
void CoordinateImpl::boundary_velocity_(torch::Tensor const& w, int axis,
                                        bool inverse) const {
  TORCH_CHECK(axis >= 1 && axis <= 3, "Invalid boundary axis");
  auto c = cosine_cell_kj;
  auto s = (1. - c.square()).sqrt();
  // g22 = g33 = 1 and g23 = c, so |v|^2 = vx^2 + vy^2 + vz^2 + 2*c*vy*vz.
  // On x2/x3 the unmixed tangential component has unit scale, not s.
  // The only nonorthogonal metric is g23. Velocities in orthogonal
  // spherical/cylindrical coordinates already use physical components.
  if (axis == 2) {
    if (inverse) {
      w[IVY].div_(s);
      w[IVZ].sub_(c * w[IVY]);
    } else {
      w[IVZ].add_(c * w[IVY]);
      w[IVY].mul_(s);
    }
  } else {
    if (inverse) {
      w[IVZ].div_(s);
      w[IVY].sub_(c * w[IVZ]);
    } else {
      w[IVY].add_(c * w[IVZ]);
      w[IVZ].mul_(s);
    }
  }
}
}  // namespace snap
