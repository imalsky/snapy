// snap
#include <snap/hydro/hydro.hpp>

#include "coordinate.hpp"

namespace snap {

void CartesianImpl::reset() {
  auto const &op = options;

  // metric
  cosine_cell_kj =
      register_buffer("cosine_cell_kj",
                      torch::zeros({op->nc3(), op->nc2(), 1}, torch::kFloat64));
  cosine_face2_kj = register_buffer(
      "cosine_face2_kj",
      torch::zeros({op->nc3(), op->nc2() + 1, 1}, torch::kFloat64));
  cosine_face3_kj = register_buffer(
      "cosine_face3_kj",
      torch::zeros({op->nc3() + 1, op->nc2(), 1}, torch::kFloat64));

  // dimension 1
  // Global cell width, not this block's own; see dx1() in coordinate.hpp.
  auto dx = op->dx1();
  if (op->stretched_x1()) {
    int n = op->nc1();
    auto fe = op->global_x1f().narrow(
        0, op->ix1() - op->nghost() + op->x1f_ghosts() - 1, n + 3);
    auto ve = 0.5 * (fe.slice(0, 0, n + 2) + fe.slice(0, 1, n + 3));
    dx1f = register_buffer("dx1f", x1f.slice(0, 1, n + 1) - x1f.slice(0, 0, n));
    x1v = register_buffer("x1v",
                          0.5 * (x1f.slice(0, 0, n) + x1f.slice(0, 1, n + 1)));
    dx1v = register_buffer("dx1v", ve.slice(0, 1, n + 1) - ve.slice(0, 0, n));
  } else {
    dx1f =
        register_buffer("dx1f", torch::ones(op->nc1(), torch::kFloat64) * dx);
    x1v = register_buffer("x1v", 0.5 * (x1f.slice(0, 0, op->nc1()) +
                                        x1f.slice(0, 1, op->nc1() + 1)));
    dx1v =
        register_buffer("dx1v", torch::ones(op->nc1(), torch::kFloat64) * dx);
  }

  // dimension 2
  dx = op->dx2();
  dx2f = register_buffer("dx2f", torch::ones(op->nc2(), torch::kFloat64) * dx);
  x2v = register_buffer("x2v", 0.5 * (x2f.slice(0, 0, op->nc2()) +
                                      x2f.slice(0, 1, op->nc2() + 1)));
  dx2v = register_buffer("dx2v", torch::ones(op->nc2(), torch::kFloat64) * dx);

  // dimension 3
  dx = op->dx3();
  dx3f = register_buffer("dx3f", torch::ones(op->nc3(), torch::kFloat64) * dx);
  x3v = register_buffer("x3v", 0.5 * (x3f.slice(0, 0, op->nc3()) +
                                      x3f.slice(0, 1, op->nc3() + 1)));
  dx3v = register_buffer("dx3v", torch::ones(op->nc3(), torch::kFloat64) * dx);

  // register buffers defined in the base class
  register_buffer("x1f", x1f);
  register_buffer("x2f", x2f);
  register_buffer("x3f", x3f);
}

void CartesianImpl::reset_coordinates(std::array<MeshGenerator, 3> meshgens) {
  CoordinateImpl::reset_coordinates(meshgens);

  // dimension 1
  dx1f.copy_(x1f.slice(0, 1, options->nc1() + 1) -
             x1f.slice(0, 0, options->nc1()));
  x1v.copy_(0.5 * (x1f.slice(0, 0, options->nc1()) +
                   x1f.slice(0, 1, options->nc1() + 1)));
  dx1v.slice(0, 1, options->nc1())
      .copy_(x1v.slice(0, 1, options->nc1()) -
             x1v.slice(0, 0, options->nc1() - 1));

  // dimension 2
  dx2f.copy_(x2f.slice(0, 1, options->nc2() + 1) -
             x2f.slice(0, 0, options->nc2()));
  x2v.copy_(0.5 * (x2f.slice(0, 0, options->nc2()) +
                   x2f.slice(0, 1, options->nc2() + 1)));
  dx2v.slice(0, 1, options->nc2())
      .copy_(x2v.slice(0, 1, options->nc2()) -
             x2v.slice(0, 0, options->nc2() - 1));

  // dimension 3
  dx3f.copy_(x3f.slice(0, 1, options->nc3() + 1) -
             x3f.slice(0, 0, options->nc3()));
  x3v.copy_(0.5 * (x3f.slice(0, 0, options->nc3()) +
                   x3f.slice(0, 1, options->nc3() + 1)));
  dx3v.slice(0, 1, options->nc3())
      .copy_(x3v.slice(0, 1, options->nc3()) -
             x3v.slice(0, 0, options->nc3() - 1));
}
}  // namespace snap
