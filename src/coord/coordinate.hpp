#pragma once

// C/C++
#include <algorithm>
#include <cmath>
#include <functional>
#include <iosfwd>
#include <vector>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include <snap/mesh/mesh_functions.hpp>

#include "coordgen.hpp"

// arg
#include <snap/add_arg.h>

namespace snap {
struct LayoutOptionsImpl;
using LayoutOptions = std::shared_ptr<LayoutOptionsImpl>;
using IndexRange = std::vector<torch::indexing::TensorIndex>;

struct EquationOfStateOptionsImpl;
using EquationOfStateOptions = std::shared_ptr<EquationOfStateOptionsImpl>;

class MeshBlockImpl;

struct CoordinateOptionsImpl {
  static std::shared_ptr<CoordinateOptionsImpl> create() {
    return std::make_shared<CoordinateOptionsImpl>();
  }

  static std::shared_ptr<CoordinateOptionsImpl> from_yaml(
      std::string const& filename);

  CoordinateOptionsImpl() = default;
  std::shared_ptr<CoordinateOptionsImpl> clone() const {
    return std::make_shared<CoordinateOptionsImpl>(*this);
  }

  void report(std::ostream& os) const {
    os << "-- coordinate options --\n";
    os << "* type = " << type() << "\n"
       << "* x1min = " << x1min() << "\n"
       << "* x2min = " << x2min() << "\n"
       << "* x3min = " << x3min() << "\n"
       << "* x1max = " << x1max() << "\n"
       << "* x2max = " << x2max() << "\n"
       << "* x3max = " << x3max() << "\n"
       << "* nx1 = " << nx1() << "\n"
       << "* nx2 = " << nx2() << "\n"
       << "* nx3 = " << nx3() << "\n"
       << "* nghost = " << nghost() << "\n";
    if (stretched_x1())
      os << "* x1f = " << x1f_interior().size() << " stretched faces\n";
  }

  int nc1() const { return nx1() > 1 ? nx1() + 2 * nghost() : 1; }
  int nc2() const { return nx2() > 1 ? nx2() + 2 * nghost() : 1; }
  int nc3() const { return nx3() > 1 ? nx3() + 2 * nghost() : 1; }

  void repartition(LayoutOptions const& layout);

  //! Adopt this block as the global grid when none was declared.
  void resolve_global_grid();

  //! Cell width taken from the GLOBAL grid, never from this block's own bounds.
  /*!
   * A block that forms its own `(x2max - x2min) / nx2` computes a cell width
   * that depends on how the mesh was decomposed: two decompositions then
   * disagree about the same physical cell by ~1 ULP, and every metric term
   * built on it inherits that. The global expression is the same
   * arithmetic for every block and every decomposition.
   */
  double dx1() const {
    TORCH_CHECK(global_nx1() > 0,
                "dx1() before resolve_global_grid(): global_nx1 is unset");
    return (global_x1max() - global_x1min()) / global_nx1();
  }
  double dx2() const {
    TORCH_CHECK(global_nx2() > 0,
                "dx2() before resolve_global_grid(): global_nx2 is unset");
    return (global_x2max() - global_x2min()) / global_nx2();
  }
  double dx3() const {
    TORCH_CHECK(global_nx3() > 0,
                "dx3() before resolve_global_grid(): global_nx3 is unset");
    return (global_x3max() - global_x3min()) / global_nx3();
  }

  bool stretched_x1() const { return !x1f_interior().empty(); }

  static constexpr double kX1MaxWidthRatio = 1.5;

  int x1f_ghosts() const { return std::max(nghost(), 3) + 1; }

  torch::Tensor global_x1f() const;

  //! Index of this block's FIRST INTERIOR CELL in the global grid.
  int ix1() const {
    return stretched_x1() ? _face_index(x1min())
                          : _offset(x1min() - global_x1min(), dx1());
  }
  int ix2() const { return _offset(x2min() - global_x2min(), dx2()); }
  int ix3() const { return _offset(x3min() - global_x3min(), dx3()); }

  ADD_ARG(std::string, type) = "cartesian";
  ADD_ARG(double, global_x1min) = 0.;
  ADD_ARG(double, global_x2min) = 0.;
  ADD_ARG(double, global_x3min) = 0.;
  ADD_ARG(double, global_x1max) = 1.;
  ADD_ARG(double, global_x2max) = 1.;
  ADD_ARG(double, global_x3max) = 1.;
  ADD_ARG(double, x1min) = 0.;
  ADD_ARG(double, x2min) = 0.;
  ADD_ARG(double, x3min) = 0.;
  ADD_ARG(double, x1max) = 1.;
  ADD_ARG(double, x2max) = 1.;
  ADD_ARG(double, x3max) = 1.;
  ADD_ARG(int, global_nx1) = 0;  // 0 == no global grid declared
  ADD_ARG(int, global_nx2) = 0;  // 0 == no global grid declared
  ADD_ARG(int, global_nx3) = 0;  // 0 == no global grid declared
  ADD_ARG(int, nx1) = 1;
  ADD_ARG(int, nx2) = 1;
  ADD_ARG(int, nx3) = 1;
  ADD_ARG(int, nghost) = 1;
  ADD_ARG(int, interp_order) = 2;

  ADD_ARG(std::vector<double>, x1f_interior) = {};

 private:
  //! Fill one axis' sentinel from this block, or check this block against it.
  static void _resolve_axis(char const* ax, double lo, double hi, int nx,
                            double& glo, double& ghi, int& gnx);

  //! `repartition` places a block at an exact multiple of the cell width, so
  //! the quotient is an integer up to round-off; round rather than truncate.
  static int _offset(double span, double dx) {
    return dx == 0. ? 0 : static_cast<int>(std::lround(span / dx));
  }

  int _face_index(double x) const;

  void _resolve_x1_faces();
};
using CoordinateOptions = std::shared_ptr<CoordinateOptionsImpl>;

class CoordinateImpl {
 public:
  //! Create and register a `Coordinate` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `Coordinate` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for the created module
   * \return          created `Coordinate` module
   */
  static std::shared_ptr<CoordinateImpl> create(
      CoordinateOptions const& opts, torch::nn::Module* p,
      std::string const& name = "coord");

  //! data
  torch::Tensor cosine_cell_kj, cosine_face2_kj, cosine_face3_kj;

  //! options with which this `Coordinate` was constructed
  CoordinateOptions options;

  //! non-owning reference to parent
  MeshBlockImpl const* pmb = nullptr;

  CoordinateImpl() : options(CoordinateOptionsImpl::create()) {}
  explicit CoordinateImpl(const CoordinateOptions& options_,
                          torch::nn::Module* p = nullptr);

  //! data
  torch::Tensor x1f, x2f, x3f;
  torch::Tensor x1v, x2v, x3v;
  torch::Tensor dx1f, dx2f, dx3f;
  torch::Tensor dx1v, dx2v, dx3v;

  virtual ~CoordinateImpl() = default;

  //! faces for the per-face x1 rows: r^3/3 on a radial x1, whose cells hold
  //! r^2 dr averages (#280); undefined where the shipped rows are exact
  torch::Tensor x1_row_faces(std::string const& interp) const;

  int il() const { return options->nx1() > 1 ? options->nghost() : 0; }

  int iu() const {
    return options->nx1() > 1 ? options->nghost() + options->nx1() - 1 : 0;
  }

  int jl() const { return options->nx2() > 1 ? options->nghost() : 0; }

  int ju() const {
    return options->nx2() > 1 ? options->nghost() + options->nx2() - 1 : 0;
  }

  int kl() const { return options->nx3() > 1 ? options->nghost() : 0; }

  int ku() const {
    return options->nx3() > 1 ? options->nghost() + options->nx3() - 1 : 0;
  }

  void print(std::ostream& stream) const;
  virtual void reset_coordinates(std::array<MeshGenerator, 3> meshgens);

  //! This block's slice of the global face array for one axis.
  static torch::Tensor block_faces_(double gmin, double gmax, int gnx, int nx,
                                    int ix, int nghost);

  //! dx1f, x1v and dx1v of this block, uniform or stretched (#280)
  std::array<torch::Tensor, 3> x1_metrics_() const;

  //! module methods
  virtual torch::Tensor center_width1() const;
  virtual torch::Tensor center_width1(int is, int ie) const {
    return center_width1().slice(2, is, ie);
  }

  virtual torch::Tensor center_width2() const;
  virtual torch::Tensor center_width2(int is, int ie) const {
    return center_width2().slice(1, is, ie);
  }

  virtual torch::Tensor center_width3() const;
  virtual torch::Tensor center_width3(int is, int ie) const {
    return center_width3().slice(0, is, ie);
  }

  virtual torch::Tensor center_distance1() const;
  virtual torch::Tensor center_distance1(int is, int ie) const {
    return center_distance1().slice(2, is, ie);
  }

  virtual torch::Tensor center_distance2() const;
  virtual torch::Tensor center_distance2(int is, int ie) const {
    return center_distance2().slice(1, is, ie);
  }

  virtual torch::Tensor center_distance3() const;
  virtual torch::Tensor center_distance3(int is, int ie) const {
    return center_distance3().slice(0, is, ie);
  }

  virtual torch::Tensor face_area1() const;
  torch::Tensor face_area1(int is, int ie) const {
    return face_area1().slice(2, is, ie);
  }

  virtual torch::Tensor face_area2() const;
  torch::Tensor face_area2(int js, int je) const {
    return face_area2().slice(1, js, je);
  }

  virtual torch::Tensor face_area3() const;
  torch::Tensor face_area3(int ks, int ke) const {
    return face_area3().slice(0, ks, ke);
  }

  virtual torch::Tensor cell_volume() const;

  //! x1 second central moment of an x2/x3 face's own area measure
  /*!
   * The finite-volume flux through an x2 (or x3) face is the FACE AVERAGE of
   * the point flux, and that average is weighted by the face's area measure.
   * For a product of two fields, the average of the product exceeds the
   * product of the averages by `sigma1^2 a_1 b_1` (issue #289), with
   * `sigma1^2` the second central moment of the face measure along x1, taken
   * about the face's area centroid. A Cartesian x2/x3 face carries a uniform
   * x1 weight, so `sigma1^2 = dx1^2/12`; curved grids override this.
   *
   * \return one value per x1 cell, shaped like `dx1f`
   */
  virtual torch::Tensor face_moment2_x1() const;

  //! `face_moment2_x1` for an x2/x3 face whose area density grows like r
  /*!
   * On the spherical-polar and cubed-sphere grids an x2 or x3 face spans
   * `dA = r dr x (angular)`, so its x1 weight is `w(r) = r` -- the
   * non-orthogonal and `sqrt(g)` factors live in the angular part, are
   * constant along x1 on one face, and cancel between the numerator and the
   * denominator of the average. With `h = rp - rm` and `rbar = (rm+rp)/2`,
   * the r-weighted moments are exact:
   *   r_c      = (3 rbar^2 + h^2/4) / (3 rbar)
   *   sigma1^2 = <r^2> - r_c^2 = (h^2/12) (1 - h^2/(12 rbar^2)) .
   * The second, cancellation-free form is what is evaluated here, so the
   * metric only shifts the Cartesian `h^2/12` by a relative `h^2/(12 rbar^2)`
   * and recovers it exactly as `h/rbar -> 0`.
   */
  static torch::Tensor radial_face_moment2_(torch::Tensor const& x1f, int nc1);

  //! x1 offset between where a CELL average lives and where a FACE average does
  /*!
   * A finite-volume cell stores the average over the CELL measure, whose x1
   * centroid is `r_v`; the exact flux through an x2/x3 face is the average
   * over the FACE measure, whose x1 centroid is `r_c`. On a curved grid the
   * two measures carry different powers of r, so `r_v != r_c` and evaluating
   * the flux from the cell value misses `(r_v - r_c) d_1(p u)` -- a SECOND
   * O(dx1^2) term, of the same order as the covariance and of relative size
   * ~L/r with L the local gradient scale (issue #289 item 2). It is exactly
   * zero in Cartesian, where the two measures are the same uniform weight.
   *
   * \return one value per x1 cell, shaped like `dx1f`
   */
  virtual torch::Tensor face_centroid_shift_x1() const;

  //! true when `face_centroid_shift_x1` is identically zero (the base
  //! measure, Cartesian), a host constant of the coordinate type: the flux
  //! covariance then skips its centroid parts without touching the device
  virtual bool face_centroid_shift_zero() const { return true; }

  //! `face_centroid_shift_x1` when the cell measure is r^2 dr and the face
  //! measure is r dr (spherical-polar and cubed-sphere). Exact and
  //! cancellation-free: `h^2 (12 rbar^2 - h^2) / (12 rbar (12 rbar^2 + h^2))`.
  static torch::Tensor radial_face_centroid_shift_(torch::Tensor const& x1f,
                                                   int nc1);

  virtual torch::Tensor find_cell_index(torch::Tensor const& coords) const;

  virtual std::array<double, 3> vec_from_cartesian(
      std::array<double, 3> vec) const {
    return {vec[0], vec[1], vec[2]};
  }

  virtual void interp_ghost(torch::Tensor var,
                            std::tuple<int, int, int> const&) const {}

  //! Cell-centered contravariant velocity <-> boundary orthonormal frame.
  //! Axis is 1, 2, or 3; the normal component retains its coordinate slot.
  void boundary_velocity_(torch::Tensor const& prim, int axis,
                          bool inverse = false) const;

  //! project contravariant velocity to a local orthogonal frame at face 1
  virtual void prim2local1_(torch::Tensor const& prim) const {}

  //! project contravariant velocity to a local orthogonal frame at face 2
  virtual void prim2local2_(torch::Tensor const& prim) const {}

  //! project contravariant velocity to a local orthogonal frame at face 3
  virtual void prim2local3_(torch::Tensor const& prim) const {}

  //! project fluxes from local orthogonal frame to global contravariant frame
  //! at face 1
  virtual void flux2global1_(torch::Tensor const& flux) const {}

  //! project fluxes from local orthogonal frame to global contravariant frame
  //! at face 2
  virtual void flux2global2_(torch::Tensor const& flux) const {}

  //! project fluxes from local orthogonal frame to global contravariant frame
  //! at face 3
  virtual void flux2global3_(torch::Tensor const& flux) const {}

  //! fluxes -> flux divergence
  torch::Tensor divergence(torch::Tensor flux1, torch::Tensor flux2,
                           torch::Tensor flux3) const;

  //! cell-centered vector curl
  virtual torch::Tensor curl(torch::Tensor velocity) const;

  virtual torch::Tensor forward(torch::Tensor prim, torch::Tensor flux1,
                                torch::Tensor flux2, torch::Tensor flux3,
                                torch::Tensor face_pressure1 = torch::Tensor());

 protected:
  //! one cached face_area1/2/3() or cell_volume() and its grid inputs
  struct GeomCache {
    torch::Tensor value, dx1, dx2, dx3;
    c10::Device device = c10::kCPU;
    c10::ScalarType dtype = c10::kDouble;
    int64_t v1 = -1, v2 = -1, v3 = -1, vv = -1;
  };
  mutable GeomCache _area1_cache, _area2_cache, _area3_cache, _vol_cache;
  //! the cached value of `build()`, rebuilt when dx1f/dx2f/dx3f change
  torch::Tensor cached_geometry_(
      GeomCache& c, std::function<torch::Tensor()> const& build) const;
};
using Coordinate = std::shared_ptr<CoordinateImpl>;

class CartesianImpl : public torch::nn::Cloneable<CartesianImpl>,
                      public CoordinateImpl {
 public:
  using CoordinateImpl::forward;

  CartesianImpl() = default;
  explicit CartesianImpl(const CoordinateOptions& options_,
                         torch::nn::Module* p = nullptr)
      : CoordinateImpl(options_, p) {
    reset();
  }
  void reset() override;
  void pretty_print(std::ostream& stream) const override {
    stream << "Cartesian coordinate:" << std::endl;
    print(stream);
  }

  void reset_coordinates(std::array<MeshGenerator, 3> meshgens) override;
};
TORCH_MODULE(Cartesian);

class CylindricalImpl : public torch::nn::Cloneable<CylindricalImpl>,
                        public CoordinateImpl {
 public:
  using CoordinateImpl::forward;

  CylindricalImpl() = default;
  explicit CylindricalImpl(const CoordinateOptions& options_,
                           torch::nn::Module* p = nullptr)
      : CoordinateImpl(options_, p) {
    reset();
  }
  void reset() override {}
  //! r-weighted faces, and the base shift needs dx1f, which is not built here
  bool face_centroid_shift_zero() const override { return false; }
  void pretty_print(std::ostream& stream) const override {
    stream << "Cylindrical coordinate:" << std::endl;
    print(stream);
  }
};
TORCH_MODULE(Cylindrical);

}  // namespace snap

#undef ADD_ARG
