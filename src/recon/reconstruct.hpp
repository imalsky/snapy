#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// snap
#include "interpolation.hpp"

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

struct EquationOfStateOptionsImpl;
using EquationOfStateOptions = std::shared_ptr<EquationOfStateOptionsImpl>;

struct ReconstructOptionsImpl {
  static std::shared_ptr<ReconstructOptionsImpl> create() {
    auto op = std::make_shared<ReconstructOptionsImpl>();
    op->interp() = InterpOptionsImpl::create();
    return op;
  }
  static std::shared_ptr<ReconstructOptionsImpl> from_yaml(
      std::string const& filename, std::string const& section);
  static std::shared_ptr<ReconstructOptionsImpl> from_yaml(
      const YAML::Node& node, std::string const& section);

  ReconstructOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- reconstruction options --\n";
    interp()->report(os);
    os << "* is_boundary_lower = " << (is_boundary_lower() ? "true" : "false")
       << "\n"
       << "* is_boundary_upper = " << (is_boundary_upper() ? "true" : "false")
       << "\n"
       << "* shock = " << (shock() ? "true" : "false") << "\n"
       << "* density_floor = " << density_floor() << "\n"
       << "* pressure_floor = " << pressure_floor() << "\n"
       << "* limiter = " << (limiter() ? "true" : "false") << "\n";
  }

  //! configure options
  ADD_ARG(bool, is_boundary_lower) = false;
  ADD_ARG(bool, is_boundary_upper) = false;
  ADD_ARG(bool, shock) = true;
  ADD_ARG(double, density_floor) = 1.e-10;
  ADD_ARG(double, pressure_floor) = 1.e-10;
  ADD_ARG(bool, limiter) = false;

  //! abstract submodules
  ADD_ARG(InterpOptions, interp) = nullptr;
};
using ReconstructOptions = std::shared_ptr<ReconstructOptionsImpl>;

class HydroImpl;

class ReconstructImpl : public torch::nn::Cloneable<ReconstructImpl> {
 public:
  //! Create and register a Reconstruct module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts      options for creating the `Reconstruct` module
   * \param[in] p         parent module for registering the created module
   * \param[in] name      name for the created module
   * \return              created `Reconstruct` module
   */
  static std::shared_ptr<ReconstructImpl> create(
      ReconstructOptions const& opts, torch::nn::Module* p,
      std::string const& name = "recon");

  //! options with which this `Reconstruction` was constructed
  ReconstructOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  //! concrete submodules
  Interp pinterp1 = nullptr;
  Interp pinterp2 = nullptr;

  //! Constructor to initialize the layers
  ReconstructImpl() : options(ReconstructOptionsImpl::create()) {}
  explicit ReconstructImpl(const ReconstructOptions& options_,
                           torch::nn::Module* p = nullptr);
  void reset() override;

  void set_x1_faces(torch::Tensor const& x1f);

  //! w -> [wl, wr]
  // floor=false skips the reconstruction-stage density/pressure floors (the
  // caller reconstructs perturbation variables that are legitimately
  // negative and enforces positivity itself); the scalar floor is kept.
  torch::Tensor forward(torch::Tensor w, int dim, bool floor = true);
};

TORCH_MODULE(Reconstruct);
}  // namespace snap

#undef ADD_ARG
