#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/thermo/thermo.hpp>
#include <kintera/utils/format.hpp>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/recon/reconstruct.hpp>
#include <snap/riemann/riemann_solver.hpp>

// arg
#include <snap/add_arg.h>

namespace snap {
class MeshBlockImpl;

struct ScalarOptionsImpl {
  static std::shared_ptr<ScalarOptionsImpl> create() {
    auto op = std::make_shared<ScalarOptionsImpl>();
    op->recon() = ReconstructOptionsImpl::create();
    op->riemann() = RiemannSolverOptionsImpl::create();
    op->riemann()->type() = "upwind";
    return op;
  }
  static std::shared_ptr<ScalarOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);

  ScalarOptionsImpl() = default;
  void report(std::ostream& os) const {
    os << "-- scalar options --\n";
    os << "* verbose = " << (verbose() ? "true" : "false") << "\n"
       << "* nvar = " << nvar() << "\n"
       << "* names = " << fmt::format("{}", names()) << "\n"
       << "* upper-bound = " << upper_bound() << "\n";
    if (recon()) {
      recon()->report(os);
    }
    if (riemann()) {
      riemann()->report(os);
    }
    if (thermo()) {
      os << "-- thermo options --\n";
      thermo()->report(os);
    }
    if (kinetics()) {
      os << "-- kinetics options --\n";
      kinetics()->report(os);
    }
  }

  ADD_ARG(bool, verbose) = false;
  ADD_ARG(int, nvar) = 0;
  ADD_ARG(std::vector<std::string>, names);

  //! Thermodynamics options
  ADD_ARG(kintera::ThermoOptions, thermo) = nullptr;

  //! Kinetics options
  ADD_ARG(kintera::KineticsOptions, kinetics) = nullptr;

  //! Upper bound on the mixing ratio, enforced as positivity of the complement.
  //! Negative disables it; there is no universal default because a passive
  //! scalar need not be a fraction.
  ADD_ARG(double, upper_bound) = -1.;

  //! submodules options
  ADD_ARG(ReconstructOptions, recon) = nullptr;
  ADD_ARG(RiemannSolverOptions, riemann) = nullptr;
};
using ScalarOptions = std::shared_ptr<ScalarOptionsImpl>;

using Variables = std::map<std::string, torch::Tensor>;

class ScalarImpl : public torch::nn::Cloneable<ScalarImpl> {
 public:
  //! \brief Create and register a `Scalar` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for creating the `Scalar` module
   * \param[in] p     parent module for registering the created module
   * \param[in] name  name for registering the created module
   * \return          created `Scalar` module
   */
  static std::shared_ptr<ScalarImpl> create(ScalarOptions const& opts,
                                            torch::nn::Module* p,
                                            std::string const& name = "scalar");

  //! options with which this `Scalar` was constructed
  ScalarOptions options;

  //! non-owning reference to parent meshblock
  MeshBlockImpl const* pmb = nullptr;

  //! submodules
  Coordinate pcoord = nullptr;
  Reconstruct precon = nullptr;
  Reconstruct precon1 = nullptr;
  RiemannSolver priemann = nullptr;
  torch::Tensor _flux1, _flux2, _flux3, _div;
  //! cumulative count of (cell, tracer) entries with positivity theta < 1
  //! (tracer and its complement; diagnostic, EOS limiter on)
  torch::Tensor _positivity_hits;

  kintera::ThermoX pthermo = nullptr;
  kintera::Kinetics pkinetics = nullptr;

  //! Constructor to initialize the layers
  ScalarImpl() : options(ScalarOptionsImpl::create()) {}
  explicit ScalarImpl(const ScalarOptions& options_,
                      torch::nn::Module* p = nullptr);
  void reset() override;

  int nvar() const { return options ? options->nvar() : 0; }
  virtual double max_time_step(torch::Tensor w) const { return 1.e9; }

  torch::Tensor get_buffer(std::string var) const {
    return named_buffers()[var];
  }

  //! Advance the conserved variables by one time step.
  torch::Tensor forward(double dt, torch::Tensor scalar_u,
                        Variables const& other);
};

TORCH_MODULE(Scalar);
}  // namespace snap
