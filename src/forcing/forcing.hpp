#pragma once

// C/C++
#include <array>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/thermo/thermo.hpp>
#include <kintera/utils/format.hpp>

// arg
#include <snap/add_arg.h>

namespace YAML {
class Node;
}  // namespace YAML

namespace snap {

class HydroImpl;

////// (1) Constant Gravity //////

struct ConstGravityOptionsImpl {
  static std::shared_ptr<ConstGravityOptionsImpl> create() {
    return std::make_shared<ConstGravityOptionsImpl>();
  }
  static std::shared_ptr<ConstGravityOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  ConstGravityOptionsImpl() = default;
  std::shared_ptr<ConstGravityOptionsImpl> clone() const {
    return std::make_shared<ConstGravityOptionsImpl>(*this);
  }

  void report(std::ostream& os) const {
    os << "-- constant gravity options --\n";
    os << "* grav1 = " << grav1() << "\n"
       << "* grav2 = " << grav2() << "\n"
       << "* grav3 = " << grav3() << "\n"
       << "* non-hydrostatic = " << non_hydrostatic() << "\n"
       << "* gravity-work = " << gravity_work() << "\n"
       << "* gravity-work-fixer = " << gravity_work_fixer() << "\n";
  }

  ADD_ARG(double, non_hydrostatic) = 1.;
  //! x1 gravity work in the energy equation (#283):
  //!   "face" (default): face-mass-flux work in every cell (the form before
  //!     #283), plus the work of the corrected potential energy while
  //!     SNAP_GRAVITY_WORK_RADIAL_EXACT is on (the default; derivation
  //!     curved-gravity-work-weight.md sec 7)
  //!   "cell": -g rho v at cell centres, the form the implicit matrix
  //!     linearises; mass moved by sedimentation or the positivity limiter
  //!     keeps its face-form booking
  //!   "face-wallc": face-mass-flux work, cell work in the two x1 wall cells
  ADD_ARG(std::string, gravity_work) = "face";
  //! with "cell": add the step's global E+PE defect back as uniform heat per
  //! unit mass (one global reduction per step)
  ADD_ARG(bool, gravity_work_fixer) = true;
  ADD_ARG(double, grav1) = 0.;
  ADD_ARG(double, grav2) = 0.;
  ADD_ARG(double, grav3) = 0.;
};
using ConstGravityOptions = std::shared_ptr<ConstGravityOptionsImpl>;

class ConstGravityImpl : public torch::nn::Cloneable<ConstGravityImpl> {
 public:
  //! options with which this `ConstGravity` was constructed
  ConstGravityOptions options;

  // Constructor to initialize the layers
  ConstGravityImpl() : options(ConstGravityOptionsImpl::create()) {}
  explicit ConstGravityImpl(ConstGravityOptions const& options_)
      : options(options_) {
    reset();
  }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(ConstGravity);

////// (2) Coriolis //////

struct CoriolisOptionsImpl {
  static std::shared_ptr<CoriolisOptionsImpl> create() {
    return std::make_shared<CoriolisOptionsImpl>();
  }
  static std::shared_ptr<CoriolisOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  CoriolisOptionsImpl() = default;
  std::shared_ptr<CoriolisOptionsImpl> clone() const {
    return std::make_shared<CoriolisOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "-- coriolis options --\n";
    os << "* omega1 = " << omega1() << "\n"
       << "* omega2 = " << omega2() << "\n"
       << "* omega3 = " << omega3() << "\n"
       << "* type = " << type() << "\n"
       << "* traditional = " << (traditional() ? "true" : "false") << "\n";
  }

  ADD_ARG(double, omega1) = 0.;
  ADD_ARG(double, omega2) = 0.;
  ADD_ARG(double, omega3) = 0.;

  ADD_ARG(std::string, type) = "xyz";
  ADD_ARG(bool, traditional) = false;
};
using CoriolisOptions = std::shared_ptr<CoriolisOptionsImpl>;

class Coriolis123Impl : public torch::nn::Cloneable<Coriolis123Impl> {
 public:
  //! options with which this `Coriolis123` was constructed
  CoriolisOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  // Constructor to initialize the layers
  Coriolis123Impl() : options(CoriolisOptionsImpl::create()) {}
  explicit Coriolis123Impl(CoriolisOptions const& options_,
                           torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(Coriolis123);

class CoriolisXYZImpl : public torch::nn::Cloneable<CoriolisXYZImpl> {
 public:
  //! data
  torch::Tensor omega1, omega2, omega3, alpha, beta;
  torch::Tensor contra_to_spherical, spherical_to_covariant;

  //! cubed-sphere metadata
  int face_id = -1;
  bool cubed_sphere = false;
  bool shallow_water = false;
  bool traditional = false;

  //! options with which this `CoriolisXYZ` was constructed
  CoriolisOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  // Constructor to initialize the layers
  CoriolisXYZImpl() : options(CoriolisOptionsImpl::create()) {}
  explicit CoriolisXYZImpl(CoriolisOptions const& options_,
                           torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(CoriolisXYZ);

//////// (3) Diffusion ////////

struct DiffusionOptionsImpl {
  static std::shared_ptr<DiffusionOptionsImpl> create() {
    return std::make_shared<DiffusionOptionsImpl>();
  }
  static std::shared_ptr<DiffusionOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  DiffusionOptionsImpl() = default;
  std::shared_ptr<DiffusionOptionsImpl> clone() const {
    return std::make_shared<DiffusionOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    auto show = [&os](char const* name, torch::Tensor const& cells,
                      torch::Tensor const& table) {
      os << "* " << name << " = ";
      if (cells.defined()) {
        os << cells.numel() << " cells\n";
      } else if (table.defined()) {
        os << "table of " << table.size(-1) << " points in x1\n";
      } else {
        os << "none (uniform)\n";
      }
    };
    os << "-- diffusion options --\n";
    os << "* nu_iso = " << nu_iso() << "\n"
       << "* kappa_iso = " << kappa_iso() << "\n"
       << "* dynamic = " << (dynamic() ? "true" : "false") << "\n";
    show("nu_scale_x1", nu_scale_x1(), nu_scale_x1_table());
    show("kappa_scale_x1", kappa_scale_x1(), kappa_scale_x1_table());
  }

  ADD_ARG(double, nu_iso) = 0.;
  ADD_ARG(double, kappa_iso) = 0.;
  //! nu_iso/kappa_iso as dynamic coefficients: flux = -mu*stress, -k*grad T
  ADD_ARG(bool, dynamic) = false;
  //! Prescribed x1 wall temperatures in kelvin; zero retains default heat BC.
  ADD_ARG(double, temperature_inner) = 0.;
  ADD_ARG(double, temperature_outer) = 0.;
  ADD_ARG(std::string, wall_velocity) = "stress-free";
  //! Strang-split, second-order L-stable radial heat solve (dry ideal gas).
  ADD_ARG(bool, implicit_radial) = false;
  //! x1 profiles multiplying the kinematic nu_iso/kappa_iso, one value per x1
  //! cell centre of the block (ghosts included), set before it is built;
  //! refused when x1 is split into several blocks (give a table instead)
  ADD_ARG(torch::Tensor, nu_scale_x1);
  ADD_ARG(torch::Tensor, kappa_scale_x1);
  //! the same as a table (2, n): x1 knots, scale; interpolated onto the cells
  ADD_ARG(torch::Tensor, nu_scale_x1_table);
  ADD_ARG(torch::Tensor, kappa_scale_x1_table);
};
using DiffusionOptions = std::shared_ptr<DiffusionOptionsImpl>;

class DiffusionImpl : public torch::nn::Cloneable<DiffusionImpl> {
 public:
  //! options with which this `Diffusion` was constructed
  DiffusionOptions options;

  // Constructor to initialize the layers
  DiffusionImpl() : options(DiffusionOptionsImpl::create()) {}
  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  explicit DiffusionImpl(DiffusionOptions const& options_,
                         torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
  double max_time_step(torch::Tensor w) const;
  void radial_conduction_step(torch::Tensor u, double dt);
  void begin_step(torch::Tensor u, double dt);
  void finish_step(torch::Tensor u, double dt);
  void restore_step(torch::Tensor u) const;

 private:
  torch::Tensor forward_cubed_sphere(torch::Tensor du, torch::Tensor w,
                                     torch::Tensor temp, double dt);
  torch::Tensor split_initial_;
  //! refuses a profile changed since reset; every entry point that reads the
  //! cached profiles (forward, max_time_step) calls it first
  void check_profiles() const;

  //! the x1 profiles on this block's cells (float64, CPU), their maxima, and
  //! copies in the state's device and dtype
  torch::Tensor nu_scale_, kappa_scale_, nu_scale_w_, kappa_scale_w_;
  double nu_scale_max_ = 1., kappa_scale_max_ = 1.;
  //! the profile options reset took, their versions and a copy of their
  //! values, to refuse a change
  std::array<torch::Tensor, 4> profile_options_, profile_values_;
  std::array<int64_t, 4> profile_versions_{};
};
TORCH_MODULE(Diffusion);

//////// (5) Body Heating ////////

struct BodyHeatOptionsImpl {
  static std::shared_ptr<BodyHeatOptionsImpl> create() {
    return std::make_shared<BodyHeatOptionsImpl>();
  }
  static std::shared_ptr<BodyHeatOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  BodyHeatOptionsImpl() = default;
  std::shared_ptr<BodyHeatOptionsImpl> clone() const {
    return std::make_shared<BodyHeatOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "-- body heating options --\n";
    os << "* dTdt = " << dTdt() << "\n"
       << "* pmin = " << pmin() << "\n"
       << "* pmax = " << pmax() << "\n";
  }

  ADD_ARG(double, dTdt) = 0.0;
  ADD_ARG(double, pmin) = 0.0;
  ADD_ARG(double, pmax) = 1.0;
};
using BodyHeatOptions = std::shared_ptr<BodyHeatOptionsImpl>;

class BodyHeatImpl : public torch::nn::Cloneable<BodyHeatImpl> {
 public:
  //! options with which this `BodyHeat` was constructed
  BodyHeatOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  // Constructor to initialize the layers
  BodyHeatImpl() : options(BodyHeatOptionsImpl::create()) {}
  explicit BodyHeatImpl(BodyHeatOptions const& options_,
                        torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(BodyHeat);

//////// (6) Top Cooling ////////

struct TopCoolOptionsImpl {
  static std::shared_ptr<TopCoolOptionsImpl> create() {
    return std::make_shared<TopCoolOptionsImpl>();
  }
  static std::shared_ptr<TopCoolOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  TopCoolOptionsImpl() = default;
  std::shared_ptr<TopCoolOptionsImpl> clone() const {
    return std::make_shared<TopCoolOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "-- top cooling options --\n";
    os << "* flux = " << flux() << "\n"
       << "* depth = " << depth() << "\n";
    if (thickness() > 0.) os << "* thickness = " << thickness() << "\n";
  }

  ADD_ARG(double, flux) = 0.0;
  ADD_ARG(int, depth) = 1;
  //! the layer in x1 length units; used instead of depth when > 0 (#280)
  ADD_ARG(double, thickness) = 0.0;
};
using TopCoolOptions = std::shared_ptr<TopCoolOptionsImpl>;

class TopCoolImpl : public torch::nn::Cloneable<TopCoolImpl> {
 public:
  //! options with which this `TopCool` was constructed
  TopCoolOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  // Constructor to initialize the layers
  TopCoolImpl() : options(TopCoolOptionsImpl::create()) {}
  explicit TopCoolImpl(TopCoolOptions const& options_,
                       torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(TopCool);

//////// (7) Bottom Heating ////////

struct BotHeatOptionsImpl {
  static std::shared_ptr<BotHeatOptionsImpl> create() {
    return std::make_shared<BotHeatOptionsImpl>();
  }
  static std::shared_ptr<BotHeatOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  BotHeatOptionsImpl() = default;
  std::shared_ptr<BotHeatOptionsImpl> clone() const {
    return std::make_shared<BotHeatOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "-- bottom heating options --\n";
    os << "* flux = " << flux() << "\n"
       << "* depth = " << depth() << "\n";
    if (thickness() > 0.) os << "* thickness = " << thickness() << "\n";
  }

  ADD_ARG(double, flux) = 0.0;
  ADD_ARG(int, depth) = 1;
  //! the layer in x1 length units; used instead of depth when > 0 (#280)
  ADD_ARG(double, thickness) = 0.0;
};
using BotHeatOptions = std::shared_ptr<BotHeatOptionsImpl>;

class BotHeatImpl : public torch::nn::Cloneable<BotHeatImpl> {
 public:
  //! options with which this `BotHeat` was constructed
  BotHeatOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  // Constructor to initialize the layers
  BotHeatImpl() : options(BotHeatOptionsImpl::create()) {}
  explicit BotHeatImpl(BotHeatOptions const& options_,
                       torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(BotHeat);

//////// (8) Relax Bottom Composition ////////

struct RelaxBotCompOptionsImpl {
  static std::shared_ptr<RelaxBotCompOptionsImpl> create() {
    return std::make_shared<RelaxBotCompOptionsImpl>();
  }
  static std::shared_ptr<RelaxBotCompOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  RelaxBotCompOptionsImpl() = default;
  std::shared_ptr<RelaxBotCompOptionsImpl> clone() const {
    return std::make_shared<RelaxBotCompOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "-- relax bottom composition options --\n";
    os << "* tau = " << tau() << "\n"
       << "* species = " << fmt::format("{}", species()) << "\n"
       << "* xfrac = " << fmt::format("{}", xfrac()) << "\n";
  }

  ADD_ARG(double, tau) = 0.0;
  ADD_ARG(std::vector<std::string>, species) = {};
  ADD_ARG(std::vector<double>, xfrac) = {};
};
using RelaxBotCompOptions = std::shared_ptr<RelaxBotCompOptionsImpl>;

class RelaxBotCompImpl : public torch::nn::Cloneable<RelaxBotCompImpl> {
 public:
  //! options with which this `RelaxBotComp` was constructed
  RelaxBotCompOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  //! thermodynamic conversion modules
  kintera::ThermoY pthermo_y = nullptr;
  kintera::ThermoX pthermo_x = nullptr;

  //! configured species indices in thermodynamic ordering
  std::vector<int64_t> species_ids;

  // Constructor to initialize the layers
  RelaxBotCompImpl() : options(RelaxBotCompOptionsImpl::create()) {}
  explicit RelaxBotCompImpl(RelaxBotCompOptions const& options_,
                            torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(RelaxBotComp);

//////// (9) Relax Bottom Temperature ////////

struct RelaxBotTempOptionsImpl {
  static std::shared_ptr<RelaxBotTempOptionsImpl> create() {
    return std::make_shared<RelaxBotTempOptionsImpl>();
  }
  static std::shared_ptr<RelaxBotTempOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  RelaxBotTempOptionsImpl() = default;
  std::shared_ptr<RelaxBotTempOptionsImpl> clone() const {
    return std::make_shared<RelaxBotTempOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "-- relax bottom temperature options --\n";
    os << "* tau = " << tau() << "\n"
       << "* btemp = " << btemp() << "\n"
       << "* at-face = " << (at_face() ? "true" : "false") << "\n";
  }

  ADD_ARG(double, tau) = 0.0;
  ADD_ARG(double, btemp) = 300.0;

  //! relax the extrapolated bottom-FACE temperature instead of the first
  //! cell-centre temperature
  ADD_ARG(bool, at_face) = false;
};
using RelaxBotTempOptions = std::shared_ptr<RelaxBotTempOptionsImpl>;

class RelaxBotTempImpl : public torch::nn::Cloneable<RelaxBotTempImpl> {
 public:
  //! options with which this `RelaxBotTemp` was constructed
  RelaxBotTempOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  // Constructor to initialize the layers
  RelaxBotTempImpl() : options(RelaxBotTempOptionsImpl::create()) {}
  explicit RelaxBotTempImpl(RelaxBotTempOptions const& options_,
                            torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(RelaxBotTemp);

//////// (10) Relax Bottom Velocity ////////

struct RelaxBotVeloOptionsImpl {
  static std::shared_ptr<RelaxBotVeloOptionsImpl> create() {
    return std::make_shared<RelaxBotVeloOptionsImpl>();
  }
  static std::shared_ptr<RelaxBotVeloOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  RelaxBotVeloOptionsImpl() = default;
  std::shared_ptr<RelaxBotVeloOptionsImpl> clone() const {
    return std::make_shared<RelaxBotVeloOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "-- relax bottom velocity options --\n";
    os << "* tau = " << tau() << "\n"
       << "* bvx = " << bvx() << "\n"
       << "* bvy = " << bvy() << "\n"
       << "* bvz = " << bvz() << "\n";
  }

  ADD_ARG(double, tau) = 0.0;
  ADD_ARG(double, bvx) = 0.0;
  ADD_ARG(double, bvy) = 0.0;
  ADD_ARG(double, bvz) = 0.0;
};
using RelaxBotVeloOptions = std::shared_ptr<RelaxBotVeloOptionsImpl>;

class RelaxBotVeloImpl : public torch::nn::Cloneable<RelaxBotVeloImpl> {
 public:
  //! options with which this `RelaxBotVelo` was constructed
  RelaxBotVeloOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  // Constructor to initialize the layers
  RelaxBotVeloImpl() : options(RelaxBotVeloOptionsImpl::create()) {}
  explicit RelaxBotVeloImpl(RelaxBotVeloOptions const& options_,
                            torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(RelaxBotVelo);

/////// (11) Top Sponge Layer ////////

struct TopSpongeLyrOptionsImpl {
  static std::shared_ptr<TopSpongeLyrOptionsImpl> create() {
    return std::make_shared<TopSpongeLyrOptionsImpl>();
  }
  static std::shared_ptr<TopSpongeLyrOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  TopSpongeLyrOptionsImpl() = default;
  std::shared_ptr<TopSpongeLyrOptionsImpl> clone() const {
    return std::make_shared<TopSpongeLyrOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "-- top sponge layer options --\n";
    os << "* tau = " << tau() << "\n"
       << "* width = " << width() << "\n";
  }

  ADD_ARG(double, tau) = 0.0;
  ADD_ARG(double, width) = 0.0;
};
using TopSpongeLyrOptions = std::shared_ptr<TopSpongeLyrOptionsImpl>;

class TopSpongeLyrImpl : public torch::nn::Cloneable<TopSpongeLyrImpl> {
 public:
  //! options with which this `TopSpongeLyr` was constructed
  TopSpongeLyrOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  // Constructor to initialize the layers
  TopSpongeLyrImpl() : options(TopSpongeLyrOptionsImpl::create()) {}
  explicit TopSpongeLyrImpl(TopSpongeLyrOptions const& options_,
                            torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(TopSpongeLyr);

//////// (12) Bottom Sponge Layer ////////

struct BotSpongeLyrOptionsImpl {
  static std::shared_ptr<BotSpongeLyrOptionsImpl> create() {
    return std::make_shared<BotSpongeLyrOptionsImpl>();
  }
  static std::shared_ptr<BotSpongeLyrOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  BotSpongeLyrOptionsImpl() = default;
  std::shared_ptr<BotSpongeLyrOptionsImpl> clone() const {
    return std::make_shared<BotSpongeLyrOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "-- bottom sponge layer options --\n";
    os << "* tau = " << tau() << "\n"
       << "* width = " << width() << "\n";
  }

  ADD_ARG(double, tau) = 0.0;
  ADD_ARG(double, width) = 0.0;
};
using BotSpongeLyrOptions = std::shared_ptr<BotSpongeLyrOptionsImpl>;

class BotSpongeLyrImpl : public torch::nn::Cloneable<BotSpongeLyrImpl> {
 public:
  //! options with which this `BotSpongeLyr` was constructed
  BotSpongeLyrOptions options;

  //! non-owning reference to parent
  HydroImpl const* phydro = nullptr;

  // Constructor to initialize the layers
  BotSpongeLyrImpl() : options(BotSpongeLyrOptionsImpl::create()) {}
  explicit BotSpongeLyrImpl(BotSpongeLyrOptions const& options_,
                            torch::nn::Module* p = nullptr);
  void reset() override;

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(BotSpongeLyr);

//////// (13) Plume Forcing ////////

struct PlumeForcingOptionsImpl {
  static std::shared_ptr<PlumeForcingOptionsImpl> create() {
    return std::make_shared<PlumeForcingOptionsImpl>();
  }
  static std::shared_ptr<PlumeForcingOptionsImpl> from_yaml(
      YAML::Node const& forcing);

  PlumeForcingOptionsImpl() = default;
  std::shared_ptr<PlumeForcingOptionsImpl> clone() const {
    return std::make_shared<PlumeForcingOptionsImpl>(*this);
  }
  void report(std::ostream& os) const {
    os << "-- plume forcing options --\n";
    os << "* entrainment = " << entrainment() << "\n"
       << "* N2 = " << N2() << "\n";
  }

  ADD_ARG(double, entrainment) = 0.1;
  ADD_ARG(double, N2) = 0.0;
};
using PlumeForcingOptions = std::shared_ptr<PlumeForcingOptionsImpl>;

class PlumeForcingImpl : public torch::nn::Cloneable<PlumeForcingImpl> {
 public:
  //! options with which this `PlumeForcing` was constructed
  PlumeForcingOptions options;

  // Constructor to initialize the layers
  PlumeForcingImpl() : options(PlumeForcingOptionsImpl::create()) {}
  explicit PlumeForcingImpl(PlumeForcingOptions const& options_) { reset(); }
  void reset() override {}

  torch::Tensor forward(torch::Tensor du, torch::Tensor w, torch::Tensor temp,
                        double dt);
};
TORCH_MODULE(PlumeForcing);

}  // namespace snap

#undef ADD_ARG
