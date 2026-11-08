// C/C++
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/snap.h>

#include <snap/hydro/balance_column.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

struct RunConfig {
  std::string input_file;
  std::string restart_file;
};

RunConfig ParseArguments(int argc, char** argv,
                         std::string const& default_input) {
  RunConfig cfg{default_input, ""};
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if ((arg == "-r" || arg == "--restart") && i + 1 < argc) {
      cfg.restart_file = argv[++i];
    } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
      cfg.input_file = argv[++i];
    } else {
      cfg.input_file = arg;
    }
  }
  return cfg;
}

int species_offset(std::vector<std::string> const& species,
                   std::string const& name) {
  for (int n = 1; n < species.size(); ++n) {
    if (species[n] == name) return n - 1;
  }
  return -1;
}

torch::Tensor bryan_saturation_pressure(torch::Tensor const& temp) {
  constexpr double kT3 = 273.16;
  constexpr double kP3 = 611.7;
  constexpr double kBeta = 24.845;
  constexpr double kEps = 0.621;
  constexpr double kGamma = 1.4;
  constexpr double kRcpVapor = 1.166;
  constexpr double kRcpLiquid = 3.46;
  constexpr double kDelta =
      (kRcpLiquid - kRcpVapor) * kEps / (1. - 1. / kGamma);

  auto reduced_temp = temp / kT3;
  return kP3 * torch::exp(kBeta * (1. - 1. / reduced_temp) -
                          kDelta * torch::log(reduced_temp));
}

void set_user_output_callback(MeshBlock block,
                              std::vector<std::string> const& species,
                              double p0) {
  int iH2O = species_offset(species, "H2O");
  int iH2Oc = species_offset(species, "H2O(l)");

  block->user_output_callback = [iH2O, iH2Oc, p0](Variables const& vars) {
    constexpr double kRd = 287.;
    constexpr double kEps = 0.621;
    constexpr double kGamma = 1.4;
    constexpr double kRcpVapor = 1.166;
    constexpr double kRcpLiquid = 3.46;
    constexpr double kBeta = 24.845;
    constexpr double kT3 = 273.16;
    constexpr double kDelta =
        (kRcpLiquid - kRcpVapor) * kEps / (1. - 1. / kGamma);

    constexpr double kRv = kRd / kEps;
    constexpr double kCpd = kGamma / (kGamma - 1.) * kRd;
    constexpr double kCpLiquid = kRcpLiquid * kCpd;

    auto w = vars.at("hydro_w");
    auto qtol = torch::zeros_like(w[IDN]);
    auto qv = torch::zeros_like(w[IDN]);
    auto qc = torch::zeros_like(w[IDN]);

    if (iH2O >= 0) qv = w[ICY + iH2O];
    if (iH2Oc >= 0) qc = w[ICY + iH2Oc];
    qtol = qv + qc;

    auto qd = torch::clamp_min(1. - qtol, 1.e-12);
    auto feps = 1. + qv * (1. / kEps - 1.) - qc;
    auto temp = w[IPR] / (w[IDN] * kRd * feps);

    auto eta = qv / (qd * kEps);
    auto xgas = 1. + eta;
    auto pd = w[IPR] / xgas;
    auto pv = w[IPR] * eta / xgas;
    auto rh = torch::clamp_min(pv / bryan_saturation_pressure(temp), 1.e-12);

    auto cpt = kCpd * qd + kCpLiquid * qtol;
    auto lv = kRv * (kBeta * kT3 - kDelta * temp);
    auto theta_e = temp * torch::pow(p0 / pd, kRd * qd / cpt) *
                   torch::pow(rh, -kRv * qv / cpt) *
                   torch::exp(lv * qv / (cpt * temp));

    Variables out;
    out["qtol"] = qtol;
    out["theta_e"] = theta_e;
    return out;
  };
}

torch::Tensor surface_mass_fractions(std::vector<std::string> const& species,
                                     int nc3, int nc2, double qt,
                                     torch::TensorOptions const& options) {
  int ny = static_cast<int>(species.size()) - 1;
  auto yfrac = torch::zeros({ny, nc3, nc2}, options);

  int iH2O = species_offset(species, "H2O");
  if (iH2O >= 0) {
    yfrac[iH2O].fill_(qt);
  }

  return yfrac;
}

void solve_virtual_temperature_perturbation(
    kintera::ThermoX& thermo_x, torch::Tensor const& temp0,
    torch::Tensor const& pres, torch::Tensor const& xfrac0,
    torch::Tensor const& target_tv, torch::Tensor const& mask, double dT,
    double Rd, torch::Tensor& temp_out, torch::Tensor& xfrac_out) {
  auto temp_lo = temp0.clone();
  auto temp_hi = temp0 + std::max(5.0, 2.0 * std::abs(dT));

  for (int iter = 0; iter < 32; ++iter) {
    auto temp_mid = 0.5 * (temp_lo + temp_hi);
    auto xtrial = xfrac0.clone();
    thermo_x->forward(temp_mid, pres, xtrial);

    auto conc = thermo_x->compute("TPX->V", {temp_mid, pres, xtrial});
    auto dens = thermo_x->compute("V->D", {conc});
    auto tv_mid = pres / (dens * Rd);
    auto too_cold = torch::logical_and(mask, tv_mid < target_tv);

    temp_lo = torch::where(too_cold, temp_mid, temp_lo);
    temp_hi =
        torch::where(torch::logical_and(mask, torch::logical_not(too_cold)),
                     temp_mid, temp_hi);
  }

  temp_out = torch::where(mask, 0.5 * (temp_lo + temp_hi), temp0);
  xfrac_out = xfrac0.clone();
  thermo_x->forward(temp_out, pres, xfrac_out);
}

// Leftover |a|/g after a projection; 1e-13 g over 330 s is 3e-10 m/s.
constexpr double kBalanceRtol = 1.e-13;

torch::Tensor interior(MeshBlock block, torch::Tensor const& w) {
  auto pcoord = block->pcoord;
  auto op = pcoord->options;
  return w.narrow(1, pcoord->kl(), op->nx3())
      .narrow(2, pcoord->jl(), op->nx2())
      .narrow(3, pcoord->il(), op->nx1());
}

// The saturation adjustment of the last RK stage (MeshBlockImpl::forward):
// ThermoY at fixed density and internal energy, species written back, applied
// here through the same W -> U -> W round trip and kept on interior cells.
void saturate(MeshBlock block, torch::Tensor w,
              std::shared_ptr<kintera::ThermoYImpl> const& thermo_y) {
  auto peos = block->phydro->peos;
  auto u = peos->compute("W->U", {w.clone()});
  int ny = u.size(0) - ICY;

  auto ke = peos->compute("U->K", {u});
  auto rho = u[IDN] + u.narrow(0, ICY, ny).sum(0);
  auto ie = u[IPR] - ke;
  auto yfrac = u.narrow(0, ICY, ny) / rho;

  auto sub = block->part({0, 0, 0}, PartOptions().exterior(false));
  auto sub3 = block->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  thermo_y->forward(rho.index(sub3), ie.index(sub3), yfrac.index(sub),
                    /*warm_start=*/true);
  u.narrow(0, ICY, ny).copy_(yfrac * rho);

  interior(block, w).copy_(interior(block, peos->compute("U->W", {u})));
}

double project(MeshBlock block, torch::Tensor w, double grav, int* sweeps) {
  auto pcoord = block->pcoord;
  auto wi = interior(block, w);
  auto dx = pcoord->dx1f.narrow(0, pcoord->il(), wi.size(3)).contiguous();
  torch::Tensor balanced;
  double residual;
  std::tie(balanced, residual, *sweeps) =
      balance_column(wi.contiguous(), dx, grav, /*wall_clamp=*/true,
                     kBalanceRtol, /*max_iter=*/400, pcoord->options->type(),
                     pcoord->options->stretched_x1());
  wi.copy_(balanced);
  return residual;
}

// problem/balance-ic (#250 D1): project the IC onto the discrete hydrostatic
// balance. balance_column holds T and mass fractions fixed and the run's first
// saturation adjustment moves the column off that balance again, so a moist
// column alternates the two until saturation no longer moves it.
void balance_ic(MeshBlock block, torch::Tensor w, double grav, int max_pass) {
  auto op = block->pcoord->options;
  TORCH_CHECK(op->nx1() == op->global_nx1(),
              "bryan: balance-ic needs the whole x1 column in one block, nx1=",
              op->nx1(), " global_nx1=", op->global_nx1());
  TORCH_CHECK(block->phydro->options->wb_wall_clamp(),
              "bryan: balance-ic needs dynamics/wb-wall-clamp: true");

  auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(
      block->named_modules()["hydro.eos.thermo"]);
  int sweeps = 0;
  if (!thermo_y || thermo_y->options->reactions().empty()) {
    project(block, w, grav, &sweeps);
    return;
  }

  constexpr double kEps = std::numeric_limits<double>::epsilon();
  double residual = -1.;  // no projection yet
  for (int pass = 0;; ++pass) {
    auto before = interior(block, w).clone();
    saturate(block, w, thermo_y);
    auto after = interior(block, w);
    double drho = (after[IDN] - before[IDN]).abs().max().item<double>() /
                  before[IDN].abs().max().item<double>();
    double dp =
        ((after[IPR] - before[IPR]) / before[IPR]).abs().max().item<double>();
    double dp_abs = (after[IPR] - before[IPR]).abs().max().item<double>();
    std::cout << std::scientific << std::setprecision(6)
              << "bryan: balance-ic pass " << pass << " residual " << residual
              << " sweeps " << sweeps << " | saturation max|drho|/max|rho| "
              << drho << " max|dp/p| " << dp << " max|dp| " << dp_abs
              << std::endl;
    if (pass > 0 && drho <= 4. * kEps && dp <= 1.e-13 &&
        residual <= kBalanceRtol) {
      return;
    }
    TORCH_CHECK(pass < max_pass, "bryan: balance-ic did not converge in ",
                max_pass, " passes: saturation still moves the projected ",
                "column by max|drho|/max|rho| = ", drho, ", max|dp/p| = ", dp);
    residual = project(block, w, grav, &sweeps);
  }
}

void initialize_block(MeshBlock block, Variables& vars,
                      YAML::Node const& config, torch::Device const& device) {
  auto pcoord = block->pcoord;
  auto peos = block->phydro->peos;
  auto modules = block->named_modules();
  auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(
      modules["hydro.eos.thermo"]);

  kintera::ThermoX thermo_x(thermo_y->options);
  thermo_x->to(device);

  auto const& species = thermo_y->options->species();
  int ny = static_cast<int>(species.size()) - 1;
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int il = pcoord->il();
  int iu = pcoord->iu();
  int nvar = peos->nvar();

  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  auto w = torch::zeros({nvar, nc3, nc2, nc1}, options);

  double Ps = config["problem"]["p0"].as<double>();
  double Ts = config["problem"]["Ts"].as<double>();
  double xc = config["problem"]["xc"].as<double>();
  double zc = config["problem"]["zc"].as<double>();
  double xr = config["problem"]["xr"].as<double>();
  double zr = config["problem"]["zr"].as<double>();
  double dT = config["problem"]["dT"].as<double>();
  double qt = config["problem"]["qt"].as<double>();
  double grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  auto temp_state = torch::zeros({nc3, nc2, nc1}, options);
  auto pres_state = torch::zeros({nc3, nc2, nc1}, options);
  auto xfrac_state = torch::zeros(
      std::vector<int64_t>{nc3, nc2, nc1, static_cast<int64_t>(species.size())},
      options);

  auto temp = Ts * torch::ones({nc3, nc2}, options);
  auto pres = Ps * torch::ones({nc3, nc2}, options);
  auto yfrac = surface_mass_fractions(species, nc3, nc2, qt, options);
  auto xfrac = thermo_y->compute("Y->X", {yfrac});
  thermo_x->forward(temp, pres, xfrac);

  double dz = pcoord->dx1f[il].item<double>();
  thermo_x->extrapolate_dz(
      temp, pres, xfrac,
      kintera::ExtrapOptions().dz(0.5 * dz).grav(grav).ds_dz(0.).rainout(
          false));

  for (int i = il; i <= iu; ++i) {
    temp_state.select(2, i).copy_(temp);
    pres_state.select(2, i).copy_(pres);
    xfrac_state.select(2, i).copy_(xfrac);

    if (i < iu) {
      // step centre i -> i+1 by the centre spacing, not the cell width
      dz = pcoord->dx1v[i + 1].item<double>();
      thermo_x->extrapolate_dz(
          temp, pres, xfrac,
          kintera::ExtrapOptions().dz(dz).grav(grav).ds_dz(0.).rainout(false));
    }
  }

  auto x2 = pcoord->x2v.view({1, nc2}).expand({nc3, nc2});
  double Rd = kintera::constants::Rgas / thermo_x->mu[0].item<double>();

  for (int i = il; i <= iu; ++i) {
    double x1 = pcoord->x1v[i].item<double>();
    auto L = torch::sqrt(torch::square((x2 - xc) / xr) +
                         std::pow((x1 - zc) / zr, 2));
    auto mask = L < 1.;
    auto amp = dT * torch::square(torch::cos(0.5 * M_PI * L)) / 300.;

    auto temp_i = temp_state.select(2, i);
    auto pres_i = pres_state.select(2, i);
    auto xfrac_i = xfrac_state.select(2, i);
    auto conc_i = thermo_x->compute(
        "TPX->V", std::vector<torch::Tensor>{temp_i, pres_i, xfrac_i});
    auto dens_i = thermo_x->compute("V->D", std::vector<torch::Tensor>{conc_i});
    auto target_tv = pres_i / (dens_i * Rd) * (1. + amp);

    torch::Tensor temp_new;
    torch::Tensor xfrac_new;
    solve_virtual_temperature_perturbation(thermo_x, temp_i, pres_i, xfrac_i,
                                           target_tv, mask, dT, Rd, temp_new,
                                           xfrac_new);
    temp_i.copy_(temp_new);
    xfrac_i.copy_(xfrac_new);
  }

  for (int i = il; i <= iu; ++i) {
    auto temp_i = temp_state.select(2, i);
    auto pres_i = pres_state.select(2, i);
    auto xfrac_i = xfrac_state.select(2, i);
    auto conc_i = thermo_x->compute(
        "TPX->V", std::vector<torch::Tensor>{temp_i, pres_i, xfrac_i});

    w[IPR].select(2, i).copy_(pres_i);
    w[IDN].select(2, i).copy_(
        thermo_x->compute("V->D", std::vector<torch::Tensor>{conc_i}));
    w.narrow(0, ICY, ny)
        .select(3, i)
        .copy_(thermo_x->compute("X->Y", std::vector<torch::Tensor>{xfrac_i}));
  }

  if (config["problem"]["balance-ic"].as<bool>(false)) {
    balance_ic(block, w, grav,
               config["problem"]["balance-ic-passes"].as<int>(8));
  }

  vars["hydro_w"] = w;
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  auto args = ParseArguments(argc, argv, "bryan.yaml");
  auto config = YAML::LoadFile(args.input_file);

  auto mesh = Mesh(MeshOptionsImpl::from_yaml(args.input_file));
  auto device = torch::Device(mesh->options->device_str());
  if (device.is_cuda()) {
    std::cout << "Running on CUDA" << std::endl;
  }
  mesh->to(device);

  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    auto modules = mesh->blocks[i]->named_modules();
    auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(
        modules["hydro.eos.thermo"]);
    set_user_output_callback(mesh->blocks[i], thermo_y->options->species(),
                             config["problem"]["p0"].as<double>());
    if (args.restart_file.empty()) {
      initialize_block(mesh->blocks[i], vars[i], config, device);
    }
  }

  double current_time = args.restart_file.empty()
                            ? mesh->initialize(vars)
                            : mesh->initialize(vars, args.restart_file.c_str());
  mesh->make_outputs(vars, current_time);

  int cycle = mesh->blocks.front()->cycle;
  while (!mesh->blocks.front()->pintg->stop(cycle, current_time)) {
    auto dt = mesh->max_time_step(vars);
    mesh->print_cycle_info(vars, current_time, dt);

    for (int stage = 0; stage < mesh->blocks.front()->pintg->stages.size();
         ++stage) {
      mesh->forward(vars, dt, stage);
    }

    int redo = mesh->check_redo(vars);
    if (redo > 0) continue;
    if (redo < 0) break;

    ++cycle;
    mesh->set_cycle(cycle);
    current_time += dt;
    mesh->make_outputs(vars, current_time);
  }

  return mesh->finalize(vars, current_time);
}
