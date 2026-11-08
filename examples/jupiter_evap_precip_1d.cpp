// C/C++
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

#include <kintera/kinetics/evolve_implicit.hpp>
#include <kintera/kinetics/kinetics.hpp>
#include <kintera/thermo/thermo.hpp>

// snap
#include <snap/input/command_line.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

namespace {

std::string input_filename_or_default(CommandLine const* cli) {
  if (cli != nullptr && cli->input_filename != nullptr) {
    return cli->input_filename;
  }
  return "jupiter_evap_precip_1d.yaml";
}

int species_offset(std::vector<std::string> const& species,
                   std::string const& name) {
  auto it = std::find(species.begin() + 1, species.end(), name);
  if (it == species.end()) return -1;
  return static_cast<int>(it - species.begin()) - 1;
}

double required_problem_value(YAML::Node const& config,
                              std::string const& name) {
  auto node = config["problem"][name];
  if (!node) {
    throw std::runtime_error("problem." + name + " is required");
  }
  return node.as<double>();
}

double optional_problem_value(YAML::Node const& config, std::string const& name,
                              double default_value) {
  return config["problem"][name].as<double>(default_value);
}

void set_user_output_callback(MeshBlock block,
                              std::vector<std::string> const& species) {
  int iH2O = species_offset(species, "H2O");
  int iNH3 = species_offset(species, "NH3");
  int iH2S = species_offset(species, "H2S");
  int iH2Olp = species_offset(species, "H2O(l,p)");
  int iNH3sp = species_offset(species, "NH3(s,p)");
  int iNH4SHsp = species_offset(species, "NH4SH(s,p)");

  block->user_output_callback = [=](Variables const& vars) {
    auto const& w = vars.at("hydro_w");
    auto zeros = torch::zeros_like(w[IDN]);

    auto h2o_v = iH2O >= 0 ? w[ICY + iH2O] : zeros;
    auto nh3_v = iNH3 >= 0 ? w[ICY + iNH3] : zeros;
    auto h2s_v = iH2S >= 0 ? w[ICY + iH2S] : zeros;
    auto h2o_p = iH2Olp >= 0 ? w[ICY + iH2Olp] : zeros;
    auto nh3_p = iNH3sp >= 0 ? w[ICY + iNH3sp] : zeros;
    auto nh4sh_p = iNH4SHsp >= 0 ? w[ICY + iNH4SHsp] : zeros;

    Variables out;
    out["q_vapor"] = h2o_v + nh3_v + h2s_v;
    out["q_precip"] = h2o_p + nh3_p + nh4sh_p;
    out["q_H2Olp"] = h2o_p;
    out["q_NH3sp"] = nh3_p;
    out["q_NH4SHsp"] = nh4sh_p;
    return out;
  };
}

void add_pocket_species(torch::Tensor& w, int offset,
                        torch::Tensor const& amp) {
  if (offset >= 0) {
    w[ICY + offset] += amp;
  }
}

void initialize_block(MeshBlock block, Variables& vars,
                      YAML::Node const& config, torch::Device const& device) {
  auto phydro = block->phydro;
  auto pcoord = block->pcoord;
  auto peos = phydro->peos;
  auto modules = block->named_modules();
  auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(
      modules["hydro.eos.thermo"]);

  kintera::ThermoX thermo_x(thermo_y->options);
  thermo_x->to(device);

  auto const& species = thermo_y->options->species();
  int ny = static_cast<int>(species.size()) - 1;
  int nvar = peos->nvar();
  int nc1 = pcoord->options->nc1();
  int nc2 = pcoord->options->nc2();
  int nc3 = pcoord->options->nc3();
  int il = pcoord->il();
  int iu = pcoord->iu();

  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  auto w = torch::zeros({nvar, nc3, nc2, nc1}, options);

  double Ps = required_problem_value(config, "Ps");
  double Ts = required_problem_value(config, "Ts");
  double Tmin = optional_problem_value(config, "Tmin", 110.);
  double grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();

  auto temp = Ts * torch::ones({nc3, nc2}, options);
  auto pres = Ps * torch::ones({nc3, nc2}, options);
  auto xfrac =
      torch::zeros({nc3, nc2, static_cast<int64_t>(species.size())}, options);

  for (int n = 1; n <= ny; ++n) {
    auto xmixr = config["problem"]["x" + species[n]].as<double>(0.);
    xfrac.select(2, n).fill_(xmixr);
  }
  xfrac.select(2, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);

  auto dz = pcoord->dx1f[il].item<double>();
  thermo_x->extrapolate_dz(
      temp, pres, xfrac,
      kintera::ExtrapOptions().dz(0.5 * dz).grav(grav).ds_dz(0.).rainout(true));

  int i = il;
  for (; i <= iu; ++i) {
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
    w[IPR].select(2, i).copy_(pres);
    w[IDN].select(2, i).copy_(thermo_x->compute("V->D", {conc}));
    w.narrow(0, ICY, ny).select(3, i).copy_(thermo_x->compute("X->Y", {xfrac}));

    if ((temp < Tmin).any().item<bool>()) {
      ++i;
      break;
    }
    if (i < iu) {
      // step centre i -> i+1 by the centre spacing, not the cell width
      dz = pcoord->dx1v[i + 1].item<double>();
      thermo_x->extrapolate_dz(
          temp, pres, xfrac,
          kintera::ExtrapOptions().dz(dz).grav(grav).ds_dz(0.).rainout(true));
    }
  }

  for (; i <= iu; ++i) {
    dz = pcoord->dx1v[i].item<double>();  // centre i-1 -> i
    auto mu = (thermo_x->mu * xfrac).sum(-1);
    pres *= torch::exp(-grav * mu * dz / (kintera::constants::Rgas * temp));
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
    w[IPR].select(2, i).copy_(pres);
    w[IDN].select(2, i).copy_(thermo_x->compute("V->D", {conc}));
    w.narrow(0, ICY, ny).select(3, i).copy_(thermo_x->compute("X->Y", {xfrac}));
  }

  double z0 = required_problem_value(config, "pocket-center");
  double width = required_problem_value(config, "pocket-width");
  auto z = pcoord->x1v.view({1, 1, nc1}).expand({nc3, nc2, nc1});
  auto distance = torch::abs((z - z0) / width);
  auto shape = torch::where(distance < 1.,
                            torch::square(torch::cos(0.5 * M_PI * distance)),
                            torch::zeros_like(distance));

  add_pocket_species(w, species_offset(species, "H2O(l,p)"),
                     optional_problem_value(config, "qH2Olp", 0.) * shape);
  add_pocket_species(w, species_offset(species, "NH3(s,p)"),
                     optional_problem_value(config, "qNH3sp", 0.) * shape);
  add_pocket_species(w, species_offset(species, "NH4SH(s,p)"),
                     optional_problem_value(config, "qNH4SHsp", 0.) * shape);

  vars["hydro_w"] = w;
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  auto cli = CommandLine::ParseArguments(argc, argv);
  if (!cli) return 0;

  auto infile = input_filename_or_default(cli);
  auto config = YAML::LoadFile(infile);

  auto op_block = MeshBlockOptionsImpl::from_yaml(infile);
  auto block = MeshBlock(op_block);

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available() && op_block->layout()->device() == "cuda") {
    std::cout << "Running on CUDA" << std::endl;
    int device_id = op_block->layout()->device_id();
    if (device_id < 0) device_id = op_block->layout()->local_rank();
    device = torch::Device(torch::kCUDA, device_id);
  }
  block->to(device);

  auto modules = block->named_modules();
  auto thermo_y = std::dynamic_pointer_cast<kintera::ThermoYImpl>(
      modules["hydro.eos.thermo"]);
  set_user_output_callback(block, thermo_y->options->species());

  Variables vars;
  double current_time = 0.;
  if (cli->restart_filename == nullptr) {
    initialize_block(block, vars, config, device);
    current_time = block->initialize(vars);
    block->make_outputs(vars, current_time);
  } else {
    current_time = block->initialize(vars, cli->restart_filename);
  }

  auto peos = block->phydro->peos;
  auto op_kinet = kintera::KineticsOptionsImpl::from_yaml(infile);
  auto kinet = kintera::Kinetics(op_kinet);
  kinet->to(device);
  kintera::ThermoX thermo_x(thermo_y->options);
  thermo_x->to(device);
  int ny = static_cast<int>(thermo_y->options->species().size()) - 1;

  while (!block->pintg->stop(block->cycle, current_time)) {
    auto dt = block->max_time_step(vars);
    block->print_cycle_info(vars, current_time, dt);

    for (int stage = 0; stage < block->pintg->stages.size(); ++stage) {
      block->forward(vars, dt, stage);
    }

    auto& hydro_u = vars["hydro_u"];
    auto& hydro_w = vars["hydro_w"];
    // Kinetics needs the final dynamics state, including saturation adjustment.
    peos->forward(hydro_u, hydro_w);
    auto temp = peos->compute("W->T", {hydro_w});
    auto pres = hydro_w[IPR];
    auto xfrac = thermo_y->compute("Y->X", {hydro_w.narrow(0, ICY, ny)});
    auto conc = thermo_x->compute("TPX->V", {temp, pres, xfrac});
    auto cp_vol = thermo_x->compute("TV->cp", {temp, conc});
    auto conc_kinet = conc.slice(-1, 1, conc.size(-1));
    auto [rate, rc_ddC, rc_ddT, rate_constants] =
        kinet->forward_with_rate_constants(temp, pres, conc_kinet);
    auto jac = kinet->jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT,
                               rate_constants);
    auto del_conc = kintera::evolve_implicit(rate, kinet->stoich, jac, dt);
    std::vector<int64_t> view_shape(del_conc.dim(), 1);
    view_shape[del_conc.dim() - 1] = -1;
    auto del_rho =
        del_conc / thermo_y->inv_mu.narrow(0, 1, ny).view(view_shape);
    hydro_u.narrow(0, ICY, ny) += del_rho.permute({3, 0, 1, 2});

    int redo = block->check_redo(vars);
    if (redo > 0) continue;
    if (redo < 0) break;

    ++block->cycle;
    current_time += dt;
    block->make_outputs(vars, current_time);
  }

  int status = block->finalize(vars, current_time);
  if (status == 0) {
    std::cout << "Completed jupiter_evap_precip_1d at time=" << current_time
              << " seconds" << std::endl;
  }

  CommandLine::Destroy();
  return status;
}
