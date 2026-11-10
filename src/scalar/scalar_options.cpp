// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/input/check_keys.hpp>

#include "scalar.hpp"

namespace snap {

ScalarOptions ScalarOptionsImpl::from_yaml(std::string const& filename,
                                           bool verbose) {
  auto op = ScalarOptionsImpl::create();
  op->verbose() = verbose;

  auto config = YAML::LoadFile(filename);
  auto node = config["scalar"];
  if (!node) {
    return op;
  }

  check_keys(node, "scalar",
             {"verbose", "nvar", "upper-bound", "names", "reconstruct",
              "riemann-solver"});
  op->verbose() = node["verbose"].as<bool>(verbose);
  op->nvar() = node["nvar"].as<int>(0);
  op->upper_bound() = node["upper-bound"].as<double>(-1.);
  if (node["names"]) {
    op->names() = node["names"].as<std::vector<std::string>>();
    op->nvar() = op->names().size();
  }

  if (node["reconstruct"]) {
    op->recon() = ReconstructOptionsImpl::create();
    auto recon = node["reconstruct"];
    check_keys(recon, "scalar/reconstruct", {"shock", "type"});
    op->recon()->shock() = recon["shock"].as<bool>(false);
    op->recon()->interp()->type() = recon["type"].as<std::string>("dc");
  } else {
    op->recon() = nullptr;
  }

  // op->thermo() = kintera::ThermoOptionsImpl::from_yaml(filename);
  // op->kinetics() = kintera::KineticsOptionsImpl::from_yaml(filename);

  if (node["riemann-solver"]) {
    check_keys(node["riemann-solver"], "scalar/riemann-solver",
               {"type", "dir"});
    op->riemann() = RiemannSolverOptionsImpl::from_yaml(node["riemann-solver"]);
  } else {
    op->riemann() = RiemannSolverOptionsImpl::create();
    op->riemann()->type() = "upwind";
  }

  if (op->names().empty()) {
    op->names().reserve(op->nvar());
    for (int n = 0; n < op->nvar(); ++n) {
      op->names().push_back("scalar" + std::to_string(n));
    }
  }

  return op;
}

}  // namespace snap
