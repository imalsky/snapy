// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {

BotHeatOptions BotHeatOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["bot-heat"]) return nullptr;

  auto node = forcing["bot-heat"];
  check_keys(node, "forcing/bot-heat", {"flux", "depth"});
  auto op = BotHeatOptionsImpl::create();

  op->flux() = node["flux"].as<double>(0.0);
  op->depth() = node["depth"].as<int>(1);

  TORCH_CHECK(op->flux() >= 0., "BotHeat flux must be positive");
  TORCH_CHECK(op->depth() > 0., "BotHeat depth must be greater than zero");

  return op;
}

BotHeatImpl::BotHeatImpl(BotHeatOptions const& options_, torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void BotHeatImpl::reset() {
  TORCH_CHECK(phydro, "[BotHeat] Parent Hydro is null");
  TORCH_CHECK(!(phydro->pmb && phydro->pmb->pcoord &&
                phydro->pmb->pcoord->options->stretched_x1()),
              "[BotHeat] not supported on a stretched x1 (geometry/x1f or "
              "cells/x1rat, #280)");
}

torch::Tensor BotHeatImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  // Applies at the physical lower x1 boundary only: under an x1-decomposed
  // layout (nb1 > 1) a rank whose lower face is an internal block interface
  // must not force there.
  if (!phydro->pmb->options->is_physical_boundary(0, 0, -1)) return du;

  auto pcoord = phydro->pmb->pcoord;

  int il = pcoord->il();
  auto dz = pcoord->dx1f[il];
  du[IPR].narrow(-1, il, options->depth()) +=
      dt * options->flux() / (dz * options->depth());
  return du;
}

}  // namespace snap
