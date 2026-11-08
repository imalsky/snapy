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

TopCoolOptions TopCoolOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["top-cool"]) return nullptr;

  auto node = forcing["top-cool"];
  check_keys(node, "forcing/top-cool", {"flux", "depth"});
  auto op = TopCoolOptionsImpl::create();

  op->flux() = node["flux"].as<double>(0.0);
  op->depth() = node["depth"].as<int>(1);

  TORCH_CHECK(op->flux() <= 0., "TopCool flux must be negative");
  TORCH_CHECK(op->depth() > 0., "TopCool depth must be greater than zero");

  return op;
}

TopCoolImpl::TopCoolImpl(TopCoolOptions const& options_, torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void TopCoolImpl::reset() {
  TORCH_CHECK(phydro, "[TopCool] Parent Hydro is null");
  TORCH_CHECK(!(phydro->pmb && phydro->pmb->pcoord &&
                phydro->pmb->pcoord->options->stretched_x1()),
              "[TopCool] not supported on a stretched x1 (geometry/x1f or "
              "cells/x1rat, #280)");
}

torch::Tensor TopCoolImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  // Applies at the physical upper x1 boundary only: under an x1-decomposed
  // layout (nb1 > 1) a rank whose upper face is an internal block interface
  // must not force there.
  if (!phydro->pmb->options->is_physical_boundary(0, 0, 1)) return du;

  auto pcoord = phydro->pmb->pcoord;

  int iu = pcoord->iu();
  auto dz = pcoord->dx1f[iu];
  du[IPR].slice(-1, iu + 1 - options->depth(), iu + 1) +=
      dt * options->flux() / (dz * options->depth());
  return du;
}

}  // namespace snap
