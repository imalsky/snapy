#include <cmath>

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
  check_keys(node, "forcing/bot-heat", {"flux", "depth", "thickness"});
  TORCH_CHECK(!(node["depth"] && node["thickness"]),
              "BotHeat: give depth (cells) or thickness (x1 length), not both");
  auto op = BotHeatOptionsImpl::create();

  op->flux() = node["flux"].as<double>(0.0);
  op->depth() = node["depth"].as<int>(1);
  op->thickness() = node["thickness"].as<double>(0.0);

  TORCH_CHECK(op->flux() >= 0., "BotHeat flux must be positive");
  TORCH_CHECK(op->depth() > 0., "BotHeat depth must be greater than zero");
  TORCH_CHECK(!node["thickness"] ||
                  (std::isfinite(op->thickness()) && op->thickness() > 0.),
              "BotHeat thickness must be finite and greater than zero");

  return op;
}

BotHeatImpl::BotHeatImpl(BotHeatOptions const& options_, torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void BotHeatImpl::reset() {
  TORCH_CHECK(phydro, "[BotHeat] Parent Hydro is null");
  if (!phydro->pmb || !phydro->pmb->pcoord) return;
  auto const& op = phydro->pmb->pcoord->options;
  TORCH_CHECK(options->thickness() > 0. || !op->stretched_x1(),
              "[BotHeat] an integer depth is not supported on a stretched x1 "
              "(geometry/x1f or cells/x1rat, #280): it counts cells; give "
              "thickness, the layer in x1 length units");
  TORCH_CHECK(options->thickness() <= op->global_x1max() - op->global_x1min(),
              "[BotHeat] thickness exceeds the x1 column");
}

torch::Tensor BotHeatImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  auto pcoord = phydro->pmb->pcoord;
  // a layer in x1 length (#280): each cell takes its overlap with it
  if (options->thickness() > 0.) {
    int il = pcoord->il(), iu = pcoord->iu();
    double lo = pcoord->options->global_x1min();
    double hi = lo + options->thickness();
    auto xf = pcoord->x1f.slice(0, il, iu + 2);
    auto overlap =
        (xf.slice(0, 1).clamp_max(hi) - xf.slice(0, 0, -1).clamp_min(lo))
            .clamp_min(0.);
    du[IPR].slice(-1, il, iu + 1) += dt * options->flux() /
                                     options->thickness() * overlap /
                                     pcoord->dx1f.slice(0, il, iu + 1);
    return du;
  }

  // Applies at the physical lower x1 boundary only: under an x1-decomposed
  // layout (nb1 > 1) a rank whose lower face is an internal block interface
  // must not force there.
  if (!phydro->pmb->options->is_physical_boundary(0, 0, -1)) return du;

  int il = pcoord->il();
  auto dz = pcoord->dx1f[il];
  du[IPR].narrow(-1, il, options->depth()) +=
      dt * options->flux() / (dz * options->depth());
  return du;
}

}  // namespace snap
