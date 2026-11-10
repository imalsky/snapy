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

TopCoolOptions TopCoolOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["top-cool"]) return nullptr;

  auto node = forcing["top-cool"];
  check_keys(node, "forcing/top-cool", {"flux", "depth", "thickness"});
  TORCH_CHECK(!(node["depth"] && node["thickness"]),
              "TopCool: give depth (cells) or thickness (x1 length), not both");
  auto op = TopCoolOptionsImpl::create();

  op->flux() = node["flux"].as<double>(0.0);
  op->depth() = node["depth"].as<int>(1);
  op->thickness() = node["thickness"].as<double>(0.0);

  TORCH_CHECK(op->flux() <= 0., "TopCool flux must be negative");
  TORCH_CHECK(op->depth() > 0., "TopCool depth must be greater than zero");
  TORCH_CHECK(!node["thickness"] ||
                  (std::isfinite(op->thickness()) && op->thickness() > 0.),
              "TopCool thickness must be finite and greater than zero");

  return op;
}

TopCoolImpl::TopCoolImpl(TopCoolOptions const& options_, torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void TopCoolImpl::reset() {
  TORCH_CHECK(phydro, "[TopCool] Parent Hydro is null");
  if (!phydro->pmb || !phydro->pmb->pcoord) return;
  auto const& op = phydro->pmb->pcoord->options;
  TORCH_CHECK(options->thickness() > 0. || !op->stretched_x1(),
              "[TopCool] an integer depth is not supported on a stretched x1 "
              "(geometry/x1f or cells/x1rat, #280): it counts cells; give "
              "thickness, the layer in x1 length units");
  TORCH_CHECK(options->thickness() <= op->global_x1max() - op->global_x1min(),
              "[TopCool] thickness exceeds the x1 column");
}

torch::Tensor TopCoolImpl::forward(torch::Tensor du, torch::Tensor w,
                                   torch::Tensor temp, double dt) {
  auto pcoord = phydro->pmb->pcoord;
  // a layer in x1 length (#280): each cell takes its overlap with it
  if (options->thickness() > 0.) {
    int il = pcoord->il(), iu = pcoord->iu();
    double hi = pcoord->options->global_x1max();
    double lo = hi - options->thickness();
    auto xf = pcoord->x1f.slice(0, il, iu + 2);
    auto overlap =
        (xf.slice(0, 1).clamp_max(hi) - xf.slice(0, 0, -1).clamp_min(lo))
            .clamp_min(0.);
    du[IPR].slice(-1, il, iu + 1) += dt * options->flux() /
                                     options->thickness() * overlap /
                                     pcoord->dx1f.slice(0, il, iu + 1);
    return du;
  }

  // Applies at the physical upper x1 boundary only: under an x1-decomposed
  // layout (nb1 > 1) a rank whose upper face is an internal block interface
  // must not force there.
  if (!phydro->pmb->options->is_physical_boundary(0, 0, 1)) return du;

  int iu = pcoord->iu();
  auto dz = pcoord->dx1f[iu];
  du[IPR].slice(-1, iu + 1 - options->depth(), iu + 1) +=
      dt * options->flux() / (dz * options->depth());
  return du;
}

}  // namespace snap
