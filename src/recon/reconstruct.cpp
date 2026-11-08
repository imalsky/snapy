// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/eos/equation_of_state.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>

#include "reconstruct.hpp"

namespace snap {
ReconstructOptions ReconstructOptionsImpl::from_yaml(
    std::string const& filename, std::string const& section) {
  auto op = ReconstructOptionsImpl::create();

  auto config = YAML::LoadFile(filename);
  if (!config["dynamics"]) return op;
  if (!config["dynamics"]["reconstruct"]) return op;
  return from_yaml(config["dynamics"]["reconstruct"], section);
}

ReconstructOptions ReconstructOptionsImpl::from_yaml(
    const YAML::Node& node, std::string const& section) {
  auto op = ReconstructOptionsImpl::create();

  check_keys(node, "dynamics/reconstruct", {"vertical", "horizontal"});
  if (!node[section]) return op;

  check_keys(node[section], "dynamics/reconstruct/" + section,
             {"shock", "type", "scale"});
  op->shock() = node[section]["shock"].as<bool>(false);
  op->interp() = InterpOptionsImpl::create();
  op->interp()->type() = node[section]["type"].as<std::string>("dc");
  op->interp()->scale() = node[section]["scale"].as<bool>(false);

  return op;
}

/*
 * |<---- nghost --->|<--- interior -->|<---- nghost --->|
 * |-----|-----|-----|-----|-----|-----|-----|-----|-----|
 * |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |
 * |-----|-----|-----|-----|-----|-----|-----|-----|-----|
 *                      ^           ^
 *                      |           |
 *                      il          iu
 */
void _apply_inplace(int dim, int il, int iu, const torch::Tensor& w,
                    Interp& pinterp, torch::Tensor wlr) {
  if (il > iu) return;

  auto outl = wlr[IRT].slice(dim, il - 1, iu + 1);
  auto outr = wlr[ILT].slice(dim, il, iu + 2);

  pinterp->forward(w, dim, outl, outr);

  // populate dummy regions
  wlr[IRT].slice(dim, 0, il) = wlr[IRT].select(dim, il).unsqueeze(dim);
  wlr[IRT].slice(dim, iu + 1) = wlr[IRT].select(dim, iu).unsqueeze(dim);

  wlr[ILT].slice(dim, 0, il) = wlr[ILT].select(dim, il).unsqueeze(dim);
  wlr[ILT].slice(dim, iu + 1) = wlr[ILT].select(dim, iu).unsqueeze(dim);
}

ReconstructImpl::ReconstructImpl(const ReconstructOptions& options_,
                                 torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void ReconstructImpl::reset() {
  pinterp1 = InterpImpl::create(options->interp(), this, "interp1");
  pinterp2 = InterpImpl::create(options->interp(), this, "interp2");
}

void ReconstructImpl::set_x1_faces(torch::Tensor const& x1f) {
  pinterp1->set_x1_coeffs(x1f);
  pinterp2->set_x1_coeffs(x1f);
}

torch::Tensor ReconstructImpl::forward(torch::Tensor w, int dim, bool floor) {
  auto vec = w.sizes().vec();
  vec.insert(vec.begin(), 2);

  auto result = torch::zeros(vec, w.options());

  auto dim_size = w.size(dim);
  int nghost = pinterp1->stencils() / 2 + 1;
  int il = nghost;
  int iu = dim_size - nghost;
  int nvar = w.size(0);

  TORCH_CHECK(il <= iu, "il > iu");

  if (phydro == nullptr) {
    _apply_inplace(dim, il, iu, w, pinterp1, result);
    result.clamp_min_(0.);
    return result;
  }

  if (options->shock()) {
    _apply_inplace(dim, il, iu, w, pinterp1, result);
    // the species faces get the floor of the non-shock path below: a WENO
    // undershoot would otherwise carry vapor up the gradient. clamp_min keeps
    // -0., so non-negative faces are unchanged bit for bit
    if (phydro->options->eos()->limiter() && nvar > ICY) {
      result.narrow(1, ICY, nvar - ICY).clamp_min_(0.);
    }
    return result;
  }

  /* modify velocity/pressure variables
  if (dim_size > 2 * nghost) {
    if (options->is_boundary_lower()) {
      il += nghost;
    } else if (options->is_boundary_upper()) {
      iu -= nghost;
    }
  } else {
    if (options->is_boundary_lower() && !options->is_boundary_upper()) {
      il += nghost;
    } else if (!options->is_boundary_lower() && options->is_boundary_upper()) {
      iu -= nghost;
    } else if (options->is_boundary_lower() && options->is_boundary_upper()) {
      int len1 = dim_size / 2;
      int len2 = dim_size - len1;
      il += len1;
      iu -= len2;
    }
  }

  // interior
  auto w_ = w.narrow(0, index::IVX, 4);
  auto wlr_ = result.narrow(1, index::IVX, 4);
  _apply_inplace(dim, il, iu, w_, pinterp2, wlr_);*/

  auto eos =
      phydro ? phydro->options->eos() : EquationOfStateOptionsImpl::create();

  // density
  _apply_inplace(dim, il, iu, w.narrow(0, IDN, 1), pinterp1,
                 result.narrow(1, IDN, 1));
  if (eos->limiter() && floor) {
    result.select(1, IDN).clamp_min_(eos->density_floor());
  }

  // velocity/pressure
  int len = std::min((int)IPR, nvar - 1);
  _apply_inplace(dim, il, iu, w.narrow(0, IVX, len), pinterp2,
                 result.narrow(1, IVX, len));
  if (eos->limiter() && floor && result.size(1) > IPR) {
    result.select(1, IPR).clamp_min_(eos->pressure_floor());
  }

  int ny = nvar - ICY;
  if (ny <= 0) return result;

  // others
  _apply_inplace(dim, il, iu, w.narrow(0, ICY, ny), pinterp1,
                 result.narrow(1, ICY, ny));
  if (eos->limiter()) {
    result.narrow(1, ICY, ny).clamp_min_(0.);
  }

  return result;
}

std::shared_ptr<ReconstructImpl> ReconstructImpl::create(
    ReconstructOptions const& opts, torch::nn::Module* p,
    std::string const& name) {
  TORCH_CHECK(p, "[Reconstruct] Parent module is null");
  TORCH_CHECK(opts, "[Reconstruct] Options pointer is null");

  return p->register_module(name, Reconstruct(opts, p));
}

}  // namespace snap
