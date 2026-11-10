// snap
#include "bc_func.hpp"

#include <snap/coord/coordinate.hpp>
#include <snap/eos/equation_of_state.hpp>

BC_FUNCTION(custom_inner, var, dim, op) {}
BC_FUNCTION(custom_outer, var, dim, op) {}

BC_FUNCTION(reflecting_inner, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nghost = op.nghost();

  var.narrow(dim, 0, nghost) = var.narrow(dim, nghost, nghost).flip(dim);

  // normal velocities
  if (op.type() == snap::kConserved || op.type() == snap::kPrimitive) {
    var[4 - dim].narrow(dim - 1, 0, nghost) *= -1;
  }
}

BC_FUNCTION(reflecting_outer, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, nc - nghost, nghost) =
      var.narrow(dim, nc - 2 * nghost, nghost).flip(dim);

  // normal velocities
  if (op.type() == snap::kConserved || op.type() == snap::kPrimitive) {
    var[4 - dim].narrow(dim - 1, nc - nghost, nghost) *= -1;
  }
}

BC_FUNCTION(periodic_inner, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, 0, nghost) = var.narrow(dim, nc - 2 * nghost, nghost);
}

BC_FUNCTION(periodic_outer, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, nc - nghost, nghost) = var.narrow(dim, nghost, nghost);
}

BC_FUNCTION(extrapolation_inner, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, 0, nghost) = var.narrow(dim, nghost, 1);
}

BC_FUNCTION(extrapolation_outer, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc = var.size(dim);
  int nghost = op.nghost();

  var.narrow(dim, nc - nghost, nghost) = var.narrow(dim, nc - nghost - 1, 1);
}

BC_FUNCTION(solid_inner, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nghost = op.nghost();

  std::vector<int64_t> shape(var.dim(), -1);
  shape[dim] = nghost;

  var.narrow(dim, 0, nghost) = 1;
}

BC_FUNCTION(solid_outer, var, dim, op) {
  if (var.size(dim) == 1) return;
  int nc1 = var.size(dim);
  int nghost = op.nghost();

  std::vector<int64_t> shape(var.dim(), -1);
  shape[dim] = nghost;

  var.narrow(dim, nc1 - nghost, nghost) = 1;
}

namespace {
using namespace snap;

void check_background(torch::Tensor const& w, EquationOfStateImpl* eos) {
  TORCH_CHECK(torch::isfinite(w).all().item<bool>(),
              "outflow: nonfinite initial background (initialize ghosts too)");
  TORCH_CHECK(
      (w[IDN] >= eos->options->density_floor()).all().item<bool>() &&
          (w[IPR] >= eos->options->pressure_floor()).all().item<bool>(),
      "outflow: invalid background density/pressure; initialize ghosts");
  int ny = w.size(0) - ICY;
  if (ny) {
    auto q = w.narrow(0, ICY, ny);
    TORCH_CHECK(
        (q >= 0).all().item<bool>() && (q.sum(0) <= 1).all().item<bool>(),
        "outflow: invalid background composition");
  }
}

void radiating(torch::Tensor const& var, int dim, BoundaryFuncOptions op,
               bool outer) {
  if (var.size(dim) == 1) return;
  // kScalar without transported-tracer context means an auxiliary field,
  // such as the flux positivity factor. It always uses extrapolation.
  if (op.type() == kScalar ||
      (op.eos && op.eos->options->type() == "shallow-water")) {
    if (outer)
      extrapolation_outer(var, dim, op);
    else
      extrapolation_inner(var, dim, op);
    return;
  }
  TORCH_CHECK(op.eos && op.coord && op.reference.defined(),
              "outflow requires EOS, coordinate and saved initial background");
  auto type = op.eos->options->type();
  TORCH_CHECK(
      type == "ideal-gas" || type == "ideal-moist" || type == "moist-mixture",
      "Unsupported outflow EOS: ", type);
  TORCH_CHECK(op.type() == kPrimitive,
              "outflow: prepare primitive context with apply_boundaries");
  TORCH_CHECK(op.reference.sizes() == var.sizes(),
              "outflow: background shape mismatch");
  check_background(op.reference, op.eos);
  TORCH_CHECK(torch::isfinite(var).all().item<bool>(),
              "outflow: nonfinite hydro input");

  int ng = op.nghost(), axis = 4 - dim, vn = IVX + axis - 1;
  int src = outer ? var.size(dim) - ng - 1 : ng;
  int dst = outer ? var.size(dim) - ng : 0;
  double sign = outer ? 1. : -1.;
  auto interior = [&](torch::Tensor t) { return t.narrow(dim, src, 1); };
  auto ghost = [&](torch::Tensor t) { return t.narrow(dim, dst, ng); };
  auto w = var.clone(), b = op.reference.clone();
  // The source plane's cells off the block interior (the other axes' ghost
  // rows) hold stale or forced values that the exchange or the next face
  // overwrites; only the interior cells must be a valid gas state.
  auto in_plane =
      torch::ones_like(var[IDN].narrow(dim - 1, src, 1), torch::kBool);
  for (int d = 1; d <= 3; ++d) {
    if (d == dim || var.size(d) == 1) continue;
    int lo = d == 3 ? op.coord->il() : d == 2 ? op.coord->jl() : op.coord->kl();
    int hi = d == 3 ? op.coord->iu() : d == 2 ? op.coord->ju() : op.coord->ku();
    auto keep = torch::zeros_like(in_plane);
    keep.narrow(d - 1, lo, hi - lo + 1).fill_(true);
    in_plane = in_plane & keep;
  }
  auto sound = [&](torch::Tensor t, bool background) {
    auto a = op.eos->compute("W->A", {t});
    auto c = op.eos->compute("WA->L", {t, a});
    // EOS sound speed has spatial axes only, unlike the 4D hydro tensor.
    c = c.narrow(dim - 1, src, 1);
    auto ok = torch::isfinite(c) & (c > 0);
    TORCH_CHECK((background ? ok : ok | ~in_plane).all().item<bool>(),
                "outflow: invalid EOS sound speed");
    return std::make_pair(c, ok);
  };
  auto [c0, ok0] = sound(b, true);
  auto [c, ok] = sound(w, false);
  // an off-interior cell without a valid state takes the background there
  interior(w).copy_(torch::where(ok.unsqueeze(0), interior(w), interior(b)));
  c = torch::where(ok, c, c0);
  op.coord->boundary_velocity_(w, axis);
  op.coord->boundary_velocity_(b, axis);
  auto wi = interior(w), bi = interior(b);
  auto d = wi - bi;
  auto un = sign * wi[vn];
  auto advect = (un >= 0);
  auto impedance = bi[IDN] * c0;
  auto plus = torch::where(un + c >= 0, sign * d[vn] + d[IPR] / impedance, 0.);
  auto minus = torch::where(un - c >= 0, sign * d[vn] - d[IPR] / impedance, 0.);
  auto entropy = torch::where(advect, d[IDN] - d[IPR] / c0.square(), 0.);
  d.mul_(advect.unsqueeze(0));
  d[vn].copy_(sign * 0.5 * (plus + minus));
  d[IPR].copy_(0.5 * impedance * (plus - minus));
  d[IDN].copy_(entropy + d[IPR] / c0.square());

  TORCH_CHECK(torch::isfinite(d).all().item<bool>(),
              "outflow: nonfinite characteristic perturbation");
  auto bg = ghost(b);
  auto alpha = torch::ones_like(bg[IDN]);
  // A single factor per ghost scales every characteristic, including tracers.
  auto lower = [&](torch::Tensor base, torch::Tensor delta, double floor) {
    auto bound =
        torch::where(delta < 0,
                     (base - floor) / torch::where(delta < 0, -delta,
                                                   torch::ones_like(delta)),
                     torch::ones_like(base));
    alpha = torch::minimum(alpha, torch::where(bound < 1, 0.99 * bound, bound));
  };
  lower(bg[IDN], d[IDN], op.eos->options->density_floor());
  lower(bg[IPR], d[IPR], op.eos->options->pressure_floor());
  int ny = var.size(0) - ICY;
  for (int n = 0; n < ny; ++n) lower(bg[ICY + n], d[ICY + n], 0.);
  if (ny)
    lower(1. - bg.narrow(0, ICY, ny).sum(0), -d.narrow(0, ICY, ny).sum(0), 0.);
  torch::Tensor dr, rb;
  if (op.tracers.defined()) {
    TORCH_CHECK(op.tracer_reference.defined() &&
                    op.tracer_reference.sizes() == op.tracers.sizes(),
                "outflow: missing or mismatched tracer background");
    TORCH_CHECK(torch::isfinite(op.tracers).all().item<bool>() &&
                    torch::isfinite(op.tracer_reference).all().item<bool>(),
                "outflow: nonfinite tracer input/background");
    TORCH_CHECK((op.tracer_reference >= 0).all().item<bool>(),
                "outflow: invalid tracer background");
    dr = (interior(op.tracers) - interior(op.tracer_reference)) * advect;
    rb = ghost(op.tracer_reference);
    for (int n = 0; n < dr.size(0); ++n) lower(rb[n], dr[n], 0.);
  }
  alpha.clamp_(0., 1.);
  ghost(w).copy_(bg + alpha.unsqueeze(0) * d);
  op.coord->boundary_velocity_(w, axis, true);
  TORCH_CHECK(torch::isfinite(ghost(w)).all().item<bool>(),
              "outflow: nonfinite reconstructed state");
  ghost(var).copy_(ghost(w));
  if (dr.defined()) ghost(op.tracers).copy_(rb + alpha * dr);
}
}  // namespace

BC_FUNCTION(outflow_inner, var, dim, op) { radiating(var, dim, op, false); }
BC_FUNCTION(outflow_outer, var, dim, op) { radiating(var, dim, op, true); }

bool is_outflow(bcfunc_t const& func) {
  using Pointer =
      void (*)(torch::Tensor const&, int, snap::BoundaryFuncOptions);
  auto ptr = func.target<Pointer>();
  return ptr && (*ptr == outflow_inner || *ptr == outflow_outer);
}

bool is_reflecting(bcfunc_t const& func) {
  using Pointer =
      void (*)(torch::Tensor const&, int, snap::BoundaryFuncOptions);
  auto ptr = func.target<Pointer>();
  return ptr && (*ptr == reflecting_inner || *ptr == reflecting_outer);
}

bool is_custom_noop(bcfunc_t const& func) {
  using Pointer =
      void (*)(torch::Tensor const&, int, snap::BoundaryFuncOptions);
  auto ptr = func.target<Pointer>();
  return ptr && (*ptr == custom_inner || *ptr == custom_outer);
}
