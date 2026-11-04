// snap
#include "scalar.hpp"

#include <snap/hydro/flux_positivity.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/layout/layout.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/recon/interpolation.hpp>

#include "scalar_fused.hpp"

namespace snap {
ScalarImpl::ScalarImpl(const ScalarOptions& options_, torch::nn::Module* p)
    : options(options_) {
  pmb = dynamic_cast<MeshBlockImpl const*>(p);
  reset();
}

void ScalarImpl::reset() {
  if (nvar() <= 0) return;

  TORCH_CHECK(pmb != nullptr, "[Scalar] Parent MeshBlock is null");
  TORCH_CHECK(options->riemann()->type() == "upwind",
              "[Scalar] Passive scalars currently require the upwind solver");

  // Zero is rejected rather than read as "off": it would pin every tracer to
  // zero and freeze transport, and it is far likelier to have meant "off".
  TORCH_CHECK(options->upper_bound() != 0.,
              "[Scalar] upper-bound = 0 pins the tracer to zero and freezes "
              "transport; use a negative value to disable the bound");
  TORCH_CHECK(
      options->upper_bound() <= 0. || pmb->phydro->options->eos()->limiter(),
      "[Scalar] upper-bound needs the eos limiter: it bounds r from "
      "above by the positivity of the complement, and enforcing one "
      "side alone leaves the other unbounded");

  pcoord = pmb->pcoord;
  precon = ReconstructImpl::create(options->recon(), this);
  priemann = RiemannSolverImpl::create(options->riemann(), this);

  if (options->thermo()) {
    pthermo = kintera::ThermoXImpl::create(options->thermo(), this);
  }
  if (options->kinetics()) {
    pkinetics = kintera::KineticsImpl::create(options->kinetics(), this);
  }

  auto nc1 = pcoord->options->nc1();
  auto nc2 = pcoord->options->nc2();
  auto nc3 = pcoord->options->nc3();

  _flux1 = register_buffer(
      "F1", nc1 > 1 ? torch::zeros({nvar(), nc3, nc2, nc1}, torch::kFloat64)
                    : torch::Tensor());
  _flux2 = register_buffer(
      "F2", nc2 > 1 ? torch::zeros({nvar(), nc3, nc2, nc1}, torch::kFloat64)
                    : torch::Tensor());
  _flux3 = register_buffer(
      "F3", nc3 > 1 ? torch::zeros({nvar(), nc3, nc2, nc1}, torch::kFloat64)
                    : torch::Tensor());
  _div = register_buffer(
      "D", torch::zeros({nvar(), nc3, nc2, nc1}, torch::kFloat64));
  _positivity_hits =
      register_buffer("positivity_hits", torch::zeros({1}, torch::kInt64));
}

torch::Tensor ScalarImpl::forward(double dt, torch::Tensor u,
                                  Variables const& other) {
  enum { DIM1 = 3, DIM2 = 2, DIM3 = 1 };
  TORCH_CHECK(pmb != nullptr, "[Scalar] Parent MeshBlock is null");
  TORCH_CHECK(other.count("hydro_u"),
              "[Scalar] hydro_u is required for passive scalar transport");

  auto rho = other.at("hydro_u")[IDN].unsqueeze(0);
  auto r = u / rho;

  // fused path (SNAP_SCALAR_FUSED=1, CUDA, float64): the same arithmetic in a
  // few kernels (scalar_fused.cu); the reconstruction is fused only for
  // WENO5, whose weights take fewer divisions (round-off differences)
  bool fused = scalar_fused_enabled() && u.is_cuda() &&
               u.scalar_type() == torch::kFloat64;
  auto weno5 =
      fused ? std::dynamic_pointer_cast<Weno5InterpImpl>(precon->pinterp1)
            : nullptr;
  auto recon = [&](int dim) {
    if (!weno5) return precon->forward(r, dim);
    auto rc = r.contiguous();
    auto vec = rc.sizes().vec();
    vec.insert(vec.begin(), 2);
    auto faces = torch::empty(vec, rc.options());
    at::native::call_scalar_weno5(faces.device().type(), faces, rc, weno5->cm,
                                  weno5->cp, weno5->options->scale(), dim,
                                  torch::Tensor());
    return faces;
  };

  if (_flux1.defined()) {
    if (weno5) {
      auto rc = r.contiguous();
      auto flux = torch::empty_like(rc);
      at::native::call_scalar_weno5(flux.device().type(), flux, rc, weno5->cm,
                                    weno5->cp, weno5->options->scale(), DIM1,
                                    pmb->phydro->flux1()[IDN].contiguous());
      _flux1.set_(flux);
    } else {
      auto rlr = precon->forward(r, DIM1);
      _flux1.set_(priemann->forward(rlr[ILT], rlr[IRT], DIM1,
                                    pmb->phydro->flux1()[IDN]));
    }
  }

  auto playout = pmb->get_layout();
  torch::Tensor rtmp2, rtmp3;
  Variables send_vars2, send_vars3;
  SyncOptions sync_opts;
  sync_opts.cross_panel_only(true).interpolate(false).type(kScalar);
  std::vector<CommWorkPtr> works2, works3;

  if (_flux2.defined()) {
    rtmp2 = recon(DIM2);
    if (playout->options->type() == "cubed-sphere") {
      // Ship the two reconstructed states under DIRECTION-SUFFIXED keys, as the
      // hydro does (hydro_forward.cpp). The ':' suffix selects the directional
      // partial send/recv in CubedSphereLayoutImpl::serialize/deserialize,
      // whose mirrored skip rules resolve the L/R roles across a panel edge;
      // one un-suffixed key took no directional branch and swapped them at a
      // flip_flag seam, making the upwind solver pick the downwind state.
      send_vars2["scalar_wl:+"] = rtmp2[ILT];
      send_vars2["scalar_wr:-"] = rtmp2[IRT];
      pmb->begin_exchange(send_vars2, sync_opts.dim(DIM2));
    }
  }

  if (_flux3.defined()) {
    rtmp3 = recon(DIM3);
    if (playout->options->type() == "cubed-sphere") {
      // See the DIM2 comment above.
      send_vars3["scalar_wl:+"] = rtmp3[ILT];
      send_vars3["scalar_wr:-"] = rtmp3[IRT];
      pmb->begin_exchange(send_vars3, sync_opts.dim(DIM3));
    }
  }

  if (playout->options->type() == "cubed-sphere") {
    bool exchange_dim2 = _flux2.defined();
    bool exchange_dim3 = _flux3.defined();
    if (exchange_dim2) {
      pmb->launch_exchange(sync_opts.dim(DIM2), works2);
    }
    if (exchange_dim3) {
      pmb->launch_exchange(sync_opts.dim(DIM3), works3);
    }
    if (exchange_dim2)
      pmb->finalize_exchange(send_vars2, sync_opts.dim(DIM2), works2);
    if (exchange_dim3)
      pmb->finalize_exchange(send_vars3, sync_opts.dim(DIM3), works3);
  }

  // SNAP_FLUX_COVARIANCE (#289): a scalar carries the tracer row of the face
  // covariance (HydroImpl::scalar_flux_covariance), not r times the dry-mass
  // row that the hydro's mass flux carries, so it rides on the mass flux
  // without that row and adds its own: the flux of a hydro species with the
  // same mass fraction
  bool cov = false;
  // the fused kernel takes the covariance factors of a column that is not
  // split in x1 (HydroImpl::scalar_flux_covariance_factors); with a split
  // the covariance takes the default path, which reads the seam rows
  auto [x1_below, x1_above] = pmb->phydro->x1_neighbors();
  bool x1_whole = x1_below < 0 && x1_above < 0;
  auto lateral = [&](torch::Tensor const& rlr, int dim,
                     torch::Tensor const& mass) {
    if (fused &&
        (x1_whole || !pmb->phydro->flux_covariance_dry(dim).defined())) {
      // the same flux from one kernel (scalar_fused.cu)
      auto factors = pmb->phydro->scalar_flux_covariance_factors(dim);
      cov = cov || !factors.empty();
      auto vel =
          factors.empty()
              ? mass.contiguous()
              : (mass - pmb->phydro->flux_covariance_dry(dim)).contiguous();
      auto rl = rlr[ILT].contiguous();
      auto rr = rlr[IRT].contiguous();
      auto flux = torch::empty_like(rl);
      at::native::call_scalar_lateral_flux(flux.device().type(), flux, rl, rr,
                                           vel, factors);
      return flux;
    }
    auto dcov = pmb->phydro->scalar_flux_covariance(rlr[ILT], rlr[IRT], dim);
    if (!dcov.defined())
      return priemann->forward(rlr[ILT], rlr[IRT], dim, mass);
    cov = true;
    auto vel = mass - pmb->phydro->flux_covariance_dry(dim);
    return priemann->forward(rlr[ILT], rlr[IRT], dim, vel) + dcov;
  };

  if (_flux2.defined()) {
    _flux2.set_(lateral(rtmp2, DIM2, pmb->phydro->flux2()[IDN]));
  }

  if (_flux3.defined()) {
    _flux3.set_(lateral(rtmp3, DIM3, pmb->phydro->flux3()[IDN]));
  }

  // Tracer flux positivity limiter: same scheme (and same rationale) as the
  // hydro species channels -- see flux_positivity.hpp and hydro_forward.cpp
  // step (4.C). It follows the existing EOS limiter setting, and runs whenever
  // the scalar carries its covariance row, which can export tracer a cell
  // does not have.
  auto sync_theta = [&](torch::Tensor theta) {
    Variables tvars;
    tvars["scalar_theta"] = theta;
    SyncOptions theta_opts;
    theta_opts.interpolate(false).type(kScalar);  // see hydro_forward.cpp 4.C
    pmb->exchange(tvars, theta_opts);

    BoundaryFuncOptions bops;
    bops.nghost(pcoord->options->nghost());
    bops.type(kScalar);
    for (int i = 0; i < pmb->options->bfuncs().size(); ++i) {
      if (pmb->options->bfuncs()[i] == nullptr) continue;
      pmb->options->bfuncs()[i](theta, 3 - i / 2, bops);
    }
    return theta;
  };

  auto cells = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  // fused path: the face areas and volumes the libtorch path uses, laid out
  // whole for the kernels
  at::native::ScalarBox box{pcoord->il(), pcoord->iu(), pcoord->jl(),
                            pcoord->ju(), pcoord->kl(), pcoord->ku()};
  torch::Tensor fa1, fa2, fa3, fvol;
  if (fused) {
    auto n1 = u.size(3), n2 = u.size(2), n3 = u.size(1);
    if (_flux1.defined())
      fa1 = pcoord->face_area1().expand({n3, n2, n1 + 1}).contiguous();
    if (_flux2.defined())
      fa2 = pcoord->face_area2().expand({n3, n2 + 1, n1}).contiguous();
    if (_flux3.defined())
      fa3 = pcoord->face_area3().expand({n3 + 1, n2, n1}).contiguous();
    fvol = pcoord->cell_volume().expand({n3, n2, n1}).contiguous();
  }
  if (pmb->phydro->options->eos()->limiter() || cov) {
    if (fused) {
      auto uc = u.contiguous();
      auto theta = torch::empty_like(uc);
      double margin = kPositivityRoundoffUlp * machine_epsilon(u.scalar_type());
      at::native::call_scalar_theta(theta.device().type(), theta, uc, _flux1,
                                    _flux2, _flux3, fa1, fa2, fa3, fvol, dt,
                                    1. - margin, box);
      _positivity_hits += (theta.index(cells) < 1.).sum();
      at::native::call_scalar_theta_scale(theta.device().type(),
                                          sync_theta(theta), _flux1, _flux2,
                                          _flux3, box);
    } else {
      auto theta = flux_positivity_theta(u, _flux1, _flux2, _flux3, pcoord, dt);
      _positivity_hits += (theta.index(cells) < 1.).sum();
      flux_positivity_scale_(sync_theta(theta), _flux1, _flux2, _flux3, pcoord);
    }
  }

  // r <= b is positivity of the complement b*rho - s, whose flux is b*F_mass -
  // F_s, so the same limiter applies to it -- a SECOND theta, not the one
  // above. Scaling the complement blends F_s toward b*F_mass, the donor-cell
  // flux of a tracer at the bound; scaling F_s toward zero would bound it too
  // but stop a plateau advecting.
  //
  // ⚠ This weakens the lower bound above from unconditional to conditional on
  // the mass Courant number staying below 1: past that, theta can leave a cell
  // exporting tracer it does not have. The premise also assumes rho moves by
  // -dt*div(F_mass) alone, which the implicit vertical correction and any
  // forcing writing du[IDN] both break.
  if (options->upper_bound() > 0.) {
    auto b = options->upper_bound();
    auto mf = [&](int dim) {
      return (dim == DIM1   ? pmb->phydro->flux1()
              : dim == DIM2 ? pmb->phydro->flux2()
                            : pmb->phydro->flux3())[IDN]
          .unsqueeze(0);
    };
    torch::Tensor g1, g2, g3, h1, h2, h3;
    if (_flux1.defined()) h1 = (g1 = b * mf(DIM1) - _flux1).clone();
    if (_flux2.defined()) h2 = (g2 = b * mf(DIM2) - _flux2).clone();
    if (_flux3.defined()) h3 = (g3 = b * mf(DIM3) - _flux3).clone();

    auto theta = flux_positivity_theta(b * rho - u, g1, g2, g3, pcoord, dt);
    _positivity_hits += (theta.index(cells) < 1.).sum();
    flux_positivity_scale_(sync_theta(theta), g1, g2, g3, pcoord);

    // Add the CHANGE, never recompute F_s = b*F_mass - g. The two are equal in
    // algebra only: recomputing carries an error absolute in b*F_mass, so it
    // perturbs EVERY face even where theta is 1, and in float32 it annihilates
    // the flux of a species whose ratio is near the epsilon of b. This form is
    // bitwise identity where nothing was limited.
    if (_flux1.defined()) _flux1.add_(h1 - g1);
    if (_flux2.defined()) _flux2.add_(h2 - g2);
    if (_flux3.defined()) _flux3.add_(h3 - g3);
  }

  if (fused) {
    auto div = torch::empty_like(u, at::MemoryFormat::Contiguous);
    auto ds = torch::empty_like(div);
    at::native::call_scalar_divergence(div.device().type(), div, ds, _flux1,
                                       _flux2, _flux3, fa1, fa2, fa3, fvol, dt,
                                       box);
    _div.set_(div);
    return ds;
  }
  _div.set_(pcoord->divergence(_flux1, _flux2, _flux3));
  auto ds = torch::zeros_like(_div);
  auto interior = pmb->part({0, 0, 0}, PartOptions().exterior(false));
  ds.index(interior) = -dt * _div.index(interior);
  return ds;
}

std::shared_ptr<ScalarImpl> ScalarImpl::create(ScalarOptions const& opts,
                                               torch::nn::Module* p,
                                               std::string const& name) {
  TORCH_CHECK(p, "[Scalar] Parent module is null");
  TORCH_CHECK(opts, "[Scalar] Options pointer is null");

  return p->register_module(name, Scalar(opts, p));
}

}  // namespace snap
