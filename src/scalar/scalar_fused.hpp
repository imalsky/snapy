#pragma once

// torch
#include <ATen/native/DispatchStub.h>
#include <torch/torch.h>

// Fused CUDA kernels for passive-scalar transport, used when
// SNAP_SCALAR_FUSED=1 and the scalar lives on a CUDA device. Each kernel
// computes, per element, the same operations in the same order as the libtorch
// expressions it replaces, with every product, sum and quotient rounded
// separately (no contraction), so it reproduces the default path bit for bit;
// the one exception is the WENO5 reconstruction (call_scalar_weno5), whose
// weights take fewer divisions and so differ at round-off (and which stays
// finite on values above about 1e290, where the default formula overflows).

namespace snap {
//! true when SNAP_SCALAR_FUSED=1 (read once)
bool scalar_fused_enabled();
}  // namespace snap

namespace at::native {

//! x2/x3 face flux of every scalar: the upwind flux of the face states rl, rr
//! with velocity vel, plus, when `factors` is not empty, the per-tracer part
//! of HydroImpl::scalar_flux_covariance built from its tracer-independent
//! factors (HydroImpl::scalar_flux_covariance_factors). flux, rl, rr:
//! (nvar, nc3, nc2, nc1); vel: (nc3, nc2, nc1).
using scalar_lateral_flux_fn = void (*)(
    torch::Tensor& flux, torch::Tensor const& rl, torch::Tensor const& rr,
    torch::Tensor const& vel, std::vector<torch::Tensor> const& factors);

DECLARE_DISPATCH(scalar_lateral_flux_fn, call_scalar_lateral_flux);

//! Box of the interior cells: il..iu (x1, last dim), jl..ju (x2), kl..ku (x3).
struct ScalarBox {
  int il, iu, jl, ju, kl, ku;
};

//! flux_positivity_theta on every cell (the outflow of the faces the
//! divergence uses, summed x1, x2, x3 as there): theta, u: (nvar, nc3, nc2,
//! nc1); f1, f2, f3 may be undefined; a1 (nc3, nc2, nc1+1), a2 (nc3, nc2+1,
//! nc1), a3 (nc3+1, nc2, nc1) the face areas, vol (nc3, nc2, nc1);
//! keep = 1 - margin.
using scalar_theta_fn = void (*)(
    torch::Tensor& theta, torch::Tensor const& u, torch::Tensor const& f1,
    torch::Tensor const& f2, torch::Tensor const& f3, torch::Tensor const& a1,
    torch::Tensor const& a2, torch::Tensor const& a3, torch::Tensor const& vol,
    double dt, double keep, ScalarBox box);

//! flux_positivity_scale_: each face flux in il..iu+1 (and likewise in x2, x3)
//! times the donor cell's theta.
using scalar_theta_scale_fn = void (*)(torch::Tensor const& theta,
                                       torch::Tensor& f1, torch::Tensor& f2,
                                       torch::Tensor& f3, ScalarBox box);

//! CoordinateImpl::divergence (into div, every cell) and ds = -dt * div on the
//! interior, 0 in the ghost cells.
using scalar_divergence_fn = void (*)(
    torch::Tensor& div, torch::Tensor& ds, torch::Tensor const& f1,
    torch::Tensor const& f2, torch::Tensor const& f3, torch::Tensor const& a1,
    torch::Tensor const& a2, torch::Tensor const& a3, torch::Tensor const& vol,
    double dt, ScalarBox box);

//! ReconstructImpl::forward of a scalar with WENO5 along tensor dimension
//! `dim` (3: x1, 2: x2, 1: x3): every face's left (ILT) and right (IRT) state
//! with the same edge copies and the clamp at 0; the WENO5 weights take fewer
//! divisions than the default path's (round-off differences; the x1 upwind
//! flux of columns longer than 1024 cells keeps the default arithmetic).
//! faces: (2, nvar, nc3, nc2, nc1). When `vel` is defined (x1 only) the
//! kernel writes instead the upwind flux vel * (ui * wl + (1 - ui) * wr) into
//! `faces` shaped (nvar, nc3, nc2, nc1).
using scalar_weno5_fn = void (*)(torch::Tensor& faces, torch::Tensor const& w,
                                 torch::Tensor const& cm,
                                 torch::Tensor const& cp, bool scale, int dim,
                                 torch::Tensor const& vel);
DECLARE_DISPATCH(scalar_weno5_fn, call_scalar_weno5);

//! MeshBlock's implicit x1 tracer transfer, added into ds:
//! P = (M > 0 ? r_below : r) * M, each donor capped by at::native::donor_cap
//! (faces il+1..iu) against held = max(s + ds, 0) * vol, then
//! ds += (P - P_above) / vol, r_below and P_above zero past the column ends.
//! ds, s, r: (nvar, nc3, nc2, nc1); m, vol: (nc3, nc2, nc1).
using scalar_implicit_transfer_fn = void (*)(
    torch::Tensor& ds, torch::Tensor const& s, torch::Tensor const& r,
    torch::Tensor const& m, torch::Tensor const& vol, int il, int iu);
DECLARE_DISPATCH(scalar_implicit_transfer_fn, call_scalar_implicit_transfer);

//! The integrator's stage average w0 u0 + w1 u1 + w2 u2 (summed left to
//! right, as there) into out.
using scalar_average3_fn = void (*)(torch::Tensor& out, torch::Tensor const& u0,
                                    torch::Tensor const& u1,
                                    torch::Tensor const& u2, double w0,
                                    double w1, double w2);
DECLARE_DISPATCH(scalar_average3_fn, call_scalar_average3);

//! MeshBlockImpl's carry of a dry-density source into the scalars, added into
//! ds: s_base = w0 s0 + w1 s + w2 base (the stage average), remove = (rho_base
//! == 0 ? r : s_base / rho_base), carry = (dry < 0 ? remove : r), ds +=
//! carry * dry. ds, s0, s, base, r: (nvar, nc3, nc2, nc1); rho_base, dry:
//! (nc3, nc2, nc1). base may be ds itself (read before the add).
using scalar_carry_dry_fn = void (*)(torch::Tensor& ds, torch::Tensor const& s0,
                                     torch::Tensor const& s,
                                     torch::Tensor const& base,
                                     torch::Tensor const& r,
                                     torch::Tensor const& rho_base,
                                     torch::Tensor const& dry, double w0,
                                     double w1, double w2);
DECLARE_DISPATCH(scalar_carry_dry_fn, call_scalar_carry_dry);

DECLARE_DISPATCH(scalar_theta_fn, call_scalar_theta);
DECLARE_DISPATCH(scalar_theta_scale_fn, call_scalar_theta_scale);
DECLARE_DISPATCH(scalar_divergence_fn, call_scalar_divergence);

}  // namespace at::native
