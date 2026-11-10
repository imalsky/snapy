#pragma once

// torch
#include <torch/torch.h>

namespace snap {

//! #289 (SNAP_WB_REF4): fourth-order x1 reference DENSITY for the
//! well-balanced reconstruction, consistent between cell and face. The face
//! pressures (the hydrostatic scan) are never changed. Applied after the
//! kernel (hydro_ref_x1_impl.h); a guarded cell or face keeps the kernel's
//! value. See docs/derivations/wb-ref4.md.
//! Read once per process from SNAP_WB_REF4; off unless it or
//! SNAP_X1_CENTROID_EXACT (which implies it) is set.
bool wb_ref4_enabled();

//! the per-grid stencils, built once on the CPU in double and moved to the
//! reference's device and dtype
struct WbRef4Stencils {
  int is = 0, iu = -1, nc1 = 0;
  //! false: a clamped wall with fewer than four owned cells; nothing changes
  bool usable = false;
  //! a physical x1 wall whose stencils stay inside the owned cells
  bool clamp_in = false, clamp_out = false;
  //! filter F = (-1, 4, 10, 4, -1)/16 of rho/p on owned cells, with cubic
  //! extrapolation past a clamped wall: (nown, K) cell indices and weights
  torch::Tensor fidx, fwt;
  //! range-guard neighbours i-1, i, i+1 of each owned cell: (nown, 3)
  torch::Tensor nidx;
  //! face density: derivative at the face of the quartic primitive through
  //! five faces (four cells), (nown + 1, 4) cell indices and weights
  torch::Tensor aidx, awt;
  //! true for the cells that count for the resolution flag
  torch::Tensor counted;
  //! rebuild for a reference of another device or dtype; an unusable cache
  //! has no weights to move and is never stale (#298)
  bool stale(torch::Tensor const& ref) const {
    return usable && (fwt.device() != ref.device() ||
                      fwt.scalar_type() != ref.scalar_type());
  }
};

//! x1f: the nc1 + 1 face positions; is/iu: first/last owned cell
WbRef4Stencils wb_ref4_stencils(torch::Tensor const& x1f, int is, int iu,
                                bool clamp_in, bool clamp_out,
                                torch::TensorOptions const& options);

//! cell part, before any x1 seam exchange of (pref, dref): dref of the owned
//! cells, in place. Returns the resolution
//! flag (nc3, nc2, nc1): cells (dilated by two) where the column has fewer
//! than two cells per pressure scale height and the kernel's values stay.
torch::Tensor wb_ref4_cells(WbRef4Stencils const& st, torch::Tensor const& w,
                            torch::Tensor const& psf_lo,
                            torch::Tensor const& psf_hi, torch::Tensor pref,
                            torch::Tensor dref);

//! face part, after the x1 seam exchange: dsf on the owned faces is..iu+1 is
//! the fourth-order face value of dref itself. Needs is >= 1 (a ghost cell).
void wb_ref4_faces(WbRef4Stencils const& st, torch::Tensor const& dref,
                   torch::Tensor dsf, torch::Tensor const& flag);

}  // namespace snap
