#pragma once

// C/C++
#include <string>
#include <tuple>

// torch
#include <torch/torch.h>

namespace snap {

//! \brief Project a ghost-free x1 column onto the well-balanced scheme's own
//!        discrete hydrostatic balance, at fixed temperature.
//!
//! At rest the well-balanced x1 reconstruction reconstructs p' = p - pref(p),
//! where pref is the reference it rebuilds from the state on every call. A
//! column marched through the hydrostatic ODE satisfies the CONTINUUM balance
//! to O(dz^2), not this discrete one, so p' is not constant and its variation
//! is a standing force on an atmosphere that is supposed to be at rest. This
//! iterates
//!
//!     p_i <- pref_i(p) + C,     rho_i <- p_i / (p_i/rho_i)_0
//!
//! to the fixed point p' == C. C is a free gauge (any constant p' is balanced)
//! and is read per column at the TOP cell, the lowest-pressure one, so the
//! column moves least where it is thinnest; anchoring it at the bottom
//! diverges on a deep column. p/rho is held fixed per cell, which for
//! `ideal-gas` and `ideal-moist` at fixed mass fractions holds the TEMPERATURE
//! fixed exactly -- only p and rho move, together.
//!
//! \param w        (nvar, nc3, nc2, nx1) primitive state, nvar > IPR. The last
//!                 dimension is the WHOLE column with NO ghost cells: every
//!                 value the reference reads is then one this caller owns.
//!                 Channels other than IDN and IPR are carried through
//!                 untouched, and `w` itself is not modified.
//! \param dx1f     (nx1,) cell widths, same dtype and device as `w`. Whether
//!                 the grid counts as uniform is decided here by the SAME test
//!                 the solver applies to its own grid, and it selects between
//!                 two different reference operators -- so a caller whose
//!                 ghost-free dx1f would classify differently from the block's
//!                 padded one would be balancing against the wrong operator.
//! \param grav     downward gravity MAGNITUDE (> 0), as the kernel takes it.
//! \param wall_clamp  must be true, and must match `dynamics/wb-wall-clamp` in
//!                 the config that will run the result. With the clamp off the
//!                 reference's two cells at each wall are quadratured from
//!                 faces BELOW the wall -- ghost faces, which a ghost-free
//!                 column does not have -- so the fixed point found here would
//!                 not be the one the solver later enforces.
//! \param rtol     bound on the residual below, dimensionless.
//! \param max_iter sweeps before giving up.
//!
//! \return (balanced w, residual, sweeps). The residual is
//!         max |p' - C| / (rho g dz), which bounds what the leftover imbalance
//!         can still do: it is the acceleration the column can still feel, in
//!         units of g. A dimensionless bound is the only portable one -- 1e-9
//!         Pa is below double precision on a column whose bottom cell is
//!         1.2e7 Pa. The residual is measured BEFORE the update that is
//!         skipped on convergence, so it describes the state that is returned.
//!         `sweeps` counts updates applied, so a column already at the fixed
//!         point comes back with 0.
//!
//! \throws if the sweeps do not reach `rtol`: a column that did not converge is
//!         not balanced, and starting a run from it is the bug this exists to
//!         remove.
//!
//! Both x1 ends are treated as PHYSICAL boundaries, which is what a column
//! bounded by walls is. The caller owns four conditions this cannot check:
//!
//!  * the x1 boundaries really are physical, not periodic;
//!  * every block the column is later split into has at least five x1 cells --
//!    below that the reference falls back to a two-point mean at the wall, and
//!    the balance found here is not the one that block will enforce;
//!  * the `uniform` classification below, which is made ONCE for the whole
//!    column, is the same one every block will make for itself. The solver
//!    classifies per block, so a grid that is uniform over part of the x1 range
//!    and stretched over the rest can give a block inside the uniform part the
//!    six-face rows while this took the log-mean branch for the column -- and
//!    then the fixed point found here belongs to an operator no block applies.
//!    The two branches are different operators, not different accuracies.
//!  * a moist caller re-applies saturation after the projection and iterates
//!    the two while saturation still changes the state (examples/bryan.cpp).
//!
//! And one consequence rather than an obligation: the fixed point is p' == C
//! for a FREE constant C, so nothing ties the result to the input. The column
//! moves
//! -- 2.5e-4 of p on a Neptune-like column, 1.2e-2 on a 300 km giant-planet
//! one -- and no surface pressure, column mass or anchor level survives that.
//! An IC builder that pinned something must re-pin it against the BALANCED
//! column, which means running the projection inside its own fixed-point loop.
//!
//! What transfers from this ghost-free column to a real block is the CELL
//! PRESSURE reference `pref`, and it transfers bitwise -- that is the property
//! the whole arrangement rests on, and `tests/test_balance_column.cpp` gates
//! it. The face density `dsf` is the one reference output that does NOT, at
//! the bottom-most cell only, because its one-sided fallback keys on the
//! absolute index i > 0 rather than on i > il. `dsf` plays no part in the
//! fixed point, so the balance is unaffected; nothing else here depends on it.
//!
//! `stretched` is the grid key's classification (#280): pass
//! `coord->options->stretched_x1()` of the grid the column comes from. A
//! stretched x1 is non-uniform in every block whatever its widths, and its
//! cell pressure is the six-face degree-5 Lagrange cell average with per-cell
//! rows (hydro_ref_x1_stretched_rows); a column of unequal widths takes them
//! whatever the flag.
//!
//! Under SNAP_X1_CENTROID_EXACT a spherical-polar block's reference works on
//! plain means converted from its r^2 means, which this column does not model:
//! the caller then passes geometry "cartesian", and anything else (unknown, or
//! spherical-polar, whose rest state is the solver's own) is refused.
std::tuple<torch::Tensor, double, int> balance_column(
    torch::Tensor const& w, torch::Tensor const& dx1f, double grav,
    bool wall_clamp = true, double rtol = 1.e-10, int max_iter = 120,
    std::string const& geometry = "", bool stretched = false);

}  // namespace snap
