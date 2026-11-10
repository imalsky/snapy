// C/C++
#include <limits>

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>

#include "balance_column.hpp"
#include "hydro_dispatch.hpp"

namespace snap {

std::tuple<torch::Tensor, double, int> balance_column(
    torch::Tensor const& w, torch::Tensor const& dx1f, double grav,
    bool wall_clamp, double rtol, int max_iter, std::string const& geometry,
    bool stretched) {
  TORCH_CHECK(grav > 0., "balance_column: grav is a downward magnitude (> 0)");
  TORCH_CHECK(w.dim() == 4, "balance_column: w must be (nvar, nc3, nc2, nx1)");
  TORCH_CHECK(w.size(0) > IPR, "balance_column: w has no pressure channel");
  TORCH_CHECK(dx1f.dim() == 1 && dx1f.size(0) == w.size(-1),
              "balance_column: dx1f must be (nx1,) and match w's last dim");
  TORCH_CHECK(
      dx1f.scalar_type() == w.scalar_type() && dx1f.device() == w.device(),
      "balance_column: dx1f and w must share dtype and device");
  TORCH_CHECK(rtol > 0. && max_iter > 0,
              "balance_column: rtol and max_iter must be positive");
  // without the clamp the reference's wall rows quadrature faces below the
  // wall, which a ghost-free column does not have
  TORCH_CHECK(wall_clamp,
              "balance_column needs dynamics/wb-wall-clamp: without it the "
              "reference reads faces outside the column at each wall, so the "
              "balance found here is not the one the solver would enforce.");
  // no r^2 -> plain-mean conversion here (balance_column.hpp)
  TORCH_CHECK(!x1_centroid_exact_enabled() || geometry == "cartesian",
              "balance_column: under SNAP_X1_CENTROID_EXACT only a column "
              "declared geometry='cartesian' is balanced here, got '",
              geometry,
              "'; a spherical-polar column's rest state is the solver's own "
              "reference");

  int nc1 = w.size(-1);
  // below five the kernel's own thin-block fallback takes over and the answer,
  // though self-consistent, is not the one any real block would enforce; at
  // nc1 == 1 the gauge cell is the only cell and this would return "already
  // balanced" having done nothing
  TORCH_CHECK(nc1 >= 5, "balance_column: a column needs at least 5 cells, got ",
              nc1);
  auto wb = w.contiguous().clone();
  auto dxf = dx1f.contiguous();
  auto rho = wb[IDN];
  auto prs = wb[IPR];
  TORCH_CHECK(prs.min().item<double>() > 0. && rho.min().item<double>() > 0.,
              "balance_column: p and rho must be positive everywhere");
  auto rt = (prs / rho).clone();  // p/rho per cell, the invariant, taken ONCE

  auto sizes = w.sizes().slice(1).vec();
  auto psf_lo = torch::empty(sizes, w.options());
  auto psf_hi = torch::empty(sizes, w.options());
  auto pref = torch::empty(sizes, w.options());
  auto dsf = torch::empty(sizes, w.options());
  auto dref = torch::empty(sizes, w.options());
  torch::Tensor anchor;  // undefined: the kernel builds its own top anchor

  // unequal widths take the stretched rows (#280), whatever the key
  auto d = dxf.to(torch::kCPU).to(torch::kFloat64);
  stretched = stretched || (d.max() - d.min()).item<double>() >=
                               1.e-10 * d.mean().item<double>();
  int x1_grid = stretched ? kX1RefStretched : kX1RefUniform;
  // #280 (iv): the same rows a block builds from the same widths
  torch::Tensor rows;
  if (stretched)
    rows = hydro_ref_x1_stretched_rows(dxf, nc1 - 1, true, true, wall_clamp)
               .to(w.options());
  auto wgt = grav * dxf;

  // check, then update: the residual therefore describes the state returned,
  // and a column already at the fixed point comes back after zero updates
  double err = std::numeric_limits<double>::infinity();
  int sweeps = 0;
  for (; sweeps <= max_iter; ++sweeps) {
    at::native::call_hydro_ref_x1(wb.device().type(), wb, dxf, anchor, psf_lo,
                                  psf_hi, pref, dsf, dref, nc1 - 1, grav,
                                  x1_grid, /*phys_in=*/true, /*phys_out=*/true,
                                  wall_clamp, rows);
    auto pp = prs - pref;
    auto c = pp.narrow(-1, nc1 - 1, 1);  // one gauge per column, at its top
    err = ((pp - c).abs() / (rho * wgt)).max().item<double>();
    if (err < rtol || sweeps == max_iter) break;
    prs.copy_(pref + c);
    rho.copy_(prs / rt);
    // a diverging column walks its top cells to vacuum, and once rho goes
    // negative the residual above goes negative with it and reads as converged
    TORCH_CHECK(
        prs.min().item<double>() > 0.,
        "balance_column: the projection drove the pressure non-positive "
        "on sweep ",
        sweeps, " -- the column is diverging, not converging.");
  }

  TORCH_CHECK(err < rtol, "balance_column: no discrete balance after ", sweeps,
              " sweeps; residual bound on |a|/g is ", err, " > ", rtol);
  return {wb, err, sweeps};
}

}  // namespace snap
