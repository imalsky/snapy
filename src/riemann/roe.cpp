// snap
#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "../eos/ideal_moist.hpp"
#include "riemann_dispatch.hpp"
#include "riemann_solver.hpp"

namespace snap {

void RoeSolverImpl::reset() {
  TORCH_CHECK(phydro, "[RoeSolver] parent is nullptr");
}

torch::Tensor RoeSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                     int dim, torch::Tensor flx,
                                     torch::Tensor face_pressure) {
  auto peos = phydro->peos;
  auto ideal_moist = peos->options->type() == "ideal-moist";
  int nvapor = 0;
  torch::Tensor inv_mu_ratio_m1, cv_ratio_m1, u0;
  if (ideal_moist) {
    auto moist = dynamic_cast<IdealMoistImpl*>(peos.get());
    TORCH_CHECK(moist != nullptr, "[RoeSolver] ideal-moist EOS cast failed");
    nvapor = moist->pthermo->options->vapor_ids().size() - 1;
    inv_mu_ratio_m1 = moist->inv_mu_ratio_m1.to(wl);
    cv_ratio_m1 = moist->cv_ratio_m1.to(wl);
    u0 = moist->u0.to(wl);
  }

  auto eil = peos->compute("W->I", {wl});
  auto eir = peos->compute("W->I", {wr});
  auto gammal = peos->compute("W->A", {wl});
  auto gammar = peos->compute("W->A", {wr});
  auto cl = peos->compute("WA->L", {wl, gammal});
  auto cr = peos->compute("WA->L", {wr, gammar});

  // solve in the face-local orthonormal frame, as lmars.cpp does
  auto pcoord = phydro->pmb->pcoord;
  for (auto const& w : {wl, wr}) {
    if (dim == 1) pcoord->prim2local3_(w);
    if (dim == 2) pcoord->prim2local2_(w);
    if (dim == 3) pcoord->prim2local1_(w);
  }

  auto face_pressure_out =
      face_pressure.defined() ? face_pressure : torch::empty_like(wl[IDN]);
  auto elr = torch::stack({eil, eir});
  auto glr = torch::stack({gammal, gammar});
  auto clr = torch::stack({cl, cr});
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(flx.sizes(), /*squash_dims=*/0)
                  .add_output(flx)
                  .add_owned_output(face_pressure_out.unsqueeze(0))
                  .add_input(wl)
                  .add_input(wr)
                  .add_input(elr)
                  .add_input(glr)
                  .add_input(clr)
                  .build();

  at::native::call_roe(flx.device().type(), iter, dim, ideal_moist, nvapor,
                       peos->options->gammad(), inv_mu_ratio_m1, cv_ratio_m1,
                       u0);

  if (dim == 1) pcoord->flux2global3_(flx);
  if (dim == 2) pcoord->flux2global2_(flx);
  if (dim == 3) pcoord->flux2global1_(flx);
  return flx;
}

}  // namespace snap
