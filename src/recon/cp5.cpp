// torch
#include <ATen/TensorIterator.h>

// snap
#include <snap/snap.h>

#include "interpolation.hpp"
#include "recon_dispatch.hpp"
#include "recon_dispatch_nu.hpp"
#include "x1_coeffs.hpp"

namespace snap {

void Center5InterpImpl::reset() {
  cm = register_buffer("cm", torch::tensor({-1. / 20., 9. / 20., 47. / 60.,
                                            -13. / 60., 1. / 30.},
                                           torch::kFloat64));
  cp = register_buffer("cp", cm.flip({0}));

  x1_reset_tables(*this, cm_x1, cp_x1, kPoly5NuCoeff);
}

void Center5InterpImpl::set_x1_coeffs(torch::Tensor const& x1f) {
  x1_set_coeffs(*this, x1f, &cp5_x1_coeffs);
}

void Center5InterpImpl::forward_nu(torch::Tensor const& w, int dim,
                                   torch::Tensor const& out,
                                   torch::Tensor const& src,
                                   torch::Tensor& cache) {
  x1_forward_nu(w, dim, out, src, cache, "cp5",
                [&](at::TensorIterator& iter, torch::Tensor const& c) {
                  at::native::call_poly5_nu(out.device().type(), iter, c, dim);
                });
}

void Center5InterpImpl::left(torch::Tensor w, int dim,
                             torch::Tensor const& out) {
  if (has_x1_coeffs()) {
    forward_nu(w, dim, out, cm_x1, cm_x1_cast);
    return;
  }

  auto iter = x1_uniform_iterator(w, dim, out);

  at::native::call_poly5(out.device().type(), iter, cm, dim);
}

void Center5InterpImpl::right(torch::Tensor w, int dim,
                              torch::Tensor const& out) {
  if (has_x1_coeffs()) {
    forward_nu(w, dim, out, cp_x1, cp_x1_cast);
    return;
  }

  auto iter = x1_uniform_iterator(w, dim, out);

  at::native::call_poly5(out.device().type(), iter, cp, dim);
}
}  // namespace snap
