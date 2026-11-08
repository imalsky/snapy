#pragma once

#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using recon_poly_nu_fn = void (*)(at::TensorIterator &iter, at::Tensor payload,
                                  int dim);

using recon_weno_nu_fn = void (*)(at::TensorIterator &iter, at::Tensor payload,
                                  int dim, bool scale);

DECLARE_DISPATCH(recon_poly_nu_fn, call_poly5_nu);
DECLARE_DISPATCH(recon_weno_nu_fn, call_weno5_nu);

}  // namespace at::native
