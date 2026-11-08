#include "recon_dispatch_nu.hpp"

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

#include "interp_impl_nu.h"
#include "x1_coeffs.hpp"

namespace snap {
namespace {

void check_chunk(char const* base, int64_t nrow, int64_t rowbytes, char** data,
                 const int64_t* strides, int64_t n, char const* what) {
  char const* lo = data[2];
  char const* hi = data[2] + (n - 1) * strides[2];
  if (hi < lo) std::swap(lo, hi);
  TORCH_CHECK((lo - base) % rowbytes == 0 && (hi - base) % rowbytes == 0, "[",
              what, "] coefficient row pointer is not row aligned");
  TORCH_CHECK(lo >= base && (hi - base) / rowbytes < nrow, "[", what,
              "] coefficient row pointer is outside the table");
}

template <int NCoeff, typename Op>
void call_nu_cpu(at::TensorIterator& iter, torch::Tensor const& coeff, int dim,
                 char const* what, Op op) {
  x1_coeffs_check_rows(iter, coeff, dim, NCoeff, what);
  TORCH_CHECK(iter.ninputs() == 2, "[", what,
              "] the CPU iterator needs the coefficient row view; "
              "build it with snap::x1_nu_iterator");
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_nu_cpu", [&] {
    int stride1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out = at::native::ensure_nonempty_stride(iter.output(), 0);
    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);

    auto const* base =
        reinterpret_cast<char const*>(coeff.data_ptr<scalar_t>());
    int64_t nrow = coeff.size(0);
    int64_t rowbytes = NCoeff * static_cast<int64_t>(sizeof(scalar_t));

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          if (n <= 0) return;
          check_chunk(base, nrow, rowbytes, data, strides, n, what);
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto w = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto c =
                reinterpret_cast<scalar_t const*>(data[2] + i * strides[2]);
            op(out, w, c, stride1, stride2, stride_out, nvar);
          }
        },
        grain_size);
  });
}

}  // namespace

void call_weno5_nu_cpu(at::TensorIterator& iter, torch::Tensor coeff, int dim,
                       bool scale) {
  call_nu_cpu<kWeno5NuCoeff>(
      iter, coeff, dim, "weno5_nu",
      [scale](auto* out, auto const* w, auto const* c, int stride1, int stride2,
              int stride_out, int nvar) {
        interp_weno5_nu_impl(out, w, c, stride1, stride2, stride_out, nvar,
                             scale);
      });
}

void call_poly5_nu_cpu(at::TensorIterator& iter, torch::Tensor coeff, int dim) {
  call_nu_cpu<kPoly5NuCoeff>(
      iter, coeff, dim, "poly5_nu",
      [](auto* out, auto const* w, auto const* c, int stride1, int stride2,
         int stride_out, int nvar) {
        interp_poly5_nu_impl(out, w, c, stride1, stride2, stride_out, nvar);
      });
}

void call_weno5_nu_mps(at::TensorIterator&, torch::Tensor, int, bool) {
  TORCH_CHECK(false,
              "[weno5_nu] a stretched x1 grid is not supported on the MPS "
              "backend in stage 1 of snapy#280; use CPU or CUDA");
}

void call_poly5_nu_mps(at::TensorIterator&, torch::Tensor, int) {
  TORCH_CHECK(false,
              "[poly5_nu] a stretched x1 grid is not supported on the MPS "
              "backend in stage 1 of snapy#280; use CPU or CUDA");
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_poly5_nu);
DEFINE_DISPATCH(call_weno5_nu);

REGISTER_ALL_CPU_DISPATCH(call_poly5_nu, &snap::call_poly5_nu_cpu);
REGISTER_ALL_CPU_DISPATCH(call_weno5_nu, &snap::call_weno5_nu_cpu);

REGISTER_MPS_DISPATCH(call_poly5_nu, &snap::call_poly5_nu_mps);
REGISTER_MPS_DISPATCH(call_weno5_nu, &snap::call_weno5_nu_mps);

}  // namespace at::native
