#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <limits>

#include <snap/utils/loops.cuh>

#include "interp_impl_nu.cuh"
#include "recon_dispatch_nu.hpp"
#include "x1_coeffs.hpp"

namespace snap {
namespace {

template <typename scalar_t, int Arity, typename func_t>
void stencil_kernel_tiled_nu(at::TensorIterator& iter, int dim, int block_x,
                             size_t shared_bytes, const func_t& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);
  TORCH_CHECK(block_x >= 1 && block_x <= 1024);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  int64_t numel = iter.input().numel();

  TORCH_INTERNAL_ASSERT(numel >= 0 &&
                        numel <= std::numeric_limits<int32_t>::max());
  if (numel == 0) {
    return;
  }

  int len[3] = {1, 1, 1};
  int ndim = iter.input().dim();
  len[3 + dim - ndim] = at::native::ensure_nonempty_size(iter.input(), dim);
  native::_left_shift<3>(len, dim + 1 - ndim);
  dim3 line_block(len[2], len[1], len[0]);
  TORCH_CHECK(line_block.y == 1 && line_block.z == 1);
  TORCH_CHECK(line_block.x > static_cast<unsigned>(block_x));

  if (ndim <= 3) {
    len[3 - ndim] = at::native::ensure_nonempty_size(iter.input(), 0);
  }
  for (int i = 1; i < ndim; ++i) {
    len[3 + i - ndim] = at::native::ensure_nonempty_size(iter.input(), i);
  }
  native::_left_shift<3>(len, dim + 1 - ndim);

  dim3 grid(len[2] / line_block.x, len[1] / line_block.y,
            len[0] / line_block.z);
  TORCH_CHECK(grid.x == 1);
  dim3 block(block_x, line_block.y, line_block.z);

  auto stream = at::cuda::getCurrentCUDAStream();
  native::reduce_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
      numel, [=] __device__(int bid, scalar_t* smem) {
        auto offsets = offset_calc.get(bid);
        f(data.data(), offsets.data(), smem);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int recon_tile_width_nu(int64_t nline, int nvar, int stencil,
                        size_t scalar_size, size_t* shared_bytes) {
  if (nline <= 1024) {
    return 0;
  }
  TORCH_CHECK(nline <= std::numeric_limits<int>::max(),
              "recon line does not fit in int: ", nline);
  TORCH_CHECK(nvar > 0 && stencil >= 2);
  constexpr size_t kSharedLimit = 48 * 1024;
  int tile = 256;
  auto bytes = [&](int t) -> size_t {
    return static_cast<size_t>(t + stencil - 1) * static_cast<size_t>(nvar) *
           scalar_size;
  };
  while (tile > 1 && bytes(tile) > kSharedLimit) {
    tile >>= 1;
  }
  TORCH_CHECK(bytes(tile) <= kSharedLimit,
              "recon tile does not fit in shared memory, nvar=", nvar,
              " stencil=", stencil);
  *shared_bytes = bytes(tile);
  return tile;
}

}  // namespace

template <int Stencil, int NCoeff, typename Op>
void call_nu_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim,
                  bool scale, char const* what) {
  at::cuda::CUDAGuard device_guard(iter.device());
  x1_coeffs_check_rows(iter, coeff, dim, NCoeff, what);

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_nu_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);
    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);
    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();
    int64_t nline64 = at::native::ensure_nonempty_size(iter.input(), dim);
    size_t shared = 0;
    int tile = recon_tile_width_nu(nline64, nvar, Stencil,
                                   sizeof(scalar_t), &shared);

    if (tile == 0) {
      // One launch holds every variable's whole line in shared memory; more
      // variables than fit go in parts of at most `per`. The rows are in
      // global memory and indexed by face, so each part reads the same rows.
      int per = native::stencil_vars_per_launch(nline64, 0, sizeof(scalar_t));
      for (int v0 = 0; v0 < nvar; v0 += per) {
        int nv = std::min(per, nvar - v0);
        int64_t off_in = static_cast<int64_t>(v0) * stride_in2;
        int64_t off_out = static_cast<int64_t>(v0) * stride_out2;
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, /*buffers=*/0,
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out =
                  reinterpret_cast<scalar_t*>(data[0] + strides[0]) + off_out;
              auto w =
                  reinterpret_cast<scalar_t*>(data[1] + strides[1]) + off_in;
              interp_nu_untiled<NCoeff>(out, w, c, nv, stride_in1, stride_in2,
                                        stride_out1, stride_out2, scale, smem);
            },
            nv);
      }
      return;
    }

    int nline = static_cast<int>(nline64);
    stencil_kernel_tiled_nu<scalar_t, 2>(
        iter, dim, tile, shared,
        [=] __device__(char* const data[2], unsigned int strides[2],
                       scalar_t* smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_line_tiled_nu<scalar_t, Stencil, NCoeff, Op>(
              out, w, c, nvar, stride_in1, stride_in2, stride_out1,
              stride_out2, scale, nline, smem);
        });
  });
}

void call_weno5_nu_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim,
                        bool scale) {
  call_nu_cuda<kWeno5Stencil, kWeno5NuCoeff, Weno5NuOp>(
      iter, coeff, dim, scale, "weno5_nu");
}

void call_poly5_nu_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim) {
  call_nu_cuda<kPoly5NuCoeff, kPoly5NuCoeff, PolyOp<5>>(
      iter, coeff, dim, false, "poly5_nu");
}

}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_poly5_nu, &snap::call_poly5_nu_cuda);
REGISTER_CUDA_DISPATCH(call_weno5_nu, &snap::call_weno5_nu_cuda);

}  // namespace at::native
