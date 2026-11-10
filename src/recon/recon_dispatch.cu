// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <limits>

// snap
#include <snap/utils/loops.cuh>
#include "recon_dispatch.hpp"
#include "interp_impl.cuh"

namespace snap {

// 0 means the line fits in one CUDA block, so the caller keeps stencil_kernel.
// Otherwise returns the tile width and the dynamic shared-memory size for
// (tile + stencil - 1) inputs per variable plus ncoeff weights.
int recon_tile_width(int64_t nline, int nvar, int stencil, int64_t ncoeff,
                     size_t scalar_size, size_t* shared_bytes) {
  if (nline <= 1024) {
    return 0;
  }
  TORCH_CHECK(nline <= std::numeric_limits<int>::max(),
              "recon line does not fit in int: ", nline);
  TORCH_CHECK(nvar > 0 && stencil >= 2 && ncoeff > 0);
  constexpr size_t kSharedLimit = 48 * 1024;
  int tile = 256;
  auto bytes = [&](int t) -> size_t {
    return (static_cast<size_t>(t + stencil - 1) * static_cast<size_t>(nvar) +
            static_cast<size_t>(ncoeff)) *
           scalar_size;
  };
  while (tile > 1 && bytes(tile) > kSharedLimit) {
    tile >>= 1;
  }
  TORCH_CHECK(bytes(tile) <= kSharedLimit,
              "recon tile does not fit in shared memory, nvar=", nvar,
              " stencil=", stencil, " ncoeff=", ncoeff);
  *shared_bytes = bytes(tile);
  return tile;
}

// Same grid as stencil_kernel (one block per line), but block.x is the
// tile width instead of the line length. grid.x stays 1: blockIdx is the
// line id, so a second block along the line would read the wrong offset.
template <typename scalar_t, int Arity, typename func_t>
void stencil_kernel_tiled(at::TensorIterator& iter, int dim, int block_x,
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

  dim3 grid(len[2] / line_block.x, len[1] / line_block.y, len[0] / line_block.z);
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

template <int N>
void call_poly_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_poly_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();
    int64_t nline64 = at::native::ensure_nonempty_size(iter.input(), dim);
    size_t shared = 0;
    int tile = recon_tile_width(nline64, nvar, N, coeff.numel(),
                                sizeof(scalar_t), &shared);

    if (tile == 0) {
      int per = native::stencil_vars_per_launch(nline64, coeff.numel(),
                                                sizeof(scalar_t));
      if (nvar <= per) {
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] GPU_LAMBDA(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
              auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
              interp_poly_impl<scalar_t, N>(out, w, c, nvar, stride_in1,
                                            stride_in2, stride_out1,
                                            stride_out2, smem);
            });
        return;
      }
      // More variables than one launch's shared memory holds: launch on
      // parts of at most `per` variables. Every output is computed as in the
      // single launch.
      for (int v0 = 0; v0 < nvar; v0 += per) {
        int nv = std::min(per, nvar - v0);
        int64_t off_in = static_cast<int64_t>(v0) * stride_in2;
        int64_t off_out = static_cast<int64_t>(v0) * stride_out2;
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] GPU_LAMBDA(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out =
                  reinterpret_cast<scalar_t*>(data[0] + strides[0]) + off_out;
              auto w =
                  reinterpret_cast<scalar_t*>(data[1] + strides[1]) + off_in;
              interp_poly_impl<scalar_t, N>(out, w, c, nv, stride_in1,
                                            stride_in2, stride_out1,
                                            stride_out2, smem);
            },
            nv);
      }
      return;
    }

    TORCH_CHECK(coeff.numel() == N);
    int nline = static_cast<int>(nline64);
    stencil_kernel_tiled<scalar_t, 2>(
        iter, dim, tile, shared,
        [=] __device__ (char* const data[2], unsigned int strides[2], scalar_t *smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_line_tiled<scalar_t, N, N, PolyOp<N>>(
              out, w, c, nvar, stride_in1, stride_in2, stride_out1, stride_out2,
              /*scale=*/false, nline, smem);
        });
  });
}

void call_weno3_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim, bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_weno3_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();
    int64_t nline64 = at::native::ensure_nonempty_size(iter.input(), dim);
    size_t shared = 0;
    int tile = recon_tile_width(nline64, nvar, kWeno3Stencil, kWeno3Coeff,
                                sizeof(scalar_t), &shared);

    if (tile == 0) {
      int per = native::stencil_vars_per_launch(nline64, coeff.numel(),
                                                sizeof(scalar_t));
      if (nvar <= per) {
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
              auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
              interp_weno3_impl(out, w, c, nvar, stride_in1, stride_in2,
                                stride_out1, stride_out2, scale, smem);
            });
        return;
      }
      // More variables than one launch's shared memory holds: launch on
      // parts of at most `per` variables. Every output is computed as in the
      // single launch.
      for (int v0 = 0; v0 < nvar; v0 += per) {
        int nv = std::min(per, nvar - v0);
        int64_t off_in = static_cast<int64_t>(v0) * stride_in2;
        int64_t off_out = static_cast<int64_t>(v0) * stride_out2;
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out =
                  reinterpret_cast<scalar_t*>(data[0] + strides[0]) + off_out;
              auto w =
                  reinterpret_cast<scalar_t*>(data[1] + strides[1]) + off_in;
              interp_weno3_impl(out, w, c, nv, stride_in1, stride_in2,
                                stride_out1, stride_out2, scale, smem);
            },
            nv);
      }
      return;
    }

    TORCH_CHECK(coeff.numel() == kWeno3Coeff);
    int nline = static_cast<int>(nline64);
    stencil_kernel_tiled<scalar_t, 2>(
        iter, dim, tile, shared,
        [=] __device__ (char* const data[2], unsigned int strides[2], scalar_t *smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_line_tiled<scalar_t, kWeno3Stencil, kWeno3Coeff, Weno3Op>(
              out, w, c, nvar, stride_in1, stride_in2, stride_out1, stride_out2,
              scale, nline, smem);
        });
  });
}

void call_weno5_cuda(at::TensorIterator& iter, at::Tensor coeff, int dim, bool scale) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "call_weno5_cuda", [&]() {
    int stride_in1 = at::native::ensure_nonempty_stride(iter.input(), dim);
    int stride_in2 = at::native::ensure_nonempty_stride(iter.input(), 0);

    int stride_out1 = at::native::ensure_nonempty_stride(iter.output(), dim);
    int stride_out2 = at::native::ensure_nonempty_stride(iter.output(), 0);

    int nvar = at::native::ensure_nonempty_size(iter.output(), 0);
    auto c = coeff.data_ptr<scalar_t>();
    int64_t nline64 = at::native::ensure_nonempty_size(iter.input(), dim);
    size_t shared = 0;
    int tile = recon_tile_width(nline64, nvar, kWeno5Stencil, kWeno5Coeff,
                                sizeof(scalar_t), &shared);

    // 1024 threads is the CUDA block limit. This launch is the one that
    // already ships, so a line that fits stays on it.
    if (tile == 0) {
      int per = native::stencil_vars_per_launch(nline64, coeff.numel(),
                                                sizeof(scalar_t));
      if (nvar <= per) {
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
              auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
              interp_weno5_impl(out, w, c, nvar, stride_in1, stride_in2,
                                stride_out1, stride_out2, scale, smem);
            });
        return;
      }
      // More variables than one launch's shared memory holds: launch on
      // parts of at most `per` variables. Every output is computed as in the
      // single launch.
      for (int v0 = 0; v0 < nvar; v0 += per) {
        int nv = std::min(per, nvar - v0);
        int64_t off_in = static_cast<int64_t>(v0) * stride_in2;
        int64_t off_out = static_cast<int64_t>(v0) * stride_out2;
        native::stencil_kernel<scalar_t, 2>(
            iter, dim, coeff.numel(),
            [=] __device__(char* const data[2], unsigned int strides[2],
                           scalar_t* smem) {
              auto out =
                  reinterpret_cast<scalar_t*>(data[0] + strides[0]) + off_out;
              auto w =
                  reinterpret_cast<scalar_t*>(data[1] + strides[1]) + off_in;
              interp_weno5_impl(out, w, c, nv, stride_in1, stride_in2,
                                stride_out1, stride_out2, scale, smem);
            },
            nv);
      }
      return;
    }

    TORCH_CHECK(coeff.numel() == kWeno5Coeff);
    int nline = static_cast<int>(nline64);
    stencil_kernel_tiled<scalar_t, 2>(
        iter, dim, tile, shared,
        [=] __device__ (char* const data[2], unsigned int strides[2], scalar_t *smem) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto w = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          interp_line_tiled<scalar_t, kWeno5Stencil, kWeno5Coeff, Weno5Op>(
              out, w, c, nvar, stride_in1, stride_in2, stride_out1, stride_out2,
              scale, nline, smem);
        });
  });
}
}  // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_poly3, &snap::call_poly_cuda<3>);
REGISTER_CUDA_DISPATCH(call_poly5, &snap::call_poly_cuda<5>);
REGISTER_CUDA_DISPATCH(call_weno3, &snap::call_weno3_cuda);
REGISTER_CUDA_DISPATCH(call_weno5, &snap::call_weno5_cuda);

}  // namespace at::native
