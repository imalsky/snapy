// torch
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// snap
#include "hydro_dispatch.hpp"
#include "hydro_ref_x1_impl.h"

namespace snap {

template <typename T>
__global__ void hydro_ref_x1_cuda_kernel(T const *w, T const *dx1f,
                                         T const *anchor, T *psf_lo, T *psf_hi,
                                         T *pref, T *dsf, T *dref, int ncolumns,
                                         int nc1, int iu, T grav, int x1_grid,
                                         bool phys_in, bool phys_out,
                                         bool wall_clamp, T const *rows) {
  int column = blockIdx.x;

  if (threadIdx.x == 0) {
    hydro_ref_x1_scan_impl(w, dx1f, anchor, psf_lo, psf_hi, column, ncolumns,
                           nc1, iu, grav);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < nc1; i += blockDim.x) {
    hydro_ref_x1_cell_impl(w, dx1f, psf_lo, psf_hi, pref, dsf, dref, column, i,
                           ncolumns, nc1, iu, grav, x1_grid, phys_in, phys_out,
                           wall_clamp, rows);
  }
}

void hydro_ref_x1_cuda(torch::Tensor const &w, torch::Tensor const &dx1f,
                       torch::Tensor const &anchor, torch::Tensor const &psf_lo,
                       torch::Tensor const &psf_hi, torch::Tensor const &pref,
                       torch::Tensor const &dsf, torch::Tensor const &dref,
                       int iu, double grav, int x1_grid, bool phys_in,
                       bool phys_out, bool wall_clamp,
                       torch::Tensor const &rows) {
  TORCH_CHECK(x1_grid != kX1RefStretched || rows.defined(),
              "hydro_ref_x1: a stretched x1 needs its six-face rows");
  TORCH_CHECK(!rows.defined() || (rows.device() == w.device() &&
                                  rows.scalar_type() == w.scalar_type() &&
                                  rows.is_contiguous()),
              "hydro_ref_x1: the stretched rows must be contiguous and on the "
              "state's own device and dtype; got ",
              rows.device(), "/", rows.scalar_type(), " against ", w.device(),
              "/", w.scalar_type());

  at::cuda::CUDAGuard device_guard(w.device());
  int ncolumns = w.size(1) * w.size(2);
  int nc1 = w.size(3);
  int threads = 256;
  int blocks = ncolumns;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "hydro_ref_x1_cuda", [&] {
    auto anchor_ptr = anchor.defined() ? anchor.data_ptr<scalar_t>() : nullptr;
    auto rows_ptr = rows.defined() ? rows.data_ptr<scalar_t>() : nullptr;
    hydro_ref_x1_cuda_kernel<<<blocks, threads, 0, stream>>>(
        w.data_ptr<scalar_t>(), dx1f.data_ptr<scalar_t>(), anchor_ptr,
        psf_lo.data_ptr<scalar_t>(), psf_hi.data_ptr<scalar_t>(),
        pref.data_ptr<scalar_t>(), dsf.data_ptr<scalar_t>(),
        dref.data_ptr<scalar_t>(), ncolumns, nc1, iu, scalar_t(grav), x1_grid,
        phys_in, phys_out, wall_clamp, rows_ptr);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace snap

namespace at::native {

REGISTER_CUDA_DISPATCH(call_hydro_ref_x1, &snap::hydro_ref_x1_cuda);

} // namespace at::native
