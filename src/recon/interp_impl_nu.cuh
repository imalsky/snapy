#pragma once

#include <cuda_runtime.h>

#include "interp_impl.cuh"
#include "x1_coeffs_device.h"


#define INP(j, i) (inp[(j) * stride_in2 + (i) * stride_in1])
#define OUT(j, i) (out[(j) * stride_out2 + (i) * stride_out1])
#define SQR(x) ((x) * (x))

namespace snap {


template <typename T>
__device__ T interp_shared_weno5_nu_impl(T const *line, T const *row, int v,
                                         int start, int axis_size, bool scale) {
  T const *c1 = row;
  T const *c2 = c1 + 5;
  T const *c3 = c2 + 5;
  T const *c4 = c3 + 5;
  T const *c5 = c4 + 5;
  T const *c6 = c5 + 5;
  T const *c7 = c6 + 5;
  T const *c8 = c7 + 5;
  T const *c9 = c8 + 5;
  T const *d = c9 + 5;

  T const *src = line + v * axis_size + start;
  T vscale = scale ? (fabs(src[0]) + fabs(src[1]) + fabs(src[2]) +
                      fabs(src[3]) + fabs(src[4])) /
                         5.0
                   : 1.0;

  if (vscale == 0.0) return 0.0;

  T phi[5];
  for (int k = 0; k < 5; ++k) {
    phi[k] = src[k] / vscale;
  }

  T p0 = _vvdot_c<5>(phi, c1);
  T p1 = _vvdot_c<5>(phi, c2);
  T p2 = _vvdot_c<5>(phi, c3);

  T beta0 =
      13. / 12. * SQR(_vvdot_c<5>(phi, c4)) + .25 * SQR(_vvdot_c<5>(phi, c5));
  T beta1 =
      13. / 12. * SQR(_vvdot_c<5>(phi, c6)) + .25 * SQR(_vvdot_c<5>(phi, c7));
  T beta2 =
      13. / 12. * SQR(_vvdot_c<5>(phi, c8)) + .25 * SQR(_vvdot_c<5>(phi, c9));

  T alpha0 = d[0] / SQR(beta0 + 1e-6);
  T alpha1 = d[1] / SQR(beta1 + 1e-6);
  T alpha2 = d[2] / SQR(beta2 + 1e-6);

  return vscale * (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) /
         (alpha0 + alpha1 + alpha2);
}

template <typename T>
__device__ void interp_weno5_nu_impl(T *out, T *inp, T const *coeff, int nvar,
                                     int stride_in1, int stride_in2,
                                     int stride_out1, int stride_out2,
                                     bool scale, T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;

  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[id + j * nt] = INP(j, id);
  }

  bool active = id <= nt - kWeno5Stencil;

  __syncthreads();

  if (!active) return;

  T const *row = coeff + static_cast<int64_t>(id) * kWeno5NuCoeff;
  for (int j = 0; j < nvar; ++j) {
    OUT(j, id) = interp_shared_weno5_nu_impl(sinp, row, j, id, nt, scale);
  }
};

template <typename T>
__device__ void interp_poly5_nu_impl(T *out, T *inp, T const *coeff, int nvar,
                                     int stride_in1, int stride_in2,
                                     int stride_out1, int stride_out2,
                                     T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;

  T *sinp = smem;
  for (int j = 0; j < nvar; ++j) {
    sinp[id + j * nt] = INP(j, id);
  }

  bool active = id <= nt - kPoly5NuCoeff;

  __syncthreads();

  if (!active) return;

  T const *row = coeff + static_cast<int64_t>(id) * kPoly5NuCoeff;
  for (int j = 0; j < nvar; ++j) {
    OUT(j, id) = interp_shared_poly_coeff_impl<T, 5>(sinp, row, j, id, nt);
  }
};

template <int NCoeff, typename T>
__device__ void interp_nu_untiled(T *out, T *inp, T const *coeff, int nvar,
                                  int stride_in1, int stride_in2,
                                  int stride_out1, int stride_out2,
                                  bool scale, T *smem) {
  if constexpr (NCoeff == kWeno5NuCoeff) {
    interp_weno5_nu_impl(out, inp, coeff, nvar, stride_in1, stride_in2,
                         stride_out1, stride_out2, scale, smem);
  } else {
    interp_poly5_nu_impl(out, inp, coeff, nvar, stride_in1, stride_in2,
                         stride_out1, stride_out2, smem);
  }
}

struct Weno5NuOp {
  template <typename T>
  __device__ static T eval(T const *line, T const *row, int v, int start,
                           int axis, bool scale) {
    return interp_shared_weno5_nu_impl(line, row, v, start, axis, scale);
  }
};

template <typename T, int Stencil, int NCoeff, typename Op>
__device__ void interp_line_tiled_nu(T *out, T *inp, T const *coeff, int nvar,
                                     int stride_in1, int stride_in2,
                                     int stride_out1, int stride_out2,
                                     bool scale, int nline, T *smem) {
  int id = threadIdx.x;
  int nt = blockDim.x;
  int overhang = Stencil - 1;
  int nout = nline - overhang;

  for (int start = 0; start < nout; start += nt) {
    int n_out = nout - start;
    if (n_out > nt) n_out = nt;
    int window = n_out + overhang;

    T *sinp = smem;
    for (int j = 0; j < nvar; ++j) {
      for (int i = id; i < window; i += nt) {
        sinp[j * window + i] = INP(j, start + i);
      }
    }

    __syncthreads();

    if (id < n_out) {
      T const *row = coeff + static_cast<int64_t>(start + id) * NCoeff;
      for (int j = 0; j < nvar; ++j) {
        OUT(j, start + id) =
            Op::template eval<T>(sinp, row, j, id, window, scale);
      }
    }

    __syncthreads();
  }
}

}  // namespace snap

#undef SQR
#undef INP
#undef OUT
