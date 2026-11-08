#pragma once

#include "interp_impl.h"
#include "x1_coeffs.hpp"

#define INP(j, i) (inp[(j) * stride2 + (i) * stride1])
#define OUT(j) (out[(j) * stride_out])
#define SQR(x) ((x) * (x))

namespace snap {

template <typename T>
void interp_weno5_nu_impl(T* out, T const* inp, T const* coeff, int stride1,
                          int stride2, int stride_out, int nvar, double scale) {
  T const* c1 = coeff;
  T const* c2 = c1 + 5;
  T const* c3 = c2 + 5;
  T const* c4 = c3 + 5;
  T const* c5 = c4 + 5;
  T const* c6 = c5 + 5;
  T const* c7 = c6 + 5;
  T const* c8 = c7 + 5;
  T const* c9 = c8 + 5;
  T const* d = c9 + 5;

  T phi[5];

  for (int j = 0; j < nvar; ++j) {
    T vscale = scale ? (fabs(INP(j, 0)) + fabs(INP(j, 1)) + fabs(INP(j, 2)) +
                        fabs(INP(j, 3)) + fabs(INP(j, 4))) /
                           5.0
                     : 1.0;

    if (vscale != 0.0) {
      for (int k = 0; k < 5; ++k) {
        phi[k] = INP(j, k) / vscale;
      }
    } else {
      OUT(j) = 0.0;
      continue;
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

    OUT(j) = vscale * (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) /
             (alpha0 + alpha1 + alpha2);
  }
};

template <typename T>
void interp_poly5_nu_impl(T* out, T const* inp, T const* coeff, int stride1,
                          int stride2, int stride_out, int nvar) {
  for (int j = 0; j < nvar; ++j) {
    OUT(j) = 0.;
    for (int i = 0; i < kPoly5NuCoeff; ++i) {
      OUT(j) += coeff[i] * INP(j, i);
    }
  }
};

}  // namespace snap

#undef OUT
#undef SQR
#undef INP
