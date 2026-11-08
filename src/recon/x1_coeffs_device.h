#pragma once

#include <configure.h>

namespace snap {

constexpr int kWeno5NuCoeff = 48;
constexpr int kPoly5NuCoeff = 5;

template <int N, typename T>
DISPATCH_MACRO inline T _vvdot_c(T const* v1, T const* v2) {
  T out = 0.;
  for (int i = 0; i < N; ++i) {
    out += v1[i] * v2[i];
  }
  return out;
}

}  // namespace snap
