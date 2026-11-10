#pragma once

// torch
#include <torch/torch.h>

namespace snap {

//! x1 slopes of SNAP_X1_MASS_COVARIANCE; `stretched` is the grid key (#280)
inline torch::Tensor x1_cov_d1_centred(torch::Tensor const& a,
                                       torch::Tensor const& x1v,
                                       bool stretched) {
  int n1 = a.size(-1);
  auto d = torch::empty_like(a);
  if (stretched) {
    auto hm = x1v.narrow(0, 1, n1 - 2) - x1v.narrow(0, 0, n1 - 2);
    auto hp = x1v.narrow(0, 2, n1 - 2) - x1v.narrow(0, 1, n1 - 2);
    d.narrow(-1, 1, n1 - 2)
        .copy_(-hp / (hm * (hm + hp)) * a.narrow(-1, 0, n1 - 2) +
               (hp - hm) / (hm * hp) * a.narrow(-1, 1, n1 - 2) +
               hm / (hp * (hm + hp)) * a.narrow(-1, 2, n1 - 2));
  } else {
    auto dx = x1v.narrow(0, 2, n1 - 2) - x1v.narrow(0, 0, n1 - 2);
    d.narrow(-1, 1, n1 - 2)
        .copy_((a.narrow(-1, 2, n1 - 2) - a.narrow(-1, 0, n1 - 2)) / dx);
  }
  d.select(-1, 0).copy_(d.select(-1, 1));
  d.select(-1, n1 - 1).copy_(d.select(-1, n1 - 2));
  return d;
}

//! one-sided slope at wall cell i from cells i, i + s, i + 2 s
inline torch::Tensor x1_cov_one_sided(torch::Tensor const& a,
                                      torch::Tensor const& x1v, int i, int s,
                                      bool stretched) {
  if (stretched) {
    auto p = x1v[i + s] - x1v[i], q = x1v[i + 2 * s] - x1v[i];
    return -(p + q) / (p * q) * a.select(-1, i) +
           q / (p * (q - p)) * a.select(-1, i + s) -
           p / (q * (q - p)) * a.select(-1, i + 2 * s);
  }
  auto h = x1v[i + s] - x1v[i];
  return s *
         (-3. * a.select(-1, i) + 4. * a.select(-1, i + s) -
          a.select(-1, i + 2 * s)) /
         (2. * s * h);
}

}  // namespace snap
