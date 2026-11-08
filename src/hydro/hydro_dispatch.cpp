// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>

// C/C++
#include <cmath>
#include <limits>

// snap
#include "hydro_dispatch.hpp"
#include "hydro_ref_x1_impl.h"

namespace snap {

void hydro_ref_x1_cpu(torch::Tensor const& w, torch::Tensor const& dx1f,
                      torch::Tensor const& anchor, torch::Tensor const& psf_lo,
                      torch::Tensor const& psf_hi, torch::Tensor const& pref,
                      torch::Tensor const& dsf, torch::Tensor const& dref,
                      int iu, double grav, int x1_grid, bool phys_in,
                      bool phys_out, bool wall_clamp,
                      torch::Tensor const& rows) {
  TORCH_CHECK(x1_grid != kX1RefStretched || rows.defined(),
              "hydro_ref_x1: a stretched x1 needs its six-face rows");
  int ncolumns = w.size(1) * w.size(2);
  int nc1 = w.size(3);
  AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "hydro_ref_x1_cpu", [&] {
    auto anchor_ptr = anchor.defined() ? anchor.data_ptr<scalar_t>() : nullptr;
    auto rows_ptr = rows.defined() ? rows.data_ptr<scalar_t>() : nullptr;
    at::parallel_for(0, ncolumns, 0, [&](int64_t begin, int64_t end) {
      for (int64_t column = begin; column < end; ++column) {
        hydro_ref_x1_impl(w.data_ptr<scalar_t>(), dx1f.data_ptr<scalar_t>(),
                          anchor_ptr, psf_lo.data_ptr<scalar_t>(),
                          psf_hi.data_ptr<scalar_t>(),
                          pref.data_ptr<scalar_t>(), dsf.data_ptr<scalar_t>(),
                          dref.data_ptr<scalar_t>(), static_cast<int>(column),
                          ncolumns, nc1, iu, scalar_t(grav), x1_grid, phys_in,
                          phys_out, wall_clamp, rows_ptr);
      }
    });
  });
}

void hydro_ref_x1_mps(torch::Tensor const& w, torch::Tensor const& dx1f,
                      torch::Tensor const& anchor_in,
                      torch::Tensor const& psf_lo, torch::Tensor const& psf_hi,
                      torch::Tensor const& pref, torch::Tensor const& dsf,
                      torch::Tensor const& dref, int iu, double grav,
                      int x1_grid, bool phys_in, bool phys_out, bool wall_clamp,
                      torch::Tensor const& /*rows*/) {
  TORCH_CHECK(x1_grid != kX1RefStretched,
              "a stretched x1 (geometry/x1f or cells/x1rat) is not supported "
              "on MPS (#280); run it on CPU or CUDA");
  bool uniform = x1_grid == kX1RefUniform;
  int nc1 = w.size(-1);
  int il = nc1 - 1 - iu;
  auto rho = w[IDN];
  auto dp = grav * rho * dx1f;
  auto cum = torch::cumsum(dp, -1);

  auto anchor = anchor_in.defined()
                    ? anchor_in
                    : (w[IPR].select(-1, iu) *
                       torch::exp(-grav * 0.5 * dx1f[iu] /
                                  (w[IPR].select(-1, iu) / rho.select(-1, iu))))
                          .unsqueeze(-1);
  auto cum_iu = cum.select(-1, iu).unsqueeze(-1);
  psf_lo.copy_(anchor + cum_iu - cum + dp);
  psf_hi.copy_(psf_lo - dp);
  psf_lo.clamp_min_(std::numeric_limits<double>::min());
  psf_hi.clamp_min_(std::numeric_limits<double>::min());

  if (uniform) {
    pref.copy_(0.5 * (psf_lo + psf_hi));
    auto faces = torch::cat({psf_lo, psf_hi.narrow(-1, nc1 - 1, 1)}, -1);
    constexpr double w6[6] = {11. / 1440., -31. / 480., 401. / 720.,
                              401. / 720., -31. / 480., 11. / 1440.};
    auto six = w6[0] * faces.narrow(-1, 0, nc1 - 4);
    for (int k = 1; k < 6; ++k) {
      six += w6[k] * faces.narrow(-1, k, nc1 - 4);
    }
    auto lo = torch::minimum(psf_lo, psf_hi).narrow(-1, 2, nc1 - 4);
    auto hi = torch::maximum(psf_lo, psf_hi).narrow(-1, 2, nc1 - 4);
    auto mid = pref.narrow(-1, 2, nc1 - 4);
    mid.copy_(torch::where((six >= lo) & (six <= hi), six, mid));

    constexpr double w6e[2][6] = {
        {95. / 288., 1427. / 1440., -133. / 240., 241. / 720., -173. / 1440.,
         3. / 160.},
        {-3. / 160., 637. / 1440., 511. / 720., -43. / 240., 77. / 1440.,
         -11. / 1440.},
    };
    // At a PHYSICAL wall the stencil never crosses it.
    // A one-sided row spans six faces and so overruns the owned range at the
    // OPPOSITE end on a block under five cells -- a seam face, and readable,
    // unless that end is a wall as well. See hydro_ref_x1_impl.h.
    bool thin = (iu - il + 1) < 5;
    bool fits =
        iu >= 4;  // the rows must fit the array; see hydro_ref_x1_impl.h
    bool clamp_in = wall_clamp && phys_in && fits && !(thin && phys_out);
    bool clamp_out = wall_clamp && phys_out && fits && !(thin && phys_in);
    if (clamp_in) {
      for (int j : {il, il + 1}) {
        int sigma = j - il;
        auto val = w6e[sigma][0] * faces.select(-1, il);
        for (int m = 1; m < 6; ++m)
          val += w6e[sigma][m] * faces.select(-1, il + m);
        auto flo = torch::minimum(psf_lo.select(-1, j), psf_hi.select(-1, j));
        auto fhi = torch::maximum(psf_lo.select(-1, j), psf_hi.select(-1, j));
        auto cur = pref.select(-1, j);
        cur.copy_(
            torch::where((val >= flo) & (val <= fhi), val,
                         0.5 * (psf_lo.select(-1, j) + psf_hi.select(-1, j))));
      }
    }
    if (wall_clamp && phys_in && !clamp_in) {  // no row fits: 2-point mean
      for (int j : {il, il + 1})
        pref.select(-1, j).copy_(0.5 *
                                 (psf_lo.select(-1, j) + psf_hi.select(-1, j)));
    }
    if (wall_clamp && phys_out && !clamp_out) {
      for (int j : {iu - 1, iu})
        pref.select(-1, j).copy_(0.5 *
                                 (psf_lo.select(-1, j) + psf_hi.select(-1, j)));
    }
    if (clamp_out) {
      int s0 = iu + 1 - 5;
      for (int j : {iu - 1, iu}) {
        int row = 4 - (j - s0);
        auto val = w6e[row][5] * faces.select(-1, s0);
        for (int m = 1; m < 6; ++m)
          val += w6e[row][5 - m] * faces.select(-1, s0 + m);
        auto flo = torch::minimum(psf_lo.select(-1, j), psf_hi.select(-1, j));
        auto fhi = torch::maximum(psf_lo.select(-1, j), psf_hi.select(-1, j));
        auto cur = pref.select(-1, j);
        cur.copy_(
            torch::where((val >= flo) & (val <= fhi), val,
                         0.5 * (psf_lo.select(-1, j) + psf_hi.select(-1, j))));
      }
    }
    if (!phys_in) {
      for (int j : {0, 1}) {
        auto val = w6e[j][0] * faces.select(-1, 0);
        for (int m = 1; m < 6; ++m) val += w6e[j][m] * faces.select(-1, m);
        auto flo = torch::minimum(psf_lo.select(-1, j), psf_hi.select(-1, j));
        auto fhi = torch::maximum(psf_lo.select(-1, j), psf_hi.select(-1, j));
        auto cur = pref.select(-1, j);
        cur.copy_(torch::where((val >= flo) & (val <= fhi), val, cur));
      }
    }
    if (!phys_out) {
      for (int j : {nc1 - 2, nc1 - 1}) {
        int sigma = j - (nc1 - 5);
        auto val = w6e[4 - sigma][5] * faces.select(-1, nc1 - 5);
        for (int m = 1; m < 6; ++m) {
          val += w6e[4 - sigma][5 - m] * faces.select(-1, nc1 - 5 + m);
        }
        auto flo = torch::minimum(psf_lo.select(-1, j), psf_hi.select(-1, j));
        auto fhi = torch::maximum(psf_lo.select(-1, j), psf_hi.select(-1, j));
        auto cur = pref.select(-1, j);
        cur.copy_(torch::where((val >= flo) & (val <= fhi), val, cur));
      }
    }
  } else {
    auto ratio = psf_lo / psf_hi;
    pref.copy_(torch::where((ratio - 1.).abs() < 1e-6, 0.5 * (psf_lo + psf_hi),
                            dp / torch::log(ratio)));
  }

  auto rop = (rho / w[IPR]).clone();
  auto lo_edge = rop.narrow(-1, 0, 1);
  auto hi_edge = rop.narrow(-1, nc1 - 1, 1);
  auto pad = torch::cat({lo_edge, lo_edge, rop, hi_edge, hi_edge}, -1);
  // past a clamped wall: rho/p continued from the two cells next to it,
  // linearly where it rises towards the wall and ln-linearly where it falls,
  // else the wall cell (hydro_ref_x1_wall_rop); pad index p holds cell p - 2
  auto wall_rop = [](torch::Tensor const& r0, torch::Tensor const& r1, int k) {
    auto e = torch::where(r1 > r0, r0 * (r0 / r1).pow(double(k)),
                          r0 + double(k) * (r0 - r1));
    return torch::where(e > 0., e, r0);
  };
  if (wall_clamp && phys_in) {
    auto r0 = rop.narrow(-1, il, 1);
    for (int j = -2; j < il; ++j) {
      auto v = r0;
      if (il + 1 <= iu) {
        v = wall_rop(r0, rop.narrow(-1, il + 1, 1), il - j);
      }
      pad.narrow(-1, j + 2, 1).copy_(v);
    }
  }
  if (wall_clamp && phys_out) {
    auto r0 = rop.narrow(-1, iu, 1);
    for (int j = iu + 1; j < nc1 + 2; ++j) {
      auto v = r0;
      if (iu - 1 >= il) {
        v = wall_rop(r0, rop.narrow(-1, iu - 1, 1), j - iu);
      }
      pad.narrow(-1, j + 2, 1).copy_(v);
    }
  }
  auto rs = (pad.narrow(-1, 0, nc1) + 4. * pad.narrow(-1, 1, nc1) +
             6. * pad.narrow(-1, 2, nc1) + 4. * pad.narrow(-1, 3, nc1) +
             pad.narrow(-1, 4, nc1)) /
            16.;
  auto rf = rs.clone();
  rf.narrow(-1, 1, nc1 - 1)
      .copy_(0.5 * (rs.narrow(-1, 0, nc1 - 1) + rs.narrow(-1, 1, nc1 - 1)));
  dref.copy_(pref * rs);
  dsf.copy_(psf_lo * rf);
}

torch::Tensor hydro_ref_x1_stretched_rows(torch::Tensor const& dx1f, int iu,
                                          bool phys_in, bool phys_out,
                                          bool wall_clamp) {
  auto h = dx1f.to(torch::kCPU, torch::kFloat64).contiguous();
  int nc1 = h.size(0);
  auto hp = h.data_ptr<double>();
  auto rows = torch::zeros({nc1, 7}, torch::kFloat64);
  auto r = rows.accessor<double, 2>();
  long double const tq[3] = {0.5L - 0.1L * std::sqrt(15.0L), 0.5L,
                             0.5L + 0.1L * std::sqrt(15.0L)};
  long double const wq[3] = {5.0L / 18.0L, 8.0L / 18.0L, 5.0L / 18.0L};
  for (int i = 0; i < nc1; ++i) {
    int start =
        hydro_ref_x1_six_face_start(i, nc1, iu, phys_in, phys_out, wall_clamp);
    r[i][0] = start;
    if (start < 0) continue;
    long double t[6];
    for (int k = 0; k < 6; ++k) {
      long double o = 0.0L;
      for (int j = i; j < start + k; ++j) o += hp[j];
      for (int j = start + k; j < i; ++j) o -= hp[j];
      t[k] = o / hp[i];
    }
    for (int m = 0; m < 6; ++m) {
      long double wm = 0.0L;
      for (int q = 0; q < 3; ++q) {
        long double l = 1.0L;
        for (int k = 0; k < 6; ++k)
          if (k != m) l *= (tq[q] - t[k]) / (t[m] - t[k]);
        wm += wq[q] * l;
      }
      r[i][1 + m] = static_cast<double>(wm);
    }
  }
  return rows;
}

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_hydro_ref_x1);
REGISTER_ALL_CPU_DISPATCH(call_hydro_ref_x1, &snap::hydro_ref_x1_cpu);
REGISTER_MPS_DISPATCH(call_hydro_ref_x1, &snap::hydro_ref_x1_mps);

}  // namespace at::native
