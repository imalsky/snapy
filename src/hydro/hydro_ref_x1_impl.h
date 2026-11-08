#pragma once

// base
#include <configure.h>

// C/C++
#include <cmath>
#include <limits>

// snap
#include <snap/snap.h>

#include "hydro_dispatch.hpp"

namespace snap {

template <typename T>
inline DISPATCH_MACRO T hydro_ref_x1_face(T const* psf_lo, T const* psf_hi,
                                          int flat, int face, int nc1) {
  if (face < nc1) return psf_lo[flat + face];
  return psf_hi[flat + nc1 - 1];
}

template <typename T>
inline DISPATCH_MACRO void hydro_ref_x1_scan_impl(T const* w, T const* dx1f,
                                                  T const* anchor, T* psf_lo,
                                                  T* psf_hi, int column,
                                                  int ncolumns, int nc1, int iu,
                                                  T grav) {
  int ncells = ncolumns * nc1;
  int flat = column * nc1;
  T top_anchor;
  if (anchor) {
    top_anchor = anchor[column];
  } else {
    T rho_top = w[IDN * ncells + flat + iu];
    T pres_top = w[IPR * ncells + flat + iu];
    T rdt_top = pres_top / rho_top;
    top_anchor = pres_top * exp(-grav * T(0.5) * dx1f[iu] / rdt_top);
  }

  T face = top_anchor;
  T min_positive = std::numeric_limits<T>::min();
  for (int i = iu; i >= 0; --i) {
    T dp = grav * w[IDN * ncells + flat + i] * dx1f[i];
    T lo = face + dp;
    T hi = face;
    psf_lo[flat + i] = lo > min_positive ? lo : min_positive;
    psf_hi[flat + i] = hi > min_positive ? hi : min_positive;
    face = lo;
  }

  face = top_anchor;
  for (int i = iu + 1; i < nc1; ++i) {
    T dp = grav * w[IDN * ncells + flat + i] * dx1f[i];
    T lo = face;
    T hi = lo - dp;
    psf_lo[flat + i] = lo > min_positive ? lo : min_positive;
    psf_hi[flat + i] = hi > min_positive ? hi : min_positive;
    face = hi;
  }
}

//! rho/p k cells past a wall, from the wall cell r0 and its neighbour r1:
//! linear, r0 + k (r0 - r1), where rho/p rises towards the wall (r1 <= r0),
//! and ln-linear, r0 (r0 / r1)^k, where it falls (r1 > r0). Each is the
//! branch that stays closer to r0, so the continuation lies within
//! [r0 (r0 / r1)^k, r0 (1 + k)] for r1 >= 0: positive, and bounded above when
//! r1 is nearly empty. A non-positive value keeps the wall cell.
template <typename T>
inline DISPATCH_MACRO T hydro_ref_x1_wall_rop(T r0, T r1, T k) {
  T e = r1 > r0 ? r0 * pow(r0 / r1, k) : r0 + k * (r0 - r1);
  return e > T(0) ? e : r0;
}

template <typename T>
inline DISPATCH_MACRO T hydro_ref_x1_wall_steps(T const* dx1f, int jw, int dir,
                                                int k) {
  T d = T(0.5) * dx1f[jw];
  for (int m = 1; m <= k; ++m) {
    T hm = dx1f[jw - dir * (m - 1)];
    d += (m < k) ? hm : T(0.5) * hm;
  }
  return d / (T(0.5) * (dx1f[jw] + dx1f[jw - dir]));
}

//! rho/p smoothed by a 5-point binomial along x1: the reference must
//! track the column profile at LARGE scales only. An unsmoothed local rho/p
//! makes rho' degenerate with the pressure perturbation, so entropy/buoyancy
//! anomalies bypass the high-order reconstruction; a bottom-anchored
//! isentrope reference errs by orders of magnitude on a stratified column.
//! Past a clamped wall (ext_lo/ext_hi: the wall side owns at least two cells)
//! the stencil continues rho/p from the two cells next to the wall instead of
//! repeating the wall cell (hydro_ref_x1_wall_rop). Either continuation is
//! linear in index to O(dz^2), so the wall cells keep the interior's O(dz^2)
//! bias; a repeated wall cell is exact only for constant rho/p and leaves an
//! O(dz) error at the first faces (docs/derivations/wb-ref-wall.md). Only
//! owned cells are read.
template <typename T>
inline DISPATCH_MACRO T hydro_ref_x1_rop_smooth(
    T const* w, int ncells, int flat, int nc1, int i, int jlo, int jhi,
    bool ext_lo, bool ext_hi, T const* dx1f = nullptr, bool stretched = false) {
  auto rop = [&](int j) {
    return w[IDN * ncells + flat + j] / w[IPR * ncells + flat + j];
  };
  T v[5];
  for (int m = -2; m <= 2; ++m) {
    int j = i + m;
    if (j < jlo) {
      v[m + 2] =
          ext_lo ? hydro_ref_x1_wall_rop(rop(jlo), rop(jlo + 1),
                                         stretched ? hydro_ref_x1_wall_steps(
                                                         dx1f, jlo, -1, jlo - j)
                                                   : T(jlo - j))
                 : rop(jlo);
    } else if (j > jhi) {
      v[m + 2] =
          ext_hi ? hydro_ref_x1_wall_rop(rop(jhi), rop(jhi - 1),
                                         stretched ? hydro_ref_x1_wall_steps(
                                                         dx1f, jhi, +1, j - jhi)
                                                   : T(j - jhi))
                 : rop(jhi);
    } else {
      v[m + 2] = rop(j);
    }
  }
  return (v[0] + T(4) * v[1] + T(6) * v[2] + T(4) * v[3] + v[4]) / T(16);
}

inline DISPATCH_MACRO int hydro_ref_x1_six_face_start(int i, int nc1, int iu,
                                                      bool phys_in,
                                                      bool phys_out,
                                                      bool wall_clamp) {
  int il = nc1 - 1 - iu;
  bool thin = (iu - il + 1) < 5;
  bool fits = iu >= 4;
  bool wall_in = wall_clamp && phys_in && i >= il && i < il + 2;
  bool wall_out = wall_clamp && phys_out && i > iu - 2 && i <= iu;
  int start = -1;
  if (i >= 2 && i < nc1 - 2 && !wall_in && !wall_out) start = i - 2;
  if (wall_in && fits && !(thin && phys_out)) start = il;
  if (wall_out && fits && !(thin && phys_in)) start = iu + 1 - 5;
  if (!phys_in && i < 2) start = 0;
  if (!phys_out && i >= nc1 - 2) start = nc1 - 5;
  return start;
}

template <typename T>
inline DISPATCH_MACRO void hydro_ref_x1_cell_impl(
    T const* w, T const* dx1f, T const* psf_lo, T const* psf_hi, T* pref,
    T* dsf, T* dref, int column, int i, int ncolumns, int nc1, int iu, T grav,
    int x1_grid, bool phys_in, bool phys_out, bool wall_clamp, T const* rows) {
  bool uniform = x1_grid == kX1RefUniform;
  int ncells = ncolumns * nc1;
  int flat = column * nc1;
  int il = nc1 - 1 - iu;
  // the six-face stencil never crosses a physical wall
  // A one-sided row spans SIX faces, so on a block with fewer than five x1
  // cells it reaches one face past the owned range -- at the OPPOSITE end from
  // the wall it is protecting. That is a seam face, i.e. the neighbour's own
  // data, and reading it is right; it is only unreadable when that end is a
  // physical wall too, which means the whole column is under five cells.
  bool thin = (iu - il + 1) < 5;
  // and the row has to FIT THE ARRAY. The inner row reads faces il..il+5, the
  // outer iu-4..iu+1, and with nc1 == il + iu + 1 both fit exactly when
  // iu >= 4. Below that the outer row's start goes negative -- and
  // hydro_ref_x1_face saturates above nc1 but has no lower guard, so it would
  // read before this column. nx1 >= 5 always satisfies it; a block thin enough
  // to fail it has fewer cells than it has ghosts.
  bool fits = iu >= 4;
  bool wall_in = wall_clamp && phys_in && i >= il && i < il + 2;
  bool wall_out = wall_clamp && phys_out && i > iu - 2 && i <= iu;
  bool clamp_in = wall_in && fits && !(thin && phys_out);
  bool clamp_out = wall_out && fits && !(thin && phys_in);
  T lo = psf_lo[flat + i];
  T hi = psf_hi[flat + i];
  T cell_pref = T(0.5) * (lo + hi);

  if (uniform) {
    constexpr double w6[6] = {11. / 1440., -31. / 480., 401. / 720.,
                              401. / 720., -31. / 480., 11. / 1440.};
    T lower = lo < hi ? lo : hi;
    T upper = lo > hi ? lo : hi;
    auto use_six_faces = [&](int start, double const* weights,
                             bool reverse = false) {
      T value = T(0);
      for (int m = 0; m < 6; ++m) {
        value += T(weights[reverse ? 5 - m : m]) *
                 hydro_ref_x1_face(psf_lo, psf_hi, flat, start + m, nc1);
      }
      if (value >= lower && value <= upper) cell_pref = value;
    };
    if (i >= 2 && i < nc1 - 2 && !wall_in && !wall_out) {
      use_six_faces(i - 2, w6);
    }

    constexpr double w6e[2][6] = {
        {95. / 288., 1427. / 1440., -133. / 240., 241. / 720., -173. / 1440.,
         3. / 160.},
        {-3. / 160., 637. / 1440., 511. / 720., -43. / 240., 77. / 1440.,
         -11. / 1440.},
    };
    if (clamp_in) {
      int sigma = i - il;
      use_six_faces(il, w6e[sigma]);
    }
    if (clamp_out) {
      int s = iu + 1 - 5;
      int row = 4 - (i - s);
      use_six_faces(s, w6e[row], true);
    }
    if (!phys_in && i < 2) {
      use_six_faces(0, w6e[i]);
    }
    if (!phys_out && i >= nc1 - 2) {
      int sigma = i - (nc1 - 5);
      int row = 4 - sigma;
      use_six_faces(nc1 - 5, w6e[row], true);
    }
  } else if (x1_grid == kX1RefStretched) {
    T lower = lo < hi ? lo : hi;
    T upper = lo > hi ? lo : hi;
    int start = static_cast<int>(rows[7 * i]);
    if (start >= 0) {
      T value = T(0);
      for (int m = 0; m < 6; ++m) {
        value += rows[7 * i + 1 + m] *
                 hydro_ref_x1_face(psf_lo, psf_hi, flat, start + m, nc1);
      }
      if (value >= lower && value <= upper) cell_pref = value;
    }
  } else {
    T dp = grav * w[IDN * ncells + flat + i] * dx1f[i];
    T ratio = lo / hi;
    cell_pref =
        fabs(ratio - T(1)) < T(1.e-6) ? T(0.5) * (lo + hi) : dp / log(ratio);
  }

  pref[flat + i] = cell_pref;
  int jlo = (wall_clamp && phys_in) ? il : 0;
  int jhi = (wall_clamp && phys_out) ? iu : nc1 - 1;
  bool ext_lo = wall_clamp && phys_in && il + 1 <= iu;
  bool ext_hi = wall_clamp && phys_out && iu - 1 >= il;
  bool stretched = x1_grid == kX1RefStretched;
  T rs = hydro_ref_x1_rop_smooth(w, ncells, flat, nc1, i, jlo, jhi, ext_lo,
                                 ext_hi, dx1f, stretched);
  T rf = rs;
  if (i > 0) {
    T rs_dn = hydro_ref_x1_rop_smooth(w, ncells, flat, nc1, i - 1, jlo, jhi,
                                      ext_lo, ext_hi, dx1f, stretched);
    rf = stretched
             ? (dx1f[i] * rs_dn + dx1f[i - 1] * rs) / (dx1f[i - 1] + dx1f[i])
             : T(0.5) * (rs_dn + rs);
  }
  dref[flat + i] = cell_pref * rs;
  dsf[flat + i] = lo * rf;
}

template <typename T>
inline DISPATCH_MACRO void hydro_ref_x1_impl(
    T const* w, T const* dx1f, T const* anchor, T* psf_lo, T* psf_hi, T* pref,
    T* dsf, T* dref, int column, int ncolumns, int nc1, int iu, T grav,
    int x1_grid, bool phys_in, bool phys_out, bool wall_clamp, T const* rows) {
  hydro_ref_x1_scan_impl(w, dx1f, anchor, psf_lo, psf_hi, column, ncolumns, nc1,
                         iu, grav);
  for (int i = 0; i < nc1; ++i) {
    hydro_ref_x1_cell_impl(w, dx1f, psf_lo, psf_hi, pref, dsf, dref, column, i,
                           ncolumns, nc1, iu, grav, x1_grid, phys_in, phys_out,
                           wall_clamp, rows);
  }
}

}  // namespace snap
