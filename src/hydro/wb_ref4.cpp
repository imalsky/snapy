// C/C++
#include <algorithm>
#include <array>
#include <cctype>
#include <vector>

// snap
#include <snap/snap.h>

#include <snap/coord/x1_centroid.hpp>
#include <snap/layout/layout.hpp>

#include "wb_ref4.hpp"

namespace snap {

namespace {

// relative margin of the range guards: on a flat profile the two bounds
// coincide, and an exact test would flip on round-off between a split and a
// one-block column
constexpr double kMargin = 1.e-10;
// |ln(p_lo/p_hi)| above this: fewer than two cells per pressure scale height
constexpr double kMaxLogDrop = 0.5;

// d/dx of the Lagrange basis polynomial j through nodes X, at x
double lagrange_deriv(std::vector<double> const& X, int j, double x) {
  double sum = 0.;
  int n = X.size();
  for (int l = 0; l < n; ++l) {
    if (l == j) continue;
    double prod = 1. / (X[j] - X[l]);
    for (int m = 0; m < n; ++m) {
      if (m == j || m == l) continue;
      prod *= (x - X[m]) / (X[j] - X[m]);
    }
    sum += prod;
  }
  return sum;
}

torch::Tensor to_index(std::vector<std::vector<int64_t>> const& rows, int k) {
  auto t = torch::zeros({(int64_t)rows.size(), k}, torch::kInt64);
  auto a = t.accessor<int64_t, 2>();
  for (size_t i = 0; i < rows.size(); ++i)
    for (size_t j = 0; j < rows[i].size(); ++j) a[i][j] = rows[i][j];
  return t;
}

torch::Tensor to_weight(std::vector<std::vector<double>> const& rows, int k) {
  auto t = torch::zeros({(int64_t)rows.size(), k}, torch::kFloat64);
  auto a = t.accessor<double, 2>();
  for (size_t i = 0; i < rows.size(); ++i)
    for (size_t j = 0; j < rows[i].size(); ++j) a[i][j] = rows[i][j];
  return t;
}

// sum_k v[..., idx[n, k]] * wt[n, k] over k: (..., n)
torch::Tensor apply(torch::Tensor const& v, torch::Tensor const& idx,
                    torch::Tensor const& wt) {
  auto sizes = v.sizes().vec();
  sizes.back() = idx.size(0);
  sizes.push_back(idx.size(1));
  return (v.index_select(-1, idx.flatten()).view(sizes) * wt).sum(-1);
}

torch::Tensor in_range(torch::Tensor const& v, torch::Tensor const& lo,
                       torch::Tensor const& hi) {
  return (v >= lo - kMargin * lo.abs()) & (v <= hi + kMargin * hi.abs());
}

}  // namespace

bool wb_ref4_enabled() {
  // read once, like the flux-covariance switch: every block must build the
  // same reference or the x1 seam faces stop agreeing
  static const bool on = [] {
    auto v = get_env("SNAP_WB_REF4", "0");
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return !(v.empty() || v == "0" || v == "false" || v == "off" || v == "no");
  }();
  // SNAP_X1_CENTROID_EXACT implies it: one predicate for the solver,
  // balance_column and the tests
  return on || x1_centroid_exact_enabled();
}

WbRef4Stencils wb_ref4_stencils(torch::Tensor const& x1f, int is, int iu,
                                bool clamp_in, bool clamp_out,
                                torch::TensorOptions const& options) {
  WbRef4Stencils st;
  auto xt = x1f.to(torch::kCPU, torch::kFloat64).contiguous();
  int nc1 = xt.size(0) - 1;
  std::vector<double> xf(xt.data_ptr<double>(),
                         xt.data_ptr<double>() + nc1 + 1);
  int nown = iu - is + 1;
  st.is = is;
  st.iu = iu;
  st.nc1 = nc1;
  st.clamp_in = clamp_in;
  st.clamp_out = clamp_out;
  // the clamped stencils need four owned cells next to the wall; a block
  // between two seams reads its (exchanged) ghost cells instead
  st.usable = (nown >= 4 || !(clamp_in || clamp_out)) && nc1 >= 5;
  if (!st.usable) return st;
  auto clampi = [](int j, int lo, int hi) {
    return std::max(lo, std::min(hi, j));
  };

  // 1. filter rows: F in cell-index units; past a clamped wall the two
  //    values are cubic extrapolations of the four owned cells nearest it
  //    (exact for a cubic), elsewhere the array's own values, and past the
  //    array the edge value
  constexpr std::array<double, 5> F = {-1. / 16., 4. / 16., 10. / 16., 4. / 16.,
                                       -1. / 16.};
  constexpr std::array<std::array<double, 4>, 2> E = {
      {{4., -6., 4., -1.}, {10., -20., 15., -4.}}};
  std::vector<std::vector<int64_t>> fi, ni;
  std::vector<std::vector<double>> fw;
  int kf = 0;
  for (int i = is; i <= iu; ++i) {
    std::vector<double> row(nc1, 0.);
    for (int m = -2; m <= 2; ++m) {
      int j = i + m;
      double wm = F[m + 2];
      if (clamp_in && j < is) {
        for (int q = 0; q < 4; ++q) row[is + q] += wm * E[is - j - 1][q];
      } else if (clamp_out && j > iu) {
        for (int q = 0; q < 4; ++q) row[iu - q] += wm * E[j - iu - 1][q];
      } else {
        row[clampi(j, 0, nc1 - 1)] += wm;
      }
    }
    std::vector<int64_t> c;
    std::vector<double> v;
    for (int j = 0; j < nc1; ++j) {
      if (row[j] != 0.) {
        c.push_back(j);
        v.push_back(row[j]);
      }
    }
    kf = std::max(kf, (int)c.size());
    fi.push_back(c);
    fw.push_back(v);
    // range-guard neighbours, inside the owned cells at a clamped wall
    int lo = clamp_in ? is : 0, hi = clamp_out ? iu : nc1 - 1;
    ni.push_back({clampi(i - 1, lo, hi), i, clampi(i + 1, lo, hi)});
  }
  for (auto& c : fi) c.resize(kf, c.empty() ? 0 : c.back());
  for (auto& v : fw) v.resize(kf, 0.);
  st.fidx = to_index(fi, kf).to(options.device());
  st.fwt = to_weight(fw, kf).to(options);
  st.nidx = to_index(ni, 3).to(options.device());

  // 2. face density at faces is..iu+1: P'(x_f), P the quartic through the
  //    primitive P(x_{s+j}) = sum_{k<j} d_{s+k} dz_{s+k}, j = 0..4, window
  //    s = f-2 kept inside the owned cells at a clamped wall. The weight of
  //    cell s+k is dz_{s+k} sum_{j>k} L_j'(x_f). Uniform interior:
  //    (-1, 7, 7, -1)/12.
  std::vector<std::vector<int64_t>> ai;
  std::vector<std::vector<double>> aw;
  for (int f = is; f <= iu + 1; ++f) {
    int s = f - 2;
    if (clamp_in) s = std::max(s, is);
    if (clamp_out) s = std::min(s, iu - 3);
    s = clampi(s, 0, nc1 - 4);
    double h = xf[f + (f < nc1 ? 1 : 0)] - xf[f - (f < nc1 ? 0 : 1)];
    std::vector<double> X(5);
    for (int j = 0; j < 5; ++j) X[j] = (xf[s + j] - xf[f]) / h;
    std::vector<int64_t> idx(4);
    std::vector<double> wt(4, 0.);
    for (int k = 0; k < 4; ++k) {
      idx[k] = s + k;
      double dz = (xf[s + k + 1] - xf[s + k]) / h;
      for (int j = k + 1; j < 5; ++j) wt[k] += dz * lagrange_deriv(X, j, 0.);
    }
    ai.push_back(idx);
    aw.push_back(wt);
  }
  st.aidx = to_index(ai, 4).to(options.device());
  st.awt = to_weight(aw, 4).to(options);

  // 4. ghost cells past a clamped wall do not count for the resolution flag:
  //    the stencils there never read them, and the scan pressure falls
  //    steeply past a top wall
  auto counted = torch::ones({nc1}, torch::kBool);
  if (clamp_in) counted.narrow(0, 0, is).fill_(false);
  if (clamp_out) counted.narrow(0, iu + 1, nc1 - iu - 1).fill_(false);
  st.counted = counted.to(options.device());
  return st;
}

torch::Tensor wb_ref4_cells(WbRef4Stencils const& st, torch::Tensor const& w,
                            torch::Tensor const& psf_lo,
                            torch::Tensor const& psf_hi, torch::Tensor pref,
                            torch::Tensor dref) {
  if (!st.usable) return torch::ones_like(psf_lo, torch::kBool);
  // resolution flag, dilated by two cells
  auto bad = ((psf_lo / psf_hi).log().abs() > kMaxLogDrop) & st.counted;
  auto sizes = bad.sizes();
  auto flag =
      torch::max_pool1d(bad.to(w.dtype()).reshape({-1, 1, st.nc1}), 5, 1, 2)
          .reshape(sizes) > 0.;

  int nown = st.iu - st.is + 1;
  auto fl = flag.narrow(-1, st.is, nown);
  auto d_old = dref.narrow(-1, st.is, nown).clone();

  auto p_new = pref.narrow(-1, st.is, nown);

  // dref = pref * F(rho/p); outside the range of rho/p over cells i-1..i+1
  // the kernel's smoothed ratio stays
  auto r = w[IDN] / w[IPR];
  auto fr = apply(r, st.fidx, st.fwt);
  auto nb = r.index_select(-1, st.nidx.flatten())
                .view({r.size(0), r.size(1), nown, 3});
  auto ok = in_range(fr, std::get<0>(nb.min(-1)), std::get<0>(nb.max(-1)));
  dref.narrow(-1, st.is, nown)
      .copy_(torch::where(fl, d_old, torch::where(ok, p_new * fr, d_old)));
  return flag;
}

void wb_ref4_faces(WbRef4Stencils const& st, torch::Tensor const& dref,
                   torch::Tensor dsf, torch::Tensor const& flag) {
  if (!st.usable) return;
  int nf = st.iu - st.is + 2;
  auto face = apply(dref, st.aidx, st.awt);
  auto dl = dref.narrow(-1, st.is - 1, nf), dr = dref.narrow(-1, st.is, nf);
  auto ok = in_range(face, torch::minimum(dl, dr), torch::maximum(dl, dr)) &
            ~flag.narrow(-1, st.is - 1, nf) & ~flag.narrow(-1, st.is, nf);
  auto d = dsf.narrow(-1, st.is, nf);
  d.copy_(torch::where(ok, face, d));
}

}  // namespace snap
