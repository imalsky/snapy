#include "x1_coeffs.hpp"

#include <ATen/native/ReduceOpsUtils.h>

#include <cmath>

namespace snap {
namespace {

void primitive_row(double const* z, int n, double* w) {
  double lp[6];
  for (int j = 0; j <= n; ++j) {
    double s = 0.;
    for (int i = 0; i <= n; ++i) {
      if (i == j) continue;
      double term = 1. / (z[j] - z[i]);
      for (int l = 0; l <= n; ++l) {
        if (l == j || l == i) continue;
        term *= (0. - z[l]) / (z[j] - z[l]);
      }
      s += term;
    }
    lp[j] = s;
  }

  for (int k = 0; k < n; ++k) {
    double acc = 0.;
    for (int j = k + 1; j <= n; ++j) acc += lp[j];
    w[k] = (z[k + 1] - z[k]) * acc;
  }
}

void quad_inv_rows(double const* z, double* r1, double* r2) {
  double V[3][3];
  for (int k = 0; k < 3; ++k) {
    double p = z[k], q = z[k + 1];
    V[k][0] = 1.;
    V[k][1] = .5 * (p + q);
    V[k][2] = (p * p + p * q + q * q) / 3.;
  }

  double cof[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
      int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
      cof[i][j] = V[i1][j1] * V[i2][j2] - V[i1][j2] * V[i2][j1];
    }
  }
  double det = V[0][0] * cof[0][0] + V[0][1] * cof[0][1] + V[0][2] * cof[0][2];
  TORCH_CHECK(det != 0., "[x1_coeffs] degenerate WENO5 stencil");

  for (int j = 0; j < 3; ++j) {
    r1[j] = cof[j][1] / det;
    r2[j] = cof[j][2] / det;
  }
}

void ptype_row(double const* y, double* row) {
  double h2 = y[3] - y[2];
  double z[6];
  for (int k = 0; k < 6; ++k) z[k] = (y[k] - y[3]) / h2;

  const int k0[3] = {2, 1, 0};
  const double bsign[3] = {1., 1., -1.};

  for (int i = 0; i < 45; ++i) row[i] = 0.;

  for (int r = 0; r < 3; ++r) {
    double const* nodes = z + k0[r];

    double w[3];
    primitive_row(nodes, 3, w);

    double r1[3], r2[3];
    quad_inv_rows(nodes, r1, r2);

    for (int n = 0; n < 3; ++n) {
      int col = k0[r] + n;
      row[r * 5 + col] = w[n];
      row[(3 + 2 * r) * 5 + col] = 2. * r2[n];
      row[(4 + 2 * r) * 5 + col] = bsign[r] * (2. * r2[n] - 2. * r1[n]);
    }
  }

  double a = -z[0], b = -z[1], c = z[4], e = z[5];
  row[45] = a * b / ((a + e) * (b + e));
  row[46] = a * e * (a + b + c + e) / ((a + e) * (b + e) * (a + c));
  row[47] = c * e / ((a + c) * (a + e));
}

void window_weno5(double const* y, double* cm, double* cp) {
  ptype_row(y, cp);

  double ym[6];
  for (int k = 0; k < 6; ++k) ym[k] = -y[5 - k];

  double tmp[kWeno5NuCoeff];
  ptype_row(ym, tmp);
  for (int r = 0; r < 9; ++r) {
    for (int n = 0; n < 5; ++n) cm[r * 5 + n] = tmp[r * 5 + (4 - n)];
  }
  cm[45] = tmp[45];
  cm[46] = tmp[46];
  cm[47] = tmp[47];
}

void combine_optimal(double const* row48, double* out5) {
  for (int n = 0; n < 5; ++n) {
    out5[n] = row48[45] * row48[n] + row48[46] * row48[5 + n] +
              row48[47] * row48[10 + n];
  }
}

void window_cp5(double const* y, double* cm, double* cp) {
  double rm[kWeno5NuCoeff], rp[kWeno5NuCoeff];
  window_weno5(y, rm, rp);
  combine_optimal(rm, cm);
  combine_optimal(rp, cp);
}

torch::Tensor check_faces(torch::Tensor const& x1f) {
  TORCH_CHECK(x1f.defined(), "[x1_coeffs] face tensor is undefined");
  TORCH_CHECK(x1f.dim() == 1, "[x1_coeffs] expected a 1-D face vector, got ",
              x1f.dim(), " dimensions");
  TORCH_CHECK(x1f.numel() >= 6,
              "[x1_coeffs] a WENO5/cp5 window needs at least 6 faces, got ",
              x1f.numel());

  auto f = x1f.to(torch::kCPU, torch::kFloat64).contiguous();
  TORCH_CHECK(torch::isfinite(f).all().item<bool>(),
              "[x1_coeffs] face vector has a non-finite entry");

  auto d = f.slice(0, 1) - f.slice(0, 0, f.numel() - 1);
  TORCH_CHECK(d.min().item<double>() > 0.,
              "[x1_coeffs] face vector must be strictly increasing");
  return f;
}

template <int NCoeff, void (*Window)(double const*, double*, double*)>
std::pair<torch::Tensor, torch::Tensor> build_x1_coeffs(
    torch::Tensor const& x1f) {
  auto f = check_faces(x1f);
  int64_t nout = f.numel() - 5;
  auto cm = torch::empty({nout, NCoeff}, torch::kFloat64);
  auto cp = torch::empty({nout, NCoeff}, torch::kFloat64);
  auto const* y = f.data_ptr<double>();
  auto* pm = cm.data_ptr<double>();
  auto* pp = cp.data_ptr<double>();
  for (int64_t m = 0; m < nout; ++m)
    Window(y + m, pm + m * NCoeff, pp + m * NCoeff);
  return {cm, cp};
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> weno5_x1_coeffs(
    torch::Tensor const& x1f) {
  return build_x1_coeffs<kWeno5NuCoeff, window_weno5>(x1f);
}

std::pair<torch::Tensor, torch::Tensor> cp5_x1_coeffs(
    torch::Tensor const& x1f) {
  return build_x1_coeffs<kPoly5NuCoeff, window_cp5>(x1f);
}

void x1_reset_tables(torch::nn::Module& m, torch::Tensor& cm_x1,
                     torch::Tensor& cp_x1, int ncoeff) {
  bool installed = cm_x1.defined() && cm_x1.numel() > 0;

  cm_x1 = torch::zeros({0, ncoeff}, torch::kFloat64);
  cp_x1 = torch::zeros({0, ncoeff}, torch::kFloat64);

  if (installed) {
    cm_x1 = m.register_buffer("cm_x1", cm_x1);
    cp_x1 = m.register_buffer("cp_x1", cp_x1);
  }
}

torch::Tensor const& x1_coeffs_for(torch::Tensor const& src,
                                   torch::Tensor& cache,
                                   torch::Tensor const& w) {
  if (src.scalar_type() == w.scalar_type() && src.device() == w.device()) {
    return src;
  }
  if (!cache.defined() || cache.scalar_type() != w.scalar_type() ||
      cache.device() != w.device() || cache.sizes() != src.sizes()) {
    cache = src.to(w.device(), w.scalar_type()).contiguous();
  }
  return cache;
}

torch::Tensor x1_coeffs_row_view(torch::Tensor const& c,
                                 torch::Tensor const& out, int dim) {
  TORCH_CHECK(c.dim() == 2 && c.is_contiguous(),
              "[x1_coeffs] row view needs a contiguous 2-D table");
  TORCH_CHECK(dim >= 0 && dim < out.dim(), "[x1_coeffs] bad dim ", dim);
  TORCH_CHECK(c.size(0) == out.size(dim), "[x1_coeffs] table has ", c.size(0),
              " rows but the output has ", out.size(dim), " faces along dim ",
              dim);

  auto sizes = out.sizes().vec();
  std::vector<int64_t> strides(sizes.size(), 0);
  strides[dim] = c.size(1);
  return c.as_strided(sizes, strides, c.storage_offset());
}

at::TensorIterator x1_nu_iterator(torch::Tensor const& w, int dim,
                                  torch::Tensor const& out,
                                  torch::Tensor const& c) {
  TORCH_CHECK(dim >= 0 && dim < out.dim(), "[x1_coeffs] bad dim ", dim);
  TORCH_CHECK(c.scalar_type() == w.scalar_type() && c.device() == w.device(),
              "[x1_coeffs] coefficient table is ", c.scalar_type(), " on ",
              c.device(), " but the state is ", w.scalar_type(), " on ",
              w.device());

  std::vector<int64_t> squash_dim = {0};
  if (w.device().is_cuda()) {
    squash_dim.push_back(dim);
  }

  auto config = at::TensorIteratorConfig();
  config.resize_outputs(false)
      .check_all_same_dtype(true)
      .declare_static_shape(out.sizes(), squash_dim)
      .add_output(out)
      .add_input(w);

  torch::Tensor rows;
  if (!w.device().is_cuda()) {
    rows = x1_coeffs_row_view(c, out, dim);
    config.add_input(rows);
  }

  return config.build();
}

void x1_coeffs_check_rows(at::TensorIterator const& iter,
                          torch::Tensor const& c, int dim, int ncoeff,
                          char const* what) {
  TORCH_CHECK(c.dim() == 2 && c.size(1) == ncoeff, "[", what,
              "] expected a (nout, ", ncoeff, ") coefficient table, got ",
              c.sizes());
  TORCH_CHECK(c.is_contiguous(), "[", what,
              "] coefficient table must be contiguous");
  int64_t nout = at::native::ensure_nonempty_size(iter.output(), dim);
  TORCH_CHECK(c.size(0) == nout, "[", what, "] coefficient table has ",
              c.size(0), " rows but the output has ", nout,
              " faces along dimension ", dim);
}

}  // namespace snap
