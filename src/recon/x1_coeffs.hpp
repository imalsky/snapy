#pragma once

#include <ATen/TensorIterator.h>
#include <torch/torch.h>

#include <utility>

#include "x1_coeffs_device.h"

namespace snap {

std::pair<torch::Tensor, torch::Tensor> weno5_x1_coeffs(
    torch::Tensor const& x1f);

std::pair<torch::Tensor, torch::Tensor> cp5_x1_coeffs(torch::Tensor const& x1f);

void x1_reset_tables(torch::nn::Module& m, torch::Tensor& cm_x1,
                     torch::Tensor& cp_x1, int ncoeff);

template <typename Self, typename Builder>
void x1_set_coeffs(Self& self, torch::Tensor const& x1f, Builder build) {
  auto rows = build(x1f);

  auto opts = self.cm.options();
  auto cm_rows = rows.first.to(opts).contiguous();
  auto cp_rows = rows.second.to(opts).contiguous();
  if (self.named_buffers(/*recurse=*/false).contains("cm_x1")) {
    self.cm_x1.set_data(cm_rows);
    self.cp_x1.set_data(cp_rows);
  } else {
    self.cm_x1 = self.register_buffer("cm_x1", cm_rows);
    self.cp_x1 = self.register_buffer("cp_x1", cp_rows);
  }

  self.cm_x1_cast = torch::Tensor();
  self.cp_x1_cast = torch::Tensor();
}

torch::Tensor const& x1_coeffs_for(torch::Tensor const& src,
                                   torch::Tensor& cache,
                                   torch::Tensor const& w);

inline at::TensorIterator x1_uniform_iterator(torch::Tensor const& w, int dim,
                                              torch::Tensor const& out) {
  std::vector<int64_t> squash_dim = {0};
  if (w.device().is_cuda()) squash_dim.push_back(dim);
  return at::TensorIteratorConfig()
      .resize_outputs(false)
      .check_all_same_dtype(true)
      .declare_static_shape(out.sizes(), squash_dim)
      .add_output(out)
      .add_input(w)
      .build();
}

torch::Tensor x1_coeffs_row_view(torch::Tensor const& c,
                                 torch::Tensor const& out, int dim);

at::TensorIterator x1_nu_iterator(torch::Tensor const& w, int dim,
                                  torch::Tensor const& out,
                                  torch::Tensor const& c);

void x1_coeffs_check_rows(at::TensorIterator const& iter,
                          torch::Tensor const& c, int dim, int ncoeff,
                          char const* what);

template <typename Call>
void x1_forward_nu(torch::Tensor const& w, int dim, torch::Tensor const& out,
                   torch::Tensor const& src, torch::Tensor& cache,
                   char const* what, Call call) {
  TORCH_CHECK(dim == out.dim() - 1, "[", what,
              "] route A covers the x1 dimension only, so a module holding "
              "per-face x1 coefficients must be given the last dimension of "
              "a (nvar, nc3, nc2, nc1) state; got dim ",
              dim, " of a ", out.dim(), "-dimensional output (note Sec. 4.7)");

  TORCH_CHECK(out.size(dim) == src.size(0), "[", what,
              "] per-face x1 coefficients are installed but dimension ", dim,
              " has ", out.size(dim), " outputs while the table has ",
              src.size(0),
              " rows; route A covers the x1 dimension only, so `recon23` must "
              "not be given a face vector (note Sec. 4.7)");

  auto const& c = x1_coeffs_for(src, cache, w);
  auto iter = x1_nu_iterator(w, dim, out, c);
  call(iter, c);
}

}  // namespace snap
