#pragma once

// C/C++
#include <functional>
#include <string>
#include <unordered_map>

// torch
#include <torch/torch.h>

// snap
#include "bc.hpp"

using bcfunc_t =
    std::function<void(torch::Tensor const&, int, snap::BoundaryFuncOptions)>;

inline std::unordered_map<std::string, bcfunc_t>& get_bc_func() {
  static std::unordered_map<std::string, bcfunc_t> bcmap;
  return bcmap;
}

struct BCRegistrar {
  BCRegistrar(const std::string& name, bcfunc_t func) {
    get_bc_func()[name] = func;
  }
};

// A face function may be called again on a tangential ghost slab after
// exchange: the orthogonal size is then nghost, not the block. It must not
// depend on that shape, and a second call must match the first (write ghosts
// from the interior; do not accumulate) (#264).
#define BC_FUNCTION(name, var, dim, op)                            \
  void name(torch::Tensor const&, int, snap::BoundaryFuncOptions); \
  static BCRegistrar bc_##name(#name, name);                       \
  void name(torch::Tensor const& var, int dim, snap::BoundaryFuncOptions op)

// Face classification also works for programmatically selected built-in
// callbacks.
bool is_outflow(bcfunc_t const& func);
bool is_reflecting(bcfunc_t const& func);
