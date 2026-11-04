// C/C++
#include <cstdlib>
#include <string>

// snap
#include "scalar_fused.hpp"

namespace snap {

bool scalar_fused_enabled() {
  static bool const on = [] {
    char const* e = std::getenv("SNAP_SCALAR_FUSED");
    return e != nullptr && std::string(e) == "1";
  }();
  return on;
}

namespace {
void scalar_lateral_flux_unsupported(torch::Tensor&, torch::Tensor const&,
                                     torch::Tensor const&, torch::Tensor const&,
                                     std::vector<torch::Tensor> const&) {
  TORCH_CHECK(false, "the fused scalar path runs on CUDA devices only");
}
void scalar_theta_unsupported(torch::Tensor&, torch::Tensor const&,
                              torch::Tensor const&, torch::Tensor const&,
                              torch::Tensor const&, torch::Tensor const&,
                              torch::Tensor const&, torch::Tensor const&,
                              torch::Tensor const&, double, double,
                              at::native::ScalarBox) {
  TORCH_CHECK(false, "the fused scalar path runs on CUDA devices only");
}
void scalar_theta_scale_unsupported(torch::Tensor const&, torch::Tensor&,
                                    torch::Tensor&, torch::Tensor&,
                                    at::native::ScalarBox) {
  TORCH_CHECK(false, "the fused scalar path runs on CUDA devices only");
}
void scalar_divergence_unsupported(torch::Tensor&, torch::Tensor&,
                                   torch::Tensor const&, torch::Tensor const&,
                                   torch::Tensor const&, torch::Tensor const&,
                                   torch::Tensor const&, torch::Tensor const&,
                                   torch::Tensor const&, double,
                                   at::native::ScalarBox) {
  TORCH_CHECK(false, "the fused scalar path runs on CUDA devices only");
}
void scalar_weno5_unsupported(torch::Tensor&, torch::Tensor const&,
                              torch::Tensor const&, torch::Tensor const&, bool,
                              int, torch::Tensor const&) {
  TORCH_CHECK(false, "the fused scalar path runs on CUDA devices only");
}
void scalar_implicit_transfer_unsupported(torch::Tensor&, torch::Tensor const&,
                                          torch::Tensor const&,
                                          torch::Tensor const&) {
  TORCH_CHECK(false, "the fused scalar path runs on CUDA devices only");
}
void scalar_average3_unsupported(torch::Tensor&, torch::Tensor const&,
                                 torch::Tensor const&, torch::Tensor const&,
                                 double, double, double) {
  TORCH_CHECK(false, "the fused scalar path runs on CUDA devices only");
}
}  // namespace

}  // namespace snap

namespace at::native {

DEFINE_DISPATCH(call_scalar_lateral_flux);
REGISTER_ALL_CPU_DISPATCH(call_scalar_lateral_flux,
                          &snap::scalar_lateral_flux_unsupported);
REGISTER_MPS_DISPATCH(call_scalar_lateral_flux,
                      &snap::scalar_lateral_flux_unsupported);
DEFINE_DISPATCH(call_scalar_weno5);
REGISTER_ALL_CPU_DISPATCH(call_scalar_weno5, &snap::scalar_weno5_unsupported);
REGISTER_MPS_DISPATCH(call_scalar_weno5, &snap::scalar_weno5_unsupported);
DEFINE_DISPATCH(call_scalar_implicit_transfer);
REGISTER_ALL_CPU_DISPATCH(call_scalar_implicit_transfer,
                          &snap::scalar_implicit_transfer_unsupported);
REGISTER_MPS_DISPATCH(call_scalar_implicit_transfer,
                      &snap::scalar_implicit_transfer_unsupported);
DEFINE_DISPATCH(call_scalar_average3);
REGISTER_ALL_CPU_DISPATCH(call_scalar_average3,
                          &snap::scalar_average3_unsupported);
REGISTER_MPS_DISPATCH(call_scalar_average3, &snap::scalar_average3_unsupported);
DEFINE_DISPATCH(call_scalar_theta);
REGISTER_ALL_CPU_DISPATCH(call_scalar_theta, &snap::scalar_theta_unsupported);
REGISTER_MPS_DISPATCH(call_scalar_theta, &snap::scalar_theta_unsupported);
DEFINE_DISPATCH(call_scalar_theta_scale);
REGISTER_ALL_CPU_DISPATCH(call_scalar_theta_scale,
                          &snap::scalar_theta_scale_unsupported);
REGISTER_MPS_DISPATCH(call_scalar_theta_scale,
                      &snap::scalar_theta_scale_unsupported);
DEFINE_DISPATCH(call_scalar_divergence);
REGISTER_ALL_CPU_DISPATCH(call_scalar_divergence,
                          &snap::scalar_divergence_unsupported);
REGISTER_MPS_DISPATCH(call_scalar_divergence,
                      &snap::scalar_divergence_unsupported);

}  // namespace at::native
