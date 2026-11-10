// CUDA launches one thread block per reconstruction line. 1024 cells is the
// largest line that launch can hold. 1025 is the smallest that must take
// the tiled path. WENO5, WENO3, and polynomial degrees 3 and 5 share it.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <exception>
#include <snap/recon/interpolation.hpp>
#include <sstream>
#include <vector>

#include "cuda_test_gate.hpp"

using namespace snap;

namespace {

enum class Kind { Weno5, Weno3, Cp5, Cp3 };

char const* kind_name(Kind kind) {
  switch (kind) {
    case Kind::Weno5:
      return "weno5";
    case Kind::Weno3:
      return "weno3";
    case Kind::Cp5:
      return "cp5";
    case Kind::Cp3:
      return "cp3";
  }
  return "unknown";
}

int overhang(Kind kind) {
  switch (kind) {
    case Kind::Weno5:
    case Kind::Cp5:
      return 4;
    case Kind::Weno3:
    case Kind::Cp3:
      return 2;
  }
  return 0;
}

// left or right faces of w along dim; with x1f (the x1 faces of w's cells,
// ghosts included) the interpolator takes its stretched-x1 rows.
torch::Tensor recon(Kind kind, torch::Tensor w, int dim, torch::Device device,
                    bool left = true, torch::Tensor x1f = {}) {
  auto out_sizes = w.sizes().vec();
  out_sizes[dim] -= overhang(kind);
  auto out = torch::empty(out_sizes, w.options().device(device));
  auto input = w.to(device);
  auto run = [&](auto& interp) {
    interp->to(device, w.scalar_type());
    if (x1f.defined()) {
      interp->set_x1_coeffs(x1f);
    }
    if (left) {
      interp->left(input, dim, out);
    } else {
      interp->right(input, dim, out);
    }
  };
  if (kind == Kind::Weno5) {
    Weno5Interp interp;
    run(interp);
  } else if (kind == Kind::Weno3) {
    Weno3Interp interp;
    run(interp);
  } else if (kind == Kind::Cp5) {
    Center5Interp interp;
    run(interp);
  } else {
    Center3Interp interp;
    run(interp);
  }
  if (device.is_cuda()) {
    torch::cuda::synchronize();
  }
  return out;
}

torch::Tensor make_line(int nvar, int n, torch::TensorOptions opts) {
  auto x = torch::arange(n, opts);
  auto w = torch::empty({nvar, 1, 1, n}, opts);
  for (int v = 0; v < nvar; ++v) {
    w[v][0][0] = torch::sin((v + 1) * x / n);
  }
  return w;
}

void expect_matches_cpu(Kind kind, torch::Tensor w, int dim) {
  torch::Tensor cpu, gpu;
  ASSERT_NO_THROW(cpu = recon(kind, w, dim, torch::kCPU));
  bool launched = true;
  try {
    gpu = recon(kind, w, dim, torch::kCUDA);
  } catch (std::exception const& err) {
    launched = false;
    ADD_FAILURE() << kind_name(kind) << " dim=" << dim << " shape=" << w.sizes()
                  << " CUDA failed: " << err.what();
  }
  if (!launched) {
    return;
  }
  auto diff = (gpu.cpu() - cpu).abs().max().item<double>();
  EXPECT_LT(diff, 1e-12) << kind_name(kind) << " dim=" << dim
                         << " shape=" << w.sizes() << " max abs diff " << diff;
}

}  // namespace

TEST(recon_cuda, line_above_1024_matches_cpu) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "no CUDA device";
  }
  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  Kind kinds[] = {Kind::Weno5, Kind::Weno3, Kind::Cp5, Kind::Cp3};
  for (Kind kind : kinds) {
    for (int n : {32, 1024, 1025, 1030, 1280}) {
      expect_matches_cpu(kind, make_line(5, n, opts), /*dim=*/3);
    }
    // Two lines in one tensor, so the block index has to stay the line id.
    auto two = torch::empty({5, 1, 2, 1025}, opts);
    auto x = torch::arange(1025, opts);
    for (int v = 0; v < 5; ++v) {
      two[v][0][0] = torch::sin((v + 1) * x / 1025);
      two[v][0][1] = torch::cos((v + 1) * x / 1025);
    }
    expect_matches_cpu(kind, two, /*dim=*/3);

    // Reconstruction along dim 2, so the line stride is not 1.
    // 32 stays on the old launch and checks the harness. 1030 is tiled.
    auto narrow = torch::empty({5, 1, 32, 4}, opts);
    auto y = torch::arange(32, opts);
    for (int v = 0; v < 5; ++v) {
      for (int a = 0; a < 4; ++a) {
        narrow[v][0].select(1, a) = torch::sin((v + 1) * y / 32 + a);
      }
    }
    expect_matches_cpu(kind, narrow, /*dim=*/2);

    auto wide = torch::empty({5, 1, 1030, 4}, opts);
    auto z = torch::arange(1030, opts);
    for (int v = 0; v < 5; ++v) {
      for (int a = 0; a < 4; ++a) {
        wide[v][0].select(1, a) = torch::sin((v + 1) * z / 1030 + a);
      }
    }
    expect_matches_cpu(kind, wide, /*dim=*/2);
  }
}

namespace {

// nvar variables over many decades, as chemical tracers have, on a
// {nvar, 2, 3, 3} block with n cells along dim.
torch::Tensor many_tracers(int nvar, int dim, int n,
                           torch::TensorOptions opts) {
  std::vector<int64_t> shape = {nvar, 2, 3, 3};
  shape[dim] = n;
  auto w = torch::empty(shape, opts);
  auto x = torch::arange(w[0].numel(), opts).view(w[0].sizes());
  for (int v = 0; v < nvar; ++v) {
    w[v] =
        std::pow(10., -0.3 * (v % 60)) * (1.5 + torch::sin((v + 1) * x / 37.));
  }
  return w;
}

// Every variable of one CUDA call on w must equal, bit for bit, its own
// one-variable call.
void expect_each_as_alone(Kind kind, torch::Tensor w, int dim, bool left,
                          torch::Tensor x1f, torch::Dtype bits) {
  std::ostringstream where;
  where << kind_name(kind) << (left ? " left" : " right") << " dim=" << dim
        << " n=" << w.size(dim) << " nvar=" << w.size(0)
        << (x1f.defined() ? " stretched" : " uniform");
  torch::Tensor all;
  try {
    all = recon(kind, w, dim, torch::kCUDA, left, x1f).cpu();
  } catch (std::exception const& err) {
    ADD_FAILURE() << where.str() << " CUDA failed: " << err.what();
    return;
  }
  for (int v = 0; v < w.size(0); ++v) {
    auto one = recon(kind, w.narrow(0, v, 1).contiguous(), dim, torch::kCUDA,
                     left, x1f)
                   .cpu();
    ASSERT_TRUE(torch::equal(all.narrow(0, v, 1).contiguous().view(bits),
                             one.view(bits)))
        << where.str() << " variable " << v << " differs from its own launch";
  }
}

struct ManyCase {
  torch::Dtype dtype;
  torch::Dtype bits;
  int nvar;
};

// 57 fp64 variables fill 48 KB on a 106-cell line; 58 is one past it.
constexpr ManyCase kManyCases[] = {{torch::kFloat64, torch::kInt64, 57},
                                   {torch::kFloat64, torch::kInt64, 58},
                                   {torch::kFloat64, torch::kInt64, 90},
                                   {torch::kFloat32, torch::kInt32, 116}};

}  // namespace

// One launch holds every variable's whole line in the default 48 KB of shared
// memory, so 106-cell lines (100 layers and 3 ghosts a side) held at most 57
// fp64 (115 fp32) variables and a 69-tracer scalar could not launch. Larger
// counts are split over several launches; each variable must come out bit for
// bit as when it is reconstructed alone, along x1 (contiguous) and along a
// strided dimension, at the split boundary and well past it.
TEST(recon_cuda, many_variables_match_one_at_a_time) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "no CUDA device";
  }
  for (auto const& cs : kManyCases) {
    auto opts = torch::TensorOptions().dtype(cs.dtype);
    for (Kind kind : {Kind::Weno5, Kind::Weno3, Kind::Cp5, Kind::Cp3}) {
      for (int dim : {3, 2}) {
        for (int n : {70, 106}) {
          auto w = many_tracers(cs.nvar, dim, n, opts);
          for (bool left : {true, false}) {
            expect_each_as_alone(kind, w, dim, left, {}, cs.bits);
          }
        }
      }
    }
  }
}

// The same on a stretched x1 grid, where WENO5 and cp5 take per-face rows
// (cells growing by 1.1 per cell, so every face's row differs).
TEST(recon_cuda, many_variables_match_one_at_a_time_stretched_x1) {
  if (!snapy_cuda_test_enabled()) {
    GTEST_SKIP() << "no CUDA device";
  }
  constexpr double kRatio = 1.1;
  for (auto const& cs : kManyCases) {
    auto opts = torch::TensorOptions().dtype(cs.dtype);
    for (Kind kind : {Kind::Weno5, Kind::Cp5}) {
      for (int n : {70, 106}) {
        auto widths = torch::pow(kRatio, torch::arange(n, torch::kFloat64));
        auto x1f = torch::cat(
            {torch::zeros(1, torch::kFloat64), torch::cumsum(widths, 0)});
        auto w = many_tracers(cs.nvar, /*dim=*/3, n, opts);
        for (bool left : {true, false}) {
          expect_each_as_alone(kind, w, 3, left, x1f, cs.bits);
        }
      }
    }
  }
}
