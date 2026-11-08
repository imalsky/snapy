
#include <snap/snap.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <snap/mesh/mesh.hpp>
#include <string>
#include <vector>

using namespace snap;

namespace {

constexpr uint64_t kSeed = 20261103;
uint64_t fnv1a(const unsigned char* p, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

torch::Tensor make_state(MeshBlock const& block) {
  auto pcoord = block->pcoord;
  int nc1 = pcoord->x1v.size(0);
  int nc2 = pcoord->x2v.size(0);
  int nc3 = pcoord->x3v.size(0);
  int nvar = block->phydro->peos->nvar();
  auto opts = pcoord->x1v.options();
  auto w = torch::zeros({nvar, nc3, nc2, nc1}, opts);
  auto x1 = pcoord->x1v;
  double lo = x1.min().item<double>();
  double hi = x1.max().item<double>();
  double H = (hi - lo) > 0. ? (hi - lo) / 3. : 1.;
  auto prof = torch::exp(-(x1 - lo) / H).view({1, 1, nc1});
  auto noise = [&](double amp) {
    return amp * torch::rand({nvar == 0 ? 1 : nc3, nc2, nc1}, opts);
  };
  w[IDN].copy_(1.2 * prof.expand({nc3, nc2, nc1}) * (1. + noise(0.05)));
  if (nvar > IPR) {
    w[IPR].copy_(1.e5 * prof.expand({nc3, nc2, nc1}) * (1. + noise(0.02)));
  }
  for (int v = IVX; v <= IVZ && v < nvar; ++v) {
    w[v].copy_(noise(0.4) - 0.2);
  }
  for (int v = ICY; v < nvar; ++v) {
    w[v].copy_(1.e-3 * (1. + noise(0.1)));
  }
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: g3_decks <deck.yaml> <steps> <outdir> <tag> "
                 "[cpu|cuda]\n";
    return 2;
  }
  std::string deck = argv[1];
  int nsteps = std::atoi(argv[2]);
  std::string outdir = argv[3];
  std::string tag = argv[4];
  std::string dev = argc > 5 ? argv[5] : "cpu";

  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  auto device = dev == "cuda" ? torch::Device(torch::kCUDA, 0)
                              : torch::Device(torch::kCPU);
  if (device.is_cuda() && !torch::cuda::is_available()) {
    std::cerr << "SKIP no cuda\n";
    return 125;
  }

  torch::manual_seed(kSeed);
  if (device.is_cuda()) torch::cuda::manual_seed_all(kSeed);
  auto mesh = Mesh(MeshOptionsImpl::from_yaml(deck));
  mesh->to(device, torch::kFloat64);
  MeshVariables vars(mesh->blocks.size());
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    vars[i]["hydro_w"] = make_state(mesh->blocks[i]);
  }
  mesh->initialize(vars);
  for (int step = 0; step < nsteps; ++step) {
    auto dt = mesh->max_time_step(vars);
    for (size_t stage = 0; stage < mesh->blocks.front()->pintg->stages.size();
         ++stage) {
      mesh->forward(vars, dt, stage);
    }
  }

  std::ofstream man(outdir + "/" + tag + ".manifest");
  uint64_t all = 1469598103934665603ULL;
  size_t ntensor = 0, total = 0;
  for (size_t i = 0; i < mesh->blocks.size(); ++i) {
    std::vector<std::string> keys;
    for (auto const& kv : vars[i]) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    for (auto const& k : keys) {
      auto t = vars[i].at(k).to(torch::kCPU).to(torch::kFloat64).contiguous();
      auto n = static_cast<size_t>(t.numel()) * sizeof(double);
      auto const* p = reinterpret_cast<const unsigned char*>(t.data_ptr());

      std::string fn = tag + "." + std::to_string(i) + "." + k + ".bin";
      std::ofstream os(outdir + "/" + fn, std::ios::binary);
      os.write(reinterpret_cast<const char*>(p), n);
      os.close();
      man << i << " " << k << " " << t.numel() << " " << fn << " "
          << fnv1a(p, n) << "\n";
      for (size_t b = 0; b < n; ++b) {
        all ^= p[b];
        all *= 1099511628211ULL;
      }
      ++ntensor;
      total += n;
    }
  }
  man.close();

  std::cout << "G3 " << tag << " blocks=" << mesh->blocks.size()
            << " tensors=" << ntensor << " bytes=" << total
            << " steps=" << nsteps << " device=" << dev << " fnv1a=" << all
            << std::endl;
  return 0;
}
