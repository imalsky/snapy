// C/C++
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

// external
#include <gtest/gtest.h>

#include "cuda_test_gate.hpp"

// torch
#include <torch/torch.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/hydro/hydro_dispatch.hpp>
#include <snap/mesh/meshblock.hpp>

using namespace snap;

// The default (SNAP_WB_REF4 off) well-balanced x1 reference at a physical
// wall. A hydrostatic polytrope T = 1 - beta z (g = R = 1, rho = p / T) one
// pressure scale height deep, reflecting x1 walls, held as exact cell
// averages. Reported per face f, relative to the exact rho(z_f):
//   dsf          the face density reference itself;
//   rho_L, rho_R the face densities the solver builds from it,
//                dsf + WENO5(rho - dref) with the even-parity wall ghosts of
//                hydro_forward.cpp.
// Interior faces are O(dz^2). See docs/derivations/wb-ref-wall.md.

namespace {

constexpr int kNg = 3;
constexpr int kDim1 = 3;  // x1, as DIM1 in hydro_forward.cpp

struct Profile {
  double beta;            // T = 1 - beta z; beta = 0: isothermal
  double depth() const {  // ln(p_b / p_t) = 1
    return beta == 0. ? 1. : (1. - std::exp(-beta)) / beta;
  }
  double p(double z) const {
    return beta == 0. ? std::exp(-z) : std::pow(1. - beta * z, 1. / beta);
  }
  double rho(double z) const { return p(z) / (1. - beta * z); }
};

std::shared_ptr<MeshBlockImpl> make_block(int nx1, double depth,
                                          torch::Device device) {
  char fname[] = "/tmp/test_wb_ref_wall_XXXXXX";
  int fd = mkstemp(fname);
  EXPECT_GE(fd, 0);
  close(fd);
  std::ofstream(fname) << "reference-state: {Tref: 300., Pref: 1.e5}\n"
                          "species:\n"
                          "  - name: dry\n"
                          "    composition: {O: 0.42, N: 1.56, Ar: 0.01}\n"
                          "    cv_R: 2.5\n"
                          "geometry:\n"
                          "  type: cartesian\n"
                          "  bounds: {x1min: 0., x1max: "
                       << depth
                       << ", x2min: 0., x2max: 1., x3min: 0., x3max: 1.}\n"
                          "  cells: {nx1: "
                       << nx1
                       << ", nx2: 1, nx3: 1, nghost: 3}\n"
                          "dynamics:\n"
                          "  equation-of-state: {type: ideal-gas}\n"
                          "  reconstruct:\n"
                          "    vertical: {type: weno5, scale: false, shock: "
                          "false}\n"
                          "    horizontal: {type: weno5, scale: false, shock: "
                          "false}\n"
                          "boundary-condition:\n"
                          "  external: {x1-inner: reflecting, x1-outer: "
                          "reflecting}\n";
  auto options = MeshBlockOptionsImpl::from_yaml(fname);
  std::remove(fname);
  auto block = std::make_shared<MeshBlockImpl>(options);
  block->to(device, torch::kFloat64);
  return block;
}

struct Faces {
  torch::Tensor dsf, rl, rr;  // relative errors at faces il..iu+1
  double dz;
};

Faces face_errors(int nx1, Profile const& prof,
                  torch::Device device = torch::kCPU) {
  double depth = prof.depth();
  auto block = make_block(nx1, depth, device);
  auto coord = block->pcoord;
  int il = coord->il(), iu = coord->iu(), nc1 = coord->options->nc1();
  double dz = depth / nx1;

  // exact cell averages (8-point Gauss-Legendre per cell); the ghosts carry
  // the even mirror a reflecting wall writes
  static const double gx[4] = {0.1834346424956498, 0.5255324099163290,
                               0.7966664774136267, 0.9602898564975363};
  static const double gw[4] = {0.3626837833783620, 0.3137066458778873,
                               0.2223810344533745, 0.1012285362903763};
  auto opt = torch::TensorOptions().dtype(torch::kFloat64);
  auto w = torch::zeros({block->phydro->peos->nvar(), 1, 1, nc1}, opt);
  auto rho_t = w[IDN], prs_t = w[IPR];
  auto rho = rho_t.accessor<double, 3>();
  auto prs = prs_t.accessor<double, 3>();
  for (int i = il; i <= iu; ++i) {
    double c = (i - il + 0.5) * dz, h = 0.5 * dz, sr = 0., sp = 0.;
    for (int k = 0; k < 4; ++k)
      for (double s : {-1., 1.}) {
        sr += 0.5 * gw[k] * prof.rho(c + s * gx[k] * h);
        sp += 0.5 * gw[k] * prof.p(c + s * gx[k] * h);
      }
    rho[0][0][i] = sr;
    prs[0][0][i] = sp;
  }
  for (int m = 0; m < kNg; ++m) {
    for (auto a : {rho, prs}) {
      a[0][0][il - 1 - m] = a[0][0][il + m];
      a[0][0][iu + 1 + m] = a[0][0][iu - m];
    }
  }

  // the reference, as HydroImpl::_hydro_ref_x1 builds it for one block with
  // two physical walls (wb-wall-clamp is on by default)
  w = w.to(device);
  opt = opt.device(device);
  auto sizes = w.sizes().slice(1).vec();
  auto psf_lo = torch::empty(sizes, opt), psf_hi = torch::empty(sizes, opt),
       pref = torch::empty(sizes, opt), dsf = torch::empty(sizes, opt),
       dref = torch::empty(sizes, opt);
  at::native::call_hydro_ref_x1(device.type(), w, coord->dx1f.contiguous(),
                                torch::Tensor(), psf_lo, psf_hi, pref, dsf,
                                dref, iu, /*grav=*/1., /*uniform=*/true,
                                /*phys_in=*/true, /*phys_out=*/true,
                                /*wall_clamp=*/true, torch::Tensor());

  // the solver's face densities (hydro_forward.cpp): reconstruct rho - dref
  // with even-parity wall ghosts, then add dsf back
  auto wp = w.clone();
  wp[IPR] -= pref;
  wp[IDN] -= dref;
  for (int c : {(int)IPR, (int)IDN}) {
    wp[c].narrow(-1, il - kNg, kNg).copy_(wp[c].narrow(-1, il, kNg).flip(-1));
    wp[c]
        .narrow(-1, iu + 1, kNg)
        .copy_(wp[c].narrow(-1, iu + 1 - kNg, kNg).flip(-1));
  }
  auto wtmp = block->phydro->precon1->forward(wp, kDim1, /*floor=*/false);

  if (char const* path = std::getenv("WB_REF_WALL_DUMP")) {  // review aid
    std::FILE* fp = std::fopen(path, "ab");
    for (auto t : {psf_lo, pref, dsf, dref}) {
      auto c = t.flatten().cpu().contiguous();
      std::fwrite(c.data_ptr<double>(), sizeof(double), c.numel(), fp);
    }
    std::fclose(fp);
  }

  int nf = iu + 2 - il;
  auto exact = torch::empty({nf}, torch::kFloat64);
  for (int f = 0; f < nf; ++f) exact[f] = prof.rho(f * dz);
  auto at = [&](torch::Tensor t) {
    return t.flatten().narrow(0, il, nf).cpu();
  };
  Faces out;
  out.dsf = at(dsf) / exact - 1.;
  out.rl = (at(wtmp[ILT][IDN]) + at(dsf)) / exact - 1.;
  out.rr = (at(wtmp[IRT][IDN]) + at(dsf)) / exact - 1.;
  out.dz = dz;
  return out;
}

// largest |error| over the faces at least `k` faces from each wall (the wall
// face itself carries no mass flux and is left out of every measure)
double interior_max(torch::Tensor e, int k) {
  int nf = e.size(0);
  return e.narrow(0, k, nf - 2 * k).abs().max().item<double>();
}

}  // namespace

TEST(WbRefWall, order_table) {
  std::vector<double> betas = {0.5, 0.};
  if (char const* e = std::getenv("WB_REF_WALL_BETAS")) {  // scan aid
    betas.clear();
    for (std::string tok, s = e; !s.empty();) {
      auto k = s.find(',');
      tok = s.substr(0, k);
      betas.push_back(std::stod(tok));
      s = k == std::string::npos ? "" : s.substr(k + 1);
    }
  }
  for (double beta : betas) {
    Profile prof{beta};
    std::printf("beta %g (depth %.6f, 1 pressure scale height)\n", beta,
                prof.depth());
    std::printf(
        "  nz   face1 dsf   face2 dsf   face1 rhoL  face2 rhoL  face1 rhoR  "
        "face2 rhoR | top1 dsf    top2 dsf   | interior dsf (>=3 from wall)\n");
    for (int nz : {16, 32, 64, 128}) {
      auto e = face_errors(nz, prof);
      int nf = e.dsf.size(0);
      auto v = [&](torch::Tensor t, int f) { return t[f].item<double>(); };
      std::printf(
          "  %4d %+.3e %+.3e %+.3e %+.3e %+.3e %+.3e | %+.3e %+.3e | "
          "%.3e\n",
          nz, v(e.dsf, 1), v(e.dsf, 2), v(e.rl, 1), v(e.rl, 2), v(e.rr, 1),
          v(e.rr, 2), v(e.dsf, nf - 2), v(e.dsf, nf - 3),
          interior_max(e.dsf, 3));
    }
  }
}

// The faces next to each wall (one and two cells in) keep the interior's
// second order: the observed order nz 32 -> 64 is at least 1.7 for the face
// density reference and for the face densities the solver builds from it, and
// at nz 64 the reference there is no worse than 1.5 times the largest interior
// error. With the wall cell repeated past the wall (the clamp) the first face
// is first order, 1.39e-3 against an interior 1.45e-4 at nz 64.
void faces_next_to_the_walls_are_second_order(torch::Device device) {
  Profile prof{0.5};
  auto c = face_errors(32, prof, device), f = face_errors(64, prof, device);
  int nc = c.dsf.size(0), nf = f.dsf.size(0);
  double interior = interior_max(f.dsf, 3);
  struct Face {
    char const* name;
    int ic, iff;
  };
  for (auto face :
       {Face{"bottom 1", 1, 1}, Face{"bottom 2", 2, 2},
        Face{"top 1", nc - 2, nf - 2}, Face{"top 2", nc - 3, nf - 3}}) {
    for (auto [what, ec, ef] :
         {std::tuple{"dsf", c.dsf, f.dsf}, std::tuple{"rho_L", c.rl, f.rl},
          std::tuple{"rho_R", c.rr, f.rr}}) {
      double a = std::abs(ec[face.ic].item<double>());
      double b = std::abs(ef[face.iff].item<double>());
      double order = std::log2(a / b);
      std::printf("%-8s %-5s nz 32 %.3e  nz 64 %.3e  order %.2f\n", face.name,
                  what, a, b, order);
      EXPECT_GE(order, 1.7) << face.name << " " << what;
      if (std::string(what) == "dsf") {
        EXPECT_LE(b, 1.5 * interior)
            << face.name << " dsf " << b << " vs interior " << interior;
      }
    }
  }
}

TEST(WbRefWall, faces_next_to_the_walls_are_second_order) {
  faces_next_to_the_walls_are_second_order(torch::kCPU);
}

TEST(WbRefWall, faces_next_to_the_walls_are_second_order_cuda) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  faces_next_to_the_walls_are_second_order(torch::Device(torch::kCUDA, 0));
}

// The order table on the GPU: every face error the CPU reports, both profiles
// and every resolution, to round-off
TEST(WbRefWall, order_table_cuda_matches_cpu) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  for (double beta : {0.5, 0.}) {
    Profile prof{beta};
    for (int nz : {16, 32, 64, 128}) {
      auto a = face_errors(nz, prof);
      auto b = face_errors(nz, prof, torch::Device(torch::kCUDA, 0));
      for (auto [what, ea, eb] :
           {std::tuple{"dsf", a.dsf, b.dsf}, std::tuple{"rho_L", a.rl, b.rl},
            std::tuple{"rho_R", a.rr, b.rr}}) {
        double d = (ea - eb).abs().max().item<double>();
        std::printf("beta %g nz %3d %-5s max |CUDA - CPU| %.2e\n", beta, nz,
                    what, d);
        EXPECT_LE(d, 1.e-13) << "beta " << beta << " nz " << nz << " " << what;
      }
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
