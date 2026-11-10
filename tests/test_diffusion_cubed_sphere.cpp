#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <future>
#include <snap/coord/coord_utils.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/forcing/forcing.hpp>
#include <snap/mesh/mesh.hpp>

#include "cuda_test_gate.hpp"
#ifdef USE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

using namespace snap;
namespace {
Mesh make_mesh(int n = 12, int partitions = 1, std::vector<double> x1f = {}) {
  auto opts = MeshOptionsImpl::from_yaml("test_diffusion_cubed_sphere.yaml");
  auto c = opts->block()->coord();
  if (!x1f.empty()) {  // #280: a stretched x1
    int m = x1f.size() - 1;
    c->nx1(m).global_nx1(m).nghost(3).x1min(x1f.front()).x1max(x1f.back());
    c->global_x1min(x1f.front()).global_x1max(x1f.back()).x1f_interior(x1f);
    opts->block()->hydro()->recon1()->interp()->type("weno5");
  }
  c->nx2() = n;
  c->nx3() = n;
  c->global_nx2() = n * partitions;
  c->global_nx3() = n * partitions;
  opts->block()->layout()->px(partitions).py(partitions);
  opts->blocks_per_process(6 * partitions * partitions / get_world_size());
  // Keep the distributed test session alive between meshes. Recreating Gloo
  // against torchrun's persistent store can reuse stale rendezvous keys.
  static std::shared_ptr<ProcessGroupContext> session;
  auto mesh = Mesh(opts);
  if (get_world_size() > 1) session = mesh->blocks[0]->get_layout()->comm;
  return mesh;
}
struct State {
  torch::Tensor w, t, x;
};
std::vector<State> states(Mesh mesh, int kind) {
  std::vector<State> out;
  for (auto block : mesh->blocks) {
    auto c = block->pcoord;
    auto w = torch::zeros(
        {5, c->options->nc3(), c->options->nc2(), c->options->nc1()},
        torch::kFloat64);
    w[IDN].fill_(1.);
    w[IPR].fill_(1.e5);
    int panel = std::get<2>(
        block->get_layout()->loc_of(block->options->layout()->rank()));
    auto a = c->x2v.view({1, -1, 1}), b = c->x3v.view({-1, 1, 1}),
         r = c->x1v.view({1, 1, -1});
    auto x = torch::zeros_like(w.narrow(0, IVX, 3));
    x[0].fill_(1.);
    cs_contra_to_cart_(x, a, b, panel);
    x.mul_(r);
    auto t = torch::full_like(w[IDN], 300.);
    if (kind == 1) t += x[0] * x[1] / (r * r);  // l=2
    if (kind == 2) t += 2. / r;
    if (kind == 3) {
      auto v = w.narrow(0, IVX, 3);
      v[0] = -x[1];
      v[1] = x[0];
      v[2].zero_();
      cs_cart_to_contra_(v, a, b, panel);
    }
    if (kind ==
        4) {  // reproducible smooth shear; invariant under decomposition
      auto v = w.narrow(0, IVX, 3);
      v[0] = x[1] * x[2] / (r * r);
      v[1] = -x[0] * x[2] / (r * r);
      v[2].zero_();
      cs_cart_to_contra_(v, a, b, panel);
    }
    out.push_back({w, t, x});
  }
  return out;
}
std::vector<torch::Tensor> apply_diffusion(Mesh mesh,
                                           std::vector<State> const &state) {
#ifdef USE_CUDA
  if (state.front().w.is_cuda()) torch::cuda::synchronize();
#endif
  std::vector<std::future<torch::Tensor>> tasks;
  for (size_t i = 0; i < state.size(); ++i) {
    tasks.push_back(std::async(std::launch::async, [&, i] {
#ifdef USE_CUDA
      if (state[i].w.is_cuda()) {
        auto stream =
            c10::cuda::getStreamFromPool(false, state[i].w.get_device());
        c10::cuda::CUDAStreamGuard guard(stream);
        auto result = mesh->blocks[i]->phydro->pdiffusion->forward(
            torch::zeros_like(state[i].w), state[i].w, state[i].t, 1.);
        stream.synchronize();
        return result;
      }
#endif
      return mesh->blocks[i]->phydro->pdiffusion->forward(
          torch::zeros_like(state[i].w), state[i].w, state[i].t, 1.);
    }));
  }
  std::vector<torch::Tensor> result;
  for (auto &task : tasks) result.push_back(task.get());
  return result;
}
auto interior(MeshBlock b) {
  return b->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
}
double harmonic_error(int n) {
  auto mesh = make_mesh(n);
  auto s = states(mesh, 1);
  auto du = apply_diffusion(mesh, s);
  double error = 0., norm = 0., energy = 0., diss = 0.;
  for (size_t i = 0; i < s.size(); ++i) {
    auto b = mesh->blocks[i];
    auto sl = interior(b);
    auto c = b->pcoord;
    auto vol = c->cell_volume().index(sl), q = du[i][IPR].index(sl);
    auto f = (s[i].t - 300.).index(sl);
    auto r = c->x1v.slice(0, c->il(), c->iu() + 1).view({1, 1, -1});
    auto exact = -0.02 * 6. * f / (r * r);
    error += (vol * (q - exact).square()).sum().item<double>();
    norm += (vol * exact.square()).sum().item<double>();
    energy += (vol * q).sum().item<double>();
    diss += (vol * q * f).sum().item<double>();
  }
  EXPECT_NEAR(energy, 0., 1.e-11);
  EXPECT_LT(diss, 0.);
  return std::sqrt(error / norm);
}
// the stage-2 stretched x1 faces (#280), or the same faces mapped onto [10, 11]
MeshOptions stretched_cs_options(int nx2 = 24, bool unit = false) {
  auto f = YAML::LoadFile("test_stage2_cubed_stretched.yaml")["geometry"]["x1f"]
               .as<std::vector<double>>();
  double f0 = f.front(), f1 = f.back();
  if (unit)
    for (auto &x : f) x = 10. + (x - f0) / (f1 - f0);
  auto op = MeshOptionsImpl::from_yaml("test_diffusion_cubed_sphere.yaml");
  op->block()->coord()->nx1(30).global_nx1(30).nx2(nx2).global_nx2(nx2);
  op->block()->coord()->nx3(nx2).global_nx3(nx2).nghost(3).x1min(f.front());
  op->block()->coord()->x1max(f.back()).global_x1min(f.front());
  op->block()->coord()->global_x1max(f.back()).x1f_interior(f);
  op->block()->hydro()->recon1()->interp()->type("weno5");
  op->blocks_per_process(6);
  return op;
}
Mesh to_device(Mesh mesh, bool gpu) {
  if (gpu) {
    for (auto b : mesh->blocks) b->get_layout()->options->device("cuda");
    mesh->to(torch::Device(torch::kCUDA));
  }
  return mesh;
}
}  // namespace
TEST(CubedSphereDiffusion, constant_state_and_rigid_rotation) {
  torch::set_num_threads(1);
  for (int kind : {0, 3}) {
    auto mesh = make_mesh();
    auto s = states(mesh, kind);
    auto du = apply_diffusion(mesh, s);
    for (size_t i = 0; i < s.size(); ++i) {
      auto sl = mesh->blocks[i]->part({0, 0, 0}, PartOptions().exterior(false));
      EXPECT_LT(du[i].index(sl).abs().max().item<double>(), 2.e-11);
    }
  }
}
TEST(CubedSphereDiffusion, harmonic_convergence_and_heat_conservation) {
  torch::set_num_threads(1);
  auto e1 = harmonic_error(12), e2 = harmonic_error(24),
       e3 = harmonic_error(48);
  std::cout << "harmonic relative L2: " << e1 << ", " << e2 << ", " << e3
            << std::endl;
  EXPECT_LT(e2, e1 / 2.5);
  EXPECT_LT(e3, e2 / 2.5);
}
TEST(CubedSphereDiffusion, radial_conductive_equilibrium) {
  torch::set_num_threads(1);
  // uniform x1, then the stage-2 faces mapped onto [10, 11] on CPU and CUDA;
  // explicit, then (u1 - u0) / dt of one implicit radial step (dt = 1e3, so
  // the rounding of u ~ 2e5 stays well under the tolerance)
  for (int arm : {0, 1, 2}) {
    if (arm == 2 && !snapy_cuda_test_enabled()) continue;
    auto mesh = arm ? Mesh(stretched_cs_options(12, true)) : make_mesh();
    for (auto b : mesh->blocks) {
      b->phydro->pdiffusion->options->temperature_inner() = 300. + 2. / 10.;
      b->phydro->pdiffusion->options->temperature_outer() = 300. + 2. / 11.;
    }
    auto s = states(mesh, 2);
    for (auto &x : s)
      if (arm == 2) x.w = x.w.cuda(), x.t = x.t.cuda();
    auto du = apply_diffusion(to_device(mesh, arm == 2), s);
    for (size_t i = 0; i < s.size(); ++i) {
      EXPECT_LT(du[i][IPR]
                    .index(interior(mesh->blocks[i]))
                    .abs()
                    .max()
                    .item<double>(),
                2.e-11);
      auto b = mesh->blocks[i];
      auto eos = b->phydro->peos;
      auto w = s[i].w.clone();
      w[IPR] = eos->species_cv_ref() * (eos->options->gammad() - 1.) * s[i].t;
      auto u = eos->compute("W->U", {w}), u0 = u.clone();
      b->phydro->pdiffusion->options->implicit_radial(true);
      b->phydro->pdiffusion->radial_conduction_step(u, 1.e3);
      EXPECT_LT(
          (u - u0)[IPR].index(interior(b)).abs().max().item<double>() / 1.e3,
          2.e-11);
    }
  }
}
TEST(CubedSphereDiffusion, viscous_energy_conservation_and_dissipation) {
  torch::set_num_threads(1);
  auto mesh = make_mesh();
  auto s = states(mesh, 4);
  auto du = apply_diffusion(mesh, s);
  double energy = 0., work = 0.;
  for (size_t i = 0; i < s.size(); ++i) {
    auto sl = interior(mesh->blocks[i]);
    auto vol = mesh->blocks[i]->pcoord->cell_volume().index(sl);
    energy += (vol * du[i][IPR].index(sl)).sum().item<double>();
    // primitive contravariant velocity contracted with covariant force
    work +=
        (vol *
         (s[i].w.narrow(0, IVX, 3) * du[i].narrow(0, IVX, 3)).sum(0).index(sl))
            .sum()
            .item<double>();
  }
  EXPECT_NEAR(energy, 0., 1.e-10);
  EXPECT_LT(work, 0.);
}

// #280: v = z g(r), g = sin(2 pi (r - 10)), on the stage-2 faces mapped onto
// [10, 11] and split at their midpoints `l` times: relative L2 error of the
// viscous tendency nu (lap v + grad div v / 3)
double stretched_viscous_error(int l) {
  auto f = YAML::LoadFile("test_stage2_cubed_stretched.yaml")["geometry"]["x1f"]
               .as<std::vector<double>>();
  double f0 = f.front(), f1 = f.back(), k = 2. * M_PI, error = 0., norm = 0.;
  for (auto &x : f) x = 10. + (x - f0) / (f1 - f0);
  for (int j = 0; j < l; ++j)
    for (int i = f.size() - 2; i >= 0; --i)
      f.insert(f.begin() + i + 1, (f[i] + f[i + 1]) / 2);
  auto mesh = make_mesh(12 << l, 1, f);
  auto s = states(mesh, 0);
  auto du = std::vector<torch::Tensor>();
  for (size_t i = 0; i < s.size(); ++i) {
    auto c = mesh->blocks[i]->pcoord;
    int panel = std::get<2>(mesh->blocks[i]->get_layout()->loc_of(
        mesh->blocks[i]->options->layout()->rank()));
    auto a = c->x2v.view({1, -1, 1}), b = c->x3v.view({-1, 1, 1});
    auto r = c->x1v.view({1, 1, -1}), z = s[i].x[2];
    auto g = (k * (r - 10.)).sin(), g1 = k * (k * (r - 10.)).cos();
    auto g2 = -k * k * g, h = g1 / r, h1 = g2 / r - g1 / (r * r);
    auto v = s[i].w.narrow(0, IVX, 3);
    v[2].copy_(g.expand_as(v[2]));
    cs_cart_to_contra_(v, a, b, panel);
    auto t = (z * h1 / (3. * r)) * s[i].x;
    t[2] += g2 + 2. * g1 / r + h / 3.;
    t.mul_(0.01);  // nu_iso
    cs_cart_to_contra_(t, a, b, panel);
    coord_vec_lower_(t, c->cosine_cell_kj);
    du.push_back(t);
  }
  auto got = apply_diffusion(mesh, s);
  for (size_t i = 0; i < s.size(); ++i) {
    auto c = mesh->blocks[i]->pcoord;
    auto sl = interior(mesh->blocks[i]);
    auto l2 = [&](torch::Tensor q) {  // three cells off each x1 wall
      auto e = c->cell_volume().index(sl) * q.square().sum(0).index(sl);
      return e.narrow(-1, 3, c->iu() - c->il() - 5).sum().item<double>();
    };
    error += l2(got[i].narrow(0, IVX, 3) - du[i]);
    norm += l2(du[i]);
  }
  return std::sqrt(error / norm);
}
TEST(CubedSphereDiffusion, stretched_x1_viscosity) {
  torch::set_num_threads(1);
  double e[3] = {stretched_viscous_error(0), stretched_viscous_error(1),
                 stretched_viscous_error(2)};
  std::cout << "stretched viscous L2: " << e[0] << ", " << e[1] << ", " << e[2]
            << std::endl;
  EXPECT_GT(e[0] / e[1], 3.);
  EXPECT_GT(e[1] / e[2], 3.8);
}

TEST(CubedSphereDiffusion, split_panels_preserve_constants_and_rotation) {
  torch::set_num_threads(1);
  auto mesh = make_mesh(6, 2);
  auto s = states(mesh, 3);
  auto du = apply_diffusion(mesh, s);
  for (size_t i = 0; i < s.size(); ++i) {
    auto sl = mesh->blocks[i]->part({0, 0, 0}, PartOptions().exterior(false));
    EXPECT_LT(du[i].index(sl).abs().max().item<double>(), 2.e-10);
  }
}

TEST(CubedSphereDiffusion, piecewise_constant_panels_exchange_heat) {
  torch::set_num_threads(1);
  auto mesh = make_mesh(6);
  auto s = states(mesh, 0);
  s[0].t.fill_(301.);
  auto du = apply_diffusion(mesh, s);
  double total = 0., hot = 0.;
  for (size_t i = 0; i < s.size(); ++i) {
    auto sl = interior(mesh->blocks[i]);
    double e = (mesh->blocks[i]->pcoord->cell_volume().index(sl) *
                du[i][IPR].index(sl))
                   .sum()
                   .item<double>();
    total += e;
    if (i == 0) hot = e;
  }
  EXPECT_LT(hot, 0.);
  EXPECT_NEAR(total, 0., 1.e-11);
}

TEST(CubedSphereDiffusion, closed_shell_heat_operator_spectrum) {
  torch::set_num_threads(1);
  auto mesh = make_mesh(4);
  auto s = states(mesh, 0);
  constexpr int cells = 6 * 4 * 4 * 4;
  auto matrix = torch::zeros({cells, cells}, torch::kFloat64);
  auto volume = torch::empty({cells}, torch::kFloat64);
  for (int ib = 0; ib < 6; ++ib)
    volume.narrow(0, ib * 64, 64)
        .copy_(mesh->blocks[ib]
                   ->pcoord->cell_volume()
                   .index(interior(mesh->blocks[ib]))
                   .reshape({-1}));
  for (int col = 0; col < cells; ++col) {
    int ib = col / 64, local = col % 64;
    for (auto &state : s) state.t.zero_();
    s[ib].t[2 + local / 16][2 + (local / 4) % 4][2 + local % 4] = 1.;
    auto du = apply_diffusion(mesh, s);
    for (int row = 0; row < 6; ++row)
      matrix.select(1, col)
          .narrow(0, row * 64, 64)
          .copy_(du[row][IPR].index(interior(mesh->blocks[row])).reshape({-1}));
  }
  auto spectrum = torch::linalg_eigvals(matrix);
  double maxreal = torch::real(spectrum).max().item<double>();
  auto weighted = volume.unsqueeze(1) * matrix;
  double maxenergy = torch::linalg_eigvalsh(0.5 * (weighted + weighted.t()))
                         .max()
                         .item<double>();
  std::cout << "thermal spectral abscissa=" << maxreal
            << ", maximum energy eigenvalue=" << maxenergy << std::endl;
  EXPECT_LT(maxreal, 1.e-10);
  EXPECT_LT(maxenergy, 1.e-10);
  EXPECT_LT((volume.unsqueeze(0).matmul(matrix)).abs().max().item<double>(),
            1.e-11);
}

TEST(CubedSphereDiffusion, full_mesh_constant_state_stays_at_rest) {
  torch::set_num_threads(1);
  auto mesh = make_mesh(6);
  auto s = states(mesh, 0);
  MeshVariables vars(s.size());
  for (size_t i = 0; i < s.size(); ++i) vars[i]["hydro_w"] = s[i].w;
  mesh->initialize(vars);
  for (int cycle = 0; cycle < 3; ++cycle) {
    mesh->set_cycle(cycle + 1);
    double dt = mesh->max_time_step(vars);
    for (int stage = 0; stage < mesh->blocks[0]->pintg->stages.size(); ++stage)
      mesh->forward(vars, dt, stage);
  }
  for (size_t i = 0; i < s.size(); ++i) {
    auto v = vars[i]["hydro_u"].narrow(0, IVX, 3);
    EXPECT_LT(v.abs().max().item<double>(), 1.e-8);
    EXPECT_TRUE(torch::isfinite(vars[i]["hydro_u"]).all().item<bool>());
  }
}

TEST(CubedSphereDiffusion, radial_implicit_conservation_stability_and_order) {
  torch::set_num_threads(1);
  auto mesh = make_mesh(4);
  auto b = mesh->blocks[0];
  auto c = b->pcoord;
  auto d = b->phydro->pdiffusion;
  d->options->implicit_radial() = true;
  d->options->kappa_iso() = 1000.;
  auto s = states(mesh, 0)[0];
  auto r = c->x1v.slice(0, c->il(), c->iu() + 1);
  auto rf = c->x1f.slice(0, c->il(), c->iu() + 2);
  double cv = b->phydro->peos->species_cv_ref();
  double gas_constant = cv * (b->phydro->peos->options->gammad() - 1.);
  auto t0 = 300. + 10. * torch::cos(M_PI * (r - 10.));
  auto v = (rf.slice(0, 1) - rf.slice(0, 0, -1)) *
           (rf.slice(0, 1).square() + rf.slice(0, 1) * rf.slice(0, 0, -1) +
            rf.slice(0, 0, -1).square()) /
           3.;
  auto mat = torch::zeros({4, 4}, torch::kFloat64);
  for (int i = 0; i < 3; ++i) {
    auto conductance = 1000. / (1. / r[i] - 1. / r[i + 1]);
    mat[i][i + 1] = conductance / (cv * v[i]);
    mat[i][i] -= mat[i][i + 1];
    mat[i + 1][i] = conductance / (cv * v[i + 1]);
    mat[i + 1][i + 1] -= mat[i + 1][i];
  }
  double end = 0.2;
  auto exact = 300. + torch::matrix_exp(end * mat).matmul(t0 - 300.);
  std::vector<double> error;
  for (int steps : {2, 4, 8}) {
    auto w = s.w.clone();
    w[IPR]
        .narrow(-1, c->il(), 4)
        .copy_(gas_constant *
               t0.view({1, 1, -1}).expand_as(w[IPR].narrow(-1, c->il(), 4)));
    auto u = b->phydro->peos->compute("W->U", {w});
    auto energy0 =
        (u[IPR].index(interior(b)) * c->cell_volume().index(interior(b)))
            .sum()
            .item<double>();
    for (int step = 0; step < steps; ++step)
      d->radial_conduction_step(u, end / steps);
    auto got = b->phydro->peos
                   ->compute("W->T", {b->phydro->peos->compute("U->W", {u})})
                   .index(interior(b));
    error.push_back((got - exact).square().mean().sqrt().item<double>());
    auto energy1 =
        (u[IPR].index(interior(b)) * c->cell_volume().index(interior(b)))
            .sum()
            .item<double>();
    EXPECT_NEAR(energy1, energy0, 1.e-12 * energy0);
    d->begin_step(u, 10.);
    d->finish_step(u, 10.);
    EXPECT_TRUE(torch::isfinite(u).all().item<bool>());
    d->restore_step(u);
    auto restored =
        b->phydro->peos
            ->compute("W->T", {b->phydro->peos->compute("U->W", {u})})
            .index(interior(b));
    EXPECT_TRUE(torch::allclose(got, restored, 0., 1.e-12));
  }
  std::cout << "radial SDIRK errors: " << error[0] << ", " << error[1] << ", "
            << error[2] << std::endl;
  EXPECT_GT(error[0] / error[1], 3.5);
  EXPECT_GT(error[1] / error[2], 3.5);
}

// #280 (R19): with k linear in r on the stage-2 faces, the steady wall fluxes
// of the implicit radial solve carry k(r_wall): the wall conductivity is
// extrapolated linearly in position over the cell widths
TEST(CubedSphereDiffusion, stretched_radial_implicit_wall_k) {
  torch::set_num_threads(1);
  auto f = YAML::LoadFile("test_stage2_cubed_stretched.yaml")["geometry"]["x1f"]
               .as<std::vector<double>>();
  double r0 = f.front(), r1 = f.back();
  auto lin = [&](double x) { return 1. + 2. * (x - r0) / (r1 - r0); };
  auto rc = [&](int i) { return 0.5 * (f[i] + f[i + 1]); };
  auto scale = torch::empty({36}, torch::kFloat64);
  for (int i = 0; i < 36; ++i) scale[i] = lin(rc(std::clamp(i - 3, 0, 29)));
  for (bool gpu : {false, true}) {
    if (gpu && !snapy_cuda_test_enabled()) continue;
    auto op = stretched_cs_options(4);
    auto d = op->block()->hydro()->diffusion();
    d->nu_iso(0.).kappa_iso(1.e11).dynamic(false).implicit_radial(true);
    d->kappa_scale_x1(scale).temperature_inner(300.).temperature_outer(310.);
    auto mesh = Mesh(op);
    auto b = mesh->blocks[0];
    auto eos = b->phydro->peos;
    auto u = eos->compute("W->U", {states(mesh, 0)[0].w});
    u = u.to(to_device(mesh, gpu)->blocks[0]->pcoord->x1v.device());
    for (int n = 0; n < 40; ++n)
      b->phydro->pdiffusion->radial_conduction_step(u, 1.e9);
    auto t = eos->compute("W->T", {eos->compute("U->W", {u})})
                 .index(interior(b))
                 .cpu()[0][0];
    // k at the wall next to cell i, from the steady flux to neighbour j
    for (auto [i, j, tw] : {std::tuple{0, 1, 300.}, std::tuple{29, 28, 310.}}) {
      double rw = i ? r1 : r0, ti = t[i].item<double>();
      double flux = 0.5 * (lin(rc(i)) + lin(rc(j))) * rc(i) * rc(j) /
                    std::abs(rc(j) - rc(i)) * (ti - t[j].item<double>());
      EXPECT_NEAR(flux * std::abs(rc(i) - rw) / (rw * rc(i) * (tw - ti)),
                  lin(rw), 1.e-9 * lin(rw));
    }
  }
}

// Rayleigh-Ritz onset using the actual C++ heat/stress operator. This isolates
// diffusion accuracy; it is NOT an unconstrained compressible onset run.
TEST(CubedSphereDiffusion, thin_shell_modal_onset) {
  torch::set_num_threads(1);
  constexpr double eps = .1, amplitude = 1.e-5;
  constexpr int ell = 22, nr = 24;
  double L = ell * (ell + 1), a2 = eps * eps * L;
  double analytical = std::pow(M_PI * M_PI + a2, 3) / a2;
  for (int n : {48, 96}) {
    auto opts = MeshOptionsImpl::from_yaml("test_diffusion_cubed_sphere.yaml");
    auto c = opts->block()->coord();
    c->nx1() = nr;
    c->global_nx1() = nr;
    c->nx2() = n;
    c->global_nx2() = n;
    c->nx3() = n;
    c->global_nx3() = n;
    c->x1min() = 9.5;
    c->global_x1min() = 9.5;
    c->x1max() = 10.5;
    c->global_x1max() = 10.5;
    auto d = opts->block()->hydro()->diffusion();
    d->nu_iso() = 1.;
    d->kappa_iso() = 1.;
    d->temperature_inner() = 300.;
    d->temperature_outer() = 300.;
    auto mesh = Mesh(opts);
    auto s = states(mesh, 0);
    for (size_t ib = 0; ib < s.size(); ++ib) {
      auto b = mesh->blocks[ib];
      auto c = b->pcoord;
      auto r = c->x1v.view({1, 1, -1}), normal = s[ib].x / r;
      auto z = normal[2];
      auto prev = torch::ones_like(z), y = z.clone();
      for (int l = 2; l <= ell; ++l) {
        auto next = ((2. * l - 1.) * z * y - (l - 1.) * prev) / l;
        prev = y;
        y = next;
      }
      auto dy = ell * (prev - z * y) / (1. - z.square()).clamp_min(1.e-15);
      auto angular = -normal * z;
      angular[2] += 1.;
      angular *= dy;
      auto W = torch::sin(M_PI * (r - 9.5)),
           DW = M_PI * torch::cos(M_PI * (r - 9.5));
      auto velocity =
          amplitude * (normal * (W * y) + angular * (2. * W + r * DW) / L);
      s[ib].w.narrow(0, IVX, 3).copy_(velocity);
      int panel =
          std::get<2>(b->get_layout()->loc_of(b->options->layout()->rank()));
      cs_cart_to_contra_(s[ib].w.narrow(0, IVX, 3), c->x2v.view({1, -1, 1}),
                         c->x3v.view({-1, 1, 1}), panel);
      s[ib].t = 300. + W * y;
    }
    auto du = apply_diffusion(mesh, s);
    double normu = 0., normt = 0., cross = 0., work = 0., heat = 0.;
    for (size_t ib = 0; ib < s.size(); ++ib) {
      auto b = mesh->blocks[ib];
      auto sl = interior(b);
      auto vol = b->pcoord->cell_volume().index(sl);
      auto v = s[ib].w.narrow(0, IVX, 3), lower = v.clone();
      coord_vec_lower_(lower, b->pcoord->cosine_cell_kj);
      auto theta = (s[ib].t - 300.).index(sl);
      normu += (vol * (v * lower).sum(0).index(sl)).sum().item<double>();
      normt += (vol * theta.square()).sum().item<double>();
      cross += (vol * theta * s[ib].w[IVX].index(sl)).sum().item<double>();
      work += (vol * (v * du[ib].narrow(0, IVX, 3)).sum(0).index(sl))
                  .sum()
                  .item<double>();
      heat += (vol * theta * du[ib][IPR].index(sl)).sum().item<double>();
    }
    double ra = (-work / normu) * (-heat / normt) / (cross * amplitude / normu);
    std::cout << "modal onset N=" << n << ", nr=" << nr << ", ell=" << ell
              << ", Ra=" << ra << ", thin-shell=" << analytical << std::endl;
    EXPECT_NEAR(ra, analytical, 0.02 * analytical);
  }
}

TEST(CubedSphereDiffusion, distributed_shared_heat_flux) {
  torch::set_num_threads(1);
  auto mesh = make_mesh(6);
  auto s = states(mesh, 1);
  for (size_t i = 0; i < s.size(); ++i)
    s[i].t += 0.1 * mesh->blocks[i]->options->layout()->rank();
  auto du = apply_diffusion(mesh, s);
  auto sums = torch::zeros({2}, torch::kFloat64);
  for (size_t i = 0; i < s.size(); ++i) {
    auto b = mesh->blocks[i];
    auto sl = interior(b);
    auto energy = b->pcoord->cell_volume().index(sl) * du[i][IPR].index(sl);
    sums[0] += energy.sum();
    sums[1] += energy.abs().sum();
  }
  auto layout = mesh->blocks[0]->get_layout();
  if (layout->has_process_group()) {
    std::vector<torch::Tensor> reduction{sums};
    layout->comm->allreduce(reduction, c10d::ReduceOp::SUM);
  }
  EXPECT_GT(sums[1].item<double>(), 0.);
  EXPECT_LT(std::abs(sums[0].item<double>()), 1.e-12 * sums[1].item<double>());
}

TEST(CubedSphereDiffusion, kinematic_profiles_preserve_dynamic_products) {
  torch::set_num_threads(1);
  std::vector<torch::Tensor> reference;
  double cv = 0.;
  for (bool kinematic : {false, true}) {
    auto opts = MeshOptionsImpl::from_yaml("test_diffusion_cubed_sphere.yaml");
    auto d = opts->block()->hydro()->diffusion();
    if (kinematic) {
      auto r = 10. + (torch::arange(8, torch::kFloat64) - 1.5) * .25;
      auto rho = 1. + .1 * (r - 10.);
      d->dynamic() = false;
      d->nu_scale_x1() = 1. / rho;
      d->kappa_scale_x1() = 1. / (rho * cv);
    }
    auto mesh = Mesh(opts);
    auto s = states(mesh, 4);
    cv = mesh->blocks[0]->phydro->peos->species_cv_ref();
    for (size_t i = 0; i < s.size(); ++i) {
      auto r = mesh->blocks[i]->pcoord->x1v.view({1, 1, -1});
      s[i].w[IDN].copy_((1. + .1 * (r - 10.)).expand_as(s[i].w[IDN]));
      s[i].t += s[i].x[0] * s[i].x[1] / (r * r);
    }
    auto du = apply_diffusion(mesh, s);
    if (!kinematic)
      reference = du;
    else
      for (size_t i = 0; i < s.size(); ++i)
        EXPECT_TRUE(torch::allclose(du[i], reference[i], 1.e-11, 1.e-12));
  }
}

TEST(CubedSphereDiffusion, cuda_operator_and_radial_solve_match_cpu) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  torch::set_num_threads(1);
  for (int kind : {1, 3, 4}) {
    auto mesh = make_mesh(6);
    auto s = states(mesh, kind);
    auto reference = apply_diffusion(mesh, s);
    // Match the GPU layout exchange protocol, with one stream per block.
    for (auto block : mesh->blocks)
      block->get_layout()->options->device("cuda");
    mesh->to(torch::Device(torch::kCUDA));
    for (auto &state : s) {
      state.w = state.w.cuda();
      state.t = state.t.cuda();
      state.x = state.x.cuda();
    }
    auto result = apply_diffusion(mesh, s);
    for (size_t i = 0; i < s.size(); ++i)
      EXPECT_TRUE(
          torch::allclose(result[i].cpu(), reference[i], 1.e-8, 1.e-10));
  }
  auto mesh = make_mesh(6);
  auto b = mesh->blocks[0];
  auto s = states(mesh, 2)[0];
  auto d = b->phydro->pdiffusion;
  d->options->implicit_radial() = true;
  d->options->kappa_iso() = 1000.;
  double gas = b->phydro->peos->species_cv_ref() *
               (b->phydro->peos->options->gammad() - 1.);
  s.w[IPR].copy_(gas * s.t);
  auto initial = b->phydro->peos->compute("W->U", {s.w});
  auto reference = initial.clone();
  d->radial_conduction_step(reference, .3);
  mesh->to(torch::Device(torch::kCUDA));
  auto result = initial.cuda();
  d->radial_conduction_step(result, .3);
  EXPECT_TRUE(torch::allclose(result.cpu(), reference, 1.e-12, 1.e-9));
}

TEST(CubedSphereDiffusion, cuda_mesh_split_step_matches_cpu) {
  if (!snapy_cuda_test_enabled()) GTEST_SKIP() << "CUDA is not available";
  torch::set_num_threads(1);
  std::vector<torch::Tensor> reference;
  for (bool gpu : {false, true}) {
    auto opts = MeshOptionsImpl::from_yaml("test_diffusion_cubed_sphere.yaml");
    opts->block()->coord()->nx2(6).nx3(6).global_nx2(6).global_nx3(6);
    opts->block()->layout()->device(gpu ? "cuda" : "cpu").device_id(0);
    auto mesh = Mesh(opts);
    auto s = states(mesh, 2);
    MeshVariables vars(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
      auto b = mesh->blocks[i];
      b->phydro->pdiffusion->options->implicit_radial(true).kappa_iso(1000.);
      double gas = b->phydro->peos->species_cv_ref() *
                   (b->phydro->peos->options->gammad() - 1.);
      s[i].w[IPR].copy_(gas * s[i].t);
      vars[i]["hydro_w"] = gpu ? s[i].w.cuda() : s[i].w;
    }
    if (gpu) mesh->to(torch::Device(torch::kCUDA, 0));
    mesh->initialize(vars);
    for (int cycle = 0; cycle < 3; ++cycle) {
      mesh->set_cycle(cycle + 1);
      double dt = mesh->max_time_step(vars);
      for (int stage = 0; stage < mesh->blocks[0]->pintg->stages.size();
           ++stage)
        mesh->forward(vars, dt, stage);
    }
    for (size_t i = 0; i < s.size(); ++i) {
      auto u = vars[i]["hydro_u"].cpu();
      if (gpu)
        EXPECT_TRUE(torch::allclose(u, reference[i], 1.e-8, 1.e-8));
      else
        reference.push_back(u.clone());
    }
  }
}
