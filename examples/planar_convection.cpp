// Fully compressible planar Rayleigh--Benard onset. All evolution runs on
// the requested device; there is no CPU fallback when CUDA is requested.
// Usage: executable yaml Ra a D k nz end_diffusive seed cfl output.csv device
#include <kintera/constants.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <snap/hydro/balance_column.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;
int main(int argc, char **argv) {
  try {
    TORCH_CHECK(argc == 12,
                "usage: planar_convection yaml Ra a D k nz end_diffusive "
                "seed cfl output.csv cpu|cuda:0|cuda:1");
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    torch::NoGradGuard no_grad;
    double ra = std::stod(argv[2]), a = std::stod(argv[3]),
           dissipation = std::stod(argv[4]), k = std::stod(argv[5]),
           end = std::stod(argv[7]), seed = std::stod(argv[8]),
           cfl = std::stod(argv[9]);
    int n = std::stoi(argv[6]);
    TORCH_CHECK(ra > 0. && a > 0. && a < 2. && dissipation > 0. &&
                    dissipation < a && k > 0. && n >= 8 && n % 2 == 0 &&
                    end > 0. && seed >= 0. && cfl > 0. && cfl <= 1.,
                "invalid planar benchmark parameters");
    torch::Device device(argv[11]);
    TORCH_CHECK(!device.is_cuda() || torch::cuda::is_available(),
                "CUDA was requested but is not available");
    c10::OptionalDeviceGuard device_guard(device);
    auto opts = MeshOptionsImpl::from_yaml(argv[1]);
    auto block_opts = opts->block();
    TORCH_CHECK(get_world_size() == 1 && opts->blocks_per_process() == 1,
                "this benchmark requires one process and one block");
    block_opts->layout()
        ->device(device.is_cuda() ? "cuda" : "cpu")
        .device_id(device.is_cuda() ? device.index() : 0);
    auto co = block_opts->coord();
    co->nx1(n).global_nx1(n).nx2(2 * n).global_nx2(2 * n);
    co->x2max(2 * M_PI / k).global_x2max(2 * M_PI / k);
    block_opts->intg()->cfl(cfl);
    auto eos_opts = block_opts->hydro()->eos();
    // Match the EOS library's constant exactly, including its rounding.
    // These are benchmark units, with R=1, rather than a physical molar mass.
    eos_opts->weight(kintera::constants::Rgas);
    double gamma = eos_opts->gammad(), cp = gamma / (gamma - 1),
           gravity = cp * dissipation;
    double chi = std::sqrt(gravity * (a - dissipation) / ra);
    block_opts->hydro()->grav()->grav1(-gravity);
    block_opts->hydro()
        ->diffusion()
        ->dynamic(true)
        .nu_iso(chi)
        .kappa_iso(cp * chi)
        .temperature_inner(1 + a / 2)
        .temperature_outer(1 - a / 2);
    auto mesh = Mesh(opts);
    auto b = mesh->blocks[0];
    auto c = b->pcoord;
    int ng = co->nghost();
    auto z = c->x1v, temperature = 1 - a * z;
    TORCH_CHECK(temperature.min().item<double>() > 0.,
                "temperature extrapolation crosses zero in the ghost cells");
    auto rho = torch::pow(temperature, gravity / a - 1);
    auto col = torch::zeros({5, 1, 1, n}, torch::kFloat64);
    col[IDN][0][0].copy_(rho.slice(0, ng, ng + n));
    col[IPR][0][0].copy_((rho * temperature).slice(0, ng, ng + n));
    auto [balanced, balance_error, sweeps] = balance_column(
        col, c->dx1f.slice(0, ng, ng + n).contiguous(), gravity, true, 1.e-11,
        200, "cartesian", c->options->stretched_x1());
    // Re-pin midpoint density after the homogeneous balance projection.
    // Cubic interpolation avoids introducing an O(dz^2) reference-density bias.
    auto rb = balanced[IDN][0][0];
    double rho_mid =
        (-rb[n / 2 - 2] + 9 * rb[n / 2 - 1] + 9 * rb[n / 2] - rb[n / 2 + 1])
            .item<double>() /
        16.;
    balanced[IDN].div_(rho_mid);
    balanced[IPR].div_(rho_mid);
    auto w =
        torch::zeros({5, co->nc3(), co->nc2(), co->nc1()}, torch::kFloat64);
    for (int q : {IDN, IPR}) {
      w[q].narrow(-1, ng, n).copy_(balanced[q]);
      w[q].narrow(-1, 0, ng).copy_(w[q].narrow(-1, ng, 1));
      w[q].narrow(-1, ng + n, ng).copy_(w[q].narrow(-1, ng + n - 1, 1));
    }
    auto zz = z.view({1, 1, -1}), x = c->x2v.view({1, -1, 1});
    auto W = torch::sin(M_PI * (zz + .5)),
         DW = M_PI * torch::cos(M_PI * (zz + .5));
    auto shape = W * torch::cos(k * x);
    double Q = M_PI * M_PI + k * k;
    w[IVX].copy_(seed * chi * Q * shape);
    w[IVY].copy_(-seed * chi * Q / k * (DW - (gravity - a) / (1 - a * zz) * W) *
                 torch::sin(k * x));
    auto base_temperature = w[IPR] / w[IDN];
    w[IDN].copy_(w[IPR] /
                 (base_temperature + seed * (a - dissipation) * shape));
    mesh->to(device);
    MeshVariables vars(1);
    vars[0]["hydro_w"] = w.to(device);
    mesh->initialize(vars);
    auto sl = b->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    auto vol = c->cell_volume().index(sl), f = shape.to(device).index(sl);
    auto norm = (vol * f * f).sum(), total_volume = vol.sum();
    auto initial_rho = vars[0]["hydro_u"][IDN].index(sl).clone();
    auto target_t = (1 - a * c->x1v)
                        .view({1, 1, -1})
                        .expand_as(w[IDN].to(device))
                        .index(sl);
    std::string path(argv[10]);
    std::ofstream csv(path);
    TORCH_CHECK(csv.good(), "cannot open CSV output");
    csv << std::setprecision(16)
        << "time,diffusive_time,amplitude,vertical_rms,temperature_drift,"
           "density_drift\n";
    auto record = [&](double time) {
      auto u = vars[0]["hydro_u"];
      auto vz = (u[IVX] / u[IDN]).index(sl);
      auto primitive = b->phydro->peos->compute("U->W", {u.clone()});
      auto temp = b->phydro->peos->compute("W->T", {primitive}).index(sl);
      auto stats =
          torch::stack({(vol * f * vz).sum() / norm,
                        torch::sqrt((vol * vz * vz).sum() / total_volume),
                        (temp - target_t).abs().max(),
                        (u[IDN].index(sl) - initial_rho).abs().max()})
              .cpu();
      csv << time << ',' << time * chi;
      for (int j = 0; j < 4; ++j) csv << ',' << stats[j].item<double>();
      csv << '\n';
      csv.flush();
    };
    std::cout << std::setprecision(14) << "device=" << device << " Ra_sa=" << ra
              << " a=" << a << " D=" << dissipation << " gamma=" << gamma
              << " k=" << k << " nz=" << n << " chi=" << chi
              << " balance_residual=" << balance_error << std::endl;
    record(0.);
    auto start = std::chrono::steady_clock::now();
    double time = 0.;
    int cycle = 0, retries = 0;
    while (time * chi < end) {
      double dt = std::min(mesh->max_time_step(vars), (end - time * chi) / chi);
      TORCH_CHECK(std::isfinite(dt) && dt > 0., "invalid timestep");
      mesh->set_cycle(++cycle);
      for (;;) {
        for (int stage = 0; stage < b->pintg->stages.size(); ++stage)
          mesh->forward(vars, dt, stage);
        int redo = mesh->check_redo(vars);
        TORCH_CHECK(redo >= 0, "step retry limit");
        if (redo == 0) break;
        ++retries;
        dt *= .5;
      }
      time += dt;
      if (cycle % 25 == 0 || time * chi >= end) record(time);
      if (cycle % 1000 == 0)
        std::cout << "cycle=" << cycle << " t_diff=" << time * chi << std::endl;
    }
    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::ofstream metadata(path + ".json");
    metadata << std::setprecision(16) << "{\n"
             << "\"device\": \"" << device << "\",\n"
             << "\"Ra\": " << ra << ", \"a\": " << a
             << ", \"D\": " << dissipation << ",\n"
             << "\"gamma\": " << gamma << ", \"Pr\": 1, \"k\": " << k << ",\n"
             << "\"nz\": " << n << ", \"nx\": " << 2 * n << ", \"cfl\": " << cfl
             << ",\n"
             << "\"seed\": " << seed << ", \"chi_mid\": " << chi << ",\n"
             << "\"end_diffusive_time\": " << time * chi
             << ", \"cycles\": " << cycle << ",\n"
             << "\"retries\": " << retries << ", \"seconds\": " << elapsed
             << ",\n"
             << "\"balance_residual\": " << balance_error << "\n}\n";
    std::cout << "completed " << cycle << " cycles in " << elapsed << " s"
              << std::endl;
  } catch (std::exception const &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
