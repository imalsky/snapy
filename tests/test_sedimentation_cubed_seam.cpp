// A two-rank vertical seam is the smallest distributed case that exercises
// sedimentation, the species-flux limiter, and the cubed-layout seam repair.
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <snap/forcing/forcing.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

double global_sum(Layout const& layout, double local) {
  auto value = torch::tensor({local}, torch::dtype(torch::kFloat64));
  std::vector<torch::Tensor> values = {value};
  layout->comm->allreduce(values, c10d::ReduceOp::SUM);
  return values[0].item<double>();
}

double condensate_mass(MeshBlock const& block, Variables const& vars) {
  auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
  auto species =
      vars.at("hydro_u").narrow(0, ICY, vars.at("hydro_u").size(0) - ICY);
  return (species * block->pcoord->cell_volume())
      .index(interior)
      .sum()
      .item<double>();
}

double max_component_difference(torch::Tensor const& difference,
                                int component) {
  if (component >= difference.size(0))
    return std::numeric_limits<double>::quiet_NaN();
  return difference[component].amax().item<double>();
}

}  // namespace

int main() {
  auto options = MeshOptionsImpl::from_yaml("test_gravity_sedimentation.yaml");
  options->block()->layout()->type("cubed").pz(2);
  options->blocks_per_process(1);
  options->block()->hydro()->eos()->limiter(true);
  auto gravity = ConstGravityOptionsImpl::create();
  gravity->grav1(-1.);
  options->block()->hydro()->grav() = gravity;
  auto mesh = Mesh(options);
  auto block = mesh->blocks.front();
  auto layout = block->get_layout();

  bool ok = layout->has_process_group() && layout->options->type() == "cubed" &&
            layout->options->pz() == 2 &&
            layout->options->blocks_per_process() == 1;
  int rank = layout->options->process_rank();
  auto coord = block->pcoord;
  auto w = torch::zeros({block->phydro->peos->nvar(), coord->options->nc3(),
                         coord->options->nc2(), coord->options->nc1()},
                        torch::dtype(torch::kFloat64));
  w[IPR].fill_(1.e5);
  // A nonuniform rising column makes the cloud's net flux upward while its
  // sedimentation component is downward.  Each rank reconstructs the mixed
  // face from a different local stencil, so the saved pre-average settling
  // contribution is intentionally not the same at the shared face.  dt=1
  // makes theta active.
  double rho[] = {1.00, 1.15, 0.90, 1.05, 0.85, 1.20};
  double vx[] = {3.0, 3.4, 2.6, 3.2, 2.8, 3.1};
  double vy[] = {3.0, 3.5, 2.5, 4.0, 2.0, 3.2};
  double vz[] = {0.0, 0.5, -0.5, 1.0, -1.0, 0.3};
  double cloud[] = {0.02, 0.02, 0.20, 0.01, 0.02, 0.02};
  for (int c = 0; c < 3; ++c) {
    int g = 3 * rank + c;
    int i = coord->il() + c;
    w[IDN].select(-1, i).fill_(rho[g]);
    w[IVX].select(-1, i).fill_(vx[g]);
    w[IVY].select(-1, i).fill_(vy[g]);
    w[IVZ].select(-1, i).fill_(vz[g]);
    w[ICY + 1].select(-1, i).fill_(cloud[g]);
  }
  w[ICY].fill_(0.01);
  MeshVariables vars(1);
  vars[0]["hydro_w"] = w;
  mesh->initialize(vars);

  double mass_before = global_sum(layout, condensate_mass(block, vars[0]));
  mesh->forward(vars, 1., 0);
  auto flux = block->phydro->flux1();
  int face = rank == 0 ? coord->iu() + 1 : coord->il();
  auto seam = flux.select(-1, face).contiguous();
  auto seam_sum = seam.clone();
  std::vector<torch::Tensor> seam_values = {seam_sum};
  layout->comm->allreduce(seam_values, c10d::ReduceOp::SUM);
  // With exactly two ranks, a local seam slab equals half of the global sum
  // iff the two rank-local slabs are identical.
  auto pair_difference = 2. * (seam - seam_values[0] * 0.5).abs();
  bool identical_flux = pair_difference.amax().item<double>() == 0.;
  std::array<double, 5> seam_component_difference = {
      max_component_difference(pair_difference, IDN),
      max_component_difference(pair_difference, IVX),
      max_component_difference(pair_difference, IVY),
      max_component_difference(pair_difference, IVZ),
      max_component_difference(pair_difference, IPR),
  };
  double seam_flux_mass = seam[IDN].sum().item<double>();
  double seam_flux_momentum1 = seam[IVX].sum().item<double>();
  double seam_flux_momentum2 = seam[IVY].sum().item<double>();
  double seam_flux_momentum3 = seam[IVZ].sum().item<double>();
  double seam_flux_energy = seam[IPR].sum().item<double>();
  // This mixed flux has an upward advective cloud part and a downward
  // settling part.  Its component sums lock the two donor corrections: the
  // pre-fix single-donor carry gives different momentum/energy values.
  // The vapour (q = 0.01 everywhere) is a uniform ratio over over-drained dry
  // air, so the limiter no longer withholds it; with the vapour withheld as
  // before, the cloud's carry gives 99901.819827557352 and 493328.12156816921.
  bool mixed_carry_correct =
      std::abs(seam_flux_momentum1 - 99901.855127834307) < 1.e-7 &&
      std::abs(seam_flux_energy - 499161.89154918579) < 5.e-3;
  double mass_after = global_sum(layout, condensate_mass(block, vars[0]));
  bool conserved = std::abs(mass_after - mass_before) <= 1.e-12 * mass_before;
  double limiter_hits = global_sum(
      layout,
      static_cast<double>(block->phydro->positivity_hits().item<int64_t>()));
  bool limiter_active = limiter_hits > 0;
  ok = ok && identical_flux && conserved && limiter_active &&
       mixed_carry_correct;

  if (rank == 0) {
    std::cout
        << std::setprecision(17) << "sedimentation cubed seam: ranks="
        << layout->options->process_world_size()
        << " pz=" << layout->options->pz() << " limiter_hits=" << limiter_hits
        << " identical_flux=" << identical_flux
        << " seam_max_difference_mass=" << seam_component_difference[0]
        << " seam_max_difference_momentum1=" << seam_component_difference[1]
        << " seam_max_difference_momentum2=" << seam_component_difference[2]
        << " seam_max_difference_momentum3=" << seam_component_difference[3]
        << " seam_max_difference_energy=" << seam_component_difference[4]
        << " mixed_carry_correct=" << mixed_carry_correct
        << " seam_flux_mass=" << seam_flux_mass
        << " seam_flux_momentum1=" << seam_flux_momentum1
        << " seam_flux_momentum2=" << seam_flux_momentum2
        << " seam_flux_momentum3=" << seam_flux_momentum3
        << " seam_flux_energy=" << seam_flux_energy
        << " condensate_mass_before=" << mass_before
        << " condensate_mass_after=" << mass_after << std::endl;
  }

  if (!ok) {
    std::cerr << std::setprecision(17) << "rank=" << rank
              << " cubed=" << layout->options->type()
              << " pz=" << layout->options->pz()
              << " limiter_hits=" << limiter_hits
              << " identical_flux=" << identical_flux
              << " mass_before=" << mass_before << " mass_after=" << mass_after
              << std::endl;
  }
  return ok ? 0 : 1;
}
