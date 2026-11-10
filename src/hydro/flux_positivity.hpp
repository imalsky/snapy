#pragma once

// torch
#include <torch/torch.h>

// snap
#include <snap/coord/coordinate.hpp>

namespace snap {

//! Round-off, in ulp of the working precision, for the species mass the
//! positivity machinery withholds or repairs. flux_positivity_theta already
//! leaves this fraction of a cell's own species mass behind on purpose; a
//! withheld or repaired mass below this fraction of the cell's TOTAL gas mass
//! (rho_total * volume, all species) is reported as round-off, not as a
//! positivity event: it neither counts as severe nor marks a limiter patch
//! for MeshBlock::check_redo. The clip or repair itself is always applied.
constexpr double kPositivityRoundoffUlp = 4096.;

//! The same bound in float32, in float32 ulp. 4096 of them is 4.9e-4 of the
//! gas mass, the size of a real repair; a species update rounds by at most
//! 1 ulp of the gas mass. The margin of flux_positivity_theta stays 4096 ulp.
constexpr double kPositivityRoundoffUlpFloat = 64.;

//! A species deficit (a negative conserved density before the EOS species
//! repair), in ulp of the cell's TOTAL gas density, above which the repair
//! marks a limiter patch for MeshBlock::check_redo; a smaller one is repaired
//! without a redo. It defaults to the round-off bound above; its final value
//! is to be set from the redo logs of the TJ16 species deck.
constexpr double kSpeciesRepairMarkUlp = kPositivityRoundoffUlp;

//! The same mark in float32, in float32 ulp
constexpr double kSpeciesRepairMarkUlpFloat = kPositivityRoundoffUlpFloat;

//! machine epsilon of `dtype` (float32 for kFloat, float64 otherwise)
double machine_epsilon(c10::ScalarType dtype);

//! kPositivityRoundoffUlp (kPositivityRoundoffUlpFloat in float32) times the
//! machine epsilon of `dtype`
double positivity_roundoff(c10::ScalarType dtype);

//! kSpeciesRepairMarkUlp (kSpeciesRepairMarkUlpFloat in float32) times the
//! machine epsilon of `dtype`
double species_repair_mark(c10::ScalarType dtype);

//! \brief Per-cell positivity limiter factors for donor-form tracer fluxes.
//!
//! For each channel c and cell i, sums the outgoing (donor-side) flux over all
//! faces of the cell exactly as the divergence will apply them,
//!   out_i = sum_faces max(+/- A*F, 0),
//! and returns
//!   theta_i = min(1, u_i * V_i / (dt * out_i)),
//! the largest uniform scaling of cell i's outgoing fluxes that cannot drain
//! the cell below zero in one forward-Euler step of size dt. Applying theta of
//! the donor cell to every face (flux_positivity_scale_) then guarantees
//!   u_i + dt * du_i(transport) >= 0
//! cell by cell. All shipped integrators (rk1/rk2/rk3: wght2 == wght1 per
//! stage; rk3s4: wght2 <= wght1) form each stage as a convex combination of
//! previous states and one full-dt Euler step, so per-stage limiting at the
//! full dt preserves non-negativity of the stage updates as well.
//!
//! theta == 1 wherever the cell is not near depletion, so the high-order flux
//! is untouched almost everywhere; conservation is exact because each face is
//! scaled by a single factor shared by both adjacent cells.
//!
//! Ghost cells get theta = 1 here (their outflow sum is not computed); the
//! caller must make donor factors single-valued at internal seams by filling
//! theta's ghost layer the same way conserved-variable ghosts are filled
//! (exchange + physical boundary functions) BEFORE calling
//! flux_positivity_scale_.
//!
//! \param u      conserved tracer densities, (nchan, nc3, nc2, nc1)
//! \param flux1/2/3  channel-sliced flux views, same layout as u; undefined
//!               tensors are skipped. flux[i] is the flux through the lower
//!               face of cell i (divergence convention).
//! \param pcoord coordinate providing face_area1/2/3 and cell_volume
//! \param dt     full stage time step
//! \param drain  if given, receives dt * out_i, the species mass (not density)
//!               each cell would lose unlimited; the limiter withholds
//!               (1 - theta_i) * drain_i of it
torch::Tensor flux_positivity_theta(torch::Tensor const& u,
                                    torch::Tensor const& flux1,
                                    torch::Tensor const& flux2,
                                    torch::Tensor const& flux3,
                                    Coordinate const& pcoord, double dt,
                                    torch::Tensor* drain = nullptr);

//! \brief The share phi of a ratio q's relaxed drain that its neighbours repay.
//!
//! phi = clamp(min(q of the two neighbours along every resolved dimension) / q,
//! 0, 1), and 0 where q <= 0. A cell drained past its content gets its ratio
//! back from its neighbours (the implicit x1 refill, the inflow along x2 and
//! x3), so it may drain past its content only as far as the poorer one repays.
//! \param q  ratio to the dry density, (nchan, nc3, nc2, nc1)
torch::Tensor flux_positivity_share(torch::Tensor const& q);

//! \brief Scale each face's flux by the donor cell's theta, in place.
//!
//! The donor of a face is the cell the flux drains: the lower cell where the
//! (per-channel) flux is positive, the upper cell otherwise. Only the faces
//! the divergence consumes (lower faces il..iu+1 per dimension) are touched.
void flux_positivity_scale_(torch::Tensor const& theta,
                            torch::Tensor const& flux1,
                            torch::Tensor const& flux2,
                            torch::Tensor const& flux3,
                            Coordinate const& pcoord);

//! \brief Withhold, in place, the energy and momentum the species mass carries
//! that flux_positivity_scale_ is about to withhold (call it first).
//!
//! Per face and species, dm = (1 - theta_donor) * F_species; the energy flux
//! loses dm * hspec(donor) and the momentum flux dm * vel(donor). When fsed1
//! is given, the x1 species flux is split into its advected part F - fsed1
//! and its settling part fsed1: each loses the same share (1 - theta_donor,
//! donor of the net F) and carries hspec and vel of its own donor, picked by
//! the sign of that part.
//! \param hspec  energy per unit species mass, (ny, nc3, nc2, nc1)
//! \param vel    momentum per unit mass, (3, nc3, nc2, nc1)
//! \param flux1/2/3  full hydro fluxes; undefined tensors are skipped
//! \param fsed1  settling part of flux1's species flux, (ny, nc3, nc2, nc1);
//!               undefined: the whole flux is one part
void flux_positivity_carry_(
    torch::Tensor const& theta, torch::Tensor const& hspec,
    torch::Tensor const& vel, torch::Tensor const& flux1,
    torch::Tensor const& flux2, torch::Tensor const& flux3,
    Coordinate const& pcoord, torch::Tensor const& fsed1 = torch::Tensor());

}  // namespace snap
