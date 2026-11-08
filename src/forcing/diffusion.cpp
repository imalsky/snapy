// C/C++
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

// torch
#include <c10/core/InferenceMode.h>

// yaml
#include <yaml-cpp/yaml.h>

// snap
#include <snap/snap.h>

#include <snap/coord/coordinate.hpp>
#include <snap/hydro/hydro.hpp>
#include <snap/input/check_keys.hpp>
#include <snap/mesh/meshblock.hpp>

#include "forcing.hpp"

namespace snap {
namespace {

constexpr int kSpatialDims[] = {3, 2, 1};
using Region = std::array<int64_t, 3>;

torch::Tensor cell_width(Coordinate const& coord, int idir) {
  if (idir == 0) return coord->center_width1();
  if (idir == 1) return coord->center_width2();
  return coord->center_width3();
}

torch::Tensor center_distance(Coordinate const& coord, int idir) {
  if (idir == 0) return coord->center_distance1();
  if (idir == 1) return coord->center_distance2();
  return coord->center_distance3();
}

std::vector<torch::indexing::TensorIndex> region_index(Region const& start,
                                                       Region const& end) {
  return {torch::indexing::Slice(start[0], end[0]),
          torch::indexing::Slice(start[1], end[1]),
          torch::indexing::Slice(start[2], end[2])};
}

torch::Tensor region_slice(torch::Tensor value, Region const& start,
                           Region const& end) {
  return value.index(region_index(start, end));
}

std::pair<Region, Region> region_bounds(
    std::vector<torch::indexing::TensorIndex> const& index) {
  Region start;
  Region end;
  for (int dim = 0; dim < 3; ++dim) {
    start[dim] = index[dim].slice().start().expect_int();
    end[dim] = index[dim].slice().stop().expect_int();
  }
  return {start, end};
}

torch::Tensor spacing_slice(torch::Tensor spacing, int idir,
                            Region const& start, Region const& end) {
  auto dim = kSpatialDims[idir] - 1;
  return spacing.slice(dim, start[dim], end[dim]);
}

torch::Tensor centered_derivative(torch::Tensor value, Coordinate const& coord,
                                  int idir, Region start, Region end) {
  auto dim = kSpatialDims[idir] - 1;
  auto upper_start = start;
  auto upper = end;
  auto lower = start;
  auto lower_end = end;
  ++upper_start[dim];
  ++upper[dim];
  --lower[dim];
  --lower_end[dim];
  return (region_slice(value, upper_start, upper) -
          region_slice(value, lower, lower_end)) /
         (2. * spacing_slice(cell_width(coord, idir), idir, start, end));
}

torch::Tensor face_normal_derivative(torch::Tensor value,
                                     Coordinate const& coord, int idir,
                                     Region start, Region end) {
  auto dim = kSpatialDims[idir] - 1;
  auto lower = start;
  auto lower_end = end;
  --lower[dim];
  --lower_end[dim];
  return (region_slice(value, start, end) -
          region_slice(value, lower, lower_end)) /
         spacing_slice(center_distance(coord, idir), idir, start, end);
}

torch::Tensor face_average(torch::Tensor value, int idir, Region start,
                           Region end) {
  auto dim = kSpatialDims[idir] - 1;
  auto lower = start;
  auto lower_end = end;
  --lower[dim];
  --lower_end[dim];
  return 0.5 * (region_slice(value, start, end) +
                region_slice(value, lower, lower_end));
}

//! Linear extrapolation of `value` onto a physical wall face from the two
//! nearest ACTIVE cells. Weights come from the actual cell widths, so a
//! stretched mesh is handled correctly. Falls back to the nearest active cell
//! if the extrapolation is non-positive; that value also reads no ghost.
torch::Tensor extrapolate_to_wall(torch::Tensor value, Coordinate const& coord,
                                  int idir, Region start, Region end,
                                  bool upper) {
  auto dim = kSpatialDims[idir] - 1;
  auto near = start;
  auto near_end = end;
  auto next = start;
  auto next_end = end;
  near[dim] = upper ? end[dim] - 2 : start[dim];
  next[dim] = upper ? end[dim] - 3 : start[dim] + 1;
  near_end[dim] = near[dim] + 1;
  next_end[dim] = next[dim] + 1;

  auto width = cell_width(coord, idir);
  auto wnear = spacing_slice(width, idir, near, near_end);
  auto wnext = spacing_slice(width, idir, next, next_end);
  auto t = wnear / (wnear + wnext);

  auto vnear = region_slice(value, near, near_end);
  auto vnext = region_slice(value, next, next_end);
  auto wall = (1. + t) * vnear - t * vnext;
  return torch::where(wall > 0., wall, vnear);
}

//! Face-centred diffusion coefficient. Interior faces take the two-cell
//! average; a physical wall face is extrapolated from active cells instead,
//! because the ghost there is filled by a boundary condition to serve a
//! different operator and is not the physical state at the wall.
torch::Tensor face_coefficient(torch::Tensor value, Coordinate const& coord,
                               int idir, Region start, Region end,
                               bool wall_lower, bool wall_upper) {
  auto out = face_average(value, idir, start, end);
  auto dim = kSpatialDims[idir] - 1;
  auto nface = end[dim] - start[dim];
  if (nface < 3) return out;  // fewer than two active cells to extrapolate from

  if (wall_lower) {
    out.narrow(dim, 0, 1).copy_(
        extrapolate_to_wall(value, coord, idir, start, end, false));
  }
  if (wall_upper) {
    out.narrow(dim, nface - 1, 1)
        .copy_(extrapolate_to_wall(value, coord, idir, start, end, true));
  }
  return out;
}

//! value times an x1 profile at the faces: the coefficient of their product,
//! so an interior face takes the mean of the two cells' products. The product
//! of the two means is not the mean of the products: for a uniform product
//! (a constant dynamic coefficient over a stratified density) it adds
//! (dvalue)^2 / (4 value_a value_b) to each face, and the flux of a conductive
//! or viscous equilibrium is no longer divergence-free
//! (docs/derivations/diffusion-face-coefficient.md).
torch::Tensor face_scaled_coefficient(torch::Tensor value, torch::Tensor scale,
                                      Coordinate const& coord, int idir,
                                      Region start, Region end, bool wall_lower,
                                      bool wall_upper) {
  if (idir == 0) {
    return face_coefficient(value * scale.view({1, 1, -1}), coord, idir, start,
                            end, wall_lower, wall_upper);
  }
  // x2/x3 faces: both cells share the x1 profile, so the mean of the products
  // equals the product of the means; keep that form, bitwise as before
  auto out = face_average(value, idir, start, end) *
             scale.slice(0, start[2], end[2]).view({1, 1, -1});
  auto dim = kSpatialDims[idir] - 1;
  auto nface = end[dim] - start[dim];
  if (nface < 3 || (!wall_lower && !wall_upper)) return out;

  auto product = value * scale.view({1, 1, -1});
  if (wall_lower) {
    out.narrow(dim, 0, 1).copy_(
        extrapolate_to_wall(product, coord, idir, start, end, false));
  }
  if (wall_upper) {
    out.narrow(dim, nface - 1, 1)
        .copy_(extrapolate_to_wall(product, coord, idir, start, end, true));
  }
  return out;
}

//! a (2, n) table of x1 knots and scale: n >= 2, finite, x1 strictly
//! increasing, scale > 0
torch::Tensor checked_table(torch::Tensor const& table, char const* name) {
  TORCH_CHECK(table.dim() == 2 && table.size(0) == 2 && table.size(1) >= 2,
              "[Diffusion] ", name,
              " table must hold x1 knots and scale values, at least two of "
              "each; got ",
              table.sizes());
  auto t = table.to(torch::kCPU, torch::kFloat64).contiguous();
  TORCH_CHECK(torch::isfinite(t).all().item<bool>(), "[Diffusion] ", name,
              " table must be finite.");
  TORCH_CHECK((t[0].slice(0, 1) > t[0].slice(0, 0, -1)).all().item<bool>(),
              "[Diffusion] ", name, " x1 knots must be strictly increasing.");
  TORCH_CHECK(t[1].min().item<double>() > 0., "[Diffusion] ", name,
              " scale values must be strictly positive.");
  return t;
}

//! a checked table, linear in x1 between knots and held beyond the ends, at the
//! cell centres x1v; a centre on a knot takes that knot's value exactly
torch::Tensor profile_from_table(torch::Tensor const& table,
                                 torch::Tensor const& x1v) {
  auto x = table[0].contiguous(), s = table[1].contiguous();
  auto xv = x1v.to(torch::kCPU, torch::kFloat64).contiguous();
  auto i = (torch::searchsorted(x, xv, /*out_int32=*/false, /*right=*/true) - 1)
               .clamp(0, x.size(0) - 2);
  auto x0 = x.index_select(0, i), x1 = x.index_select(0, i + 1);
  auto f = ((xv - x0) / (x1 - x0)).clamp(0., 1.);
  return (1. - f) * s.index_select(0, i) + f * s.index_select(0, i + 1);
}

//! the four profile options, in a fixed order
std::array<torch::Tensor, 4> profile_options(DiffusionOptionsImpl const& op) {
  return {op.nu_scale_x1(), op.nu_scale_x1_table(), op.kappa_scale_x1(),
          op.kappa_scale_x1_table()};
}

//! an in-place write bumps it; reset refuses inference tensors, which keep none
int64_t version_of(torch::Tensor const& t) {
  return t.defined() ? t._version() : 0;
}

bool active(Coordinate const& coord, int idir) {
  if (idir == 0) return coord->options->nc1() > 1;
  if (idir == 1) return coord->options->nc2() > 1;
  return coord->options->nc3() > 1;
}

}  // namespace

DiffusionOptions DiffusionOptionsImpl::from_yaml(YAML::Node const& forcing) {
  if (!forcing["diffusion"]) return nullptr;

  auto node = forcing["diffusion"];
  TORCH_CHECK(!node["K"] && !node["type"],
              "DiffusionOptions: legacy 'K' and 'type' keys are unsupported; "
              "use 'nu_iso' and 'kappa_iso'.");
  check_keys(node, "forcing/diffusion",
             {"nu_iso", "kappa_iso", "dynamic", "nu_scale_x1", "kappa_scale_x1",
              "temperature_inner", "temperature_outer", "wall_velocity",
              "implicit_radial"});

  // the tables are ordinary tensors even when parsed under inference mode
  c10::InferenceMode not_inference(false);
  auto op = DiffusionOptionsImpl::create();
  auto take_non_negative = [&](char const* key) {
    if (!node[key]) return 0.;
    auto const value = node[key];
    TORCH_CHECK(value.IsScalar(), "DiffusionOptions: ", key,
                " must be a finite number >= 0.");
    double parsed = 0.;
    try {
      parsed = value.as<double>();
    } catch (YAML::Exception const&) {
      TORCH_CHECK(false, "DiffusionOptions: ", key,
                  " must be a finite number >= 0, got '", value.Scalar(), "'.");
    }
    TORCH_CHECK(std::isfinite(parsed) && parsed >= 0.,
                "DiffusionOptions: ", key,
                " must be a finite number >= 0, got ", parsed, ".");
    return parsed;
  };
  op->nu_iso() = take_non_negative("nu_iso");
  op->kappa_iso() = take_non_negative("kappa_iso");
  op->temperature_inner() = take_non_negative("temperature_inner");
  op->temperature_outer() = take_non_negative("temperature_outer");
  if (node["wall_velocity"])
    op->wall_velocity() = node["wall_velocity"].as<std::string>();
  if (node["implicit_radial"]) {
    auto value = node["implicit_radial"].as<std::string>();
    TORCH_CHECK(value == "true" || value == "false",
                "[Diffusion] implicit_radial must be true or false.");
    op->implicit_radial() = value == "true";
  }
  if (node["dynamic"]) {
    auto const flag = node["dynamic"];
    TORCH_CHECK(flag.IsScalar(),
                "DiffusionOptions: dynamic must be true or false.");
    auto const text = flag.Scalar();
    TORCH_CHECK(text == "true" || text == "false",
                "DiffusionOptions: dynamic must be true or false, got '", text,
                "'.");
    op->dynamic() = text == "true";
  }
  // {x1: [...], scale: [...]}: an x1 profile as a table in the x1 coordinate
  auto take_table = [&](char const* key) {
    if (!node[key]) return torch::Tensor();
    auto const table = node[key];
    TORCH_CHECK(table.IsMap(), "DiffusionOptions: ", key,
                " must be a table {x1: [...], scale: [...]}.");
    check_keys(table, std::string("forcing/diffusion/") + key, {"x1", "scale"});
    auto numbers = [&](char const* sub) {
      auto const list = table[sub];
      TORCH_CHECK(list && list.IsSequence(), "DiffusionOptions: ", key, ".",
                  sub, " must be a list of numbers.");
      std::vector<double> out;
      for (auto const& item : list) {
        TORCH_CHECK(item.IsScalar(), "DiffusionOptions: ", key, ".", sub,
                    " must be a list of numbers.");
        try {
          out.push_back(item.as<double>());
        } catch (YAML::Exception const&) {
          TORCH_CHECK(false, "DiffusionOptions: ", key, ".", sub,
                      " must be a list of numbers, got '", item.Scalar(), "'.");
        }
      }
      return out;
    };
    auto x1 = numbers("x1");
    auto scale = numbers("scale");
    TORCH_CHECK(x1.size() == scale.size(), "DiffusionOptions: ", key,
                " needs as many scale values as x1 knots; got ", scale.size(),
                " and ", x1.size(), ".");
    return checked_table(torch::stack({torch::tensor(x1, torch::kFloat64),
                                       torch::tensor(scale, torch::kFloat64)}),
                         key);
  };
  op->nu_scale_x1_table() = take_table("nu_scale_x1");
  op->kappa_scale_x1_table() = take_table("kappa_scale_x1");
  return op;
}

DiffusionImpl::DiffusionImpl(DiffusionOptions const& options_,
                             torch::nn::Module* p)
    : options(options_) {
  phydro = dynamic_cast<HydroImpl const*>(p);
  reset();
}

void DiffusionImpl::reset() {
  TORCH_CHECK(phydro, "[Diffusion] Parent Hydro is null");

  auto coord = phydro->pmb->pcoord;
  auto enabled = options->nu_iso() > 0. || options->kappa_iso() > 0.;
  TORCH_CHECK(!enabled || !coord->options->stretched_x1(),
              "[Diffusion] not supported on a stretched x1 (geometry/x1f or "
              "cells/x1rat, #280): its x1 differences assume uniform cells");
  bool spherical = coord->options->type() == "gnomonic-equiangle";
  TORCH_CHECK(!options->implicit_radial() ||
                  (spherical && phydro->peos->options->type() == "ideal-gas"),
              "[Diffusion] implicit_radial requires cubed sphere and dry "
              "ideal-gas EOS.");
  TORCH_CHECK(
      !enabled || coord->options->type() == "cartesian" || spherical,
      "[Diffusion] Supported coordinates: cartesian, gnomonic-equiangle.");
  TORCH_CHECK(std::isfinite(options->temperature_inner()) &&
                  options->temperature_inner() >= 0. &&
                  std::isfinite(options->temperature_outer()) &&
                  options->temperature_outer() >= 0.,
              "[Diffusion] Wall temperatures must be finite and nonnegative.");
  TORCH_CHECK(options->wall_velocity() == "stress-free" ||
                  options->wall_velocity() == "no-slip",
              "[Diffusion] wall_velocity must be stress-free or no-slip.");
  TORCH_CHECK(
      spherical || options->wall_velocity() == "stress-free",
      "[Diffusion] The no-slip wall option requires gnomonic-equiangle.");
  if (!spherical && (options->temperature_inner() > 0. ||
                     options->temperature_outer() > 0.)) {
    auto const& names = phydro->pmb->options->bcnames();
    TORCH_CHECK(coord->options->type() == "cartesian" && names.size() >= 2 &&
                    (options->temperature_inner() == 0. ||
                     names[0] == "reflecting_inner") &&
                    (options->temperature_outer() == 0. ||
                     names[1] == "reflecting_outer"),
                "[Diffusion] Cartesian prescribed temperatures require "
                "reflecting x1 walls.");
  }
  if (enabled && spherical) {
    auto op = coord->options;
    TORCH_CHECK(op->nghost() >= 2 && op->nx1() >= std::max(4, op->nghost()) &&
                    op->nx2() >= std::max(4, op->nghost()) &&
                    op->nx3() >= std::max(4, op->nghost()) &&
                    coord->x1v.min().item<double>() > 0.,
                "[Diffusion] Cubed sphere requires positive radii, nghost >= 2 "
                "and at least max(4, nghost) active cells per direction.");
    TORCH_CHECK(
        phydro->pmb->options->layout()->type() == "cubed-sphere" &&
            phydro->pmb->options->layout()->pz() == 1,
        "[Diffusion] Cubed sphere requires cubed-sphere layout and nb1=1.");
    auto const& names = phydro->pmb->options->bcnames();
    TORCH_CHECK(
        names.size() >= 2 && names[0].compare(0, 10, "reflecting") == 0 &&
            names[1].compare(0, 10, "reflecting") == 0,
        "[Diffusion] Cubed sphere currently requires reflecting radial walls.");
  }
  TORCH_CHECK(options->nu_iso() == 0. || coord->options->nghost() >= 2,
              "[Diffusion] Isotropic viscosity requires nghost >= 2.");
  TORCH_CHECK(options->kappa_iso() == 0. || phydro->peos->species_cv_ref() > 0.,
              "[Diffusion] Isotropic heat conduction requires an EOS with a "
              "positive reference specific heat at constant volume.");

  // an inference tensor keeps no version counter, so a later in-place write to
  // it could not be detected
  auto given = profile_options(*options);
  char const* const names[] = {"nu_scale_x1", "nu_scale_x1_table",
                               "kappa_scale_x1", "kappa_scale_x1_table"};
  for (size_t i = 0; i < given.size(); ++i) {
    TORCH_CHECK(!given[i].defined() || !given[i].is_inference(), "[Diffusion] ",
                names[i],
                " is an inference tensor (made under torch.inference_mode()); "
                "pass an ordinary tensor, made outside inference mode.");
  }

  // one value per x1 cell centre, ghosts included, from a tensor or a table
  auto nc1 = coord->options->nc1();
  auto layout = phydro->pmb->options->layout();
  int nb1 = layout ? layout->pz() : 1;
  auto profile = [&](torch::Tensor const& cells, torch::Tensor const& table,
                     char const* name, double* max) {
    *max = 1.;
    if (!cells.defined() && !table.defined()) return torch::Tensor();
    TORCH_CHECK(!cells.defined() || !table.defined(), "[Diffusion] ", name,
                " is given both per cell and as a table; give one.");
    TORCH_CHECK(!options->dynamic(), "[Diffusion] ", name,
                " scales the kinematic coefficient and has no meaning with "
                "dynamic: true.");
    torch::Tensor scale;
    if (table.defined()) {
      scale = profile_from_table(checked_table(table, name), coord->x1v);
    } else {
      // the options, and so this tensor, are shared by every block
      TORCH_CHECK(nb1 == 1, "[Diffusion] ", name,
                  " given per cell covers one block's x1 cells, but x1 is "
                  "split into ",
                  nb1,
                  " blocks; give it as a table in x1 instead (YAML {x1: [...], "
                  "scale: [...]}, or ",
                  name, "_table).");
      TORCH_CHECK(cells.dim() == 1 && cells.numel() == nc1, "[Diffusion] ",
                  name, " must be a 1-D profile over this block's nc1 = ", nc1,
                  " x1 cell centres, ghosts included; got ", cells.sizes());
      TORCH_CHECK(cells.scalar_type() == torch::kFloat64, "[Diffusion] ", name,
                  " must be float64.");
      scale = cells.to(torch::kCPU).clone();
      TORCH_CHECK(torch::isfinite(scale).all().item<bool>() &&
                      scale.min().item<double>() > 0.,
                  "[Diffusion] ", name,
                  " must be finite and strictly positive.");
    }
    *max = scale.max().item<double>();
    return scale;
  };
  nu_scale_ = profile(options->nu_scale_x1(), options->nu_scale_x1_table(),
                      "nu_scale_x1", &nu_scale_max_);
  kappa_scale_ =
      profile(options->kappa_scale_x1(), options->kappa_scale_x1_table(),
              "kappa_scale_x1", &kappa_scale_max_);
  profile_options_ = given;
  for (size_t i = 0; i < profile_options_.size(); ++i) {
    profile_versions_[i] = version_of(profile_options_[i]);
    // a write through outside storage (torch.from_numpy, .data) bumps no
    // version, so the values are compared as well
    profile_values_[i] =
        given[i].defined() ? given[i].detach().clone() : torch::Tensor();
  }
  nu_scale_w_ = torch::Tensor();
  kappa_scale_w_ = torch::Tensor();
}

void DiffusionImpl::check_profiles() const {
  // a profile set, replaced or changed after construction never reached reset
  auto now = profile_options(*options);
  bool any = false;
  for (size_t i = 0; i < now.size(); ++i) {
    auto const& seen = profile_options_[i];
    TORCH_CHECK(
        now[i].defined() == seen.defined() &&
            (!seen.defined() || (now[i].is_same(seen) &&
                                 version_of(now[i]) == profile_versions_[i] &&
                                 torch::equal(now[i], profile_values_[i]))),
        "[Diffusion] an x1 coefficient profile must be set before the "
        "MeshBlock is constructed and not changed afterwards.");
    any = any || seen.defined();
  }
  // reset refuses this pair; dynamic is read live, so check it again
  TORCH_CHECK(!any || !options->dynamic(),
              "[Diffusion] an x1 coefficient profile has no meaning with "
              "dynamic: true, set after the MeshBlock was constructed.");
}

torch::Tensor DiffusionImpl::forward(torch::Tensor du, torch::Tensor w,
                                     torch::Tensor temp, double dt) {
  auto pmb = phydro->pmb;
  auto coord = pmb->pcoord;

  check_profiles();
  // in the state's device and dtype, so a profile of ones changes no bit
  auto on_state = [&w](torch::Tensor const& scale, torch::Tensor& work) {
    if (scale.defined() && (!work.defined() || work.device() != w.device() ||
                            work.scalar_type() != w.scalar_type())) {
      work = scale.to(w.device(), w.scalar_type());
    }
  };
  on_state(nu_scale_, nu_scale_w_);
  on_state(kappa_scale_, kappa_scale_w_);
  if (coord->options->type() == "gnomonic-equiangle") {
    if (options->nu_iso() == 0. && options->kappa_iso() == 0.) return du;
    return forward_cubed_sphere(du, w, temp, dt);
  }

  auto [cell_start, cell_end] = region_bounds(
      pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3)));
  PartOptions div_options;
  div_options.exterior(false).ndim(3);
  for (int idir = 0; idir < 3; ++idir) {
    if (!active(coord, idir)) continue;
    if (idir == 0) div_options.extend_x1(1);
    if (idir == 1) div_options.extend_x2(1);
    if (idir == 2) div_options.extend_x3(1);
  }
  auto [div_start, div_end] = region_bounds(pmb->part({0, 0, 0}, div_options));

  torch::Tensor div_vel;
  torch::Tensor rho_cv;

  if (options->nu_iso() > 0.) {
    div_vel = torch::zeros_like(w[IDN]);
    auto div_vel_region = region_slice(div_vel, div_start, div_end);
    for (int idir = 0; idir < 3; ++idir) {
      if (active(coord, idir)) {
        div_vel_region +=
            centered_derivative(w[IVX + idir], coord, idir, div_start, div_end);
      }
    }
  }
  if (options->kappa_iso() > 0. && !options->dynamic()) {
    // Fourier flux -kappa grad T. rho*cv converts the kinematic diffusivity
    // into a conductivity; W->T is in kelvin.
    rho_cv = w[IDN] * phydro->peos->specific_heat_cv(w, temp);
  }

  std::array<torch::Tensor, 3> fluxes;
  bool has_flux = false;
  for (int idir = 0; idir < 3; ++idir) {
    if (!active(coord, idir)) continue;
    has_flux = true;

    auto dim = kSpatialDims[idir] - 1;
    auto face_start = cell_start;
    auto face_end = cell_end;
    ++face_end[dim];
    auto face_index = region_index(face_start, face_end);
    auto flux = torch::zeros_like(w);

    // x1 only: the vertical is where a reflecting wall cuts a monotone,
    // gravity-stratified profile. A lateral reflecting face may be a true
    // symmetry plane, where the two-cell average is the better estimate, and
    // reflecting is the default for every unspecified face, so extending this
    // to x2/x3 would opt configurations in silently.
    bool wall_lower = idir == 0 && pmb->options->is_wall_boundary(0, 0, -1);
    bool wall_upper = idir == 0 && pmb->options->is_wall_boundary(0, 0, 1);

    // rho at the face, times the viscosity profile when one is given; the
    // dynamic form carries no face density
    torch::Tensor rho_face;
    if (!options->dynamic()) {
      rho_face = nu_scale_w_.defined()
                     ? face_scaled_coefficient(w[IDN], nu_scale_w_, coord, idir,
                                               face_start, face_end, wall_lower,
                                               wall_upper)
                     : face_coefficient(w[IDN], coord, idir, face_start,
                                        face_end, wall_lower, wall_upper);
    }

    if (options->nu_iso() > 0.) {
      auto div_face = face_average(div_vel, idir, face_start, face_end);
      for (int ivar = 0; ivar < 3; ++ivar) {
        auto stress = face_normal_derivative(w[IVX + ivar], coord, idir,
                                             face_start, face_end);
        if (ivar == idir) {
          stress = 2. * stress - (2. / 3.) * div_face;
        } else if (active(coord, ivar)) {
          auto shear_start = face_start;
          auto shear_end = face_end;
          --shear_start[dim];
          --shear_end[dim];
          stress += 0.5 * (centered_derivative(w[IVX + idir], coord, ivar,
                                               face_start, face_end) +
                           centered_derivative(w[IVX + idir], coord, ivar,
                                               shear_start, shear_end));
        }

        auto momentum_flux = options->dynamic()
                                 ? -options->nu_iso() * stress
                                 : -options->nu_iso() * rho_face * stress;
        flux[IVX + ivar].index(face_index).copy_(momentum_flux);
        flux[IPR].index(face_index) +=
            face_average(w[IVX + ivar], idir, face_start, face_end) *
            momentum_flux;
      }
    }

    if (options->kappa_iso() > 0.) {
      auto dtdn =
          face_normal_derivative(temp, coord, idir, face_start, face_end);
      if (idir == 0) {
        // Conductive Dirichlet values are independent of the reflecting
        // hydro ghosts, which must remain suitable for hydrostatic balance.
        // Zero preserves the existing Cartesian ghost-based heat flux.
        for (int side = 0; side < 2; ++side) {
          double tw = side ? options->temperature_outer()
                           : options->temperature_inner();
          if (tw == 0. || !(side ? wall_upper : wall_lower)) continue;
          int ic = side ? coord->iu() : coord->il();
          int face = side ? coord->iu() + 1 : coord->il();
          auto tc = region_slice(temp, cell_start, cell_end)
                        .select(-1, side ? cell_end[2] - cell_start[2] - 1 : 0);
          auto distance =
              (coord->x1v[ic] - coord->x1f[face]).to(temp.options());
          dtdn.select(-1, side ? dtdn.size(-1) - 1 : 0)
              .copy_((tc - tw) / distance);
        }
      }
      if (options->dynamic()) {
        flux[IPR].index(face_index) -= options->kappa_iso() * dtdn;
      } else {
        // W->T is in kelvin, so convert thermal diffusivity to conductivity
        // using the local mixture volumetric heat capacity.
        flux[IPR].index(face_index) -=
            options->kappa_iso() *
            (kappa_scale_w_.defined()
                 ? face_scaled_coefficient(rho_cv, kappa_scale_w_, coord, idir,
                                           face_start, face_end, wall_lower,
                                           wall_upper)
                 : face_coefficient(rho_cv, coord, idir, face_start, face_end,
                                    wall_lower, wall_upper)) *
            dtdn;
      }
    }
    fluxes[idir] = flux;
  }

  if (has_flux) {
    du -= dt * coord->divergence(fluxes[0], fluxes[1], fluxes[2]);
  }
  return du;
}

double DiffusionImpl::max_time_step(torch::Tensor w) const {
  check_profiles();
  auto coord = phydro->pmb->pcoord;
  auto interior =
      phydro->pmb->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  int ndim = 0;
  double dx_min = std::numeric_limits<double>::max();

  for (int idir = 0; idir < 3; ++idir) {
    if (!active(coord, idir)) continue;
    ++ndim;
    dx_min = std::min(dx_min, cell_width(coord, idir)
                                  .expand_as(w[IDN])
                                  .index(interior)
                                  .min()
                                  .item<double>());
  }

  if (ndim == 0) return std::numeric_limits<double>::max();
  // each on its own: std::max(finite, NaN) returns the finite one
  TORCH_CHECK(
      std::isfinite(options->nu_iso()) && std::isfinite(options->kappa_iso()),
      "[Diffusion] diffusivity is not finite");
  double coeff;
  if (options->dynamic()) {
    auto rho_min = w[IDN].index(interior).min().item<double>();
    double thermal_diffusivity = 0.;
    if (options->kappa_iso() > 0.) {
      auto temp = phydro->peos->compute("W->T", {w});
      auto rho_cv_min = (w[IDN] * phydro->peos->specific_heat_cv(w, temp))
                            .index(interior)
                            .min()
                            .item<double>();
      thermal_diffusivity = options->kappa_iso() / rho_cv_min;
    }
    coeff = std::max(options->nu_iso() / rho_min, thermal_diffusivity);
  } else {
    coeff = std::max(options->nu_iso() * nu_scale_max_,
                     options->kappa_iso() * kappa_scale_max_);
  }
  if (coeff == 0.) return std::numeric_limits<double>::max();
  TORCH_CHECK(std::isfinite(coeff), "[Diffusion] diffusivity is not finite");
  double dt = dx_min * dx_min / (2. * ndim * coeff);
  // Conservative initial bound for corrected nonorthogonal gradients and
  // the 4/3 longitudinal viscous eigenvalue. Explicit stages only.
  if (coord->options->type() == "gnomonic-equiangle") dt *= 0.125;
  if (coord->options->type() == "gnomonic-equiangle" &&
      options->implicit_radial()) {
    double viscous = options->nu_iso() * nu_scale_max_;
    if (options->dynamic())
      viscous = options->nu_iso() / w[IDN].index(interior).min().item<double>();
    double rate = 0.;
    for (int idir = 0; idir < 3; ++idir) {
      auto dx = cell_width(coord, idir)
                    .expand_as(w[IDN])
                    .index(interior)
                    .min()
                    .item<double>();
      rate += (idir == 0 ? (4. / 3.) * viscous : (4. / 3.) * coeff) / (dx * dx);
    }
    dt = rate > 0. ? 1. / (16. * rate) : std::numeric_limits<double>::max();
  }
  TORCH_CHECK(std::isfinite(dt) && dt > 0.,
              "[Diffusion] time-step bound must be positive and finite, got ",
              dt);
  return dt;
}

}  // namespace snap
