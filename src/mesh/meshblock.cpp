// C/C++
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <vector>

// snap
#include <snap/coord/coord_utils.hpp>
#include <snap/hydro/flux_positivity.hpp>
#include <snap/hydro/gravity_work_radial.hpp>
#include <snap/input/read_restart_file.hpp>
#include <snap/output/output_formats.hpp>
#include <snap/scalar/scalar_fused.hpp>
#include <snap/utils/log.hpp>
#include <snap/utils/signal_handler.hpp>

#include "cycle_diagnostics.hpp"
#include "meshblock.hpp"

namespace snap {

static std::mutex meshblock_mutex;

static void set_scalar_primitive(Variables &vars, torch::Tensor const &scalar_s,
                                 torch::Tensor const &hydro_u) {
  auto scalar_r = scalar_s / hydro_u[IDN].unsqueeze(0);
  auto it = vars.find("scalar_r");
  if (it != vars.end()) {
    it->second.set_(scalar_r);
  } else {
    vars["scalar_r"] = scalar_r;
  }
}

static c10::impl::GenericDict stage_forcing_variables(
    MeshBlockImpl const &block, Variables const &vars) {
  c10::impl::GenericDict inputs(c10::StringType::get(), c10::TensorType::get());
  for (auto const &item : block.named_buffers(/*recurse=*/true)) {
    inputs.insert(item.key(), item.value());
  }
  for (auto const &[name, value] : vars) {
    inputs.insert_or_assign(name, value);
  }
  return inputs;
}

static Variables stage_forcing_result(c10::IValue const &result, size_t index) {
  TORCH_CHECK(result.isGenericDict(), "User stage forcing ", index,
              " must return Dict[str, Tensor], but returned ", result.tagKind(),
              ".");
  Variables output;
  for (auto const &item : result.toGenericDict()) {
    TORCH_CHECK(item.key().isString(), "User stage forcing ", index,
                " returned a dictionary with a non-string key.");
    TORCH_CHECK(item.value().isTensor(), "User stage forcing ", index,
                " returned a non-Tensor value for key '",
                item.key().toStringRef(), "'.");
    output.emplace(item.key().toStringRef(), item.value().toTensor());
  }
  return output;
}

MeshBlockImpl::MeshBlockImpl(MeshBlockOptions const &options_)
    : options(options_) {
  // Output-slice validation below reads the global grid directly, long before
  // pcoord exists.
  options->coord()->resolve_global_grid();

  int nc1 = options->coord()->nc1();
  int nc2 = options->coord()->nc2();
  int nc3 = options->coord()->nc3();

  if (nc1 > 1 && options->bfuncs().size() < 2) {
    throw std::runtime_error("MeshBlockImpl: bfuncs size must be at least 2");
  }

  if (nc2 > 1 && options->bfuncs().size() < 4) {
    throw std::runtime_error("MeshBlockImpl: bfuncs size must be at least 4");
  }

  if (nc3 > 1 && options->bfuncs().size() < 6) {
    throw std::runtime_error("MeshBlockImpl: bfuncs size must be at least 6");
  }

  reset();
}

MeshBlockImpl::~MeshBlockImpl() {
  // destroy signal handler
  SignalHandler::Destroy();
}

void MeshBlockImpl::set_user_stage_forcings(
    std::vector<std::string> const &filenames) {
  std::vector<std::shared_ptr<torch::jit::Module>> loaded;
  loaded.reserve(filenames.size());
  auto device = torch::Device(options->device_str());
  for (auto const &filename : filenames) {
    auto module =
        std::make_shared<torch::jit::Module>(torch::jit::load(filename));
    module->to(device);
    module->eval();
    loaded.push_back(std::move(module));
  }
  user_stage_forcings = std::move(loaded);
}

void MeshBlockImpl::reset() {
  //// ---- (1) set up distributed environment ---- ////
  if (_playout == nullptr) {
    std::unique_lock<std::mutex> lock(meshblock_mutex);
    if (_playout == nullptr) {
      _playout = LayoutImpl::create(options->layout(), this);
    }
  }
  send_bufs.resize(_playout->num_exchange_buffers());
  recv_bufs.resize(_playout->num_exchange_buffers());
  exchange_buffer_cache.clear();

  int px = options->layout()->px();
  int py = options->layout()->py();
  int pz = options->layout()->pz();

  int nranks = px * py * pz;
  if (options->layout()->type() == "cubed-sphere") {
    nranks *= 6;
  }
  int rank = options->layout()->rank();

  TORCH_CHECK(options->layout()->world_size() == nranks,
              "MeshBlockImpl: world_size (", options->layout()->world_size(),
              ") does not match layout partitioning (", nranks, ").");

  //// ---- (2) reset internal block boundaries ---- ////
  if (options->layout()->type() != "cubed-sphere") {  // slab or cubed layout
    auto [lx2, lx3, lx1] = _playout->loc_of(rank);

    // x1-dir
    if ((lx1 != 0) || options->layout()->periodic_z()) {
      options->bfuncs()[BoundaryFace::kInnerX1] = nullptr;
    }

    if ((lx1 != pz - 1) || options->layout()->periodic_z()) {
      options->bfuncs()[BoundaryFace::kOuterX1] = nullptr;
    }

    // x2-dir
    if ((lx2 != 0) || options->layout()->periodic_x()) {
      options->bfuncs()[BoundaryFace::kInnerX2] = nullptr;
    }

    if ((lx2 != px - 1) || options->layout()->periodic_x()) {
      options->bfuncs()[BoundaryFace::kOuterX2] = nullptr;
    }

    // x3-dir
    if ((lx3 != 0) || options->layout()->periodic_y()) {
      options->bfuncs()[BoundaryFace::kInnerX3] = nullptr;
    }

    if ((lx3 != py - 1) || options->layout()->periodic_y()) {
      options->bfuncs()[BoundaryFace::kOuterX3] = nullptr;
    }

    if (options->verbose()) {
      SINFO(MeshBlock) << "setting up rank bcs" << std::endl;
      for (int i = 0; i < options->bfuncs().size(); ++i) {
        if (options->bfuncs()[i] == nullptr) {
          SINFO(MeshBlock) << "  bc func " << i << ": internal/custom"
                           << std::endl;
        } else {
          SINFO(MeshBlock) << "  bc func " << i << ": external" << std::endl;
        }
      }
    }
  }

  //// --------- (3) set up output --------- ////
  for (auto const &out_op : options->outputs()) {
    auto validate_slice = [](std::vector<double> const &slices, double lower,
                             double upper, bool sum, char const *axis) {
      if (!slices.empty() && sum) {
        throw std::invalid_argument(std::string("Cannot request both slice and "
                                                "sum along ") +
                                    axis);
      }
      for (size_t i = 0; i < slices.size(); ++i) {
        double slice = slices[i];
        if (!(slice >= lower && slice < upper)) {
          throw std::invalid_argument(
              std::string("Slice out of mesh bounds: ") + axis);
        }
        if (i && slice <= slices[i - 1]) {
          throw std::invalid_argument(
              std::string("Slices must be strictly increasing: ") + axis);
        }
      }
    };
    validate_slice(out_op->x1_slice(), options->coord()->global_x1min(),
                   options->coord()->global_x1max(), out_op->output_sumx1(),
                   "x1");
    validate_slice(out_op->x2_slice(), options->coord()->global_x2min(),
                   options->coord()->global_x2max(), out_op->output_sumx2(),
                   "x2");
    validate_slice(out_op->x3_slice(), options->coord()->global_x3min(),
                   options->coord()->global_x3max(), out_op->output_sumx3(),
                   "x3");

    if (out_op->file_type() == "restart") {
      output_types.push_back(std::make_shared<RestartOutput>(out_op));
    } else if (out_op->file_type() == "netcdf") {
      output_types.push_back(std::make_shared<NetcdfOutput>(out_op));
    } else if (out_op->file_type() == "pnetcdf") {
#ifdef PNETCDFOUTPUT
      output_types.push_back(std::make_shared<PNetcdfOutput>(out_op));
#else
      throw std::runtime_error("Output type 'pnetcdf' requires PNETCDF=ON.");
#endif
      /*} else if (out_op.file_type() == "hdf5") {
        output_types.push_back(
            std::make_shared<HDF5Output>(out_op));*/
    } else {
      throw std::runtime_error("Output type '" + out_op->file_type() +
                               "' is not implemented.");
    }

    if (options->verbose()) {
      SINFO(MeshBlock) << "adding output type: " << out_op->file_type()
                       << std::endl;
    }
  }

  //// -------- (4) set up integrator -------- ////
  pintg = harp::IntegratorImpl::create(options->intg(), this);
  if (options->verbose()) {
    SINFO(MeshBlock) << "using integrator type: " << pintg->options->type()
                     << std::endl;
  }

  //// ----- (5) set up coordinate model ------ ////
  pcoord = CoordinateImpl::create(options->coord(), this);
  if (options->verbose()) {
    SINFO(MeshBlock) << "using coordinate type: " << pcoord->options->type()
                     << "\n";
  }

  //// -------- (6) set up hydro model -------- ////
  phydro = HydroImpl::create(options->hydro(), this);
  if (options->verbose()) {
    SINFO(MeshBlock) << "using hydro type: " << phydro->peos->options->type()
                     << std::endl;
  }

  //// -------- (7) set up scalar model ------- ////
  pscalar = ScalarImpl::create(options->scalar(), this);

  //// ------ (8) set up internal boundary ---- ////
  pib = InternalBoundaryImpl::create(options->ib(), this);
  if (options->verbose()) {
    SINFO(MeshBlock) << "Internal boundary max-iter: "
                     << pib->options->max_iter() << "\n";
  }

  // dimensions
  int nc1 = options->coord()->nc1();
  int nc2 = options->coord()->nc2();
  int nc3 = options->coord()->nc3();
  auto peos = phydro->peos;

  //// ---------- (9) set up hydro buffer ------ ////
  TORCH_CHECK(phydro->peos->nvar() > 0, "Hydro model must have nvar > 0.");
  _hydro_u0 = register_buffer(
      "u0",
      torch::zeros({phydro->peos->nvar(), nc3, nc2, nc1}, torch::kFloat64));

  //// ------- (10) set up scalar buffer ------- ////
  _scalar_s0 = register_buffer(
      "s0", torch::zeros({pscalar->nvar(), nc3, nc2, nc1}, torch::kFloat64));

  if (options->verbose()) {
    SINFO(MeshBlock) << "setting up buffer with shapes:" << std::endl
                     << "* hydro_u0: " << _hydro_u0.sizes() << std::endl
                     << "* scalar_s0: " << _scalar_s0.sizes() << std::endl;
  }
}

std::vector<torch::indexing::TensorIndex> MeshBlockImpl::part(
    std::tuple<int, int, int> offset, PartOptions const &opts) const {
  int nc1 = options->coord()->nc1();
  int nc2 = options->coord()->nc2();
  int nc3 = options->coord()->nc3();
  int nghost_coord = options->coord()->nghost();

  int is_ghost = opts.exterior() ? 1 : 0;

  auto [o3, o2, o1] = offset;
  int start1, len1, start2, len2, start3, len3;

  int nx1 = nc1 > 1 ? nc1 - 2 * nghost_coord : 1;
  int nx2 = nc2 > 1 ? nc2 - 2 * nghost_coord : 1;
  int nx3 = nc3 > 1 ? nc3 - 2 * nghost_coord : 1;

  // ---- dimension 1 ---- //
  int nghost = nx1 == 1 ? 0 : nghost_coord;

  if (o1 == -1) {
    start1 = nghost * (1 - is_ghost);
    len1 = std::min(nghost, opts.depth());
  } else if (o1 == 0) {
    start1 = nghost - opts.extend_x1();
    len1 = nx1 + 2 * opts.extend_x1();
  } else {  // o1 == 1
    start1 = nx1 + nghost * is_ghost;
    len1 = std::min(nghost, opts.depth());
  }

  // ---- dimension 2 ---- //
  nghost = nx2 == 1 ? 0 : nghost_coord;

  if (o2 == -1) {
    start2 = nghost * (1 - is_ghost);
    len2 = std::min(nghost, opts.depth());
  } else if (o2 == 0) {
    start2 = nghost - opts.extend_x2();
    len2 = nx2 + 2 * opts.extend_x2();
  } else {  // o2 == 1
    start2 = nx2 + nghost * is_ghost;
    len2 = std::min(nghost, opts.depth());
  }

  // ---- dimension 3 ---- //
  nghost = nx3 == 1 ? 0 : nghost_coord;

  if (o3 == -1) {
    start3 = nghost * (1 - is_ghost);
    len3 = std::min(nghost, opts.depth());
  } else if (o3 == 0) {
    start3 = nghost - opts.extend_x3();
    len3 = nx3 + 2 * opts.extend_x3();
  } else {  // o3 == 1
    start3 = nx3 + nghost * is_ghost;
    len3 = std::min(nghost, opts.depth());
  }

  auto slice1 = torch::indexing::Slice(start1, start1 + len1);
  auto slice2 = torch::indexing::Slice(start2, start2 + len2);
  auto slice3 = torch::indexing::Slice(start3, start3 + len3);
  auto slice4 = torch::indexing::Slice();

  if (opts.ndim() == 1) {
    return {slice1};
  } else if (opts.ndim() == 2) {
    return {slice2, slice1};
  } else if (opts.ndim() == 3) {
    return {slice3, slice2, slice1};
  } else if (opts.ndim() == 4) {
    return {slice4, slice3, slice2, slice1};
  } else {
    throw std::runtime_error("part: ndim must be 1-4.");
  }
}

double MeshBlockImpl::initialize(Variables &vars, char const *restart_file) {
  /*c10d::BarrierOptions op;
  op.device_ids = {options->layout()->local_rank()};
  _playout->pg->barrier(op)->wait();*/
  if (_playout->has_process_group()) {
    _playout->comm->barrier();
  }

  //// ------------ (1) Set up a signal handler ------------ ////
  SignalHandler::GetInstance();

  if (restart_file != nullptr) {
    return _init_from_restart(vars, std::string(restart_file));
  }

  initialize_local(vars);

  //// ------ (6) Exchange hydro and scalar buffers -------- ////
  if (options->verbose()) {
    SINFO(MeshBlock) << "exchanging ghost zones." << std::endl;
  }

  SyncOptions sync_opts;
  sync_opts.interpolate(true).type(kPrimitive);

  Variables sync_vars;
  sync_vars["hydro_w"] = vars.at("hydro_w");
  exchange(sync_vars, sync_opts);

  if (pscalar->nvar() > 0) {
    sync_opts.type(kScalar);
    sync_vars.clear();
    sync_vars["scalar_r"] = vars.at("scalar_r");
    exchange(sync_vars, sync_opts);
  }

  finalize_initialization(vars);

  return 0.;  // default start time is 0.0
}

void MeshBlockImpl::initialize_local(Variables &vars) {
  torch::Tensor hydro_w, scalar_r, solid;

  //// ------------ (2) Check hydro primitive ------------ ////
  int64_t nc3 = options->coord()->nc3();
  int64_t nc2 = options->coord()->nc2();
  int64_t nc1 = options->coord()->nc1();

  TORCH_CHECK(vars.count("hydro_w"),
              "initialize: hydro_w is required for hydro model.");
  hydro_w = vars.at("hydro_w");

  TORCH_CHECK(hydro_w.sizes() ==
                  std::vector<int64_t>({phydro->peos->nvar(), nc3, nc2, nc1}),
              "initialize: hydro_w has incorrect shape.", " Expected [",
              phydro->peos->nvar(), ", ", nc3, ", ", nc2, ", ", nc1,
              "] but got ", hydro_w.sizes());

  if (pscalar->nvar() > 0) {
    TORCH_CHECK(vars.count("scalar_r"), "initialize: scalar_r is required");
    scalar_r = vars.at("scalar_r");
    TORCH_CHECK(scalar_r.sizes() ==
                    std::vector<int64_t>({pscalar->nvar(), nc3, nc2, nc1}),
                "initialize: scalar_r has incorrect shape");
  }
  // Capture before any physical face fill, independently of mutable inputs.
  if (has_radiating_boundary()) {
    vars["boundary_reference_w"] = hydro_w.clone();
    if (scalar_r.defined()) vars["boundary_reference_r"] = scalar_r.clone();
  }
  apply_boundaries(vars, hydro_w, scalar_r, true);
}

void MeshBlockImpl::initialize_under_mesh(Variables &vars) {
  initialize_local(vars);

  SyncOptions prim_opts;
  prim_opts.interpolate(true).type(kPrimitive);
  Variables prim_vars;
  prim_vars["hydro_w"] = vars.at("hydro_w");
  exchange(prim_vars, prim_opts);

  if (pscalar->nvar() > 0) {
    SyncOptions scalar_opts;
    scalar_opts.interpolate(true).type(kScalar);
    Variables scalar_vars;
    scalar_vars["scalar_r"] = vars.at("scalar_r");
    exchange(scalar_vars, scalar_opts);
  }

  finalize_initialization(vars);
}

void MeshBlockImpl::finalize_initialization(Variables &vars) {
  auto hydro_w = vars.at("hydro_w");
  auto scalar_r =
      vars.count("scalar_r") ? vars.at("scalar_r") : torch::Tensor();
  auto solid = vars.count("solid") ? vars.at("solid") : torch::Tensor();
  int64_t nc3 = options->coord()->nc3();
  int64_t nc2 = options->coord()->nc2();
  int64_t nc1 = options->coord()->nc1();

  //// ------ (7) Computer hydro and scalar conserved -------- ////
  if (options->verbose()) {
    SINFO(MeshBlock) << "computing conserved variables." << std::endl;
  }

  vars["hydro_u"] = phydro->peos->compute("W->U", {hydro_w});
  if (pscalar->nvar() > 0) {
    // Seed with the conserved (dry) density, as set_scalar_primitive divides.
    vars["scalar_s"] = vars.at("hydro_u")[IDN] * scalar_r;
  }

  //// ------------- (8) Fill solid boundaries -------------- ////
  if (vars.count("solid")) {
    if (options->verbose()) {
      SINFO(MeshBlock) << "filling solid boundaries." << std::endl;
    }

    solid = vars.at("solid");
    TORCH_CHECK(solid.sizes() == std::vector<int64_t>({nc3, nc2, nc1}),
                "initialize: solid has incorrect shape.", " Expected [", nc3,
                ", ", nc2, ", ", nc1, "] but got ", solid.sizes());
    vars["fill_solid_hydro_w"] =
        torch::where(solid.unsqueeze(0).expand_as(hydro_w), hydro_w, 0.);
    vars["fill_solid_hydro_w"].narrow(0, IVX, 3).zero_();
    pib->mark_prim_solid_(hydro_w, solid);

    vars["fill_solid_hydro_u"] =
        torch::where(solid.unsqueeze(0).expand_as(vars.at("hydro_u")),
                     vars.at("hydro_u"), 0.);
    vars["fill_solid_hydro_u"].narrow(0, IVX, 3).zero_();
  } else {
    vars["fill_solid_hydro_w"] = hydro_w;
    vars["fill_solid_hydro_u"] = vars.at("hydro_u");
  }

  apply_boundaries(
      vars, vars.at("hydro_u"),
      vars.count("scalar_s") ? vars.at("scalar_s") : torch::Tensor());

  //// ---------------- (11) Start timing ----------------- ////
  _time_start = clock();

  if (options->verbose()) {
    SINFO(MeshBlock) << "initialization completed." << std::endl;
  }
}

double MeshBlockImpl::max_time_step(Variables const &vars) {
  auto dt_local = local_max_time_step(vars);
  auto const &w = vars.at("hydro_w");
  auto dt_min = torch::tensor({dt_local}, torch::dtype(torch::kFloat64));

  std::vector<at::Tensor> dt_reduce = {dt_min};
  c10d::AllreduceOptions op;
  op.reduceOp = c10d::ReduceOp::MIN;
  if (_playout->has_process_group()) {
    _playout->comm->allreduce(dt_reduce, op.reduceOp);
  }

  auto dt = dt_reduce[0].item<double>();

  if (options->verbose()) {
    SINFO(MeshBlock) << "suggested dt from hydro: " << std::scientific
                     << std::setprecision(6) << dt << std::endl;
  }
  return pow(2., -pintg->current_redo) * pintg->options->cfl() * dt;
}

double MeshBlockImpl::local_max_time_step(Variables const &vars) const {
  auto const &w = vars.at("hydro_w");
  double dt_min = 1.e9;

  // hyperbolic hydro time step
  if (vars.count("solid")) {
    dt_min = phydro->max_time_step(w, vars.at("solid"));
  } else {
    dt_min = phydro->max_time_step(w);
  }

  return dt_min;
}

void MeshBlockImpl::forward(Variables &vars, double dt, int stage) {
  // Exchange BEFORE reconstruction consumes the ghosts (same count per step):
  // host-side operator-split source terms applied between forward calls
  // (e.g. condensation on hydro_u) are then synced before use, removing a
  // one-signed seam bias on the cubed sphere.
  exchange_ghost_zones(vars);
  advance_local(vars, dt, stage);
}

void MeshBlockImpl::exchange(Variables &vars, SyncOptions const &opts) const {
  // A subdivided cubed-sphere panel packs its cross-panel strip EXTENDED into
  // its own tangential halo (cs_interp_margin), and that halo is only written
  // by deserialize(). So the intra-panel round must COMPLETE before the
  // cross-panel round packs, which one round cannot do. Corners are synthesized
  // from the edge strips, so they come after both.
  auto const &lo = *options->layout();
  if (opts.interpolate() && !opts.intra_panel_only() &&
      !opts.cross_panel_only() && lo.type() == "cubed-sphere" &&
      (lo.px() > 1 || lo.py() > 1)) {
    SyncOptions intra = opts, cross = opts;
    _exchange_once(vars, intra.intra_panel_only(true).interpolate(false));
    _exchange_once(vars, cross.cross_panel_only(true));
    if (opts.skip_corner()) _playout->fill_corners(this, vars);
    return;
  }

  _exchange_once(vars, opts);
}

void MeshBlockImpl::_exchange_once(Variables &vars,
                                   SyncOptions const &opts) const {
  std::vector<CommWorkPtr> works;
  begin_exchange(vars, opts);
  launch_exchange(opts, works);
  finalize_exchange(vars, opts, works);
}

void MeshBlockImpl::begin_exchange(Variables &vars,
                                   SyncOptions const &opts) const {
  _playout->serialize(this, vars, opts);
}

void MeshBlockImpl::launch_exchange(SyncOptions const &opts,
                                    std::vector<CommWorkPtr> &works) const {
  _playout->launch_exchange(this, opts, works);
}

void MeshBlockImpl::finalize_exchange(Variables &vars, SyncOptions const &opts,
                                      std::vector<CommWorkPtr> &works) const {
  _playout->finalize(this, vars, opts, works);
}

namespace {
// the CUDA vapor repair counts its failures on the device while in scope
struct DeferVaporCheck {
  EquationOfStateImpl *eos;
  explicit DeferVaporCheck(EquationOfStateImpl *e) : eos(e) {
    eos->defer_vapor_check(true);
  }
  ~DeferVaporCheck() { eos->defer_vapor_check(false); }
};
}  // namespace

void MeshBlockImpl::advance_local(Variables &vars, double dt, int stage) {
  TORCH_CHECK(stage >= 0 && stage < pintg->stages.size(),
              "Invalid stage: ", stage);
  DeferVaporCheck defer(phydro->peos.get());

  // Publish the stage: the vertical implicit correction is nonlinear in dt and
  // must weight dt INSIDE the solve, so it needs to know which stage it is in.
  phydro->rk_stage = stage;

  auto hydro_u = vars.at("hydro_u");
  auto scalar_s =
      vars.count("scalar_s") ? vars.at("scalar_s") : torch::Tensor();

  auto start = std::chrono::high_resolution_clock::now();
  // -------- (1) save initial state --------
  if (stage == 0) {
    if (phydro->pdiffusion && phydro->pdiffusion->options->implicit_radial()) {
      phydro->pdiffusion->begin_step(hydro_u, dt);
      apply_boundaries(vars, hydro_u, scalar_s);
      exchange_ghost_zones(vars);
    }
    _hydro_u0.copy_(hydro_u);
    // the fixer buffers are only written under the
    // fixer; with it off they stay zero, so skip three launches
    if (phydro->gravity_work_fixer()) {
      phydro->gravity_work_defect().zero_();  // a redone step starts afresh
      phydro->commit_gravity_work_fix();      // the previous step was accepted
      phydro->gravity_work_wall_mass().zero_();
    }
    if (phydro->picorr) {
      phydro->picorr->reset_dry_clamp_step();
      phydro->picorr->reset_solve_failure();
    }
    phydro->peos->reset_limiter_marks(hydro_u);
    saturation_failures();  // a failure counted before the step is not its own
    if (phydro->peos->vapor_failures().defined())
      phydro->peos->vapor_failures().zero_();  // nor is a vapor failure

    if (pscalar->nvar() > 0) {
      _scalar_s0.copy_(scalar_s);
    }
  }

  // -------- (2) set containers for future results --------
  torch::Tensor fut_hydro_du, fut_scalar_ds;

  // A dry source is inside the RK tendency. Addition carries the ratio at
  // source entry; removal carries the complete source-free stage result so it
  // cannot amplify a ratio that transport or the implicit solve just changed.
  auto carry_dry_source = [&](torch::Tensor const &dry,
                              torch::Tensor const &scalar_base) {
    if (!dry.defined() || !vars.count("scalar_r")) return;
    auto rho_base = pintg->forward(stage, _hydro_u0[IDN], hydro_u[IDN],
                                   fut_hydro_du[IDN] - dry);
    auto s_base = pintg->forward(stage, _scalar_s0, scalar_s, scalar_base);
    auto remove_r =
        torch::where(rho_base.unsqueeze(0) == 0., vars.at("scalar_r"),
                     s_base / rho_base.unsqueeze(0));
    auto carry_r =
        torch::where(dry.unsqueeze(0) < 0., remove_r, vars.at("scalar_r"));
    fut_scalar_ds.add_(carry_r * dry.unsqueeze(0));
  };

  // -------- (3) launch all jobs --------
  // (3.A) hydro forward
  fut_hydro_du = phydro->forward(dt, hydro_u, vars);
  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(MeshBlock) << "stage " << stage
                     << " hydro forward time (s): " << elapsed.count()
                     << std::endl;
    start = std::chrono::high_resolution_clock::now();
  }

  // (3.B) scalar forward
  if (pscalar->nvar() > 0) {
    fut_scalar_ds = pscalar->forward(dt, scalar_s, vars);
    auto mass_corr = phydro->implicit_mass_correction();
    if (mass_corr.defined() && mass_corr.numel() > 0 &&
        vars.count("scalar_r")) {
      // r is per dry air: move it with the clamped dry transfer, not total M
      auto M = mass_corr[IVY];  // slot IVX+(dir+1)%3 with dir == 0 (x1 only)
      auto r = vars.at("scalar_r");
      int nc1 = r.size(-1);
      if (scalar_fused_enabled() && r.is_cuda() &&
          r.scalar_type() == torch::kFloat64 && fut_scalar_ds.is_contiguous()) {
        // the same transfer in one kernel (scalar_fused.cu)
        auto vol = pcoord->cell_volume().expand(M.sizes()).contiguous();
        at::native::call_scalar_implicit_transfer(fut_scalar_ds.device().type(),
                                                  fut_scalar_ds, r.contiguous(),
                                                  M.contiguous(), vol);
      } else {
        auto r_below = torch::zeros_like(r);
        r_below.slice(-1, 1, nc1) = r.slice(-1, 0, nc1 - 1);
        auto P = torch::where(M > 0., r_below, r) * M;
        auto P_above = torch::zeros_like(P);
        P_above.slice(-1, 0, nc1 - 1) = P.slice(-1, 1, nc1);
        fut_scalar_ds.add_((P - P_above) / pcoord->cell_volume());
      }
    }
    // Native dry sources follow transport and the implicit tracer transfer.
    carry_dry_source(phydro->forcing_dry_increment(), fut_scalar_ds);
    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(MeshBlock) << "stage " << stage
                       << " scalar forward time (s): " << elapsed.count()
                       << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
  }

  // (3.C) user stage forcings
  auto apply_extra_forcing = [&](Variables const &extra_forcing,
                                 std::string const &source) {
    auto hydro_it = extra_forcing.find("hydro_du");
    if (hydro_it != extra_forcing.end()) {
      TORCH_CHECK(hydro_it->second.sizes() == fut_hydro_du.sizes(), source,
                  " returned hydro_du with shape ", hydro_it->second.sizes(),
                  ", expected ", fut_hydro_du.sizes(), ".");
      fut_hydro_du.add_(hydro_it->second);
    }

    auto scalar_it = extra_forcing.find("scalar_ds");
    if (scalar_it != extra_forcing.end()) {
      TORCH_CHECK(
          pscalar->nvar() > 0, source,
          " returned scalar_ds, but no scalar variables are configured on "
          "this MeshBlock.");
      TORCH_CHECK(fut_scalar_ds.defined(), source,
                  " returned scalar_ds before native scalar tendencies were "
                  "initialized.");
      TORCH_CHECK(scalar_it->second.sizes() == fut_scalar_ds.sizes(), source,
                  " returned scalar_ds with shape ", scalar_it->second.sizes(),
                  ", expected ", fut_scalar_ds.sizes(), ".");
      fut_scalar_ds.add_(scalar_it->second);
    }

    for (auto const &[name, _] : extra_forcing) {
      TORCH_CHECK(name == "hydro_du" || name == "scalar_ds", source,
                  " returned unsupported key '", name,
                  "'. Expected one or both of: hydro_du, scalar_ds.");
    }
  };

  // User scalar_ds is an independent additive tracer source. Keep the scalar
  // base from before the callbacks so a simultaneous dry removal does not debit
  // that explicitly supplied tracer.
  bool track_user_dry = !user_stage_forcings.empty() && pscalar->nvar() > 0 &&
                        vars.count("scalar_r");
  auto user_dry_before =
      track_user_dry ? fut_hydro_du[IDN].clone() : torch::Tensor();
  auto user_scalar_before =
      track_user_dry ? fut_scalar_ds.clone() : torch::Tensor();
  // A user forcing's species increment comes after the flux positivity
  // limiter and the implicit correction, and nothing else bounds it by what
  // they left: clip it so that u + du stays >= 0 in every species cell the
  // forcing would take below zero. The withheld amount is not applied (the
  // forcing removes less than it asked for there). Where the clip does not
  // bind, du is unchanged bit for bit.
  int nspec = hydro_u.size(0) - ICY;
  bool clip_species = !user_stage_forcings.empty() && nspec > 0 &&
                      phydro->options->eos()->limiter();
  auto species_base = clip_species ? fut_hydro_du.narrow(0, ICY, nspec).clone()
                                   : torch::Tensor();
  if (!user_stage_forcings.empty()) {
    auto inputs = stage_forcing_variables(*this, vars);
    for (size_t i = 0; i < user_stage_forcings.size(); ++i) {
      auto result = user_stage_forcings[i]->forward({inputs, dt, stage});
      apply_extra_forcing(stage_forcing_result(result, i),
                          "User stage forcing " + std::to_string(i));
    }
  }
  if (clip_species) {
    auto du = fut_hydro_du.narrow(0, ICY, nspec);
    auto floor = torch::minimum(species_base, -hydro_u.narrow(0, ICY, nspec));
    // only the binding cells are rewritten (signed zeros stay as they are)
    du.copy_(torch::where(du < floor, floor, du));
  }

  if (user_dry_before.defined()) {
    carry_dry_source(fut_hydro_du[IDN] - user_dry_before, user_scalar_before);
  }
  if (!user_stage_forcings.empty()) {
    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(MeshBlock) << "stage " << stage
                       << " user stage forcing time (s): " << elapsed.count()
                       << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
  }

  // -------- (4) multi-stage averaging --------
  hydro_u.set_(pintg->forward(stage, _hydro_u0, hydro_u, fut_hydro_du));
  if (stage + 1 == static_cast<int>(pintg->stages.size()) &&
      phydro->pdiffusion && phydro->pdiffusion->options->implicit_radial())
    phydro->pdiffusion->finish_step(hydro_u, dt);
  // skipped dry and scalar-free before the last stage: next U->W applies it
  if (hydro_u.size(0) > ICY || pscalar->nvar() > 0 ||
      stage + 1 == static_cast<int>(pintg->stages.size()))
    phydro->peos->apply_conserved_limiter_(hydro_u, /*whole_column=*/true);

  if (options->verbose()) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    SINFO(MeshBlock) << "stage " << stage
                     << " multi-stage averaging time (s): " << elapsed.count()
                     << std::endl;
    start = std::chrono::high_resolution_clock::now();
  }

  if (pscalar->nvar() > 0) {
    if (scalar_fused_enabled() && scalar_s.is_cuda() &&
        scalar_s.scalar_type() == torch::kFloat64 &&
        _scalar_s0.is_contiguous() && scalar_s.is_contiguous() &&
        fut_scalar_ds.is_contiguous()) {
      auto const &w = pintg->stages[stage];
      auto avg = torch::empty_like(scalar_s);
      at::native::call_scalar_average3(avg.device().type(), avg, _scalar_s0,
                                       scalar_s, fut_scalar_ds, w.wght0(),
                                       w.wght1(), w.wght2());
      scalar_s.set_(avg);
    } else {
      scalar_s.set_(pintg->forward(stage, _scalar_s0, scalar_s, fut_scalar_ds));
    }
    set_scalar_primitive(vars, scalar_s, hydro_u);
    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(MeshBlock) << "stage " << stage
                       << " multi-stage scalar averaging time (s): "
                       << elapsed.count() << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
  }

  // -------- (5) update ghost zones --------

  if (vars.count("solid")) {
    pib->fill_cons_solid_(hydro_u, vars.at("solid"),
                          vars.at("fill_solid_hydro_u"));
    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(MeshBlock) << "stage " << stage
                       << " fill solid hydro conserved time (s): "
                       << elapsed.count() << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
  }

  // -------- (6) saturation adjustment --------
  if (stage == pintg->stages.size() - 1 && phydro->options->eos()->thermo() &&
      phydro->options->eos()->thermo()->reactions().size() > 0) {
    phydro->peos->apply_conserved_limiter_(hydro_u, /*whole_column=*/true);

    int ny = hydro_u.size(0) - ICY;  // number of species

    auto ke = phydro->peos->compute("U->K", {hydro_u});
    auto rho = hydro_u[IDN] + hydro_u.narrow(0, ICY, ny).sum(0);
    auto ie = hydro_u[IPR] - ke;

    auto yfrac = hydro_u.narrow(0, ICY, ny) / rho;

    auto m = named_modules()["hydro.eos.thermo"];
    auto pthermo = std::dynamic_pointer_cast<kintera::ThermoYImpl>(m);

    auto sub = part({0, 0, 0}, PartOptions().exterior(false));
    auto sub3 = part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
    pthermo->forward(rho.index(sub3), ie.index(sub3), yfrac.index(sub),
                     /*warm_start=*/true);

    hydro_u.narrow(0, ICY, ny) = yfrac * rho;
    if (options->verbose()) {
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      SINFO(MeshBlock) << "stage " << stage
                       << " saturation adjustment time (s): " << elapsed.count()
                       << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
  }

  // Physical ghosts must include the final-stage species adjustment.
  // gravity-work-fixer (#283): once per step, after the last mass change, add
  // the step's global E+PE defect D back as heat, uniform per unit mass:
  // dE_i = -D m_i / sum_j m_j V_j. Momentum and mass are untouched. A Mesh
  // with several local blocks sums them first (MeshImpl::forward).
  if (stage == pintg->stages.size() - 1 && phydro->gravity_work_fixer() &&
      !defer_gravity_work_fixer) {
    std::vector<at::Tensor> reduce = {gravity_work_fixer_sums(vars)};
    if (_playout->has_process_group()) {
      _playout->comm->allreduce(reduce, c10d::ReduceOp::SUM);
    }
    apply_gravity_work_fixer(vars, reduce[0]);
  }

  apply_boundaries(vars, hydro_u, scalar_s);
}

torch::Tensor MeshBlockImpl::gravity_work_fixer_sums(
    Variables const &vars) const {
  auto hydro_u = vars.at("hydro_u");
  int ny = hydro_u.size(0) - ICY;
  auto m = hydro_u[IDN].clone();
  if (ny > 0) m += hydro_u.narrow(0, ICY, ny).sum(0);
  // immersed solid cells are refilled each step: they take no share of -D
  if (vars.count("solid")) m.masked_fill_(vars.at("solid"), 0.);
  auto in3 = part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  auto mv = (m * pcoord->cell_volume()).index(in3);
  // the mass of the cells next to the x1 walls sets the wall faces' round-off
  auto mwall = torch::zeros_like(mv.select(-1, 0).sum());
  if (phydro->is_x1_wall(0)) mwall += mv.select(-1, 0).sum();
  if (phydro->is_x1_wall(1)) mwall += mv.select(-1, -1).sum();
  // the limiter patched a cell or found a NaN: the redo check discards the step
  // (the marks ride in the one device-to-host copy below)
  auto const &marks = phydro->peos->limiter_marks();
  auto redo = phydro->picorr
                  ? phydro->picorr->solve_failed_tensor(mwall.options())
                  : torch::zeros_like(mwall);
  if (marks.defined()) redo = torch::maximum(redo, marks.any().to(redo));
  return torch::stack({phydro->gravity_work_defect()[0],
                       mv.sum().to(torch::kFloat64),
                       phydro->gravity_work_wall_mass()[0],
                       mwall.to(torch::kFloat64), redo.to(torch::kFloat64)})
      .cpu();
}

void MeshBlockImpl::apply_gravity_work_fixer(Variables &vars,
                                             torch::Tensor const &global) {
  auto hydro_u = vars.at("hydro_u");
  int ny = hydro_u.size(0) - ICY;
  auto m = hydro_u[IDN].clone();
  if (ny > 0) m += hydro_u.narrow(0, ICY, ny).sum(0);
  // immersed solid cells are refilled each step: they take no share of -D
  if (vars.count("solid")) m.masked_fill_(vars.at("solid"), 0.);
  auto in3 = part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
  // global is on the host: read it through one accessor, not five .item()s
  auto g = global.to(torch::kCPU, torch::kFloat64).contiguous();
  auto ga = g.accessor<double, 1>();
  double D = ga[0], mass = ga[1], wall = ga[2], mwall = ga[3];
  bool finite = true;
  for (int64_t i = 0; i < g.size(0); ++i)
    finite = finite && std::isfinite(ga[i]);
  if (ga[4] > 0. || !finite) {
    // a step the redo check will discard (limiter repair, NaN, non-finite
    // sums) is left to it, as with the fixer off
    phydro->gravity_work_defect().zero_();
    phydro->gravity_work_wall_mass().zero_();
    return;
  }
  // the defect leaves out boundary fluxes: refuse a step that moved mass
  // through an x1 boundary face (whatever installed it: YAML or Python).
  // A sealed wall's flux is round-off, of order eps times its cell's mass; the
  // bound allows 1e3 eps of the wall cells' mass per step, so it scales with
  // the dtype and the wall area. At large implicit steps, sealed-wall
  // round-off can exceed it; the onset depends on the grid and backend.
  // Reviewed 11H rest decks first refused near C=2500 (CUDA) / 2750 (CPU),
  // as recorded in #285; those measurements are not universal thresholds.
  double eps = machine_epsilon(hydro_u.scalar_type());
  double bound = 1.e3 * eps * mwall;
  TORCH_CHECK(wall <= bound, "const-gravity gravity-work-fixer: ", wall,
              " kg crossed an x1 boundary face this step (", wall / mwall,
              " of the x1 wall cells' mass; round-off bound ", 1.e3 * eps,
              "); the fixer's E+PE defect leaves out boundary fluxes. "
              "For open x1 boundaries set gravity-work-fixer: false. "
              "For sealed walls reduce the timestep to keep round-off below "
              "the bound. Disabling the fixer drops the E+PE correction.");
  double efix = -D / mass;  // J/kg
  hydro_u[IPR].index(in3) += m.index(in3) * efix;
  phydro->gravity_work_defect().zero_();
  phydro->gravity_work_wall_mass().zero_();
  phydro->add_gravity_work_fix(-D);  // counted once the step is accepted
}

void MeshBlockImpl::exchange_ghost_zones(Variables &vars) {
  auto hydro_u = vars.at("hydro_u");
  auto scalar_s =
      vars.count("scalar_s") ? vars.at("scalar_s") : torch::Tensor();

  SyncOptions sync_opts;
  sync_opts.interpolate(true).type(kConserved);

  Variables sync_vars;
  sync_vars["hydro_u"] = hydro_u;
  exchange(sync_vars, sync_opts);

  if (pscalar->nvar() > 0) {
    sync_opts.type(kScalar);
    sync_vars.clear();
    sync_vars["scalar_s"] = scalar_s;
    exchange(sync_vars, sync_opts);
  }

  // Tangential exchange fills face ghosts but not corners (skip_corner).
  // Re-apply each installed x1 function on the x2/x3 slabs exchange owns, so
  // the corner matches the updated column. A physical tangential face already
  // wrote that corner. Outflow is not reapplied: its background must match the
  // full block and it needs primitives (#264).
  {
    int ng = pcoord->options->nghost();
    auto const &fns = options->bfuncs();
    auto const &names = options->bcnames();
    BoundaryFuncOptions op;
    op.nghost(ng);
    op.eos = phydro->peos.get();
    op.coord = pcoord.get();
    auto physical = [&](int face) {
      if (face < 0 || face >= static_cast<int>(fns.size()) || !fns[face])
        return false;
      return !(static_cast<size_t>(face) < names.size() &&
               names[face].compare(0, 8, "periodic") == 0);
    };
    auto refresh = [&](torch::Tensor field, int type) {
      if (!field.defined() || field.numel() == 0) return;
      op.type(type);
      int nface =
          static_cast<int>(fns.size()) < 2 ? static_cast<int>(fns.size()) : 2;
      for (int f = 0; f < nface; ++f) {
        if (!fns[f] || is_outflow(fns[f])) continue;
        int spatial = 3 - f / 2;
        if (field.size(spatial) == 1) continue;
        // a reflecting wall is row-local (each x1 row
        // mirrors its own interior), so when every tangential face is
        // exchange-owned one full-extent call writes the slabs' corner values
        // and rewrites the other rows with the same bits
        {
          bool any_orth = false, all_owned = true;
          for (int orth = 1; orth <= 3; ++orth) {
            if (orth == spatial || field.size(orth) <= 1) continue;
            any_orth = true;
            int inner = (3 - orth) * 2;
            if (physical(inner) || physical(inner + 1)) all_owned = false;
          }
          if (any_orth && all_owned && static_cast<size_t>(f) < names.size() &&
              names[f].compare(0, 10, "reflecting") == 0) {
            fns[f](field, spatial, op);
            continue;
          }
        }
        for (int orth = 1; orth <= 3; ++orth) {
          if (orth == spatial || field.size(orth) <= 1) continue;
          int n = static_cast<int>(field.size(orth));
          int g = ng < n / 2 ? ng : n / 2;
          if (g <= 0) continue;
          int inner = (3 - orth) * 2;
          if (!physical(inner)) fns[f](field.narrow(orth, 0, g), spatial, op);
          if (!physical(inner + 1))
            fns[f](field.narrow(orth, n - g, g), spatial, op);
        }
      }
    };
    refresh(hydro_u, kConserved);
    refresh(scalar_s, kScalar);
  }

  if (pscalar->nvar() > 0) {
    set_scalar_primitive(vars, scalar_s, hydro_u);
  }

  if (options->verbose()) {
    SINFO(MeshBlock) << "ghost zone exchange completed." << std::endl;
  }
}

void MeshBlockImpl::make_outputs(Variables const &vars, double current_time,
                                 bool final_write) {
  for (auto &output_type : output_types) {
    output_type->AccumulateStats(vars, current_time);
    if (final_write) {
      output_type->write_output_file(this, vars, current_time, final_write);
    } else if (current_time >= output_type->next_time) {
      output_type->write_output_file(this, vars, current_time, final_write);
      output_type->ResetStats(current_time);
      output_type->next_time += output_type->options->dt();
      output_type->file_number += 1;
    }
  }
  if (options->verbose()) {
    SINFO(MeshBlock) << "output writing completed at time: " << current_time
                     << std::endl;
  }
}

void print_cycle_diagnostics(
    std::vector<std::pair<MeshBlockImpl const *, Variables const *>> const
        &blocks,
    double time, double dt, int precision, char const *energy_label) {
  auto root = blocks.front().first;
  auto pintg = root->pintg;
  if (pintg->options->ncycle_out() == 0 ||
      root->cycle % pintg->options->ncycle_out() != 0)
    return;

  SINFO() << "cycle=" << root->cycle << " redo=" << pintg->current_redo
          << std::scientific << std::setprecision(precision) << " time=" << time
          << " dt=" << dt;
  if (!blocks.front().second->count("hydro_u")) {
    SINFO() << std::endl;
    return;
  }

  torch::Tensor conserved, ke_sum, pe_sum, meters, theta_min, vic_max;
  auto add = [](torch::Tensor &total, torch::Tensor value) {
    if (total.defined())
      total += value;
    else
      total = value.clone();
  };
  for (auto const &[block, vars] : blocks) {
    auto hydro = block->phydro;
    auto coord = block->pcoord;
    auto interior = block->part({0, 0, 0}, PartOptions().exterior(false));
    auto vol = coord->cell_volume();
    auto u = vars->at("hydro_u");
    add(conserved, (u * vol).index(interior).sum({1, 2, 3}));

    // Read the conserved state; hydro_w can still describe an earlier stage.
    if (u.size(0) >= IVX + 3) {
      auto rho = u[IDN].unsqueeze(0).clone();
      for (int n = ICY; n < u.size(0); ++n) rho += u[n].unsqueeze(0);
      auto mom = u.narrow(0, IVX, 3).clone();
      coord_vec_raise_(mom, coord->cosine_cell_kj);
      auto ke = 0.5 * (u.narrow(0, IVX, 3) * mom).sum(0, true) / rho;
      add(ke_sum, (ke * vol).index(interior).sum({1, 2, 3}));
      if (hydro->options->grav() && hydro->options->grav()->grav1() != 0.) {
        auto grav1 = hydro->options->grav()->grav1();
        auto pe = rho * (-grav1 * coord->x1v);
        // SNAP_GRAVITY_WORK_RADIAL_EXACT books the work of the corrected
        // potential energy P, so log P: E + P is what that work conserves
        if (hydro->radial_exact_work()) {
          int is = coord->il(), ie = coord->iu() + 1;
          // across an x1 seam the ghost rho (exchanged after every stage) is
          // the neighbour's: P of a split column is one block's (#303); a
          // periodic x1 column keeps the one-sided slope (hydro_forward.cpp)
          auto [below, above] = hydro->x1_neighbors();
          pe.slice(-1, is, ie) -= corrected_pe_work_seam(
              rho, coord->x1f, coord->x1v, is, ie, grav1,
              x1_measure(coord->options->type()), below >= 0, above >= 0);
        }
        add(pe_sum, (pe * vol).index(interior).sum({1, 2, 3}));
      }
    }

    add(meters,
        torch::stack({hydro->lim_cut()[0].to(torch::kFloat64),
                      hydro->lim_flux()[0].to(torch::kFloat64),
                      hydro->positivity_severe()[0].to(torch::kFloat64)}));
    auto tmin = hydro->positivity_min().to(torch::kFloat64);
    theta_min =
        theta_min.defined() ? torch::minimum(theta_min, tmin) : tmin.clone();
    if (hydro->picorr) {
      auto vc = hydro->picorr->clamp_residual().to(torch::kFloat64);
      vic_max = vic_max.defined() ? torch::maximum(vic_max, vc) : vc.clone();
    }
  }

  auto layout = root->get_layout();
  auto reduce = [&](torch::Tensor &value, c10d::ReduceOp op) {
    if (!value.defined() || !layout->has_process_group()) return;
    std::vector<at::Tensor> values = {value};
    layout->comm->reduce(values, op,
                         root->options->layout()->process_root_rank());
    value = values[0];
  };
  reduce(conserved, c10d::ReduceOp::SUM);
  reduce(ke_sum, c10d::ReduceOp::SUM);
  reduce(pe_sum, c10d::ReduceOp::SUM);
  reduce(meters, c10d::ReduceOp::SUM);
  reduce(theta_min, c10d::ReduceOp::MIN);
  reduce(vic_max, c10d::ReduceOp::MAX);

  auto print = [&](char const *name, double value) {
    SINFO() << std::scientific << std::setprecision(precision) << name << value;
  };
  auto mass = conserved[IDN];
  print(" mass0=", mass.item<double>());
  if (conserved.size(0) > ICY) {
    for (int n = ICY; n < conserved.size(0); ++n) mass += conserved[n];
    print(" masst=", mass.item<double>());
  }
  if (ke_sum.defined()) print(" ke=", ke_sum[0].item<double>());
  if (root->phydro->peos->nvar() > IPR)
    print(energy_label, conserved[IPR].item<double>());
  if (pe_sum.defined()) print(" pe=", pe_sum[0].item<double>());

  SINFO() << " run-to-date:";
  double total = meters[1].item<double>();
  if (total > 0.) print(" limcut=", meters[0].item<double>() / total);
  print(" thetamin=", theta_min[0].item<double>());
  SINFO() << " thetasevere="
          << static_cast<long long>(meters[2].item<double>());
  if (vic_max.defined()) print(" vicclamp=", vic_max[0].item<double>());
  if (root->phydro->gravity_work_fixer())  // global: every block holds it
    print(" fixgrav=", root->phydro->gravity_work_fix()[0].item<double>());
  SINFO() << std::endl;
}

void MeshBlockImpl::print_cycle_info(Variables const &vars, double time,
                                     double dt) const {
  print_cycle_diagnostics({{this, &vars}}, time, dt,
                          std::numeric_limits<double>::max_digits10 - 4,
                          " ie=");
}

int MeshBlockImpl::finalize(Variables const &vars, double time) {
  // make final output
  make_outputs(vars, time, /*final_write=*/true);

  int status = 0;
  auto sig = SignalHandler::GetInstance();
  if (sig->GetSignalFlag(SIGTERM) != 0) {
    SINFO() << std::endl << "Terminating on Terminate signal" << std::endl;
  } else if (sig->GetSignalFlag(SIGINT) != 0) {
    SINFO() << std::endl << "Terminating on Interrupt signal" << std::endl;
  } else if (sig->GetSignalFlag(SIGALRM) != 0) {
    SINFO() << std::endl << "Terminating on wall-time limit" << std::endl;
  } else if (pintg->options->nlim() >= 0 && cycle >= pintg->options->nlim()) {
    SINFO() << std::endl << "Terminating on cycle limit" << std::endl;
  } else if (time >= pintg->options->tlim()) {
    SINFO() << std::endl << "Terminating on time limit" << std::endl;
  } else {
    SINFO() << std::endl << "Terminating abnormally" << std::endl;
    status = 1;
  }

  SINFO() << "time=" << time << " cycle=" << cycle << std::endl;
  SINFO() << "tlim=" << pintg->options->tlim()
          << " nlim=" << pintg->options->nlim() << std::endl;

  // ---------- timing info ----------
  clock_t tstop = clock();
  double cpu_time =
      (tstop > _time_start ? static_cast<double>(tstop - _time_start) : 1.0) /
      static_cast<double>(CLOCKS_PER_SEC);

  std::vector<at::Tensor> cells = {
      torch::tensor({_hydro_u0.size(1) * _hydro_u0.size(2) * _hydro_u0.size(3)},
                    torch::dtype(torch::kInt64))};

  c10d::ReduceOptions opsum;
  opsum.reduceOp = c10d::ReduceOp::SUM;
  opsum.rootRank = options->layout()->process_root_rank();

  if (_playout->has_process_group()) {
    _playout->comm->reduce(cells, opsum.reduceOp, opsum.rootRank);
  }

  int64_t cellcycles = cells[0].item<int64_t>() * cycle * pintg->stages.size();
  double zc_cpus = static_cast<double>(cellcycles) / cpu_time;

  SINFO() << std::endl
          << "million cells-per-cycle = " << cellcycles / 1e6 << std::endl;
  SINFO() << "cpu time used (s) = " << cpu_time << std::endl;
  SINFO() << "million cell-updates/second = " << zc_cpus / 1e6 << std::endl;

  // ------ shutdown processing group ------
  if (_playout->has_process_group()) {
    _playout->comm->barrier();
  }

  send_bufs.clear();
  send_bufs.shrink_to_fit();

  recv_bufs.clear();
  recv_bufs.shrink_to_fit();

  if (_playout->has_process_group() && _playout->comm->owns_process_group()) {
    _playout->comm->shutdown();
  }

  return status;
}

bool MeshBlockImpl::floor_hit(Variables const &vars) {
  auto hydro_u = vars.at("hydro_u");
  auto interior = part({0, 0, 0}, PartOptions().exterior(false));
  TORCH_CHECK(vars.count("hydro_w"),
              "MeshBlock::check_redo needs hydro_w to restore the primitives");
  // hydro_w is one stage stale: test the primitives as they stand
  auto w = phydro->peos->forward(hydro_u.clone()).index(interior);
  auto const &eos = phydro->peos->options;
  // negated so a NaN, which fails every comparison, counts as a hit
  bool hit = !(w[IDN].min().item<double>() > 1.001 * eos->density_floor());
  if (w.size(0) > IPR) {  // shallow water carries no pressure row
    hit = hit || !(w[IPR].min().item<double>() > 1.001 * eos->pressure_floor());
  }
  return hit;
}

bool MeshBlockImpl::vic_dry_clamp_hit() const {
  return phydro->picorr && phydro->picorr->dry_clamp_step().item<double>() > 0.;
}

bool MeshBlockImpl::limiter_patch_hit() const { return limiter_hits()[0]; }

std::array<bool, 2> MeshBlockImpl::limiter_hits() const {
  auto const &m = phydro->peos->limiter_marks();
  if (!m.defined()) return {false, false};
  auto c = m.to(torch::kCPU);  // one device-to-host read for both marks
  auto a = c.accessor<bool, 1>();
  return {a[0], a[1]};
}

int64_t MeshBlockImpl::saturation_failures() {
  // by name, as step (6) does: the EOS types keep their ThermoY private
  auto modules = named_modules();
  auto m = modules.find("hydro.eos.thermo");
  auto pthermo =
      m ? std::dynamic_pointer_cast<kintera::ThermoYImpl>(*m) : nullptr;
  return pthermo ? pthermo->take_saturation_adjustment_failures() : 0;
}

int MeshBlockImpl::apply_redo(Variables &vars, int causes) {
  if (causes) {
    SINFO(MeshBlock)
        << "Density/pressure at or within 0.1% of the floor, the VIC dry-gas "
           "clamp emptied a cell, the limiter patched one or found a NaN, or "
           "the saturation adjustment left a cell unadjusted, or the VIC "
           "solve rejected a singular/near-singular or nonfinite column. "
           "Redoing the step with smaller dt (causes:"
        << (causes & 1 ? " floor" : "") << (causes & 2 ? " clamp" : "")
        << (causes & 4 ? " limiter" : "") << (causes & 8 ? " nan" : "")
        << (causes & 16 ? " saturation" : "")
        << (causes & 32 ? " vic-solve" : "") << ")." << std::endl;
    pintg->current_redo += 1;
    if (pintg->current_redo > pintg->options->max_redo() && !(causes & 32)) {
      SINFO(MeshBlock)
          << "Maximum number of redo attempts exceeded. Terminating."
          << std::endl;
      return -1;  // terminate
    }

    // reset variables (the fixer's -D goes with them: drop its count)
    phydro->drop_gravity_work_fix();
    vars["hydro_u"].copy_(_hydro_u0);
    if (phydro->pdiffusion && phydro->pdiffusion->options->implicit_radial())
      phydro->pdiffusion->restore_step(vars["hydro_u"]);
    phydro->peos->forward(vars["hydro_u"], vars["hydro_w"]);
    if (vars.count("scalar_s")) {
      vars["scalar_s"].copy_(_scalar_s0);
      set_scalar_primitive(vars, vars["scalar_s"], vars["hydro_u"]);
    }

    // The caller advances the cycle only after an accepted step.
    // decided: floor_hit's and the restore's cons2prim must not mark the redo
    phydro->peos->reset_limiter_marks(vars["hydro_u"]);
    if (pintg->current_redo > pintg->options->max_redo()) {
      SINFO(MeshBlock)
          << "Maximum number of redo attempts exceeded. Terminating after "
             "restoring the rejected VIC step input."
          << std::endl;
      return -1;
    }
    return 1;  // redo
  }

  // good to go
  pintg->current_redo = 0;
  phydro->commit_gravity_work_fix();
  return 0;
}

torch::Tensor MeshBlockImpl::local_redo_flag_tensor(Variables const &vars) {
  // floor_hit's test, kept on the device
  auto hydro_u = vars.at("hydro_u");
  auto interior = part({0, 0, 0}, PartOptions().exterior(false));
  TORCH_CHECK(vars.count("hydro_w"),
              "MeshBlock::check_redo needs hydro_w to restore the primitives");
  // Retain the checked primitives for output and the next timestep.
  // A rejected step restores both fields in apply_redo.
  auto const &hydro_w = vars.at("hydro_w");
  {
    DeferVaporCheck defer(phydro->peos.get());
    phydro->peos->forward(hydro_u.clone(), hydro_w);
  }
  auto w = hydro_w.index(interior);
  auto const &eos = phydro->peos->options;
  // negated so a NaN, which fails every comparison, counts as a hit
  auto floor = (w[IDN].min() > 1.001 * eos->density_floor()).logical_not();
  if (w.size(0) > IPR) {  // shallow water carries no pressure row
    floor = floor.logical_or(
        (w[IPR].min() > 1.001 * eos->pressure_floor()).logical_not());
  }
  auto opts = hydro_u.options().dtype(torch::kFloat64);
  auto zero = torch::zeros({}, opts);

  auto clamp = phydro->picorr
                   ? (phydro->picorr->dry_clamp_step()[0] > 0.).to(opts)
                   : zero;

  // floor's cons2prim may mark a limiter repair: stacked after it, in order
  auto const &m = phydro->peos->limiter_marks();
  auto marks = m.defined() ? m.to(opts) : torch::zeros({2}, opts);

  // drains kintera's count exactly once, as saturation_failures() does
  auto sat = zero;
  auto modules = named_modules();
  auto mt = modules.find("hydro.eos.thermo");
  auto pthermo =
      mt ? std::dynamic_pointer_cast<kintera::ThermoYImpl>(*mt) : nullptr;
  if (pthermo && pthermo->nfail_.defined()) {
    sat = (pthermo->nfail_ > 0).to(opts).reshape({});
    pthermo->nfail_ = torch::Tensor();
  }

  // the VIC failure latch, read in the one copy below
  auto solve =
      phydro->picorr ? phydro->picorr->solve_failed_tensor(opts) : zero;
  // 7th: the step's deferred vapor-repair failures, a limiter cause
  auto const &nfv = phydro->peos->vapor_failures();
  auto vapor = nfv.defined() ? (nfv > 0).to(opts) : zero;
  auto out = torch::stack(
      {floor.to(opts), clamp, marks[0], marks[1], sat, solve, vapor});
  if (nfv.defined()) nfv.zero_();  // read once: after the stack, in order
  return out;
}

std::array<bool, 6> MeshBlockImpl::read_redo_flags(torch::Tensor const &flags) {
  auto c = flags.to(torch::kCPU);  // the one device-to-host read
  auto a = c.accessor<double, 1>();
  std::array<bool, 6> out;
  for (size_t i = 0; i < out.size(); ++i) out[i] = a[i] > 0.;
  // a vapor column the limiter could not repair: redo at a smaller dt (#317)
  if (c.size(0) > 6 && a[6] > 0.) {
    std::cerr << "[MeshBlock] rank=" << get_rank()
              << " vapor repair failed; check_redo will redo the step."
              << std::endl;
    out[2] = true;
  }
  return out;
}

std::array<bool, 6> MeshBlockImpl::local_redo_flags(Variables const &vars) {
  return read_redo_flags(local_redo_flag_tensor(vars));
}

int MeshBlockImpl::reduce_redo_flags(std::array<bool, 6> const &flags) const {
  auto flag = torch::zeros({6}, torch::dtype(torch::kFloat64));
  auto f = flag.accessor<double, 1>();
  for (size_t i = 0; i < flags.size(); ++i) f[i] = flags[i] ? 1. : 0.;
  std::vector<at::Tensor> reduced = {flag};
  if (_playout->has_process_group()) {
    _playout->comm->allreduce(reduced, c10d::ReduceOp::MAX);
  }
  auto global = reduced[0].accessor<double, 1>();
  int causes = 0;
  for (size_t i = 0; i < flags.size(); ++i) causes |= (global[i] > 0.) << i;
  return causes;
}

int MeshBlockImpl::check_redo(Variables &vars) {
  return apply_redo(vars, reduce_redo_flags(local_redo_flags(vars)));
}

double MeshBlockImpl::_init_from_restart(Variables &vars, std::string fname) {
  std::filesystem::path restart_path(fname);
  if (!restart_path.is_absolute() && !std::filesystem::exists(restart_path)) {
    restart_path = std::filesystem::path(options->output_dir()) / fname;
  }

  auto data = load_restart(restart_path.string(), options->layout()->rank());

  // check required variables
  TORCH_CHECK(data.count("hydro_u"),
              "Restart file is missing required variable: hydro_u");
  TORCH_CHECK(data.count("last_time"),
              "Restart file is missing required variable: last_time");
  TORCH_CHECK(data.count("last_cycle"),
              "Restart file is missing required variable: last_cycle");
  TORCH_CHECK(data.count("file_number"),
              "Restart file is missing required variable: file_number");
  TORCH_CHECK(data.count("next_time"),
              "Restart file is missing required variable: next_time");

  // Resume from the saved cycle so nlim-based runs do not take an extra step.
  cycle = data.at("last_cycle").item<int64_t>();

  // start timing
  _time_start = clock();
  _cycle_start = cycle;

  auto current_time = data.at("last_time").item<double>();

  // schedules restored by key; an edited block keeps its position, as before
  int nsaved = data.at("file_number").size(0);
  std::vector<bool> claimed(nsaved, false);
  std::vector<int> match(output_types.size(), -1);
  // New files append N exact keys after the N legacy keys. Old readers consume
  // only the prefix; an old writer replaces it with an N-entry legacy tensor.
  auto saved_keys =
      data.count("output_key") ? data.at("output_key") : torch::Tensor();
  bool precise_keys = saved_keys.defined() && saved_keys.dim() == 1 &&
                      saved_keys.size(0) == 2 * nsaved;
  int key_offset = precise_keys ? nsaved : 0;
  auto current_key = [precise_keys](std::shared_ptr<OutputType> const &output) {
    return precise_keys ? output->schedule_key_v2() : output->schedule_key();
  };
  if (saved_keys.defined()) {
    // a block still matching its own saved slot keeps it; nothing can steal it
    for (int n = 0; n < nsaved && n < (int)output_types.size(); ++n) {
      if (saved_keys[key_offset + n].item<int64_t>() ==
          current_key(output_types[n])) {
        match[n] = n;
        claimed[n] = true;
      }
    }
    for (int n = 0; n < output_types.size(); ++n) {
      if (match[n] >= 0) continue;
      auto key = current_key(output_types[n]);
      for (int k = 0; k < nsaved; ++k) {
        if (!claimed[k] && saved_keys[key_offset + k].item<int64_t>() == key) {
          match[n] = k;
          claimed[k] = true;
          break;
        }
      }
    }
  }
  for (int n = 0; n < output_types.size(); ++n) {
    if (match[n] < 0 && n < nsaved && !claimed[n]) {
      match[n] = n;
      claimed[n] = true;
    }
  }
  for (int n = 0; n < output_types.size(); ++n) {
    // the file name is out<n>, so the counter follows the POSITION, not the key
    if (n < nsaved) {
      output_types[n]->file_number = data.at("file_number")[n].item<int64_t>();
    }
    int m = match[n];
    if (m >= 0) {
      double saved = data.at("next_time")[m].item<double>();
      double dt = output_types[n]->options->dt();
      // A disabled output stores a next_time far past the resume. Keeping it
      // after dt shrinks means the output never fires. A time still within
      // one dt is a live cadence and stays. dt <= 0 keeps the saved time.
      if (dt > 0.0 && saved > current_time + dt) {
        output_types[n]->next_time = current_time + dt;
      } else {
        output_types[n]->next_time = saved;
      }
    } else {  // new output joins the restored grid
      auto dt = output_types[n]->options->dt();
      output_types[n]->next_time =
          dt > 0.0 ? current_time - std::fmod(current_time, dt) : current_time;
    }
  }

  bool rebuild_scalar_r =
      pscalar->nvar() > 0 && data.count("scalar_s") && data.count("hydro_u");

  // move to device
  for (auto &[name, tensor] : data) {
    if (rebuild_scalar_r && name == "scalar_r") {
      continue;
    }
    vars[name] = tensor.to(torch::Device(options->device_str()));
  }

  if (has_radiating_boundary()) {
    TORCH_CHECK(
        data.count("boundary_reference_w") &&
            (pscalar->nvar() == 0 || data.count("boundary_reference_r")),
        "Characteristic outflow restart is missing boundary reference "
        "tensors. Initialize afresh or select extrapolation.");
    for (auto key : {"boundary_reference_w", "boundary_reference_r"}) {
      if (vars.count(key))
        vars[key] = vars.at(key).to(vars.at("hydro_u").options());
    }
    // Validate without changing the serialized numerical state.
    apply_boundaries(
        vars, vars.at("hydro_u").clone(),
        vars.count("scalar_s") ? vars.at("scalar_s").clone() : torch::Tensor());
  }

  // remove timing data
  vars.erase("last_time");
  vars.erase("last_cycle");
  vars.erase("file_number");
  vars.erase("next_time");
  vars.erase("output_key");

  if (rebuild_scalar_r) {
    set_scalar_primitive(vars, vars.at("scalar_s"), vars.at("hydro_u"));
  }

  return current_time;
}

bool MeshBlockImpl::has_radiating_boundary() const {
  if (phydro->peos->options->type() == "shallow-water") return false;
  for (int f = 0; f < options->bfuncs().size(); ++f) {
    int nc = f / 2 == 0   ? pcoord->options->nc1()
             : f / 2 == 1 ? pcoord->options->nc2()
                          : pcoord->options->nc3();
    if (nc > 1 && is_outflow(options->bfuncs()[f])) return true;
  }
  return false;
}

void MeshBlockImpl::apply_boundaries(Variables &vars, torch::Tensor hydro,
                                     torch::Tensor tracers, bool primitive) {
  BoundaryFuncOptions op;
  op.nghost(pcoord->options->nghost());
  op.eos = phydro->peos.get();
  op.coord = pcoord.get();
  bool radiating = has_radiating_boundary();
  bool has_tracers = tracers.defined() && tracers.size(0) > 0;
  if (radiating) {
    TORCH_CHECK(vars.count("boundary_reference_w"),
                "outflow: missing initial boundary reference");
    op.reference = vars.at("boundary_reference_w");
    if (has_tracers) {
      TORCH_CHECK(vars.count("boundary_reference_r"),
                  "outflow: missing initial tracer reference");
      op.tracer_reference = vars.at("boundary_reference_r");
    }
  }

  torch::Tensor w, r;
  std::vector<int> pending_faces;
  // Consecutive radiating faces share one primitive conversion. Commit their
  // ghosts before another callback so it sees the selected representation and
  // the results of preceding faces, including corners.
  auto flush = [&]() {
    if (pending_faces.empty()) return;
    auto u = phydro->peos->compute("W->U", {w});
    // Tracer densities ride the conserved (dry) density, as in
    // set_scalar_primitive, not the total density w[IDN].
    auto s = has_tracers ? r * u[IDN] : torch::Tensor();
    for (int f : pending_faces) {
      int dim = 3 - f / 2, ng = op.nghost();
      int start = f % 2 ? hydro.size(dim) - ng : 0;
      hydro.narrow(dim, start, ng).copy_(u.narrow(dim, start, ng));
      if (has_tracers)
        tracers.narrow(dim, start, ng).copy_(s.narrow(dim, start, ng));
    }
    pending_faces.clear();
  };

  for (int f = 0; f < options->bfuncs().size(); ++f) {
    auto const &fn = options->bfuncs()[f];
    int dim = 3 - f / 2;
    if (!fn || hydro.size(dim) == 1) continue;
    if (radiating && is_outflow(fn)) {
      if (primitive) {
        w = hydro;
        r = tracers;
      } else if (pending_faces.empty()) {
        w = phydro->peos->compute("U->W", {hydro.clone()});
        r = has_tracers ? tracers / hydro[IDN] : torch::Tensor();
      }
      op.type(kPrimitive);
      op.tracers = has_tracers ? r : torch::Tensor();
      fn(w, dim, op);
      if (!primitive) pending_faces.push_back(f);
    } else {
      flush();
      op.tracers = torch::Tensor();
      op.type(primitive ? kPrimitive : kConserved);
      fn(hydro, dim, op);
      if (has_tracers) {
        op.type(kScalar);
        fn(tracers, dim, op);
      }
    }
  }
  flush();
}
}  // namespace snap
