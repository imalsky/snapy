// C/C++
#include <sstream>

// torch
#include <torch/extension.h>
#include <torch/nn/modules/container/any.h>

// snap
#include <snap/hydro/balance_column.hpp>
#include <snap/hydro/hydro.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_hydro(py::module& m) {
  auto pyHydroOptions =
      py::class_<snap::HydroOptionsImpl, snap::HydroOptions>(m, "HydroOptions");

  pyHydroOptions.def(py::init<>())
      .def_static("from_yaml", &snap::HydroOptionsImpl::from_yaml,
                  py::arg("filename"), py::arg("verbose") = false)
      .def("__repr__",
           [](const snap::HydroOptions& a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("HydroOptions(\n{})", ss.str());
           })
      .ADD_OPTION(bool, snap::HydroOptionsImpl, verbose)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, disable_flux_x1)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, disable_flux_x2)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, disable_flux_x3)
      .ADD_OPTION(bool, snap::HydroOptionsImpl, wb_wall_clamp)
      .ADD_OPTION(snap::ConstGravityOptions, snap::HydroOptionsImpl, grav)
      .ADD_OPTION(snap::CoriolisOptions, snap::HydroOptionsImpl, coriolis)
      .ADD_OPTION(snap::DiffusionOptions, snap::HydroOptionsImpl, diffusion)
      .ADD_OPTION(snap::EquationOfStateOptions, snap::HydroOptionsImpl, eos)
      .ADD_OPTION(snap::ReconstructOptions, snap::HydroOptionsImpl, recon1)
      .ADD_OPTION(snap::ReconstructOptions, snap::HydroOptionsImpl, recon23)
      .ADD_OPTION(snap::RiemannSolverOptions, snap::HydroOptionsImpl, riemann)
      .ADD_OPTION(snap::ImplicitOptions, snap::HydroOptionsImpl, icorr);

  ADD_SNAP_MODULE(Hydro, HydroOptions)
      .def(py::init<snap::HydroOptions, torch::nn::Module*>(),
           py::arg("options"), py::arg("block") = nullptr)
      .def("max_time_step", &snap::HydroImpl::max_time_step);

  m.def("balance_column", &snap::balance_column, py::arg("w"), py::arg("dx1f"),
        py::arg("grav"), py::arg("wall_clamp") = true, py::arg("rtol") = 1.e-10,
        py::arg("max_iter") = 120, py::arg("geometry") = "",
        py::arg("stretched") = false,
        R"(Project a ghost-free x1 column onto the well-balanced scheme's own
discrete hydrostatic balance, holding p/rho -- the temperature -- fixed per
cell. Returns (w, residual, sweeps); the residual is max|p'-C|/(rho*g*dz), the
acceleration the balanced column can still feel in units of g. `w` is
(nvar, nc3, nc2, nx1) with NO ghost cells: hand it the whole column and the p
and rho it returns do not depend on how that column is later split over blocks,
because the cell pressure reference is bitwise the same either way. Raises if it
does not converge. `wall_clamp` must be true and must match
`dynamics/wb-wall-clamp` in the card that will run the result, so a card that
turns the clamp off cannot use this primitive. See
src/hydro/balance_column.hpp for the caller's two obligations (physical x1
boundaries; >= 5 x1 cells per block). Under SNAP_X1_CENTROID_EXACT pass
geometry='cartesian'; any other geometry is refused. On a stretched x1 grid
key (#280) pass stretched=True (the coordinate's stretched_x1()), so the
column is classified as the solver classifies its blocks.)");
}
