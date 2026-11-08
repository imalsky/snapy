// torch
#include <torch/extension.h>

// snap
#include <snap/coord/coord_utils.hpp>
#include <snap/coord/coordinate.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/coord/gnomonic_equiangle.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_coord(py::module &m) {
  auto pyCoordinateOptions =
      py::class_<snap::CoordinateOptionsImpl, snap::CoordinateOptions>(
          m, "CoordinateOptions");

  pyCoordinateOptions.def(py::init<>(&snap::CoordinateOptionsImpl::create))
      .def("__repr__",
           [](const snap::CoordinateOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("CoordinateOptions(\n{})", ss.str());
           })
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x1min)
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x1max)
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x2min)
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x2max)
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x3min)
      .ADD_OPTION(double, snap::CoordinateOptionsImpl, x3max)
      .ADD_OPTION(int, snap::CoordinateOptionsImpl, nx1)
      .ADD_OPTION(int, snap::CoordinateOptionsImpl, nx2)
      .ADD_OPTION(int, snap::CoordinateOptionsImpl, nx3)
      .ADD_OPTION(int, snap::CoordinateOptionsImpl, nghost)
      .ADD_OPTION(std::vector<double>, snap::CoordinateOptionsImpl,
                  x1f_interior)
      .def("stretched_x1", &snap::CoordinateOptionsImpl::stretched_x1)
      .def("x1f_ghosts", &snap::CoordinateOptionsImpl::x1f_ghosts)
      .def("global_x1f", &snap::CoordinateOptionsImpl::global_x1f);

  auto pyCoordinate =
      py::class_<snap::CoordinateImpl, snap::Coordinate>(m, "Coordinate");

  pyCoordinate
      .def(py::init<snap::CoordinateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("hydro") = nullptr)
      .def("__repr__",
           [](const snap::CoordinateImpl &self) {
             std::stringstream ss;
             self.options->report(ss);
             return fmt::format("Coordinate(\n{})", ss.str());
           })
      .def("il", &snap::CoordinateImpl::il)
      .def("iu", &snap::CoordinateImpl::iu)
      .def("jl", &snap::CoordinateImpl::jl)
      .def("ju", &snap::CoordinateImpl::ju)
      .def("kl", &snap::CoordinateImpl::kl)
      .def("ku", &snap::CoordinateImpl::ku)
      .def(
          "center_width1",
          py::overload_cast<>(&snap::CoordinateImpl::center_width1, py::const_))
      .def(
          "center_width2",
          py::overload_cast<>(&snap::CoordinateImpl::center_width2, py::const_))
      .def(
          "center_width3",
          py::overload_cast<>(&snap::CoordinateImpl::center_width3, py::const_))
      .def("center_distance1",
           py::overload_cast<>(&snap::CoordinateImpl::center_distance1,
                               py::const_))
      .def("center_distance2",
           py::overload_cast<>(&snap::CoordinateImpl::center_distance2,
                               py::const_))
      .def("center_distance3",
           py::overload_cast<>(&snap::CoordinateImpl::center_distance3,
                               py::const_))
      .def("face_area1",
           py::overload_cast<>(&snap::CoordinateImpl::face_area1, py::const_))
      .def("face_area2",
           py::overload_cast<>(&snap::CoordinateImpl::face_area2, py::const_))
      .def("face_area3",
           py::overload_cast<>(&snap::CoordinateImpl::face_area3, py::const_))
      .def("cell_volume", &snap::CoordinateImpl::cell_volume);

  auto pyCartesian =
      py::class_<snap::CartesianImpl, snap::CoordinateImpl, torch::nn::Module,
                 std::shared_ptr<snap::CartesianImpl>>(m, "Cartesian");

  torch::python::add_module_bindings(pyCartesian)
      .def(py::init<snap::CoordinateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("hydro") = nullptr)
      .def("buffer",
           [](snap::CartesianImpl &self, std::string name) {
             return self.named_buffers()[name];
           })
      .def("module", [](snap::CartesianImpl &self, std::string name) {
        return self.named_modules()[name];
      });

  auto pyGnomonicEquiangle =
      py::class_<snap::GnomonicEquiangleImpl, snap::CoordinateImpl,
                 torch::nn::Module,
                 std::shared_ptr<snap::GnomonicEquiangleImpl>>(
          m, "GnomonicEquiangle");

  torch::python::add_module_bindings(pyGnomonicEquiangle)
      .def(py::init<snap::CoordinateOptions, torch::nn::Module *>(),
           py::arg("options"), py::arg("hydro") = nullptr)
      .def("buffer",
           [](snap::GnomonicEquiangleImpl &self, std::string name) {
             return self.named_buffers()[name];
           })
      .def("module", [](snap::GnomonicEquiangleImpl &self, std::string name) {
        return self.named_modules()[name];
      });

  auto m_coord = m.def_submodule("coord", "Coordinate submodule");
  m_coord.def("coord_vec_lower_", &snap::coord_vec_lower_)
      .def("coord_vec_raise_", &snap::coord_vec_raise_)
      .def("cs_cart_to_contra_", &snap::cs_cart_to_contra_)
      .def("cs_contra_to_cart_", &snap::cs_contra_to_cart_)
      .def("cs_sph_to_contra_", &snap::cs_sph_to_contra_)
      .def("cs_contra_to_sph_", &snap::cs_contra_to_sph_)
      .def("cs_ab_to_lonlat", &snap::cs_ab_to_lonlat)
      .def("get_cs_face_name", [](int face_id) {
        return std::string(snap::CS_FACE_NAMES[face_id]);
      });
}
