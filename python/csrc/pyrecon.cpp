// torch
#include <torch/extension.h>

// snap
#include <snap/recon/reconstruct.hpp>
#include <snap/recon/x1_coeffs.hpp>

// python
#include "pyoptions.hpp"

namespace py = pybind11;

void bind_recon(py::module &m) {
  auto pyInterpOptions =
      py::class_<snap::InterpOptionsImpl, snap::InterpOptions>(m,
                                                               "InterpOptions");

  pyInterpOptions.def(py::init<>())
      .def(py::init<std::string>())
      .def("__repr__",
           [](const snap::InterpOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("InterpOptions(\n{})", ss.str());
           })
      .ADD_OPTION(std::string, snap::InterpOptionsImpl, type)
      .ADD_OPTION(bool, snap::InterpOptionsImpl, scale);

  auto pyReconstructOptions =
      py::class_<snap::ReconstructOptionsImpl, snap::ReconstructOptions>(
          m, "ReconstructOptions");

  pyReconstructOptions.def(py::init<>())
      .def("__repr__",
           [](const snap::ReconstructOptions &a) {
             std::stringstream ss;
             a->report(ss);
             return fmt::format("ReconstructOptions(\n{})", ss.str());
           })
      .ADD_OPTION(bool, snap::ReconstructOptionsImpl, shock)
      .ADD_OPTION(snap::InterpOptions, snap::ReconstructOptionsImpl, interp);

  ADD_SNAP_MODULE(Reconstruct, ReconstructOptions)
      .def(py::init<snap::ReconstructOptions>(), py::arg("options"))
      .def("set_x1_faces", &snap::ReconstructImpl::set_x1_faces, py::arg("x1f"),
           R"(Install route A per-face x1 coefficients from a face vector.

The face vector has nc1 + 1 entries and must be strictly increasing; face k
is the lower face of cell k.  The x1 sweep then RUNS these rows (readable as
the buffers "interp1.cm_x1", "interp1.cp_x1", "interp2.cm_x1", "interp2.cp_x1",
which exist only after this call), x2/x3 are refused, and so is a module whose
type is not weno5 or cp5 (note Sec. 4.7, design note bc29594f).)");

  // Free builders: route A coefficients from any face vector, with no mesh
  // and no block (snapy#280 Sec. 4.2).  See src/recon/x1_coeffs.hpp for the
  // 48-column layout and the face convention.
  m.def(
      "weno5_x1_coeffs",
      [](torch::Tensor const &x1f) {
        auto r = snap::weno5_x1_coeffs(x1f);
        return py::make_tuple(r.first, r.second);
      },
      py::arg("x1f"),
      R"(Per-face WENO5 rows for a face vector.

Returns (cm, cp), each (x1f.numel() - 5, 48) float64.  Row m belongs to the
five cells bounded by x1f[m:m+6]; cm targets face x1f[m+2] (the IRT state) and
cp targets face x1f[m+3] (the ILT state).  Columns 0-44 are the nine 5-entry
rows c1..c9 the kernel walks, columns 45-47 are the linear weights d0,d1,d2.)");

  m.def(
      "cp5_x1_coeffs",
      [](torch::Tensor const &x1f) {
        auto r = snap::cp5_x1_coeffs(x1f);
        return py::make_tuple(r.first, r.second);
      },
      py::arg("x1f"),
      R"(Per-face cp5 rows for a face vector.

Returns (cm, cp), each (x1f.numel() - 5, 5) float64, with the same face
convention as weno5_x1_coeffs.)");
}
