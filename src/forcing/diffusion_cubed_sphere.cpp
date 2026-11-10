// Conservative physical diffusion on a nonorthogonal cubed sphere.
#include <algorithm>
#include <array>
#include <cmath>
#include <snap/coord/coord_utils.hpp>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/mesh/meshblock.hpp>
#include <vector>

#include "forcing.hpp"

namespace snap {
namespace {
using Box = std::array<int64_t, 3>;  // k,j,i
using Tensor = torch::Tensor;
Tensor sub(Tensor t, Box lo, Box hi) {
  for (int d = 0; d < 3; ++d) t = t.narrow(d - 3, lo[d], hi[d] - lo[d]);
  return t;
}
Tensor vector_dot(Tensor a, Tensor b) { return (a * b).sum(0); }
std::array<Tensor, 3> basis(Tensor a, Tensor b, Tensor r, int panel) {
  auto shape = torch::broadcast_tensors({a, b, r})[0];
  std::array<Tensor, 3> out;
  for (int d = 0; d < 3; ++d) {
    out[d] = torch::zeros({3, shape.size(0), shape.size(1), shape.size(2)},
                          shape.options());
    out[d][d].fill_(1.);
    cs_contra_to_cart_(out[d], a, b, panel);
  }
  return out;
}

// Four-point polynomial weights. At block boundaries use inward stencils,
// never an unfilled corner or an interpolated neighbor-panel coordinate.
// Cubic traces make the seam jump penalty O(h^3) on a smooth solution.
struct Stencil {
  Tensor indices, values;
};
// With nodes, the weights are in those positions (cell centres) and evaluated
// at `at` (faces or centres): a stretched x1 (#280), h = 1.
Stencil weights(int n, bool faces, bool derivative, double h, Tensor like,
                Tensor nodes = Tensor(), Tensor at = Tensor()) {
  TORCH_CHECK(n >= 4, "Cubed-sphere diffusion needs four stencil cells.");
  std::vector<double> x(n), t(n + (faces ? 1 : 0));
  for (int k = 0; k < n; ++k) x[k] = k;
  for (int k = 0; k < static_cast<int>(t.size()); ++k)
    t[k] = k - (faces ? 0.5 : 0.);
  if (nodes.defined()) {
    auto xn = nodes.to(torch::kCPU, torch::kFloat64).contiguous();
    auto tn = at.to(torch::kCPU, torch::kFloat64).contiguous();
    std::copy_n(xn.data_ptr<double>(), x.size(), x.begin());
    std::copy_n(tn.data_ptr<double>(), t.size(), t.begin());
  }
  int rows = n + (faces ? 1 : 0);
  auto w = torch::empty({4, rows}, torch::kFloat64);
  auto indices = torch::empty({4, rows}, torch::kInt64);
  auto ptr = w.accessor<double, 2>();
  auto iptr = indices.accessor<int64_t, 2>();
  for (int row = 0; row < rows; ++row) {
    double q = row - (faces ? 0.5 : 0.);
    int start =
        std::max(0, std::min(n - 4, static_cast<int>(std::floor(q)) - 1));
    for (int j = start; j < start + 4; ++j) {
      double value = 0., s = t[row];
      if (!derivative) {
        value = 1.;
        for (int k = start; k < start + 4; ++k)
          if (k != j) value *= (s - x[k]) / (x[j] - x[k]);
      } else
        for (int k = start; k < start + 4; ++k)
          if (k != j) {
            double term = 1. / (x[j] - x[k]);
            for (int l = start; l < start + 4; ++l)
              if (l != j && l != k) term *= (s - x[l]) / (x[j] - x[l]);
            value += term / h;
          }
      ptr[j - start][row] = value;
      iptr[j - start][row] = j;
    }
  }
  return {indices.to(like.device()), w.to(like.options())};
}
Tensor transform(Tensor t, Stencil const &stencil, int dim) {
  // Keep the same four coefficients and summation order, including the
  // one-sided rows. A dense n-by-n product wasted O(n) work per output.
  if (dim < 0) dim += t.dim();
  std::vector<int64_t> shape(t.dim(), 1);
  shape[dim] = stencil.values.size(1);
  Tensor result;
  for (int j = 0; j < 4; ++j) {
    auto term =
        t.index_select(dim, stencil.indices[j]) * stencil.values[j].view(shape);
    if (j == 0)
      result = term;
    else
      result.add_(term);
  }
  return result;
}
constexpr std::array<std::array<int, 3>, 4> offsets = {
    {{0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}}};

// Exchange outward integrated flux, reconstructed trace, dynamic coefficients,
// and centre-to-face normal distance. Cartesian channels require no rotation.
// Opposite signs of a common flux guarantee conservation at every block seam.
void share_fluxes(MeshBlockImpl const *block, std::array<Tensor, 3> &flux,
                  std::array<Tensor, 4> const &trace,
                  std::array<Tensor, 4> const &normals) {
  auto c = block->pcoord;
  Box lo{c->kl(), c->jl(), c->il()}, hi{c->ku() + 1, c->ju() + 1, c->iu() + 1};
  auto packed =
      torch::zeros({12, flux[0].size(1), flux[0].size(2), flux[0].size(3)},
                   flux[0].options());
  std::array<Tensor, 3> area{c->face_area1(), c->face_area2(), c->face_area3()};
  std::array<Tensor, 4> outgoing;
  // Snapshot ALL original face fluxes before correcting any face. A face
  // corrected in an early exchange round must not be sent in a later round.
  for (int side = 0; side < 4; ++side) {
    int dir = side < 2 ? 1 : 2, dim = 2 - dir;
    bool upper = side % 2;
    Box fl = lo, fh = hi;
    fl[dim] = upper ? hi[dim] : lo[dim];
    fh[dim] = fl[dim] + 1;
    outgoing[side] = torch::cat(
        {(upper ? 1. : -1.) * sub(flux[dir], fl, fh) * sub(area[dir], fl, fh),
         trace[side]},
        0);
  }
  Variables vars{{"diffusion_flux", packed}};
  auto layout = block->get_layout();
  auto loc = layout->loc_of(layout->options->rank());
  // The exchange API indexes full block-shaped tensors. Preserve those
  // indices and its corner/interpolation rules, but stream the four former
  // 12-channel slices through one buffer. Scalar exchange acts independently
  // on channels, so four completed rounds reproduce the old 48-channel round.
  for (int sent_side = 0; sent_side < 4; ++sent_side) {
    packed.zero_();
    int sent_dim = sent_side < 2 ? 1 : 0;
    Box cl = lo, ch = hi;
    cl[sent_dim] = sent_side % 2 ? hi[sent_dim] - 1 : lo[sent_dim];
    ch[sent_dim] = cl[sent_dim] + 1;
    sub(packed, cl, ch).copy_(outgoing[sent_side]);
    block->exchange(vars, SyncOptions().type(kScalar).phyid(12));
    for (int side = 0; side < 4; ++side) {
      auto off = std::make_tuple(offsets[side][0], offsets[side][1], 0);
      auto peer = layout->loc_of(layout->neighbor_rank(loc, off));
      int ps = std::get<2>(peer) == std::get<2>(loc)
                   ? (side ^ 1)
                   : CS_FACE_EDGES[std::get<2>(loc)][side].nside;
      if (ps != sent_side) continue;
      int dir = side < 2 ? 1 : 2, dim = 2 - dir;
      bool upper = side % 2;
      double sign = upper ? 1. : -1.;
      Box gl = lo, gh = hi, fl = lo, fh = hi;
      gl[dim] = upper ? hi[dim] : lo[dim] - 1;
      gh[dim] = gl[dim] + 1;
      fl[dim] = upper ? hi[dim] : lo[dim];
      fh[dim] = fl[dim] + 1;
      auto other = sub(packed, gl, gh);
      auto own = trace[side];
      auto f = sub(flux[dir], fl, fh);
      f.copy_(0.5 *
              (f - sign * other.narrow(0, 0, 5) / sub(area[dir], fl, fh)));
      auto n = sign * normals[side];
      auto jump = other.narrow(0, 5, 4) - own.narrow(0, 0, 4);
      auto distance = own[6] + other[11];
      auto mu = 0.5 * (own[4] + other[9]), k = 0.5 * (own[5] + other[10]);
      // Symmetric interior penalty couples even piecewise-constant panels.
      auto dv = jump.narrow(0, 1, 3);
      auto mf = -2. * mu * (dv + n * vector_dot(n, dv) / 3.) / distance;
      auto vf = 0.5 * (own.narrow(0, 1, 3) + other.narrow(0, 6, 3));
      f.narrow(0, IVX, 3).add_(sign * mf);
      f[IPR].add_(sign * (-2. * k * jump[0] / distance + vector_dot(vf, mf)));
    }
  }
}
}  // namespace

Tensor DiffusionImpl::forward_cubed_sphere(Tensor du, Tensor w, Tensor temp,
                                           double dt) {
  auto block = phydro->pmb;
  auto c = block->pcoord;
  auto op = c->options;
  int panel = std::get<2>(
      block->get_layout()->loc_of(block->options->layout()->rank()));
  Box lo{c->kl(), c->jl(), c->il()}, hi{c->ku() + 1, c->ju() + 1, c->iu() + 1};
  auto a = c->x2v.to(w.options()).view({1, -1, 1});
  auto b = c->x3v.to(w.options()).view({-1, 1, 1});
  auto r = c->x1v.to(w.options()).view({1, 1, -1});
  auto field = torch::zeros({7, w.size(1), w.size(2), w.size(3)}, w.options());
  field[0].copy_(temp);
  field.narrow(0, 1, 3).copy_(w.narrow(0, IVX, 3));
  cs_contra_to_cart_(field.narrow(0, 1, 3), a, b, panel);
  field.narrow(0, 4, 3).copy_(r * basis(a, b, r, panel)[0]);
  auto cells = sub(field, lo, hi);
  auto mu = options->dynamic() ? torch::full_like(temp, options->nu_iso())
                               : options->nu_iso() * w[IDN];
  auto k = options->dynamic() ? torch::full_like(temp, options->kappa_iso())
                              : options->kappa_iso() * w[IDN] *
                                    phydro->peos->specific_heat_cv(w, temp);
  if (nu_scale_w_.defined()) mu.mul_(nu_scale_w_.view({1, 1, -1}));
  if (kappa_scale_w_.defined()) k.mul_(kappa_scale_w_.view({1, 1, -1}));
  auto coefficients = sub(torch::stack({mu, k}), lo, hi);
  mu = Tensor();
  k = Tensor();
  std::array<double, 3> h{op->dx1(), op->dx2(), op->dx3()};
  // x1 stencils on the actual radii when x1 is stretched (#280)
  auto stencil = [&](int j, bool faces, bool derivative) {
    int m = hi[2 - j] - lo[2 - j];
    if (j || !op->stretched_x1()) return weights(m, faces, derivative, h[j], w);
    auto xv = c->x1v.slice(0, lo[2], hi[2]);
    return weights(m, faces, derivative, 1., w, xv,
                   faces ? c->x1f.slice(0, lo[2], hi[2] + 1) : xv);
  };
  std::array<Tensor, 3> flux;
  std::array<Tensor, 4> trace, normals;
  for (int dir = 0; dir < 3; ++dir) {
    int dim = 2 - dir, axis = dim - 3, n = hi[dim] - lo[dim];
    auto face = transform(cells, stencil(dir, true, false), axis);
    std::array<Tensor, 3> deriv;
    for (int j = 0; j < 3; ++j) {
      deriv[j] = j == dir ? transform(cells, stencil(j, true, true), axis)
                          : transform(face, stencil(j, false, true), -1 - j);
    }
    Box fl = lo, fh = hi;
    ++fh[dim];
    auto af = (dir == 1 ? c->x2f : c->x2v)
                  .to(w.options())
                  .slice(0, fl[1], fh[1])
                  .view({1, -1, 1});
    auto bf = (dir == 2 ? c->x3f : c->x3v)
                  .to(w.options())
                  .slice(0, fl[0], fh[0])
                  .view({-1, 1, 1});
    auto rf = (dir == 0 ? c->x1f : c->x1v)
                  .to(w.options())
                  .slice(0, fl[2], fh[2])
                  .view({1, 1, -1});
    auto bases = basis(af, bf, rf, panel);
    auto normal = bases[0];
    if (dir > 0) {
      auto cos = vector_dot(bases[1], bases[2]);
      normal =
          (bases[dir] - cos * bases[3 - dir]) / torch::sqrt(1. - cos * cos);
    }
    bases[1] = Tensor();
    bases[2] = Tensor();
    // Differentiate Cartesian velocity and the Cartesian map with exactly
    // the same stencil: every affine Cartesian velocity has exact gradient,
    // including zero-strain solid rotation on every panel.
    // The reciprocal basis is the explicit 3x3 inverse. Besides avoiding
    // general LU solves, this avoids concurrent lazy CUDA linalg loading in
    // block workers. All operations stay on the calling worker's stream.
    std::array<Tensor, 3> reciprocal;
    for (int j = 0; j < 3; ++j)
      reciprocal[j] = torch::cross(deriv[(j + 1) % 3].narrow(0, 4, 3),
                                   deriv[(j + 2) % 3].narrow(0, 4, 3), 0);
    auto determinant = vector_dot(deriv[0].narrow(0, 4, 3), reciprocal[0]);
    auto vg =
        deriv[0].narrow(0, 1, 3).unsqueeze(1) * reciprocal[0].unsqueeze(0);
    for (int j = 1; j < 3; ++j)
      vg += deriv[j].narrow(0, 1, 3).unsqueeze(1) * reciprocal[j].unsqueeze(0);
    vg /= determinant;
    reciprocal = {};
    determinant = Tensor();
    // Interpolating a curved map does not land exactly on the geometric face.
    // Correct that displacement so affine Cartesian velocity traces agree
    // across panels (in particular, rigid rotation creates no jump penalty).
    face.narrow(0, 1, 3).add_(
        (vg * (rf * bases[0] - face.narrow(0, 4, 3)).unsqueeze(0)).sum(1));
    // Linear coefficient reconstruction is positive in the interior. Physical
    // wall extrapolation falls back to the nearest cell if it is negative.
    auto cf = torch::empty({2, face.size(1), face.size(2), face.size(3)},
                           w.options());
    // linear in position: x1 cell widths from the faces (#280), equal in x2/x3
    auto h = dir ? torch::ones({n}, w.options())
                 : c->dx1f.to(w.options()).slice(0, lo[2], hi[2]);
    std::vector<int64_t> hs(4, 1);
    hs[axis + 4] = n - 1;
    auto hl = h.slice(0, 0, n - 1).view(hs), hr = h.slice(0, 1, n).view(hs);
    cf.narrow(axis, 1, n - 1)
        .copy_(hr / (hl + hr) * coefficients.narrow(axis, 0, n - 1) +
               hl / (hl + hr) * coefficients.narrow(axis, 1, n - 1));
    for (int side = 0; side < 2; ++side) {
      int near = side ? n - 1 : 0, next = side ? n - 2 : 1,
          index = side ? n : 0;
      auto v = coefficients.select(axis, near);
      auto g = h[near] / (h[near] + h[next]);
      auto extrap = (1. + g) * v - g * coefficients.select(axis, next);
      cf.select(axis, index).copy_(torch::where(extrap >= 0., extrap, v));
    }
    auto div = vg[0][0] + vg[1][1] + vg[2][2];
    auto stress = cf[0] * (vg + vg.transpose(0, 1));
    vg = Tensor();
    for (int j = 0; j < 3; ++j) stress[j][j] -= (2. / 3.) * cf[0] * div;
    div = Tensor();
    auto traction = (stress * normal.unsqueeze(0)).sum(1);
    stress = Tensor();
    Tensor heat;
    if (dir == 0) {
      // Radial resistance preserves T=A+B/r exactly for centre samples.
      auto rc = c->x1v.to(w.options()).slice(0, lo[2], hi[2]);
      heat = torch::zeros_like(face[0]);
      auto tl = cells[0].narrow(-1, 0, n - 1),
           tr = cells[0].narrow(-1, 1, n - 1);
      auto rl = rc.slice(0, 0, n - 1), rr = rc.slice(0, 1, n);
      heat.narrow(-1, 1, n - 1)
          .copy_(-cf[1].narrow(-1, 1, n - 1) * (tr - tl) * rl * rr / (rr - rl) /
                 rf.narrow(-1, 1, n - 1).square());
      for (int side = 0; side < 2; ++side) {
        int index = side ? n : 0, ic = side ? n - 1 : 0;
        double tw =
            side ? options->temperature_outer() : options->temperature_inner();
        auto wallr = rf.select(-1, index), walln = normal.select(-1, index);
        auto wallt = traction.select(-1, index);
        auto vn = vector_dot(wallt, walln);
        if (options->wall_velocity() == "stress-free")
          wallt.copy_(walln * vn);
        else {
          auto slope = cells.narrow(0, 1, 3).select(-1, ic) / (rc[ic] - wallr);
          wallt.copy_(cf[0].select(-1, index) *
                      (slope + walln * vector_dot(walln, slope) / 3.));
        }
        face.narrow(0, 1, 3).select(-1, index).zero_();
        if (tw > 0.)
          heat.select(-1, index).copy_(
              -cf[1].select(-1, index) * (cells[0].select(-1, ic) - tw) /
              ((rc[ic] - wallr) / (wallr * rc[ic])) / wallr.square());
      }
    } else {
      // Analytic angular metric. No radial derivative can contaminate a
      // horizontal heat flux, even on a strongly stratified thin shell.
      auto x = af.tan(), y = bf.tan(), s = 1. + x * x + y * y;
      auto gaa = s / (1. + x * x), gbb = s / (1. + y * y),
           gab = s * x * y / ((1. + x * x) * (1. + y * y));
      auto gnn = dir == 1 ? gaa : gbb;
      heat =
          -cf[1] / rf *
          (gnn.sqrt() * deriv[dir][0] + gab / gnn.sqrt() * deriv[3 - dir][0]);
    }
    if (dir == 0 && options->implicit_radial()) heat.zero_();
    flux[dir] = torch::zeros({5, w.size(1), w.size(2), w.size(3)}, w.options());
    auto f = sub(flux[dir], fl, fh);
    f.narrow(0, IVX, 3).copy_(-traction);
    f[IPR].copy_(heat - vector_dot(face.narrow(0, 1, 3), traction));
    // Only angular boundary slabs survive the direction loop. Clone normals
    // so a slab view does not keep the full face-normal allocation alive.
    if (dir > 0)
      for (int side = 0; side < 2; ++side) {
        int edge = 2 * (dir - 1) + side, index = side ? n : 0;
        auto shape = face.sizes().vec();
        shape[0] = 7;
        shape[axis + face.dim()] = 1;
        auto t = torch::empty(shape, w.options());
        t.narrow(0, 0, 4).copy_(face.narrow(0, 0, 4).narrow(axis, index, 1));
        t.narrow(0, 4, 2).copy_(cf.narrow(axis, index, 1));
        auto xc = cells.narrow(0, 4, 3).select(axis, side ? n - 1 : 0);
        auto xf = (rf * bases[0]).select(axis, index);
        t[6].select(axis, 0).copy_(
            vector_dot(xf - xc, normal.select(axis, index)).abs());
        trace[edge] = t;
        normals[edge] = normal.narrow(axis, index, 1).clone();
      }
  }
  cells = Tensor();
  field = Tensor();
  coefficients = Tensor();
  share_fluxes(block, flux, trace, normals);
  auto tendency = -c->divergence(flux[0], flux[1], flux[2]);
  auto momentum = tendency.narrow(0, IVX, 3);
  cs_cart_to_contra_(momentum, a, b, panel);
  coord_vec_lower_(momentum, c->cosine_cell_kj.to(w.options()));
  du.narrow(0, 0, 5).add_(dt * tendency);
  return du;
}
}  // namespace snap
