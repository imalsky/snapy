# #289: a fourth-order, cell/face-consistent x1 well-balanced reference (`SNAP_WB_REF4`)

This file derives the switch `SNAP_WB_REF4` (off unless set, read once per process) step by step, states
the oracle it is tested against, and reconciles it with the earlier unpublished prototype 473db21. The method
follows Jiheng's specification `wbref_exact_spec.md` (taken from his derivation at c5b810d, which was not
available on any GitHub fork when this was written). The exact weights and identities used below are checked
by `docs/derivations/wb_ref4_weights.py` (sympy). The code is `src/hydro/wb_ref4.cpp`, called from
`HydroImpl::_hydro_ref_x1` (`src/hydro/hydro.cpp`) and from `balance_column`.

Scope: Cartesian-exact. On spherical-polar and cubed-sphere grids the same index-space operators are applied
unchanged, which is not $r^2$-measure exact; that extension is a separate change.

## 0. Notation

- $z \equiv x_1$, gravity $g$ downward. Cell $i$ spans $[z_{i-1/2}, z_{i+1/2}]$ with width $\Delta z_i$; face $f$
  is the lower face of cell $f$. The owned cells are $i_s..i_u$.
- Ideal gas with $R_d = 1$ in the derivation, so $T = p/\rho$; $R \equiv \rho/p$.
- An overbar is a cell average; a subscript $f$ is a point value at a face.
- $\bar q = q + \tfrac{\Delta z^2}{24} q'' + O(\Delta z^4)$ for a smooth $q$ (cell average about the centre).
- $\mathcal I[\cdot]_f$: the face value the high-order (WENO5) reconstruction returns from cell averages of
  smooth data, to $O(\Delta z^5)$.

## 1. The base well-balanced x1 reconstruction

The solver reconstructs perturbations about a hydrostatic reference and restores the reference at the faces
(`hydro_forward.cpp`, step 2): $p' = \bar p - p_{\rm ref}$, $\rho' = \bar\rho - \rho_{\rm ref}$, even-parity
wall ghosts of $p'$ and $\rho'$, WENO5 of $p'$ and $\rho'$, then

$$
p^{L,R}_f = \widehat{p'}^{L,R}_f + p_{{\rm sf},f}, \qquad \rho^{L,R}_f = \widehat{\rho'}^{L,R}_f + \rho_{{\rm sf},f} .
$$

The kernel (`hydro_ref_x1_impl.h`) builds:

- **face pressure** by a top-down scan, $p_{{\rm sf},i-1/2} = p_{{\rm sf},i+1/2} + g\bar\rho_i\Delta z_i$: the
  discrete hydrostatic balance itself;
- **cell pressure** $p_{\rm ref}$: on a uniform grid the cell average of the quintic through six face
  pressures, $(11, -93, 802, 802, -93, 11)/1440$, one-sided rows at clamped walls; on a non-uniform grid the
  log-mean $\Delta p / \ln(p_{\rm lo}/p_{\rm hi})$;
- **cell density** $\rho_{{\rm ref},i} = p_{{\rm ref},i}\, r^s_i$, with $r_i = \bar\rho_i/\bar p_i$ smoothed by the
  binomial $B = (1, 4, 6, 4, 1)/16$ (edge replicated at a clamped wall);
- **face density** $\rho_{{\rm sf},f} = p_{{\rm sf},f}\,\tfrac12(r^s_{f-1} + r^s_f)$.

## 2. The base face-density offset is $O(\Delta z^2)$

Since the reconstruction treats $\rho'$ as cell averages, the face density it produces is

$$
\rho_f = \mathcal I[\bar\rho - \rho_{\rm ref}]_f + \rho_{{\rm sf},f} = \mathcal I[\bar\rho]_f + \delta\rho_f,
\qquad \delta\rho_f \equiv \rho_{{\rm sf},f} - \mathcal I[\rho_{\rm ref}]_f .
$$

So $\rho_f$ is a consistent high-order face value exactly when $\rho_{\rm sf}$ is the face value of the cell
field $\rho_{\rm ref}$ itself, whatever the accuracy of $\rho_{\rm ref}$. Step by step, for smooth $p$, $\rho$, $R$:

- Ratio of averages: $r_i = \bar\rho_i/\bar p_i = R_i + \tfrac{\Delta z^2}{24}(\rho'' - R p'')/p + O(\Delta z^4)$.
- Binomial bias: $B$ has unit sum and second moment $\sum_m m^2 w_m = 1$ (cells), so
  $B q = q + \tfrac{\Delta z^2}{2} q'' + O(\Delta z^4)$ and $r^s_i = r_i + \tfrac{\Delta z^2}{2}R''$.
- Cell reference, with $p_{\rm ref} = \bar p + c + O(\Delta z^6)$ (section 6): $\rho_{{\rm ref},i} = \bar\rho_i +
  \tfrac{\Delta z^2}{2} p R'' + O(\Delta z^4)$, so $\mathcal I[\rho_{\rm ref}]_f = \rho_f + \tfrac{\Delta z^2}{2} pR''$.
- Face reference: the mean of the two neighbouring cell values adds $\tfrac{\Delta z^2}{8}R''$, so
  $\rho_{{\rm sf},f} = p_f\big[R_f + \tfrac{\Delta z^2}{24}\tfrac{\rho'' - Rp''}{p} + \tfrac{\Delta z^2}{8}R'' + \tfrac{\Delta z^2}{2}R''\big]$.
- Subtract, with $\rho'' - Rp'' = 2p'R' + pR''$:

$$
\delta\rho_f = \frac{\Delta z^2}{12}\big(p'R' + 2pR''\big) + O(\Delta z^4).
$$

`wb_ref4_weights.py` (check 4) derives the same expression symbolically from exact cell averages of
quadratic $p$ and $R$ about the face. At a clamped wall, edge replication in $B$ is not exact even for linear
$r$, so the first two cells carry an $O(\Delta z)$ offset on top (a band of width $O(\Delta z)$).

**Where it enters.** The face pressure is untouched, so the offset enters only the x1 mass flux
$\bar u\rho_{L/R}$ at linear order, and through it the face gravity work ($-\tfrac g2(\delta F_{\rho,i-1/2} +
\delta F_{\rho,i+1/2})$ per cell). The energy flux $\bar u(\kappa p + \rho K)$ is independent of the face
density at linear order. The x2/x3 faces use the restored primitives, so this offset is separate from the
#289 x2 covariance term and adds to it.

## 3. The fix: cell density reference to fourth order

$$
\rho_{{\rm ref},i} = p_{{\rm ref},i}\,(F r)_i, \qquad F = \tfrac1{16}(-1, 4, 10, 4, -1).
$$

Properties of $F$ in cell-index units (`wb_ref4_weights.py`, check 1): unit sum, zero first, second and third
moments (fourth moment $-3/2$), so $Fq = q + O(\Delta z^4)$; zero response at the grid scale
($k\Delta z = \pi$), so it removes the 2-cell mode as $B$ does; reach $\pm2$, the same as $B$, so the ghost
depth and the x1 seam exchange are unchanged. Then $\rho_{\rm ref} = \bar\rho + O(\Delta z^4)$ and
$\rho' = O(\Delta z^4)$ on a smooth background (check 5).

**Walls.** At a clamped physical wall the two stencil values beyond the wall are cubic extrapolations, in cell
index, of the four owned cells nearest it: with $r_0..r_3$ from the wall inward, index $-1$ gets
$4r_0 - 6r_1 + 4r_2 - r_3$ and index $-2$ gets $10r_0 - 20r_1 + 15r_2 - 4r_3$ (check 2). Ghost cells are not
read. An identity follows (check 2): all five stencil values of the first two cells lie on that cubic, and
$F$ reproduces cubics, so

$$
(Fr)_{i_s} = r_{i_s}, \qquad (Fr)_{i_s+1} = r_{i_s+1}
$$

for any data, and likewise at the top. "Filter with cubic wall values" and "leave the two end cells
unfiltered" are the same reference (see section 9).

Elsewhere (seams, unclamped sides) the array's own, exchanged, values are read; past the array the edge
value is replicated.

## 4. The fix: face density as the face value of the same cell field

$$
\rho_{{\rm sf},f} = \mathcal I_4[\rho_{\rm ref}]_f \equiv P'(z_f),
$$

where $P$ is the quartic through the primitive values $P(z_{s+j}) = \sum_{k<j}\rho_{{\rm ref},s+k}\Delta z_{s+k}$,
$j = 0..4$ (five faces enclosing four cells), window $s = f - 2$, kept inside the owned cells at a clamped
wall. The weight of cell $s+k$ is $\Delta z_{s+k}\sum_{j>k}L_j'(z_f)$ with $L_j$ the Lagrange basis on the five
faces; this uses physical spacing and is exact for cubic point profiles on any grid. On a uniform grid
(check 3):

- interior: $(-1, 7, 7, -1)/12$;
- wall face, window starting at the wall: $(25, -23, 13, -3)/12$;
- the next face: $(3, 13, -5, 1)/12$ (and their mirror images at the top).

Then $\delta\rho_f = \mathcal I_4[\rho_{\rm ref}]_f - \mathcal I[\rho_{\rm ref}]_f = O(\Delta z^4)$, and the face
density the solver uses is $\mathcal I[\bar\rho]_f + O(\Delta z^4)$.

**Ordering.** The cell part (sections 3 and 5) runs right after the kernel, before the x1 seam ghost-row
exchange of $(p_{\rm ref}, \rho_{\rm ref})$, so the exchanged rows carry it. The face part runs after the
exchange, so a column split along x1 gets the same faces as one block. The in-process split test
(`test_pref_local_seam`, 4 blocks against 1 over 200 steps, tolerance $10^{-12}$) passes with the switch set.

## 5. Non-uniform x1 grids: the cell pressure

A keyed stretched x1 (`cells/x1rat` or `geometry/x1f`, #280) already has a
sixth-order cell pressure: the cell average of the degree-5 interpolant through six faces, built from the
widths (#280 (iv)). There the switch keeps it, so REF4 contributes only the density reference of section 3
and the face density of section 4, and `balance_column` does not run REF4: the stretched rest state is the
same with the switch on or off. $F$ stays in cell-index units; on a keyed
stretched family the measured order of $N^2_{\rm eff}$ is 4.06/4.04 (1 e-fold) and 3.78/3.89 (3 e-folds) at
$h_{\max}/h_{\min} = 4$. On a Cartesian x1 `SNAP_X1_CENTROID_EXACT` has no content of its own and runs as
`SNAP_WB_REF4`.

## 6. Exact discrete balance is kept

At rest $p' = $ const, so $p_L = p_R = p_{\rm sf} + c$, $\bar u = 0$ and the LMARS face pressure is
$p_{\rm sf} + c$; the momentum-flux difference across cell $i$ is $g\bar\rho_i\Delta z_i$ and cancels the gravity
source exactly. The density reference is not part of this pairing: with $\bar u = 0$ the mass flux and the
face gravity work vanish whatever the face density is. The fix never changes $p_{\rm sf}$ or $p_{\rm ref}$, so
the discrete rest state $\bar p = p_{\rm ref} + c$ is the base's.

Why $p_{\rm ref} = \bar p + c + O(\Delta z^6)$ on a hydrostatic background: the scan gives
$p_{{\rm sf},i-1/2} - p_{{\rm sf},i+1/2} = g\bar\rho_i\Delta z_i = \int_{\rm cell} g\rho\,dz$ exactly, so the scan
faces are the exact face pressures up to one column constant $c$ (the top anchor); the six-face quadrature
then has error $O(\Delta z^6)$. The constant changes $\rho_{\rm ref}$ by the smooth $cR$, which section 4
carries consistently because the face value is taken of the same $\rho_{\rm ref}$.

## 7. Guards

A guarded cell or face keeps the kernel's value. None fires on a resolved smooth column.

- Range, cells: $(Fr)_i$ outside $[\min, \max]$ of $r$ over cells $i-1, i, i+1$ (owned cells only at a clamped
  wall), with a relative margin $10^{-10}$, keeps the kernel's smoothed ratio. The margin matters: on a smooth
  monotone column $(Fr)_{i_s} = r_{i_s}$ is the bound itself and round-off puts it one ulp outside.
- Range, faces: $\mathcal I_4[\rho_{\rm ref}]_f$ outside $[\min, \max]$ of its two neighbouring cell references,
  same margin, keeps the kernel's face value.
- Resolution: a cell with $\lvert\ln(p_{{\rm sf},i-1/2}/p_{{\rm sf},i+1/2})\rvert > 0.5$ (fewer than two cells per
  pressure scale height) is flagged, the flag is dilated by two cells, and ghost cells past a clamped wall do
  not count; flagged cells keep the kernel's reference, and faces next to a flagged cell keep the kernel's
  face value.
- Seams: each block computes the flag from its own scan pressures, ghost cells included, and the flag is not
  exchanged. The flag of a cell or a seam face reads $p_{\rm sf}$ at most three cells away, so with nghost
  $\ge 3$ (checked at setup) the two sides of an x1 seam agree, except at an exact tie
  $\lvert\ln(p_{{\rm sf},i-1/2}/p_{{\rm sf},i+1/2})\rvert = 0.5$, where the two scans' last-ulp difference can
  put the cell on either side (`test_x1_seam_split_wb_ref4`: the flag switching on next to a seam).
- A block with a clamped wall and fewer than four owned cells keeps the kernel's reference. Its stencil cache
  is then "not usable" and holds no weights, so the check that rebuilds the cache for another device or dtype
  (`WbRef4Stencils::stale`) never reads them; before #298 it read the device of the undefined weight tensor and
  the second RK stage threw (`tests/test_stencil_cache_small_block.py`: nx1 2–5 per block now run).

The threshold, the dilation and the margin are design choices, not derived.

## 8. The oracle

Cartesian, one-step, linear. Neutral polytrope with $\gamma = 5/3$, $m = 1/(\gamma-1)$, $g = m+1$:
$T_0 = 1 + L_z - z$, $\rho_0 = T_0^m$, $p_0 = T_0^{m+1}$; the column is $n$ pressure e-folds deep,
$L_z = e^{n/(m+1)} - 1$; $L_x = 2L_z$, $n_x = 2n_z$; reflecting x1 walls, periodic x2, WENO5, LMARS, face
gravity work, the #289 flux covariance on.

- Mode, seeded as momentum: $\rho_0 v_1 = Ak\sin(qz)\cos(kx)$, $\rho_0 v_2 = -Aq\cos(qz)\sin(kx)$,
  $q = \pi/L_z$, $k = 2\pi/L_x$, $A = 10^{-4}$.
- Cell-average IC: exact cell averages (4x4 Gauss-Legendre) of $\rho$, $m_1$, $m_2$ and $E$; primitives
  $\bar\rho$, $\bar m/\bar\rho$, $\bar p = (\gamma-1)(\bar E - \lvert\bar m\rvert^2/2\bar\rho)$.
- One RK3 step at $\Delta t = 0.3\Delta z/\sqrt{\gamma T_{\rm bottom}}$; with $s = \ln(p\rho^{-\gamma})$ from the
  end-of-step conserved state, $\delta s_i = [(s^1 - s^0)_{\rm mode} - (s^1 - s^0)_{\rm rest}]/\Delta t$.
- $N^2_{\rm eff} = \tfrac g\gamma \sum_i(-\delta s_i)w_i / \sum_i w_i^2$, $w = v_1$ at $t = 0$, all interior cells.
  Reported as $N^2_{\rm eff}n_z^2$; the order is that of $\lvert N^2_{\rm eff}\rvert$ itself between doublings.

**No diagnostic floor.** $\nabla\cdot m = 0$ pointwise, so $\partial_t\bar\rho = 0$ exactly. The linear energy
tendency is $-\nabla\cdot(\kappa p_0 v) - g m_1 = -\kappa\, m\cdot\nabla T_0 - g m_1 = (\kappa - g)m_1 = 0$, since
$p_0v = T_0 m$, $T_0' = -1$ and $\kappa = \gamma/(\gamma-1) = g$. So the exact cell-average tendencies vanish at
linear order and every measured $\delta s$ is scheme error, at every depth.

## 9. Results

Code at the commit that adds this file (src unchanged since 1454878), CPU Release build. "off" = switch
unset, bitwise equal to 8cea3ae (the pre-rebase head of #293, merged to main as aea71ed); "on" = `SNAP_WB_REF4=1`. $N^2_{\rm eff}n_z^2$ at $n_z$ = 64 / 128 / 256 and the
observed order $p$ of $\lvert N^2_{\rm eff}\rvert$:

| depth | off (RED) | $p$ off | on (GREEN) | $p$ on | spec, on |
|---|---|---|---|---|---|
| 1 e-fold | +0.1090 / +0.0873 / +0.0756 | 2.32, 2.21 | -0.0707 / -0.0358 / -0.0180 | 2.98, 2.99 | -0.071 / -0.036 / -0.018 |
| 2 e-folds | +0.2921 / +0.2656 / +0.2485 | 2.14, 2.10 | -0.1172 / -0.0604 / -0.0306 | 2.96, 2.98 | -0.119 / -0.061 / -0.031 |
| 3 e-folds | +0.5870 / +0.5598 / +0.5326 | 2.07, 2.07 | -0.2257 / -0.1194 / -0.0614 | 2.92, 2.96 | -0.235 / -0.122 / -0.062 |
| 5 e-folds | +1.5600 / +1.7330 / +1.7072 | 1.85, 2.02 | -0.9454 / -0.5492 / -0.2961 | 2.78, 2.89 | -1.046 / -0.580 / -0.305 |

Off, $N^2_{\rm eff}n_z^2$ tends to a constant (order 2); on, it falls like $\Delta z$ (order of $N^2_{\rm eff}$
about 3). The base column agrees with the spec's base column to the printed digits at 1 e-fold (+0.109 / +0.087 /
+0.076), within 2% at 2 and 3 e-folds, and within 7% at 5 e-folds ($n_z$ 64: +1.560 against +1.458); the
differences are not diagnosed (the spec's base is main 117e449, this one 8cea3ae, #293 before its rebase to main aea71ed).

Split into the 3 cells at each wall ("band") and the rest ("interior"), same denominator, on:

| depth | band, on | interior, on | interior, off |
|---|---|---|---|
| 1 e-fold | -0.0711 / -0.0359 / -0.0180 | +0.0003 / +0.0001 / +0.0000 | +0.0590 / +0.0622 / +0.0631 |
| 5 e-folds | -0.9363 / -0.5452 / -0.2950 | -0.0091 / -0.0039 / -0.0011 | +0.9107 / +1.3099 / +1.4813 |

The interior offset is removed; what is left is the order-3 wall band, as in the spec. Its source is not
identified (hypothesis: the even-parity perturbation ghosts or the reflecting-wall reconstruction; the spec
finds it mostly present with the covariance off and in the base too).

Linearity: at $2A$, 1 e-fold, $n_z$ 64 / 128, on: -0.070748 / -0.035790 against -0.070749 / -0.035792 at $A$.

The ctest `test_wb_ref4_order` runs this oracle at 1 and 3 e-folds, $n_z$ 32 / 64 / 128, and asserts order
$\ge 2.75$ with the switch and $< 2.5$ at the last doubling without it.

## 10. Reconciliation with the prototype 473db21

473db21 (local, not published) implemented the same two operators: $F$ on the owned interior cells with the
two end cells unfiltered, and the $(-1, 7, 7, -1)/12$ face value with one-sided cubic rows at the ends.
By the identity of section 3 its end cells are the spec's, and its face windows are the spec's clamped windows;
on the uniform Cartesian oracle above its $N^2_{\rm eff}$ agrees with this implementation to the printed
digits (bitwise at 9 of the 16 points, $\le 4\times10^{-7}$ relative at the rest).

Its report of "first order, sign flipped" was the diagnostic, not the scheme:

- the order it tabulated was that of $\varepsilon_{\rm eff}n_z^2$, not of $\varepsilon_{\rm eff}$; its 0.95-0.99 is an
  order 2.95-2.99 of the residual itself, the spec's order 3 (rerun on this code: +0.02989 / +0.01544 /
  +0.00783 / +0.00394 at $n_z$ 16-128, 1 H);
- its $\varepsilon_{\rm eff} \propto +\sum T_0\,\dot s\, w$ has the opposite sign convention to
  $N^2_{\rm eff} \propto -\sum\dot s\, w$, so the switch's negative $N^2_{\rm eff}$ reads positive there.

Its harness also differed ($\gamma = 1.4$, $g = 1$, velocity instead of momentum seeded at cell centres, a
different projection), so its numbers are not comparable digit by digit with the spec's.

What this implementation adds over 473db21: the face part after the seam exchange (473db21 built faces before
it, one-sided at block ends, so a split column could differ from one block); the per-cell resolution flag
instead of a whole-column 2-cells-per-$H$ test; the range guards of the spec with the $10^{-10}$ margin;
physical-spacing face weights.

## 11. Tests that pin the kernel's reference

Both run unchanged with the switch off and a second time with it set (`*_wb_ref4` ctests):

- `test_balance_column`: the switch changes no cell pressure, so the balanced columns are the kernel's.
- `test_face_floor`: with the switch the density reference follows the dipped cell, so $\rho'$ stays small,
  the reconstructed face densities either side stay positive (about $1.4\times10^{-8}$ and $5.6\times10^{-8}$,
  where the kernel's reference gives $-4.2\times10^{-8}$) and the floor does not fire; the switched mass and
  momentum fluxes at that face are pinned instead.

## 12. Not covered

- The $r^2$-measure (spherical-polar, cubed-sphere) version of sections 3-5.
- Multi-species columns use the dry-density channel, as the kernel does; untested with the switch.
- Multi-process x1 seams are covered only by the in-process split test.
