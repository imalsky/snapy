# Issue #289, item 2: the $O(\Delta x_1^2)$ error of the horizontal energy flux on curved grids and on $x_3$

> **Read §2A before implementing.** On a curved grid the complete $O(\Delta r^2)$
> correction has **two** terms, eq. (2A.9): the #289 covariance term *and* a
> centroid term of the same order. §2 derives the first; §2A derives the second
> and corrects the earlier claim that the first was the whole story. In
> Cartesian the second vanishes identically and §2 is complete.

**Scope.** Issue [#289](https://github.com/chengcli/snapy/issues/289) item 2: derive
the face average of the horizontal energy flux *with the face metric* for
spherical-$x_1$ and cubed-sphere grids, extend it to the $x_3$ direction, and
state the discrete form that is implemented.

**Base commit.** Everything quoted from the source below was read at
`117e449a620bd7fd50abe19239ab660a56d5d4cb` (`chengcli/snapy` `main`), confirmed
on 2026-10-09 by both

```
git ls-remote https://github.com/chengcli/snapy main
gh api repos/chengcli/snapy/commits/main --jq .sha
```

Every `file:line` below is a line number **at that commit**.

**Provenance.** Derived from the formula stated in #289 and from the snapy
source only. No other implementation or derivation of this term was read.

---

## 1. Starting definitions

### 1.1 The finite-volume update

For a conserved density $U$ with flux $\mathbf{f}$, the finite-volume balance over
cell $(k,j,i)$ is exact:

$$
\frac{\mathrm{d}}{\mathrm{d}t}\Big(V_{kji}\,\bar U_{kji}\Big)
= -\oint_{\partial\Omega_{kji}} \mathbf{f}\cdot\hat{\mathbf{n}}\,\mathrm{d}A ,
\qquad
\bar U_{kji} \equiv \frac{1}{V_{kji}}\int_{\Omega_{kji}} U\,\mathrm{d}V .
$$

snapy evaluates exactly this shape, in `CoordinateImpl::divergence`
(`src/coord/coordinate.cpp:461-505`):

```cpp
  if (flux2.defined() > 0) {                                  // :492
    dflx.slice(DIM2, sj, ej) +=
        face_area2(sj + 1, ej + 1) * flux2.slice(DIM2, sj + 1, ej + 1) -
        face_area2(sj, ej) * flux2.slice(DIM2, sj, ej);
  }
  if (flux3.defined() > 0) {                                  // :498
    dflx.slice(DIM3, sk, ek) +=
        face_area3(sk + 1, ek + 1) * flux3.slice(DIM3, sk + 1, ek + 1) -
        face_area3(sk, ek) * flux3.slice(DIM3, sk, ek);
  }
  return dflx / vol;                                          // :504
```

and the tendency is $-\mathrm{d}t$ times that divergence on the interior
(`src/hydro/hydro_forward.cpp:418`, `:427-429`). So the semi-discrete energy
balance is

$$
\frac{\mathrm{d}\bar E_{kji}}{\mathrm{d}t}
= -\frac{1}{V_{kji}}\Big[
A^{(2)}_{j+\frac12}F^{(2)}_{j+\frac12} - A^{(2)}_{j-\frac12}F^{(2)}_{j-\frac12}
+ A^{(3)}_{k+\frac12}F^{(3)}_{k+\frac12} - A^{(3)}_{k-\frac12}F^{(3)}_{k-\frac12}
+ (x_1\ \text{terms})\Big].
\tag{1.1}
$$

The identity (1.1) holds **only if** $A^{(2)}F^{(2)}$ is the true surface
integral, i.e. if

$$
\boxed{\;F^{(2)}_{j+\frac12}
= \big\langle f_E^{(2)} \big\rangle_{A}
\equiv \frac{1}{A^{(2)}_{j+\frac12}}\int_{S_{j+1/2}} f_E^{(2)}\,\mathrm{d}A \;}
\tag{1.2}
$$

— the **area-weighted average over the face** of the point flux. This is the
definition #289 starts from:

> The finite-volume flux through it is the x1 average over the face of the point
> flux gamma/(gamma-1) p u. The code instead evaluates the flux from x1-averaged
> states.

### 1.2 The flux that is actually evaluated

The energy flux normal to an $x_2$ face is $f_E^{(2)}=(E+p)\,u_n$ with
$E+p = \rho h + \tfrac12\rho|\mathbf{u}|^2$ and $h=e+p/\rho$ the specific
enthalpy. LMARS writes it as (`src/riemann/lmars_impl.h:28-31`, `:61`, `:76`)

```cpp
  hl += 0.5 * (SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ])) +
        wli[IPR] / wli[IDN];                                  // :28-29
  ...
  FLX(IPR) = ubar * wli[IDN] * hl;                            // :61
```

i.e. $F^{(2)} = \bar u_n\,\bar\rho\,\bar h$, every factor taken from the
**reconstructed face state**, where $\bar u_n$ is LMARS' acoustically corrected
normal velocity (`lmars_impl.h:42`) and the normal row is selected by
`ivx = IPR - dim` (`lmars_impl.h:20`): for `dim == 2` that is `IVY`, for
`dim == 1` (the $x_3$ direction, `DIM3 = 1`) it is `IVZ`
(`src/snap.h:38-40`). Keeping the enthalpy part and writing
$\rho h = \gamma/(\gamma-1)\,p$ for an ideal gas, the code evaluates

$$
F^{(2)}_{\rm code} = \frac{\gamma}{\gamma-1}\,\bar p\,\frac{\bar m}{\bar\rho},
\qquad m\equiv\rho u_n ,
\tag{1.3}
$$

whereas (1.2) demands $\frac{\gamma}{\gamma-1}\langle p\,u_n\rangle_A$. The
missing piece is

$$
\Delta F \;\equiv\; \frac{\gamma}{\gamma-1}\Big[\langle p u_n\rangle_A
- \bar p\,\frac{\bar m}{\bar\rho}\Big].
\tag{1.4}
$$

### 1.3 The face measure $\mathrm{d}A$, per grid

Write the general curvilinear area element of a surface $x^2=\text{const}$ as
$\mathrm{d}A_2=\sqrt{g}\,\sqrt{g^{22}}\,\mathrm{d}x^1\mathrm{d}x^3$. Rather than
re-deriving it per grid, read the *discrete* measure off snapy's own
`face_area2()`/`face_area3()`, which is what (1.1) actually uses.

**(a) Cartesian** (`src/coord/coordinate.cpp:429-435`):

```cpp
torch::Tensor CoordinateImpl::face_area2() const {            // :429
  return dx3f.outer(dx1f).unsqueeze(1).expand({-1, x2f.size(0), -1});
}
torch::Tensor CoordinateImpl::face_area3() const {            // :433
  return dx2f.outer(dx1f).unsqueeze(0).expand({x3f.size(0), -1, -1});
}
```

so $\mathrm{d}A_2 = \mathrm{d}x_1\,\mathrm{d}x_3$ and
$\mathrm{d}A_3=\mathrm{d}x_1\,\mathrm{d}x_2$: the $x_1$ weight is **uniform**,

$$
w_2(x_1)=w_3(x_1)=1 .
\tag{1.5}
$$

**(b) Spherical-polar**, $(x_1,x_2,x_3)=(r,\theta,\phi)$
(`src/coord/spherical_polar.cpp:185-198`):

```cpp
torch::Tensor SphericalPolarImpl::face_area2() const {        // :185
  auto radial = 0.5 * (x1f.slice(0, 1, options->nc1() + 1).square() -
                       x1f.slice(0, 0, options->nc1()).square());
  return radial.unsqueeze(0).unsqueeze(1) *
         x2f.sin().abs().unsqueeze(0).unsqueeze(2) *
         dx3f.unsqueeze(1).unsqueeze(2);
}
torch::Tensor SphericalPolarImpl::face_area3() const {        // :193
  auto radial = 0.5 * (x1f.slice(0, 1, options->nc1() + 1).square() -
                       x1f.slice(0, 0, options->nc1()).square());
  return (radial.unsqueeze(0).unsqueeze(1) * dx2f.unsqueeze(0).unsqueeze(2))
      .expand({x3f.size(0), -1, -1});
}
```

The `radial` factor is exactly $\tfrac12(r_+^2-r_-^2)=\int_{r_-}^{r_+} r\,\mathrm{d}r$.
Hence

$$
\mathrm{d}A_2 = r\,\sin\theta_{j+\frac12}\,\mathrm{d}r\,\mathrm{d}\phi,
\qquad
\mathrm{d}A_3 = r\,\mathrm{d}r\,\mathrm{d}\theta,
\qquad\Longrightarrow\qquad
w_2(r)=w_3(r)=r .
\tag{1.6}
$$

Note that the $r^2$ of the *$x_1$* face (`spherical_polar.cpp:178-183`) and of
the cell volume (`:200-211`) does **not** appear here: an $x_2$ or $x_3$ face
integrates $\mathrm{d}r$ against exactly **one** power of $r$, the one that comes
from the transverse arc length.

**(c) Cubed sphere** (gnomonic-equiangular), $(x_1,x_2,x_3)=(r,\alpha,\beta)$
(`src/coord/gnomonic_equiangle.cpp:197-203`):

```cpp
torch::Tensor GnomonicEquiangleImpl::face_area2() const {     // :197
  return (x1v * dx1f).unsqueeze(0).unsqueeze(1) * dx3f_ang_face2_kj;
}
torch::Tensor GnomonicEquiangleImpl::face_area3() const {     // :201
  return (x1v * dx1f).unsqueeze(0).unsqueeze(1) * dx2f_ang_face3_kj;
}
```

Here `x1v` is the arithmetic mid-radius on this grid (`:36-37`), so

$$
\texttt{x1v}\cdot\texttt{dx1f} = \tfrac12(r_-+r_+)(r_+-r_-)
= \tfrac12\big(r_+^2-r_-^2\big) = \int_{r_-}^{r_+} r\,\mathrm{d}r ,
\tag{1.7}
$$

**the same radial measure as (1.6), exactly** — not to leading order. The
companion factors `dx3f_ang_face2_kj` (`:119`) and `dx2f_ang_face3_kj` (`:105`)
are the great-circle *angles* subtended by the face's transverse edge; they are
built from $\tan x_2$, $\tan x_3$ and the panel geometry and carry the
non-orthogonality $g_{23}=\cos\vartheta$ (`:83`,
`_set_face2_metric` at `:233-245`) and $\sqrt{g}\propto$ `sine_face2_kj`
(`:84`). Crucially **they do not depend on $r$.** So, with $\psi$ denoting the
transverse coordinate,

$$
\mathrm{d}A_2 = r\,W_2(\psi)\,\mathrm{d}r\,\mathrm{d}\psi,
\qquad \partial_r W_2 = 0 ,
\tag{1.8}
$$

where $W_2$ holds every angular, $\sqrt{g}$ and non-orthogonal factor.

### 1.4 The face-average operator and its moments

For any in-face coordinate $\xi$ with $x_1$-weight $w$, define

$$
\langle a\rangle_A \equiv \frac{\int a\,w\,\mathrm{d}x_1}{\int w\,\mathrm{d}x_1},
\qquad
x_1^{c} \equiv \frac{\int x_1 w\,\mathrm{d}x_1}{\int w\,\mathrm{d}x_1}
\ \ (\textbf{area centroid}),
\qquad
\zeta \equiv x_1-x_1^{c} .
\tag{1.9}
$$

Moments: $\mu_0=1$, and **by the choice of $x_1^c$**

$$
\mu_1 \equiv \langle\zeta\rangle_A = 0,
\qquad
\sigma_1^2 \equiv \mu_2 = \langle\zeta^2\rangle_A,
\qquad
\mu_3 = \langle\zeta^3\rangle_A .
\tag{1.10}
$$

$\mu_1=0$ is the only property of $x_1^c$ used below; everything else follows.

---

## 1A. The cell values are density-weighted (Favre) -- and this corrects §2A

**This section was added after a defect was found in §2 and §2A, and it changes
the covariance part of every row. It does not change the centroid part.**

§1.4 and §2 treat each factor of a face flux as an independent **volume average**
of a primitive. That is wrong for the per-mass primitives. snapy stores the
conserved densities and divides by the density to get them, at
`src/eos/ideal_moist_impl.h`:

```cpp
  // den -> mixr
  for (int n = 0; n < nmass; ++n) {
    PRIM(ICY + n) = CONS(ICY + n) / PRIM(IDN);      // :40
  }

  // mom -> vel
  PRIM(IVX) = CONS(IVX) / PRIM(IDN);                // :44
  PRIM(IVY) = CONS(IVY) / PRIM(IDN);                // :45
  PRIM(IVZ) = CONS(IVZ) / PRIM(IDN);                // :46
```

The inverse confirms the convention (`src/eos/ideal_moist.cpp:141-150`):
`cons[IDN] = (1 - sum q) * prim[IDN]`, `cons[ICY+n] = q * prim[IDN]`,
`cons[IVX..] = v * prim[IDN]`; and `:172-174` builds `prim[IDN]` as dry plus
species, so it is the **total** density.

So the stored per-mass quantities are **Favre (density-weighted) averages**:

$$
\bar u=\frac{\langle\rho u\rangle_V}{\langle\rho\rangle_V},\qquad
\bar q_n=\frac{\langle\rho q_n\rangle_V}{\langle\rho\rangle_V},\qquad
\bar h=\frac{\langle\rho h\rangle_V}{\langle\rho\rangle_V},
\tag{1A.1}
$$

the last because $\rho h=I+p$ is itself a volume-averaged conserved-like density
(`CONS(IPR)` is $E=I+\mathrm{ke}$, and the kinetic part is dropped by A3).
Therefore, **identically and not just to $O(h^2)$**,

$$
\boxed{\;\bar\rho\,\bar u=\langle\rho u\rangle_V,\qquad
\bar\rho\,\bar q_n=\langle\rho q_n\rangle_V,\qquad
\bar\rho\,\bar h=\langle\rho h\rangle_V\;}
\tag{1A.2}
$$

**Consequence: every covariance pair containing $\rho'$ must be dropped.** The
product $\bar\rho\bar u$ already *is* the average of $\rho u$; adding
$\sigma^2\rho'u'$ on top of it double-counts. What survives is the Favre
covariance between two per-mass quantities. Writing $\{\cdot\}$ for the
$\rho$-weighted face average,

$$
\langle\rho\,\phi\,\psi\rangle_A-\bar\rho\,\bar\phi\,\bar\psi
=\langle\rho\rangle_A\big(\{\phi\psi\}-\{\phi\}\{\psi\}\big)
=\rho\,\sigma_c^2\,\phi'\psi' + O(h^4),
\tag{1A.3}
$$

using that the second central moment of *any* smooth positive weight on an
interval of width $h$ is $h^2/12+O(h^4)$, so the $\rho$-weighted moment equals
$\sigma_c^2$ to the order kept.

The corrected rows as coded are collected in Zoey Hu's §4B; §4C records what they superseded and settles the energy row. §4A's row table is **superseded** for
the covariance column; its centroid column, its telescoping statements and its
rest-balance proof stand unchanged, because the centroid term $-\delta\,\partial_1F$
is a property of the *measure*, not of how the factors are averaged.

---

## 2. Every expansion, written out

### 2.1 One factor

Assume $a\in C^4$ across the face (§4, A1). Taylor about $x_1^c$:

$$
a(\zeta) = a_c + a'_c\zeta + \tfrac12 a''_c\zeta^2
+ \tfrac16 a'''_c\zeta^3 + O(\zeta^4),
$$

so, applying $\langle\cdot\rangle_A$ term by term and using $\mu_1=0$:

$$
\langle a\rangle_A
= a_c + a'_c\underbrace{\mu_1}_{=0} + \tfrac12 a''_c\,\sigma_1^2
+ \tfrac16 a'''_c\,\mu_3 + O(\mu_4).
\tag{2.1}
$$

### 2.2 Size of $\mu_3$ and $\mu_4$ — why they drop out

For the **Cartesian** weight (1.5) the measure is symmetric about the face
mid-point, so $\mu_3=0$ **exactly**.

For the **curved** weight $w=r$ (1.6)/(1.8), put $r=\bar r+s$ with
$\bar r=\tfrac12(r_-+r_+)$, $h=r_+-r_-$, $s\in[-h/2,h/2]$. Normalisation:

$$
\int_{-h/2}^{h/2}(\bar r+s)\,\mathrm{d}s = \bar r h .
$$

First moment about the mid-point:

$$
\langle s\rangle = \frac{1}{\bar r h}\int_{-h/2}^{h/2}s(\bar r+s)\,\mathrm{d}s
= \frac{1}{\bar r h}\Big(0 + \frac{h^3}{12}\Big)
= \frac{h^2}{12\,\bar r}\;\equiv\;\delta ,
\tag{2.2}
$$

so $x_1^c = \bar r+\delta$ and $\zeta=s-\delta$. Second moment about the
mid-point:

$$
\langle s^2\rangle = \frac{1}{\bar r h}\int_{-h/2}^{h/2}s^2(\bar r+s)\,\mathrm{d}s
= \frac{1}{\bar r h}\Big(\bar r\frac{h^3}{12}+0\Big) = \frac{h^2}{12},
\qquad
\langle s^3\rangle = \frac{1}{\bar r h}\int s^3(\bar r+s)\mathrm{d}s
= \frac{h^4}{80\,\bar r}.
$$

Hence

$$
\mu_3 = \langle(s-\delta)^3\rangle
= \langle s^3\rangle - 3\delta\langle s^2\rangle + 3\delta^2\langle s\rangle-\delta^3
= \frac{h^4}{80\bar r} - \frac{h^4}{48\bar r} + 2\delta^3
= -\frac{h^4}{120\,\bar r} + O\!\Big(\frac{h^6}{\bar r^3}\Big).
\tag{2.3}
$$

So $\mu_3 = O(h^4)$ and $\mu_4 = O(h^4)$: both enter (2.1) at the **same order as
the remainder we drop**, and neither contributes to the $O(h^2)$ term we are
after. This is the one place where the asymmetry of the curved measure could
have mattered, and it does not.

### 2.3 Two factors — the covariance identity

Apply (2.1) to the product $ab$, using
$(ab)''=a''b+2a'b'+ab''$ and
$(ab)'''=a'''b+3a''b'+3a'b''+ab'''$:

$$
\langle ab\rangle_A = a_cb_c
+ \tfrac12\sigma_1^2\big(a''b+2a'b'+ab''\big)_c
+ \tfrac16\mu_3\big(a'''b+3a''b'+3a'b''+ab'''\big)_c
+ O(h^4).
\tag{2.4}
$$

And multiplying the two single-factor expansions:

$$
\langle a\rangle_A\langle b\rangle_A
= \Big(a_c+\tfrac12\sigma_1^2a''+\tfrac16\mu_3a'''\Big)
  \Big(b_c+\tfrac12\sigma_1^2b''+\tfrac16\mu_3b'''\Big)
$$
$$
= a_cb_c
+ \tfrac12\sigma_1^2\big(a''b+ab''\big)_c
+ \tfrac16\mu_3\big(a'''b+ab'''\big)_c
+ \underbrace{\tfrac14\sigma_1^4\,a''b''}_{O(h^4)}
+ O(h^4).
\tag{2.5}
$$

Subtract (2.5) from (2.4). **The $a''b$ and $ab''$ terms cancel identically**,
and so do the $a'''b$ and $ab'''$ terms; what is left of the $\mu_3$ group is
$\tfrac12\mu_3(a''b'+a'b'')=O(h^4)$ by (2.3). Therefore

$$
\boxed{\;\langle ab\rangle_A - \langle a\rangle_A\langle b\rangle_A
= \sigma_1^2\,a'\,b' + O(h^4)\;}
\tag{2.6}
$$

This single identity **is** the whole generalisation. The metric enters in
exactly two ways and no others: through the value of $\sigma_1^2$, and through
the point $x_1^c$ at which $a'$ and $b'$ are evaluated.

### 2.4 The quotient $\bar m/\bar\rho$

$u_n=m/\rho$, and the code forms $\bar m/\bar\rho = \langle m\rangle/\langle\rho\rangle$,
not $\langle u_n\rangle$. Apply (2.6) with $a=m$, $b=1/\rho$:

$$
\langle u_n\rangle_A = \big\langle m\cdot\tfrac1\rho\big\rangle_A
= \langle m\rangle_A\Big\langle\tfrac1\rho\Big\rangle_A
+ \sigma_1^2\,m'\Big(\tfrac1\rho\Big)'
= \langle m\rangle_A\Big\langle\tfrac1\rho\Big\rangle_A
- \sigma_1^2\,\frac{m'\rho'}{\rho^2}.
\tag{2.7}
$$

Now the difference between "average of the reciprocal" and "reciprocal of the
average". With $(1/\rho)''=2\rho'^2/\rho^3-\rho''/\rho^2$, (2.1) gives

$$
\Big\langle\tfrac1\rho\Big\rangle_A
= \frac1{\rho_c} + \tfrac12\sigma_1^2\Big(\frac{2\rho'^2}{\rho^3}-\frac{\rho''}{\rho^2}\Big)
+ O(h^4),
$$

while

$$
\frac1{\langle\rho\rangle_A}
= \frac{1}{\rho_c+\tfrac12\sigma_1^2\rho''+O(h^4)}
= \frac1{\rho_c}\Big(1-\tfrac12\sigma_1^2\frac{\rho''}{\rho_c}\Big)+O(h^4)
= \frac1{\rho_c}-\tfrac12\sigma_1^2\frac{\rho''}{\rho^2}+O(h^4).
$$

Subtracting, **the $\rho''$ terms cancel** and only the squared first derivative
survives:

$$
\Big\langle\tfrac1\rho\Big\rangle_A - \frac1{\langle\rho\rangle_A}
= \sigma_1^2\,\frac{\rho'^2}{\rho^3} + O(h^4).
\tag{2.8}
$$

Insert (2.8) into (2.7). Because the whole expression is already $O(\sigma_1^2)$
we may replace $\langle m\rangle_A\to m$, $\langle\rho\rangle_A\to\rho$ inside it
(the correction is $O(h^4)$):

$$
\langle u_n\rangle_A - \frac{\bar m}{\bar\rho}
= m\,\sigma_1^2\frac{\rho'^2}{\rho^3} - \sigma_1^2\frac{m'\rho'}{\rho^2}
= -\sigma_1^2\,\frac{\rho'}{\rho^2}\Big(m'-\frac{m\rho'}{\rho}\Big).
$$

Since $u_n'=(m/\rho)'=m'/\rho-m\rho'/\rho^2$, i.e. $\rho\,u_n' = m'-m\rho'/\rho$,

$$
\boxed{\;\langle u_n\rangle_A - \frac{\bar m}{\bar\rho}
= -\,\sigma_1^2\,\frac{\rho'}{\rho}\,u_n' + O(h^4)\;}
\tag{2.9}
$$

### 2.5 Assembling $\Delta F$

Apply (2.6) to $p\,u_n$, then use $\bar p=\langle p\rangle_A$ and (2.9):

$$
\langle p\,u_n\rangle_A - \bar p\,\frac{\bar m}{\bar\rho}
= \Big[\langle p\rangle_A\langle u_n\rangle_A + \sigma_1^2 p'u_n'\Big]
  - \langle p\rangle_A\frac{\bar m}{\bar\rho}
= \langle p\rangle_A\Big[\langle u_n\rangle_A-\frac{\bar m}{\bar\rho}\Big]
  + \sigma_1^2 p'u_n'
$$
$$
= -\sigma_1^2\,p\,\frac{\rho'}{\rho}\,u_n' + \sigma_1^2\,p'\,u_n'
= \sigma_1^2\,u_n'\Big(p'-\frac{p\rho'}{\rho}\Big)
= \sigma_1^2\,p\,u_n'\Big(\frac{p'}{p}-\frac{\rho'}{\rho}\Big),
$$

that is

$$
\boxed{\;
\Delta F = \frac{\gamma}{\gamma-1}\,\sigma_1^2\;p\;
\big[\ln(p/\rho)\big]'\;u_n'
\;=\;\frac{\gamma}{\gamma-1}\,\sigma_1^2\;p\;(\ln T)'\;u_n'
\;}
\tag{2.10}
$$

the last step using $p=\rho R T$ with $R$ uniform across the face. This
reproduces #289's formula

> `Delta F = gamma/(gamma-1) (dz²/12) p (ln T)_z u_z`

with $\sigma_1^2$ in place of $\mathrm{d}z^2/12$. Note what is **absent**: no
term $\propto u_n p'$ and none $\propto u_n\rho'$ survives. They cancel in two
places — the $p''$ and $\rho''$ second-derivative groups cancel between
$\langle ab\rangle$ and $\langle a\rangle\langle b\rangle$ in (2.4)–(2.5), and the
$\rho''$ group cancels again between $\langle 1/\rho\rangle$ and
$1/\langle\rho\rangle$ in (2.8). This is #289's
"the u rho_z and u p_z cross terms cancel identically", shown rather than
asserted.

### 2.6 $\sigma_1^2$: Cartesian

With $w=1$ the centroid is the mid-point and

$$
\sigma_1^2 = \frac{1}{\Delta x_1}\int_{-\Delta x_1/2}^{+\Delta x_1/2}\zeta^2\,\mathrm{d}\zeta
= \frac{1}{\Delta x_1}\cdot\frac{2}{3}\Big(\frac{\Delta x_1}{2}\Big)^3
= \frac{\Delta x_1^{\,2}}{12}.
\tag{2.11}
$$

This is where #289's $\mathrm{d}z^2/12$ comes from: it is the second central
moment of a *uniform* face measure, nothing more.

### 2.7 $\sigma_1^2$: the curved horizontal faces — exactly

With $w(r)=r$ (true for spherical-polar *and* cubed-sphere, (1.6)–(1.8)), all
three moments are elementary and **exact**:

$$
r_+^3-r_-^3 = h\big(3\bar r^2+\tfrac{h^2}{4}\big),\qquad
r_+^2-r_-^2 = 2\bar r h,\qquad
r_+^4-r_-^4 = 2\bar r h\big(2\bar r^2+\tfrac{h^2}{2}\big),
$$

(from $r_\pm^2=\bar r^2\pm\bar r h+h^2/4$ and $r_+r_-=\bar r^2-h^2/4$), hence

$$
x_1^c = r_c = \frac{\int r\cdot r\,\mathrm{d}r}{\int r\,\mathrm{d}r}
= \frac{\tfrac13(r_+^3-r_-^3)}{\tfrac12(r_+^2-r_-^2)}
= \frac{2}{3}\,\frac{r_+^3-r_-^3}{r_+^2-r_-^2}
= \frac{3\bar r^2+h^2/4}{3\bar r}
= \bar r + \frac{h^2}{12\bar r},
\tag{2.12}
$$

$$
\langle r^2\rangle_A = \frac{\tfrac14(r_+^4-r_-^4)}{\tfrac12(r_+^2-r_-^2)}
= \frac{r_+^2+r_-^2}{2} = \bar r^2+\frac{h^2}{4},
\tag{2.13}
$$

$$
\sigma_1^2 = \langle r^2\rangle_A - r_c^2
= \Big(\bar r^2+\frac{h^2}{4}\Big)
- \frac{\big(3\bar r^2+h^2/4\big)^2}{9\bar r^2}
= \Big(\bar r^2+\frac{h^2}{4}\Big)
- \Big(\bar r^2+\frac{h^2}{6}+\frac{h^4}{144\bar r^2}\Big),
$$

$$
\boxed{\;
\sigma_1^2 = \frac{h^2}{12} - \frac{h^4}{144\,\bar r^2}
= \frac{h^2}{12}\left(1-\frac{h^2}{12\,\bar r^2}\right),
\qquad h=\Delta r,\ \ \bar r=\tfrac12(r_-+r_+)
\;}
\tag{2.14}
$$

(2.14) is an identity, not a truncation, and the right-hand form is free of the
cancellation that the middle form would suffer at $h\ll\bar r$.

**How much does the metric change the coefficient? Almost nothing, and here is
why.** The relative shift is $h^2/(12\bar r^2) = (\Delta r/\bar r)^2/12$: at
$\Delta r/r=0.1$ it is $8\times10^{-4}$ of $\Delta F$, and $\Delta F$ is itself
$O(h^2)$, so the metric's contribution to the flux is $O(h^4)$ — the same order
as everything else dropped. This is the precise content of #289's
"adds O(dr²) terms": they are $O((\Delta r/r)^2)$ *relative* to the term. The
implementation nevertheless uses the exact (2.14), because it costs nothing and
makes the statement exact rather than asymptotic.

The *evaluation point* also moves, from $\bar r$ to $r_c=\bar r+h^2/(12\bar r)$,
and neither equals snapy's `x1v`: on the spherical grid `x1v` is the **volume**
centroid (`src/coord/spherical_polar.cpp:17-21`, `:61`)

$$
r_v = \frac{\int r\cdot r^2\mathrm{d}r}{\int r^2\mathrm{d}r}
= \frac34\frac{r_+^4-r_-^4}{r_+^3-r_-^3} = \bar r+\frac{h^2}{6\bar r}+O(h^4),
\tag{2.15}
$$

i.e. twice as far out as $r_c$. Both differ from $r_c$ by $O(h^2/r)$, and since
they only ever multiply a quantity that is already $O(h^2)$, using `x1v` to
locate the derivatives in (2.10) perturbs $\Delta F$ by $O(h^4)$. The
implementation uses `x1v`, which equals $r_v$ on the spherical-polar grid but is
the arithmetic mid-radius on a gnomonic panel (§2A.1); the $O(h^4)$ conclusion is
unaffected either way, since $\bar r$ and $r_v$ both sit within $O(h^2/r)$ of
$r_c$.

### 2.8 Why $\sqrt{g}$ and $g_{23}\neq0$ do not appear

Take the cubed-sphere measure (1.8), $\mathrm{d}A_2=r\,W_2(\psi)\,\mathrm{d}r\,\mathrm{d}\psi$
with $\partial_r W_2=0$. The two-dimensional face average is

$$
\langle a\rangle_A
= \frac{\displaystyle\int\!\!\int a(r,\psi)\,r\,W_2(\psi)\,\mathrm{d}r\,\mathrm{d}\psi}
       {\displaystyle\int\!\!\int r\,W_2(\psi)\,\mathrm{d}r\,\mathrm{d}\psi}.
\tag{2.16}
$$

Once the $\psi$-covariance is dropped (§2.9) every factor is a function of $r$
alone, and then

$$
\langle a\rangle_A
= \frac{\Big(\int W_2\,\mathrm{d}\psi\Big)\int a(r)\,r\,\mathrm{d}r}
       {\Big(\int W_2\,\mathrm{d}\psi\Big)\int r\,\mathrm{d}r}
= \frac{\int a(r)\,r\,\mathrm{d}r}{\int r\,\mathrm{d}r},
$$

so **$W_2$ cancels identically between numerator and denominator**. Everything
non-orthogonal and every $\sqrt{g}$ lives in $W_2$. Hence:

> On the cubed sphere the non-orthogonal metric $g_{23}=\cos\vartheta$ and the
> $\sqrt{g}$ factors do not change $\sigma_1^2$ at all. The only metric effect
> on the coefficient is the radial weight $w(r)=r$, which is identical to
> spherical-polar, so (2.14) serves both grids.

What the cubed sphere *does* change is the **frame**: the normal velocity in
(2.10) must be the face-local orthonormal component that the energy flux
carries. `prim2local2_`/`prim2local3_`
(`src/coord/gnomonic_equiangle.cpp:270-308`) produce it, and
`flux2global2_`/`flux2global3_` (`:325-380`) rotate **only** the `IVY`/`IVZ`
momentum rows — `IDN` and `IPR` are never touched — so the energy flux is a
scalar under that rotation and $\Delta F$ may be added to it directly.

### 2.9 The transverse in-face covariance, and why it is dropped

An $x_2$ face is spanned by $x_1$ **and** $x_3$. For a separable measure
$w(x_1)W(x_3)$ with both first moments removed,
$\langle\zeta_1\zeta_3\rangle_A=\langle\zeta_1\rangle\langle\zeta_3\rangle=0$, so the
two directions do not mix and the corrections simply add:

$$
\Delta F = \frac{\gamma}{\gamma-1}\Big[
\sigma_1^2\,p\,(\ln T)_{,1}\,u_{n,1}
+ \sigma_\perp^2\,p\,(\ln T)_{,\perp}\,u_{n,\perp}\Big].
\tag{2.17}
$$

Only the $x_1$ piece is kept. The reason is the $\epsilon$ scaling: the
background is stratified in $x_1$ only, so $(\ln T_0)_{,1}=O(1)$ — set by the
lapse rate, **not** by $\epsilon=\nabla-\nabla_{\rm ad}$ — while
$(\ln T)_{,\perp}$ and $u_{n,\perp}$ are both perturbation quantities. The
$x_1$ piece is therefore **linear** in the perturbation amplitude and the
transverse piece is **quadratic**; for the linear onset problem #289 is about,
the transverse piece does not contribute at all. Dropping it is a deliberate
truncation (§4, A5), not an identity.

### 2.10 Why the term behaves like a spurious stratification

With $u_{n,1}$ carrying the mode's vertical structure and $(\ln T_0)_{,1}$ the
background lapse rate, the divergence of (2.10) acts on the linear problem
exactly like an added stable stratification. #289 reports
$\epsilon_{\rm spur} = -(\beta m^2/12)\,\mathrm{d}z^2 \approx -0.24/n_z^2$ for the
first mode, with the measured one-step tendency at $\epsilon=0$, $n_z=64$ giving
$\epsilon_{\rm spur}n_z^2=-0.240$ against $-0.241$ from the formula, and hence a
growth-rate error that collapses on $\epsilon n_z^2$. On a curved grid the same
argument goes through with $\mathrm{d}z^2/12\to\sigma_1^2$ from (2.14), i.e. with
a correction of relative size $(\Delta r/\bar r)^2/12$ to $\epsilon_{\rm spur}$.

---

## 2A. CORRECTION: a second $O(\Delta r^2)$ term on curved faces

**This section corrects §2.5–§2.8 of an earlier revision of this file, which
claimed that the covariance term (2.10) is the complete $O(h^2)$ correction on
a curved grid and that the metric enters only at $O(h^4)$. That claim was
wrong.** The covariance algebra of §2 is correct as far as it goes, but it
expands *both* sides about the face area centroid $r_c$, and a finite-volume
cell does not store the value at $r_c$.

### 2A.1 The two measures, and the two centroids

Two different measures are in play on the same cell:

$$
\text{CELL measure (what a stored value averages):}\quad
\mathrm{d}V \;\propto\; r^2\,\mathrm{d}r \times (\text{angular}),
$$
$$
\text{FACE measure (what the flux needs, eq. 1.2):}\quad
\mathrm{d}A_{2,3} \;\propto\; r\,\mathrm{d}r \times (\text{angular}).
$$

One power of $r$ apart — see `cell_volume()` versus `face_area2()`/`face_area3()`
in §1.3. Their $x_1$ centroids therefore differ. With $h=\Delta r$,
$\bar r=\tfrac12(r_-+r_+)$:

$$
r_c = \frac{2}{3}\frac{r_+^3-r_-^3}{r_+^2-r_-^2} = \frac{12\bar r^2+h^2}{12\bar r}
    = \bar r + \frac{h^2}{12\bar r}+O\!\Big(\frac{h^4}{\bar r^3}\Big)
\quad\text{(area, weight } r),
\tag{2A.1}
$$

$$
r_v = \frac{3}{4}\frac{r_+^4-r_-^4}{r_+^3-r_-^3}
    = \frac{3\bar r\,(4\bar r^2+h^2)}{12\bar r^2+h^2}
    = \bar r + \frac{h^2}{6\bar r}+O\!\Big(\frac{h^4}{\bar r^3}\Big)
\quad\text{(volume, weight } r^2),
\tag{2A.2}
$$

and doing the subtraction algebraically, so that no cancellation of two
$O(\bar r)$ numbers is needed,

$$
\boxed{\;
\delta \;\equiv\; r_v - r_c
\;=\; \frac{h^2\,\big(12\bar r^2 - h^2\big)}{12\,\bar r\,\big(12\bar r^2+h^2\big)}
\;=\; \frac{h^2}{12\,\bar r} + O\!\Big(\frac{h^4}{\bar r^3}\Big)
\;}
\tag{2A.3}
$$

$r_v$ is exactly snapy's `x1v` on the spherical-polar grid
(`src/coord/spherical_polar.cpp:17-21`, `:61`), i.e. the cell centre the code
already uses. **That identification holds for spherical-polar only.** On a
gnomonic cubed-sphere panel `x1v` is the *arithmetic* mid-radius $\bar r$, not
the volume-weighted $r_v$ (`src/coord/gnomonic_equiangle.cpp:36-37`:
`x1v = 0.5 * (x1f[0..nc1] + x1f[1..nc1+1])`), so there $r_v - \texttt{x1v} =
h^2/(6\bar r)+O(h^4)$ rather than $0$. This does **not** change $\delta$ or
$\sigma_c^2$: both are defined by the two *measures* (1.1), (2A.3), not by
`x1v`, so every formula in this section stands on both curved grids, and the
rest-balance result of §4A.3 does not depend on $\delta$'s value at all. What it
does mean is that a gnomonic initial condition sampled at `x1v` is a point value
at $\bar r$, not a cell average, so the cell-average reading of the stored state
(A4) is the one under which $\delta$ is the right correction there. In
**Cartesian** both measures are the same uniform weight, so
$r_v = r_c = $ the cell mid-point and $\delta \equiv 0$ **exactly**: nothing in
this section changes the Cartesian result.

### 2A.2 Redoing the expansion from the cell value

Let $\bar q$ be the stored cell value, $\bar q=\langle q\rangle_V$, and expand
it about $r_c$, writing $\sigma_v^2$ for the volume-weighted second central
moment and using $\delta=O(h^2)$ so that $\delta^2=O(h^4)$:

$$
\bar q = q(r_v) + \tfrac12\sigma_v^2 q'' + O(h^4)
      = q_c + \delta\,q'_c + \tfrac12\sigma_v^2 q''_c + O(h^4).
\tag{2A.4}
$$

The face average is (2.1) with the area moments:
$\langle q\rangle_A = q_c + \tfrac12\sigma_c^2 q''_c+O(h^4)$. Now
$\sigma_c^2 = h^2/12 - h^4/(144\bar r^2)$ from (2.14) and, by the same
elementary integrals with weight $r^2$,
$\sigma_v^2 = h^2/12 - h^4/(45\bar r^2)$, so

$$
\sigma_c^2-\sigma_v^2 = h^4\Big(\frac{1}{45}-\frac{1}{144}\Big)\frac{1}{\bar r^2}
= O(h^4) \;\Longrightarrow\;
\sigma_c^2=\sigma_v^2=\frac{h^2}{12}\ \text{to the order kept.}
\tag{2A.5}
$$

Take the quotient first. With $m=\rho u_n$ and
$u_n''=m''/\rho-2m'\rho'/\rho^2+2m\rho'^2/\rho^3-m\rho''/\rho^2$, so that
$m''/\rho-m\rho''/\rho^2=u_n''+2(\rho'/\rho)u_n'$:

$$
\frac{\bar m}{\bar\rho}
=\frac{m_c+\delta m'+\tfrac12\sigma_v^2m''}{\rho_c+\delta\rho'+\tfrac12\sigma_v^2\rho''}
= u_{n,c} + \delta\,u_n' + \tfrac12\sigma_v^2\Big(u_n''+\frac{2\rho'}{\rho}u_n'\Big)+O(h^4),
\tag{2A.6}
$$

every $\delta^2$ and $\delta\sigma^2$ cross term being $O(h^4)$. Multiply by
$\bar p$ from (2A.4):

$$
\bar p\,\frac{\bar m}{\bar\rho}
= p_cu_{n,c} + \delta\big(p'u_n+pu_n'\big)_c
+ \tfrac12\sigma_v^2\Big(p''u_n+pu_n''+\frac{2p\rho'}{\rho}u_n'\Big)_c+O(h^4),
\tag{2A.7}
$$

while the exact face average, from (2.4), is

$$
\langle p\,u_n\rangle_A = p_cu_{n,c}+\sigma_c^2p'u_n'
+\tfrac12\sigma_c^2\big(p''u_n+pu_n''\big)_c+O(h^4).
\tag{2A.8}
$$

Subtract. By (2A.5) the $\tfrac12(\sigma_c^2-\sigma_v^2)(p''u_n+pu_n'')$ group is
$O(h^4)$ and drops, leaving

$$
\langle pu_n\rangle_A-\bar p\frac{\bar m}{\bar\rho}
= \sigma_c^2\,p'u_n' - \sigma_v^2\,\frac{p\rho'}{\rho}u_n' - \delta\,\partial_r(pu_n)
= \sigma_c^2\,p\,u_n'\big[\ln(p/\rho)\big]' - \delta\,\partial_r\!\big(pu_n\big),
$$

using $\sigma_v^2=\sigma_c^2$ to the order kept. Multiplying by
$\kappa=\gamma/(\gamma-1)$:

$$
\boxed{\;
\Delta F = \kappa\Big[\;\sigma_c^2\;p\;\big[\ln(p/\rho)\big]_{,1}\;(u_n)_{,1}
\;-\;\delta\;\partial_1\!\big(p\,u_n\big)\;\Big]
\;}
\tag{2A.9}
$$

with $\sigma_c^2$ from (2.14) and $\delta$ from (2A.3). The first term is the
#289 covariance; the **second is new**. In Cartesian $\delta=0$ and (2A.9)
reduces to (2.10) exactly.

### 2A.3 Size: it is not a small correction

Relative to the covariance term, the centroid term is

$$
\frac{|\delta\,\partial_r(pu_n)|}{|\sigma_c^2\,p\,[\ln(p/\rho)]'u_n'|}
\;\sim\; \frac{h^2/(12\bar r)}{h^2/12}\cdot
\frac{|\partial_r(pu_n)|}{|p\,[\ln(p/\rho)]'\,u_n'|}
\;\sim\; \frac{L}{\bar r},
\tag{2A.10}
$$

with $L$ the local gradient scale. So it is **$O(L/R)$, not $O(\Delta r/R)$ and
not $O(1)$ in $\Delta r$** — it is the same order in $\Delta r$ as the term
#289 identified, and only small when the gradient scale is small compared with
the radius. Both terms are $O(h^2)$; neither may be dropped.

### 2A.4 Numerical check of (2A.9)

`derivation/verify_exact_curved.py`, 90-point Gauss–Legendre, generic smooth
profiles with non-constant $\rho$, comparing
$\kappa\langle pu\rangle_A$ (area average, weight $r$) against
$\kappa\bar p\,\bar m/\bar\rho$ (every bar a volume average, weight $r^2$):

| $\bar r$ | $h$ | relative residual, covariance only | relative residual, (2A.9) | ratio per halving |
|---|---|---|---|---|
| 1.4 | 0.1 | 2.765e-01 | 4.507e-02 | — |
| 1.4 | 0.05 | 3.095e-01 | 1.149e-02 | 3.92 |
| 1.4 | 0.025 | 3.179e-01 | 2.887e-03 | 3.98 |
| 1.4 | 0.0125 | 3.201e-01 | 7.226e-04 | 4.00 |
| 6 | 0.1 | 2.252e-01 | 3.667e-03 | 4.04 |
| 6 | 0.0125 | 2.208e-01 | 5.713e-05 | 4.00 |
| 50 | 0.1 | 5.674e-01 | 1.555e-03 | 3.96 |
| 50 | 0.0125 | 5.667e-01 | 2.813e-05 | 4.00 |

The covariance term alone leaves a relative residual that **does not converge**
(16–57 %); with the centroid term the residual falls by exactly $4\times$ per
halving, i.e. $O(h^4)$ absolute, so (2A.9) is the complete $O(h^2)$ correction.
Measured $|\text{centroid}|/|\text{covariance}|$: 0.25 at $\bar r=1.4$, 0.18 at
$\bar r=6$, 1.31 at $\bar r=50$ — confirming (2A.10) and confirming that the
ratio is set by $L/\bar r$, not by $h$. In a clean test with $\rho$ constant and
$p,u$ linear in $r$ (`derivation/verify_centroid_term.py`, where the covariance
is exact with no remainder) the covariance term alone has the **wrong sign** at
$\bar r=20$; with the centroid term the residual is $1.5\times10^{-5}$ and
fourth order. $\delta$ from (2A.3) matches the quadrature difference of the two
centroids to 5 significant digits.

### 2A.5 Properties of the new term

* **Telescoping — unchanged.** It is added to the face flux, so it telescopes
  exactly in the $x_2$/$x_3$ divergence (1.1) and in the closed-wall $E+PE$ sum,
  like the covariance term. Energy row only; mass and momentum untouched.
* **Rest state — unchanged.** $\partial_1(pu_n)$ carries $u_n$, which is exactly
  zero at rest, so $\Delta F\equiv0$ bitwise and a balanced column keeps its
  exact zero tendency.
* **Isothermal state — changed.** The covariance term vanishes when
  $[\ln(p/\rho)]'=0$; the centroid term does not. The "zero for an isothermal
  state" property of #289 therefore holds in **Cartesian only**.
* **The $x_3$ face** carries the same $\delta$: §3's table applies unchanged,
  since `face_area3` has the same radial measure as `face_area2`.
* **Relation to H3 (§7).** H3 flagged this centroid mismatch and judged it out
  of scope. For the *energy* flux that judgement was wrong: it is the same
  order, at the same face, as the term #289 is about, and it is included here.
  What remains genuinely separate is the same mismatch in the **mass and
  momentum** fluxes, which this change does not touch.

---

## 3. The $x_3$ direction

Nothing in §2 used which horizontal direction the face normal points along; it
used (i) the face measure's $x_1$ weight and (ii) the fact that $x_1$ is the
stratified direction. Both hold for $x_3$ faces too. Running the same derivation
for a face of constant $x_3$, spanned by $x_1$ and $x_2$:

| face normal | in-face coordinates | covariance kept along | $x_1$ weight $w$ | $\sigma_1^2$ |
|---|---|---|---|---|
| $x_1$ | $x_2,x_3$ | none — both perturbation-quadratic | — | — |
| $x_2$ | $x_1,x_3$ | $x_1$ | Cart. $1$; sph./CS $r$ | (2.11) / (2.14) |
| $x_3$ | $x_1,x_2$ | $x_1$ | Cart. $1$; sph./CS $r$ | (2.11) / (2.14) |

* **Cartesian $x_3$ face**: $\mathrm{d}A_3=\mathrm{d}x_1\mathrm{d}x_2$
  (`coordinate.cpp:433-435`), uniform $x_1$ weight,
  $\sigma_1^2=\Delta x_1^2/12$ by (2.11).
* **Spherical-polar $x_3$ face**: $\mathrm{d}A_3=r\,\mathrm{d}r\,\mathrm{d}\theta$
  (`spherical_polar.cpp:193-198`), $w=r$, $\sigma_1^2$ by (2.14) — *the same
  coefficient as the $x_2$ face*.
* **Cubed-sphere $x_3$ face**: `face_area3 = (x1v*dx1f) * dx2f_ang_face3_kj`
  (`gnomonic_equiangle.cpp:201-203`), again $\int r\,\mathrm{d}r$ times an
  $r$-independent angular factor by (1.7), $w=r$, $\sigma_1^2$ by (2.14).

So the $x_3$ term is (2.10) verbatim, with $u_n$ the $x_3$-face normal velocity
(`IVZ` after `prim2local3_`; `lmars_impl.h:20` with `dim == 1`,
`lmars.cpp:56-57`) and the **same** $\sigma_1^2$. The $x_1$ flux receives no
correction from this mechanism: its in-face coordinates are $x_2$ and $x_3$ and
both of those covariances are perturbation-quadratic.

### 3.1 Conservation: does it telescope?

Yes, **by construction rather than by cancellation**. $\Delta F$ is added to the
face flux $F^{(2)}$/$F^{(3)}$ *before* the divergence (1.1), so the sums
telescope exactly as the flux they correct does. Summing $V\,\mathrm{d}\bar E/\mathrm{d}t$
over an $x_2$ column leaves only the two end faces, therefore

* periodic $x_2$/$x_3$: the two ends are the same face, the contribution is zero
  to round-off;
* impermeable wall: $u_n\equiv0$ there, so $\Delta F\equiv0$ at that face and
  there is nothing to cancel;
* mass and momentum are untouched — $\Delta F$ enters the `IPR` row only.

The one requirement is that the two cells sharing a face compute the **same**
$\Delta F$. It is built from the face states and from $x_1$ differences of them,
which are single-valued at a shared $x_2$/$x_3$ face because those ghosts are
exchanged before reconstruction; and the switch is read once per process (§5), so
two ranks cannot disagree about whether a shared face carries the term.

### 3.2 The discrete balanced state

$\Delta F$ carries $u_n'$ as a factor, evaluated as a **difference of face-state
normal velocities**. In a state at rest every one of those is exactly zero, so
the difference is exactly zero in floating point and $\Delta F\equiv0$ bitwise.
The tendency of a discretely balanced hydrostatic state is therefore unchanged
bit for bit, at any stratification, and the term can neither repair nor spoil
well-balancing. A second exact zero comes from the other factor: an isothermal
state has $[\ln(p/\rho)]'=0$, so $\Delta F\equiv0$ there even in motion. Both are
what #289 means by "It is zero at rest and zero for an isothermal state", and
both are checked (§6).

---

## 4. Assumptions, and the order retained

* **A1 — Smoothness.** $p,\rho,m\in C^4$ across the face, and $\rho>0$. Used for
  (2.1) and for bounding $\mu_3,\mu_4$. Across a shock the expansion is void;
  the term is then a bounded $O(h^2)$ perturbation of a flux that the limiter
  already dominates, but no claim is made for it there.
* **A2 — Ideal gas for the $\ln T$ step.** The derivation produces
  $[\ln(p/\rho)]'$; writing it as $(\ln T)'$ needs $p=\rho RT$ with $R$ uniform
  across the face. The implementation evaluates $\ln(p/\rho)$, so it does not
  rely on this; only the *name* $\ln T$ does.
* **A3 — Enthalpy flux only.** $f_E^{(2)}=(E+p)u_n$ also has the kinetic part
  $\tfrac12\rho|\mathbf{u}|^2u_n$, whose covariance is smaller by $O(M^2)$.
  Dropped. #289's regime is $M\sim10^{-4}\!-\!10^{-3}$, so this is a relative
  $10^{-8}\!-\!10^{-6}$.
* **A4 — Reconstructed face state read as the face average.** The reconstructed
  $w_{L},w_{R}$ are taken to represent $\langle\cdot\rangle_A$ over that face
  with the face's **area** measure. In Cartesian the transverse weight is uniform
  and the reading is unambiguous. On a curved grid it is a choice at exactly the
  order in question, and §7 (H3) states what is left over under the other
  reading.
* **A5 — Transverse covariance dropped.** §2.9. Exact only to leading order in
  perturbation amplitude.
* **A6 — Uniform vs non-uniform $x_1$ spacing.** $\sigma_1^2$ is computed
  **per cell** from that cell's own $(r_-,r_+)$, so non-uniform $x_1$ is handled
  exactly by (2.11)/(2.14); no assumption of uniform $\Delta x_1$ enters the
  coefficient. The *derivatives* in (2.10) are taken by a centred difference on
  `x1v`, which is $O(h^2)$-accurate on a uniform grid and $O(h)$ on a stretched
  one. Since $\Delta F$ is already $O(h^2)$, that leaves the flux error at
  $O(h^3)$ in the worst case — still a strict improvement on the $O(h^2)$ error
  being removed.
* **Order retained.** Exactly the $O(h^2)$ covariance. Dropped, all $O(h^4)$ or
  smaller: the $\tfrac14\sigma_1^4a''b''$ term in (2.5); the $\mu_3$ and $\mu_4$
  groups via (2.3); the $O(h^4)$ residue of (2.14); the $O(h^2/r)$ offset between
  $r_c$ and `x1v` multiplied into an $O(h^2)$ term; and the replacement of
  $\langle m\rangle,\langle\rho\rangle$ by point values inside (2.9).

---

## 5. The final discrete form, exactly as coded

Implemented in `HydroImpl::_flux_covariance` (`src/hydro/hydro_forward.cpp`),
called once per horizontal direction from `HydroImpl::forward`. In the code's own
names, for an $x_2$ face at index $(k,j,i)$:

```cpp
auto wbar = 0.5 * (wl + wr);                    // the face state
auto p    = wbar[IPR];
auto lnt  = (p / wbar[IDN]).log();              // ln(p/rho) = ln(R T)
auto enth = peos->compute("W->I", {wbar}) + p;  // I + p = rho*h = gamma/(gamma-1) p
if (dim == 2) pcoord->prim2local2_(wbar); else pcoord->prim2local3_(wbar);
auto un   = wbar[dim == 2 ? IVY : IVZ];         // face-normal velocity, local frame

auto x1v  = pcoord->x1v.to(p.device(), p.scalar_type());
auto s2   = pcoord->face_moment2_x1().to(p.device(), p.scalar_type())
                   .unsqueeze(0).unsqueeze(1);  // sigma_c^2, one value per x1 cell
auto ds   = pcoord->face_centroid_shift_x1().to(p.device(), p.scalar_type())
                   .unsqueeze(0).unsqueeze(1);  // delta = r_v - r_c, per x1 cell
auto hflx = enth * un;                          // kappa p u_n, the flux averaged

auto dx1   = x1v.narrow(0, 2, n1 - 2) - x1v.narrow(0, 0, n1 - 2);
auto dlnt  = (lnt.narrow(-1, 2, n1 - 2)  - lnt.narrow(-1, 0, n1 - 2))  / dx1;
auto dun   = (un.narrow(-1, 2, n1 - 2)   - un.narrow(-1, 0, n1 - 2))   / dx1;
auto dhflx = (hflx.narrow(-1, 2, n1 - 2) - hflx.narrow(-1, 0, n1 - 2)) / dx1;

auto dflx = torch::zeros_like(p);
dflx.narrow(-1, 1, n1 - 2) =
      enth.narrow(-1, 1, n1 - 2) * s2.narrow(-1, 1, n1 - 2) * dlnt * dun
    - ds.narrow(-1, 1, n1 - 2) * dhflx;
```

and at the call sites, `_flux2[IPR] += dcov` / `_flux3[IPR] += dcov` immediately
after `priemann->forward(...)`. In symbols, for interior $x_1$ index $i$:

$$
\Delta F_{kji} = \big(I+p\big)_{kji}\;\sigma^2_{c,i}\;
\frac{\ln(p/\rho)_{k j, i+1}-\ln(p/\rho)_{k j, i-1}}
     {\texttt{x1v}_{i+1}-\texttt{x1v}_{i-1}}\;
\frac{(u_n)_{k j, i+1}-(u_n)_{k j, i-1}}
     {\texttt{x1v}_{i+1}-\texttt{x1v}_{i-1}}
\;-\;\delta_i\;
\frac{\big[(I+p)u_n\big]_{k j, i+1}-\big[(I+p)u_n\big]_{k j, i-1}}
     {\texttt{x1v}_{i+1}-\texttt{x1v}_{i-1}} .
\tag{5.1}
$$

Correspondence with (2.10): $(I+p)=\rho h=\frac{\gamma}{\gamma-1}p$ exactly for an
ideal gas, so no $\gamma$ has to be introduced and the form carries over to any
EOS that supplies an internal energy (see §7, H4);
$\sigma^2_{c,i}=$ `face_moment2_x1()`, which is `dx1f.square()/12.` in
`CoordinateImpl` (eq. 2.11) and `radial_face_moment2_(x1f, nc1)` in
`SphericalPolarImpl` and `GnomonicEquiangleImpl` (eq. 2.14); and
$\delta_i=$ `face_centroid_shift_x1()`, which is **identically zero** in
`CoordinateImpl` (Cartesian: the two measures coincide) and
`radial_face_centroid_shift_(x1f, nc1)` on both curved grids (eq. 2A.3). Setting
$\delta\equiv0$ recovers the Cartesian-only form, so a reader can check the two
limits against each other by reading one function.

Only Cartesian is covered by the base value. `CylindricalImpl::reset()` is empty
at `117e449` and `src/coord/cylindrical.cpp_` is not compiled, so cylindrical is
a stub there; it inherits the uniform coefficient and would need its own
override, derived by §2.7 from whichever power of $r$ its own `face_area2` and
`face_area3` turn out to carry, before the term should be trusted on it.

**Why `wbar` is projected on a private copy.** `LmarsSolverImpl::forward` calls
`prim2local2_(wl)`/`(wr)` in place (`src/riemann/lmars.cpp:60-61`), so reading
the solver's inputs *after* the call would also give the local-frame normal
velocity — but `plume_roe` does not project at all, so that side effect
is not a contract. `_flux_covariance` therefore runs **before** the Riemann call
on `0.5*(wl+wr)` and projects that copy itself; the result is added to the flux
after the call. `_set_face2_metric()` is idempotent, so the extra call does not
disturb the solver.

### 5.1 The environment switch

```cpp
bool HydroImpl::flux_covariance() {              // src/hydro/hydro.cpp
  static const bool on = [] {
    auto v = get_env("SNAP_FLUX_COVARIANCE", "1");
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return !(v.empty() || v == "0" || v == "false" || v == "off" || v == "no");
  }();
  return on;
}
```

* **Name:** `SNAP_FLUX_COVARIANCE`. **Default: ON** (it was OFF when this note
  was written). Any of `0`, `false`, `off`, `no`, or empty (case-insensitive)
  leaves the term out, and that path is bit-identical to `117e449`.
* Read **once** into a function-local `static`, for the reason in §3.1: every
  block of every rank must make the same choice or a shared face stops being
  single-valued. `get_env` is the tree's existing helper
  (`src/layout/layout.hpp:38-41`).

### 5.2 Stencil and ghost-cell requirements

* **Stencil:** $\{i-1,\,i,\,i+1\}$ along $x_1$, on the face states and on `x1v`.
  Three points, one direction, no transverse stencil.
* **Ghosts:** needs $n_{\rm ghost}\ge1$ on $x_1$. Interior cells run
  $i\in[\texttt{il},\texttt{iu}]=[n_{\rm ghost},\,n_{\rm ghost}+n_{x_1}-1]$, so
  $i\pm1$ is always in range. `dflx` is left zero at $i=0$ and $i=n_{c_1}-1$,
  which are $x_1$ ghosts and are never read by the interior update
  (`hydro_forward.cpp:427-429`).
* **Faces covered:** $j\in[\texttt{jl},\texttt{ju}{+}1]$ and
  $k\in[\texttt{kl},\texttt{ku}{+}1]$ — exactly the faces `divergence` consumes.
  Outside that range `_apply_inplace` replicates the L/R states
  (`src/recon/reconstruct.cpp:59-64`), and those faces are not used.
* **Guards:** returns undefined (no term) when $n_{c_1}<3$ — an unresolved $x_1$
  axis has no vertical gradient to correct — and for the `shallow-water` EOS,
  which carries no internal-energy row.
* **Device:** pure device-agnostic `torch` ops; `x1v` and `face_moment2_x1()` are
  explicitly cast to the field's device and dtype, so the CPU and CUDA paths run
  the same arithmetic on the same values.
* **Interaction with the other stages:** the positivity limiter rescales only the
  species rows and carries their enthalpy, so it never rescales $\Delta F$; and
  the vertical implicit solve does not form the $x_2$/$x_3$ fluxes, so the term
  applies unchanged under any `implicit-scheme` (#289 item 4).

---

## 6. What each test checks, and what a wrong term would look like

**(T1) Switch OFF reproduces the base commit, on a short run.**
Checks that the whole change is inert by default. The diff is purely additive and
gated, so the expectation is a **bitwise** identical state. A non-bitwise result
would localise the leak: either the gate itself, or the two new calls the term
makes that touch shared state — `prim2local2_`, which `set_`s the
`g22`/`g23`/`g33` metric buffers through `_set_face2_metric`
(`gnomonic_equiangle.cpp:233-245`), and `peos->compute("W->I", ...)`. If OFF were
not bitwise, this term could not be reviewed as opt-in at all.

**(T2) $\epsilon=0$ tendency on a balanced spherical column, OFF vs ON, at two
resolutions.** The background is *exactly adiabatic*
($\epsilon=\nabla-\nabla_{\rm ad}=0$), so $(\ln T)_{,1}\neq0$ and the term is
"armed", and the state is at rest and discretely hydrostatic. By §3.2 the
prediction is $\Delta F\equiv0$ **bitwise**, hence an identical
$\max|\mathrm{d}E/\mathrm{d}t|$ and identical momentum and energy residuals. This
is the sharpest cheap falsification available:

* if ON were *different* from OFF here, the velocity factor is not the face-normal
  difference — e.g. a $p'$-only term, or the metric moment added as a source
  rather than as a covariance — i.e. the term is not (2.10);
* if ON were *worse* than OFF, the term is breaking the discrete hydrostatic
  balance, which §3.2 says it cannot;
* the momentum and mass residuals must be *identical*, not merely small, because
  $\Delta F$ enters the `IPR` row only (§3.1).

**(T3) Cell-by-cell comparison against (5.1) on a predictable state.** T2 is a
test of a zero; this one tests the *value*. The state is built constant in $x_2$
and linear in the $x_3$ index, so the reconstruction returns the analytic face
value exactly (`interp_cp3` reproduces a linear profile exactly,
`src/recon/interp_simple.hpp:73-76`), and (5.1) can be evaluated independently in
Python — including $\sigma_1^2$ recomputed from `x1f` by (2.14) — and compared
with the measured $(\mathrm{d}u_{\rm on}-\mathrm{d}u_{\rm off})/\mathrm{d}t$. On
the spherical wedge $A^{(2)}$ varies with $\theta$, so an $x_2$-independent
$\Delta F$ still has a non-zero divergence, which is what makes the metric
observable. Failure modes it separates: a wrong **coefficient** (using
$\Delta r^2/12$ instead of (2.14), or $/24$, or the volume weight $r^2$) shows as
a near-constant relative offset; a wrong **stencil** shows as an error localised
at the $x_1$ ends; a wrong **frame** on the cubed sphere shows as an error of
relative size $g_{23}$.

**(T3b) The curved residual must converge FASTER than $\Delta r^2$.** This is the
test that separates (2A.9) from (2.10). With the covariance term alone the
residual of the horizontal energy tendency against a quadrature oracle stays
$O(\Delta r^2)$ — it is merely reduced, not removed (§2A.4: 16–57 % relative,
non-convergent). With both terms it must drop to $O(\Delta r^3)$ or better. A
fitted order of 2 on a curved grid is therefore a direct signature of a missing
centroid term; an order of 2 at *planet* radius would instead indicate a
Cartesian-side bug, because $\delta\to0$ there.

**(T4) Resolution scaling $n_z: 32\to64$.** $\sigma_1^2\propto\Delta r^2$, so the
$\Delta F$-induced tendency must fall by a factor $4$. A factor $2$ or $8$ means
the order is wrong, independently of any coefficient.

**(T5) CUDA point.** The term is device-agnostic `torch` arithmetic on tensors
already resident on the device, and no `.cu` kernel is involved, so CPU and CUDA
must agree to float64 round-off on the same deck. A discrepancy beyond that
would mean a host-only buffer (`x1v`, `face_moment2_x1()`) was silently
host-resident or a dtype promotion differed between the paths — which is exactly
what the explicit `.to(device, dtype)` casts in §5 exist to prevent.

**(T6) Formula-level quadrature check** (independent of snapy):
evaluate $\langle pu\rangle_A-\bar p\,\bar m/\bar\rho$ by high-order quadrature
with weight $w=1$ and $w=r$ on generic smooth profiles and compare with (2.10).
The residual must be $O(h^4)$ absolute, i.e. $O(h^2)$ *relative* to a term that
is itself $O(h^2)$, falling by $4\times$ per halving. If it fell by $2\times$,
(2.10) would be missing part of the $O(h^2)$ covariance; if the closed form
(2.14) disagreed with the quadrature moment, $\sigma_1^2$ would be wrong.

---

## 7. Derived, versus hypothesis

**Derived here, with the algebra above:** the metric-weighted covariance identity
(2.6); the quotient correction (2.9); the covariance part of $\Delta F$ (2.10);
the Cartesian coefficient (2.11); the exact curved coefficient (2.14) and its
Cartesian limit; the cancellation of $\sqrt{g}$ and $g_{23}$ (2.8)→(2.16); the
**second, centroid term and the complete curved $\Delta F$ (2A.9)**, with its
$O(L/R)$ relative size (2A.10) and its quadrature check (§2A.4); the $x_3$
extension (§3); exact telescoping of both terms (§3.1, §2A.5); bitwise
preservation of a balanced state (§3.2, §2A.5).

**Superseded by §2A:** any statement in §2.5–§2.8 that the covariance term is the
complete $O(h^2)$ correction on a curved face, or that the metric affects
$\Delta F$ only at $O(h^4)$. The metric affects the *coefficient* $\sigma_c^2$
only at $O(h^4)$, which is true; but it also produces the separate $O(h^2)$
centroid term (2A.9).

**Hypotheses, labelled as such:**

* **H1.** That dropping the transverse in-face covariance (§2.9) stays negligible
  once the flow is turbulent rather than a linear mode. Untested here; that is
  #289 item 5's question.
* **H2.** That $\Delta F$ is the *whole* $O(\mathrm{d}z^2)$ error of the
  horizontal energy flux. #289's own open item 5 asks this: after the Cartesian
  fix, `cell` keeps about $-1\%$ where `face` goes to about $0$.
* **H3. A separate $O(\Delta r^2)$ curvilinear term exists and is not this one.**
  If the stored cell value is read as a *volume*-weighted average (weight
  $r^2\mathrm{d}r$) rather than as the face-area average of A4 (weight
  $r\,\mathrm{d}r$), the two differ at the same order by a **linear** term: their
  centroids are $\bar r+h^2/(6\bar r)$ (2.15) and $\bar r+h^2/(12\bar r)$ (2.12),
  so the mismatch is $\simeq\big(h^2/(12\bar r)\big)\,\partial_r a$ for every
  advected quantity. That is the standard curvilinear-reconstruction error: it
  affects the mass and momentum fluxes too, it has none of the
  $(\ln T)_{,1}u_{n,1}$ spurious-stratification structure, and it is **out of
  scope for #289**. It deserves its own issue. Nothing here fixes it and nothing
  here makes it worse.
* **H4.** The $(I+p)$ form of §5 is exact for an ideal gas. For a genuinely
  non-ideal EOS the exact covariance would need that EOS's own expansion, and
  $\ln(p/\rho)$ would no longer be $\ln T$ up to a constant.

## 8. The exact cubed-sphere cell volume

This section covers the follow-on change to `GnomonicEquiangleImpl::cell_volume()` and
`face_area1()`. Line numbers refer to the commit that adds this section.

### 8.1 The exact radial integral

A gnomonic-equiangle cell is the cone over the cube-face rectangle
$[x_j, x_{j+1}]\times[y_k, y_{k+1}]$, with $x=\tan\alpha$ and $y=\tan\beta$, cut by
$r_- \le r \le r_+$. In spherical measure $\mathrm{d}V = r^2\,\mathrm{d}r\,\mathrm{d}\Omega$, and the
angular and radial integrals separate:

$$
V_{ijk} = \int_{\Omega_{jk}}\!\mathrm{d}\Omega\int_{r_-}^{r_+} r^2\,\mathrm{d}r
        = \Omega_{jk}\,\frac{r_+^3-r_-^3}{3}
        = \Omega_{jk}\,\frac{\Delta r\,(r_+^2+r_+r_-+r_-^2)}{3}. \tag{8.1}
$$

The last form avoids the cancellation in $r_+^3-r_-^3$ when $\Delta r\ll r$. The radial face at
$r_f$ has area $A_1 = r_f^2\,\Omega_{jk}$.

The solid angle of the rectangle seen from the centre is
$\Omega=\iint (1+x^2+y^2)^{-3/2}\,\mathrm{d}x\,\mathrm{d}y$. Since
$\partial_x\partial_y\arctan\!\big(xy/\sqrt{1+x^2+y^2}\big) = (1+x^2+y^2)^{-3/2}$, it is the corner sum

$$
\Omega_{jk} = G(x_{j+1},y_{k+1}) - G(x_j,y_{k+1}) - G(x_{j+1},y_k) + G(x_j,y_k),\qquad
G(x,y)=\arctan\frac{xy}{\sqrt{1+x^2+y^2}}. \tag{8.2}
$$

Over one panel it telescopes to $4G(1,1)=4\arctan(1/\sqrt3)=2\pi/3$, so the six panels give $4\pi$.

**The previous code** had two approximations:
- $V = \tfrac12(A_{1,i}+A_{1,i+1})\Delta r$, the trapezoid rule in $r$, with relative error
  $\Delta r^2/(6\bar r^2)+O(\Delta r^4)$;
- $\Omega\approx\Delta\alpha_{\rm arc}\,\Delta\beta_{\rm arc}\,\sin\theta$ at the cell centre, a
  midpoint rule for (8.2) with relative error $O(\Delta\alpha^2)$. Its six-panel sum misses $4\pi$
  by $8.3\times10^{-4}$, $2.07\times10^{-4}$ and $5.2\times10^{-5}$ at 16, 32 and 64 cells per panel edge.

**What each oracle needs.**
- For $F=r\hat r$, the $x_2$/$x_3$ faces carry no flux, so the discrete divergence is
  $(A_{1,i+1}r_+ - A_{1,i}r_-)/V = \Omega_A(r_+^3-r_-^3)/V$. It equals 3 exactly iff $V$ uses
  (8.1) with **the same** $\Omega$ as $A_1$, whichever $\Omega$ that is. This is oracle (b).
- The six-panel volume sum equals $4\pi(r_o^3-r_i^3)/3$ only if $\Omega$ is the exact (8.2). This
  is oracle (a).

So the change uses (8.2) for both (`solid_angle_kj`, built once in `reset()`,
`src/coord/gnomonic_equiangle.cpp:124-137`), with $A_1=r_f^2\,\Omega$ (`:207-209`) and (8.1)
for $V$ (`:219-227`). Changing only the radial factor would pass (b) but leave (a) off by the
midpoint error above.

**Round-off.** (8.2) differences $O(1)$ arctangents to obtain an $O(\Delta\alpha^2)$ number, so its
relative round-off is about $\epsilon_{\rm mach}/\Delta\alpha^2$. That is $\sim 10^{-13}$ at 32 cells
per edge and $\sim 10^{-10}$ at 1000.

### 8.2 Uses of `cell_volume()`: which ones need the exact value

The conservation identities need only **consistency**: every caller reads the same
`cell_volume()`, so the flux divergence and the totals it must telescope into use one $V$, before
and after this change. **Exactness** matters where $V$ meets a quantity that is not a difference
of the same face fluxes: geometric source terms, and physical integrals.

| use (file:line at this commit) | what $V$ does there | needs |
|---|---|---|
| `coord/coordinate.cpp:513` divergence | $\frac{1}{V}\sum A F$ for every conserved variable | consistency (telescoping); exactness for the free-stream identity (b) |
| `coord/gnomonic_equiangle.cpp:142-145` `x_ov_rD_kji`, `y_ov_rC_kji` | $\Delta(A_2\sin)/V$, the $x_2$/$x_3$ metric pressure source | consistency with the divergence (rest balance: both divide by the same $V$); exactness makes it the exact cell average of the metric source |
| `coord/gnomonic_equiangle.cpp:151` `vol` | shape template for the placeholder metric buffers | neither (values unused) |
| `coord/gnomonic_equiangle.cpp:450-455` radial pressure source | $(A_{1,+}p_+ - A_{1,-}p_-)/V - (p_+-p_-)/\Delta r$ | **exactness**: with (8.1) and constant $p$ it is $3p(r_+^2-r_-^2)/(r_+^3-r_-^3)$, the exact cell average of $2p/r$; $\Omega$ cancels only when $A_1$ and $V$ share it |
| `hydro/hydro_forward.cpp:572-583` face gravity work | $\frac1V\sum A_1 F_m$ and $\frac1V\sum A_1\phi F_m$ | consistency with continuity's divergence (E+PE telescopes) |
| `hydro/hydro_forward.cpp:469` positivity severity | cell mass $mV$ vs withheld outflow | consistency (a ratio test) |
| `hydro/flux_positivity.cpp:66` limiter $\theta$ | available mass $uV$ vs $\Delta t\sum A F$ | consistency: $\theta$ keeps $u\ge0$ only if it uses the divergence's $V$ |
| `hydro/hydro_forward.cpp:721` E+PE fixer total | $\sum(E+\phi m)V$ | consistency (the fixer compares changes) |
| `implicit/implicit_hydro.cpp:198-205` VIC operator, `work_lo/hi` | the linearised $x_1$ divergence and face-work weights | consistency with the explicit divergence |
| `implicit/implicit_hydro.cpp:262, 276, 327` mass-closure and face-work checks | $\sum$ cell change $\times V$ vs face mass differences | consistency (identity checks) |
| `eos/equation_of_state.cpp:272` column repair | conserve $\sum\rho V$ in a column | consistency |
| `mesh/meshblock.cpp:671` implicit tracer transfer | $(P-P_{\rm above})/V$ | consistency with the hydro mass transfer |
| `mesh/meshblock.cpp:849`, `:1025` mass / conserved totals | $\sum uV$ | **exactness** for the physical integral; consistency for drift |
| `output/load_diag_output_data.cpp:263` column paths | $\sum q V / A_1$ | **exactness** (a physical diagnostic) |

No caller depends on the trapezoid value itself. Changing the angular factor of $A_1$ moves the
$x_1$ face areas by $O(\Delta\alpha^2)$ relative. Every caller above reads $A_1$ and $V$ together,
so the consistency identities are unchanged. The hydrostatic rest check (oracle c) is the runtime
test of that.

### 8.3 What each check tests

- **(a) six-panel volume sum** $=4\pi(r_o^3-r_i^3)/3$: tests (8.1) and (8.2) together.
- **(b) $\max|\nabla\!\cdot(r\hat r)-3|$:** tests that $V$ and $A_1$ share $\Omega$ and that the radial
  integral is exact. It is RED at the trapezoid ($O((\Delta r/r)^2)$) and round-off when correct.
- **(c) hydrostatic rest:** tests that the change keeps the radial and lateral source terms
  consistent with the divergence.
- **(d) Cartesian and spherical-polar decks bitwise unchanged:** tests that the change is
  confined to the gnomonic geometry.

### 8.4 Audit: the other cubed-sphere measures and metric terms

Line numbers refer to `src/coord/gnomonic_equiangle.cpp` at e7f9904. The question is whether any other
area or metric factor uses the midpoint solid-angle or arc approximation that §8.1 removes from $A_1$ and $V$.

| item (e7f9904) | form | exact? | action |
|---|---|---|---|
| `face_area2()` :197-199, angle `dx3f_ang_face2_kj` :119 | $\tfrac12(r_+^2-r_-^2)\,\theta_{x2}$ | **exact** | none |
| `face_area3()` :201-203, angle `dx2f_ang_face3_kj` :105 | $\tfrac12(r_+^2-r_-^2)\,\theta_{x3}$ | **exact** | none |
| `x_ov_rD_kji`, `y_ov_rC_kji` :124-130 | $\Delta(A_2\sin\theta_{f2})/V$, $\Delta(A_3\sin\theta_{f3})/V$ | $\sin\theta_f$ taken at the face centre (midpoint along the face) | none: paired with the flux (below) |
| `sine_face2_kj`, `sine_face3_kj` :84, :87 in `flux2global2_/3_` :343-398 | covariant momentum flux | same midpoint $\sin\theta_f$ | none: paired with the source |
| `sine_cell_kj`, `cosine_cell_kj` :80-81 in `forward()` :400-458 and `prim2local1_` | velocity-dependent geometric sources $\rho v^2/r$, $\rho v^2\sin^2\theta$ | point values at the cell centre: second-order source quadrature, not a measure | none (they vanish at rest) |
| radial pressure source :429-440 (face-pressure form, every run with `nx1 > 1`) | $(A_{1,+}p_+-A_{1,-}p_-)/V-(p_+-p_-)/\Delta r$ | exact once $A_1$, $V$ share $\Omega$ (§8.2) | fixed by this change |
| fallback $2p/r$ :442 (only when `nc1 = 1`, `hydro/hydro.cpp:127-135`) | $2p/x_{1v}$ | midpoint in $r$ | none (single-layer only) |
| `center_width2/3()` :184-190 | $x_{1v}\times$ arc | CFL length only | none |

**$A_2$ and $A_3$ are already exact.** An $x_2$ face ($\alpha=$ const, $x=\tan\alpha$) lies in a plane through
the centre. Its two radial edges are the rays along $(1, x_f, y_k)$ and $(1, x_f, y_{k+1})$, and it is bounded by
$r_\pm$. A planar annular sector has area $\tfrac12(r_+^2-r_-^2)\,\theta$, where $\theta$ is the angle between the
two rays:
$\cos\theta = (1+x_f^2+y_ky_{k+1})/(\delta_k\delta_{k+1})$, $\delta=\sqrt{1+x_f^2+y^2}$.
That is line 119 verbatim, and `x1v*dx1f` $=\tfrac12(r_+^2-r_-^2)$ exactly. The same holds for $A_3$.

**The midpoint $\sin\theta_f$ cancels at rest.** At rest the local $x_2$-face flux is $(p,0)$ in the
$(x_2, x_3)$ momentum slots. With the face metric ($g_{23}=\cos\theta_f$, $g^{22}=1/\sin^2\theta_f$),
`flux2global2_` maps it to the contravariant pair $(p/\sin\theta_f,\ -p\cos\theta_f/\sin\theta_f)$. The covariant
lowering then gives
$x_2$: $p/\sin\theta_f - p\cos^2\theta_f/\sin\theta_f = p\sin\theta_f$, and $x_3$: $0$.
So the flux divergence of $x_2$ momentum is $p\,\Delta(A_2\sin\theta_f)/V = p\,x\_ov\_rD$, which is exactly
`src2`. The two cancel for **any** $V$ and any value of $\sin\theta_f$, as long as flux and source use the same
product, which they do. Changing $V$ and $A_1$ cannot break the lateral rest balance. The radial balance needs
$A_1$ and $V$ to share $\Omega$, which they now do.

So the remaining midpoint factors are paired consistently, and I did not change them. Making $\sin\theta_f$ a
face average would change the moving-flow metric source at $O(\Delta\alpha^2)$. That is a separate accuracy question
and not needed for any oracle here.

### 8.5 Runtime evidence for the mixed exact-$A_1$ / existing-$A_2,A_3$ geometry

**Hydrostatic shell.** This is the deck of `tests/test_cubed_sphere_cell_volume.yaml`: six panels, $8^3$ cells
per panel, $r\in[6.0,6.4]\times10^6$, isothermal, discretely balanced per column. The table gives max $|v|/c_s$
and the relative drift of $\sum\rho V$ and $\sum(E+\rho\phi)V$:

| cycles | build | max $\lvert v\rvert/c_s$ | mass drift | E+PE drift |
|---|---|---|---|---|
| 200 | e7f9904 | 1.08772e-11 | −1.21e-14 | −1.19e-14 |
| 200 | this commit | 1.08764e-11 | −1.20e-14 | −1.19e-14 |
| 1000 | e7f9904 | 1.30564e-11 | −6.05e-14 | −6.02e-14 |
| 1000 | this commit | 1.30534e-11 | −6.05e-14 | −6.03e-14 |

**A moving flow.** This is the six-panel solid-body zonal wind of `tests/test_flux_positivity_cubedsphere.yaml`
(unmodified), $v_\phi = 10\cos(\text{lat})$, 400 cycles. Two errors are tracked:
- $\delta v=\max|v(t)-v(0)|/\max|v(0)|$;
- the tracer $L_1$ error against the exact rigid rotation $\Delta\lambda = 10\,t/r$.

| cycle | $\delta v$ e7f9904 | $\delta v$ this commit | $L_1$ e7f9904 | $L_1$ this commit |
|---|---|---|---|---|
| 100 | 0.026414 | 0.026351 | 0.013352 | 0.013348 |
| 200 | 0.026991 | 0.026965 | 0.017929 | 0.017937 |
| 300 | 0.044127 | 0.044332 | 0.019531 | 0.019557 |
| 400 | 0.045423 | 0.045116 | 0.021800 | 0.021801 |

The two builds agree to within ±0.7% at every recorded time, and neither is systematically larger. At 12 cells per
panel edge, this deck's error is dominated by the scheme's own discretisation error, not by the $O(\Delta\alpha^2)$
change in $A_1$ and $V$.

---

# 4A. The remaining rows: mass, momentum and the moist tracers

Mass, the three momentum rows and the moist tracer rows, on a spherical-$x_1$
column and on a gnomonic cubed-sphere panel. Section 2A derived the **energy**
row; this section does the rest with the same rigour and the same scripts
(`docs/derivations/allrows_quadrature.py`, `docs/derivations/rest_balance.py`).

**Implemented here:** the mass row and every tracer row.
**Derived but NOT implemented:** the momentum rows -- see 4A.3.2, they need the
geometric pressure source shifted in lockstep, and that has not been compiled
or run. Doing it blind would put the exact rest balance at risk, which is the
very property at stake.

## 4A.1 Notation, and the one rule that covers every row

Two measures are in play on the same cell (§1.3, §2A.1 of the main file):

$$
\underbrace{\mathrm{d}V \propto r^2\,\mathrm{d}r}_{\text{what a cell value averages}}
\qquad
\underbrace{\mathrm{d}A_{2,3} \propto r\,\mathrm{d}r}_{\text{what a face flux needs}}
$$

one power of $r$ apart, hence two different $x_1$ centroids and the offset

$$
r_c=\frac{12\bar r^2+h^2}{12\bar r},\qquad
r_v=\frac{3\bar r(4\bar r^2+h^2)}{12\bar r^2+h^2},\qquad
\boxed{\;\delta=r_v-r_c=\frac{h^2\,(12\bar r^2-h^2)}{12\,\bar r\,(12\bar r^2+h^2)}
=\frac{h^2}{12\bar r}+O\!\Big(\frac{h^4}{\bar r^3}\Big)}
\tag{1.1}
$$

with $h=\Delta r$, $\bar r=\tfrac12(r_-+r_+)$, and
$\sigma_c^2=\tfrac{h^2}{12}\big(1-\tfrac{h^2}{12\bar r^2}\big)$ the area-weighted
second central moment. In Cartesian both measures are the same uniform weight,
so $r_v=r_c$ and $\delta\equiv0$ **exactly**: nothing in this document changes
any Cartesian result.

### 4A.1.1 The general rule

Let a row's face flux be a product of factors that the code reads from **cell**
storage, $F=a\,b\,c\cdots$. Expanding each $\bar q = q(r_v)+\tfrac12\sigma_v^2q''$
about $r_c$, using $\delta=O(h^2)$ so $\delta^2=O(h^4)$, and
$\sigma_c^2-\sigma_v^2=O(h^4)$ (§2A.5 of the main file), every second-derivative
group cancels and

$$
\boxed{\;
\big\langle F\big\rangle_A-\bar a\,\bar b\,\bar c\cdots
=\underbrace{\sigma_c^2\!\!\sum_{\text{distinct pairs}(x,y)}\!\! x'\,y'\!\!\prod_{z\neq x,y}\! z}_{\text{covariance, row-specific}}
\;\underbrace{-\;\delta\,\partial_1 F}_{\text{centroid, universal}}
\;+\;O(h^4)\;}
\tag{1.2}
$$

Two consequences worth stating before any row:

* the **centroid** piece is the *same* for every row — minus $\delta$ times the
  $x_1$ derivative of that row's own flux. It needs no per-row algebra.
* a factor that stands **alone** in the flux (the pressure in the normal-momentum
  row) has no partner, so it contributes **only** a centroid term:
  $\langle p\rangle_A-\bar p=-\delta\,\partial_1p+O(h^4)$. That single fact drives
  everything in §3.

---

## 4A.2 Row by row

$u_n$ is the face-normal velocity in the face-local orthonormal frame (`IVY`
after `prim2local2_` for an $x_2$ face, `IVZ` after `prim2local3_` for an $x_3$
face; `lmars_impl.h:20`, `lmars.cpp:54-67`). $u_t$ is either tangential
component. $q$ is any tracer mixing ratio, dry or moist.

| row | face flux $F$ | covariance part $/\sigma_c^2$ | centroid part |
|---|---|---|---|
| mass | $\rho u_n$ | $\rho'u_n'$ | $-\delta\,\partial_1(\rho u_n)$ |
| momentum, normal | $\rho u_n^2+p$ | $2\rho'u_n'u_n+\rho u_n'^2$ | $-\delta\,\partial_1(\rho u_n^2+p)$ |
| momentum, tangential | $\rho u_nu_t$ | $\rho'u_n'u_t+\rho'u_t'u_n+u_n'u_t'\rho$ | $-\delta\,\partial_1(\rho u_nu_t)$ |
| tracer (each, incl. moist) | $\rho u_nq$ | $\rho'u_n'q+\rho'q'u_n+u_n'q'\rho$ | $-\delta\,\partial_1(\rho u_nq)$ |
| energy (§2A, for reference) | $\kappa p\,u_n$ | $\kappa p\,[\ln(p/\rho)]'u_n'$ | $-\delta\,\partial_1(\kappa p\,u_n)$ |

### 4A.2.1 The tracer row, and why a uniform $q$ stays uniform

Expanding the three-factor rule and grouping,

$$
\Delta F_q=\sigma_c^2\big(\rho'u_n'q+\rho'q'u_n+u_n'q'\rho\big)-\delta\,\partial_1(\rho u_nq)
= q\,\Delta F_\rho \;+\; \sigma_c^2\,(\rho u_n)'\,q' \;-\;\delta\,\rho u_n\,q' ,
\tag{2.1}
$$

using $\Delta F_\rho=\sigma_c^2\rho'u_n'-\delta\partial_1(\rho u_n)$ from the
mass row. Setting $q'=0$ leaves exactly $q\,\Delta F_\rho$, so a **uniform
tracer stays exactly uniform**: the corrected tracer flux is the corrected mass
flux times $q$. That is the property Xi required.

Note, though, that "corrected mass flux $\times\,q$ plus the Cartesian
$\sigma^2(\rho u_n)'q'$" is **not** the whole curved answer — it omits the third
term $-\delta\,\rho u_n q'$ of (2.1). That term vanishes for uniform $q$, which
is why the shorthand passes the uniformity test, but it is $O(h^2)$ and of the
same order as the rest whenever the tracer has a vertical gradient — i.e. in
every moist deck with a background vapour profile. **It must be kept.** Each
tracer row is corrected independently with its own $q$, so vapour and condensate
rows need no cross terms.

### 4A.2.2 Quadrature verification

`allrows_quadrature.py`, 90-point Gauss–Legendre, generic non-constant
$\rho,p,u_n,u_t,q_v,q_c$, comparing the exact area average (weight $r$) against
the product of volume averages (weight $r^2$). Relative residual of
(covariance only) and (covariance + centroid), and the convergence ratio of the
latter:

| row | $\bar r$ | rel, cov only | rel, cov+centroid ($h=0.1\to0.0125$) | ratio |
|---|---|---|---|---|
| mass | 1.4 | 3.49e-02 (plateaus) | 1.05e-02 → 1.64e-04 | 4.00 |
| momentum, normal | 1.4 | **1.000** | 3.11e-03 → 4.86e-05 | 4.00 |
| momentum, tangential | 1.4 | 8.36e-02 | 1.09e-03 → 1.67e-05 | 4.00 |
| tracer, vapour | 1.4 | 9.40e-02 | 2.12e-02 → 3.33e-04 | 4.00 |
| tracer, condensate | 1.4 | 1.84e-01 | 4.08e-03 → 6.37e-05 | 4.00 |
| mass | 6.0 | 2.79e-02 | 1.53e-03 → 2.39e-05 | 4.00 |
| momentum, normal | 6.0 | **1.004** | 1.59e-03 → 2.48e-05 | 4.00 |
| momentum, tangential | 6.0 | 3.74e-02 | 4.54e-04 → 7.14e-06 | 4.00 |
| tracer, vapour | 6.0 | 3.10e-02 | 1.03e-03 → 1.61e-05 | 4.00 |
| tracer, condensate | 6.0 | 4.95e-02 | 1.33e-03 → 2.10e-05 | 4.00 |

Every row: exactly $4\times$ per halving with both terms, i.e. $O(h^4)$ absolute,
so (1.2) is the **complete** $O(h^2)$ correction for each. Covariance-only never
converges. The normal-momentum entry is the sharpest statement in the table: at
relative $1.000$ and $1.004$ the covariance term explains **none** of that row's
error, because the error is the pressure's centroid shift, which has no
covariance structure.

---

## 4A.3 The decisive gate: exact hydrostatic rest balance

### 4A.3.1 What has to cancel

At rest $u\equiv0$, so every velocity-bearing term in §2 vanishes and the only
surviving correction anywhere is the pressure's centroid shift in the
**normal-momentum** row:

$$
p^\star \;\equiv\; p-\delta\,\partial_1 p ,
\qquad
\Delta F\big|_{\rm rest}=-\delta\,\partial_1p \;=\; p^\star-p .
\tag{3.1}
$$

The other rows are untouched at rest: mass, tangential momentum, tracer and
energy all carry $u_n$ or $u_n'$ as a factor, exactly zero in floating point.
The $x_1$ flux is untouched too — an $x_1$ face's in-face coordinates are $x_2$
and $x_3$, so it has no $x_1$ centroid shift (§3 of the main file). In
particular the `face_pressure1` well-balanced path
(`spherical_polar.cpp:246-255`) is **not** modified, so the $x_1$-momentum rest
balance is untouched by construction.

That leaves one thing to prove: the **lateral** momentum row. At rest it is a
cancellation of two terms, both **linear in the same cell pressure $p$**:

$$
\mathrm{div}[{\rm IVY}]\big|_{\rm rest}
=\underbrace{\frac{A^{(2)}_{j+1}\Phi_{j+1}-A^{(2)}_j\Phi_j}{V}}_{\text{lateral pressure flux, }\Phi=c\,p}
\;-\;\underbrace{S_{\rm geom}\,p}_{\text{geometric source}} .
\tag{3.2}
$$

**Spherical-polar.** $\Phi=p$ and, from `spherical_polar.cpp:262-266`,
$S_{\rm geom}=$ `coord_src1_i * coord_src1_j`. With
$\texttt{radial}=\tfrac12(r_+^2-r_-^2)$,
$\texttt{radial\_volume}=\tfrac13(r_+^3-r_-^3)$,
$\texttt{polar\_volume}=|\cos\theta_-\!-\!\cos\theta_+|$ (`:186-187`, `:201-208`),
$A^{(2)}=\texttt{radial}\,|\sin\theta_f|\,\Delta\phi$ (`:188-190`) and
$V=\texttt{radial\_volume}\cdot\texttt{polar\_volume}\cdot\Delta\phi$ (`:210`):

$$
\frac{A^{(2)}_{j+1}-A^{(2)}_j}{V}
=\frac{\texttt{radial}\,(\sin\theta_+-\sin\theta_-)}{\texttt{radial\_volume}\cdot\texttt{polar\_volume}}
=\underbrace{\frac{\texttt{radial}}{\texttt{radial\_volume}}}_{\texttt{coord\_src1\_i}}
\cdot\underbrace{\frac{\sin\theta_+-\sin\theta_-}{\texttt{polar\_volume}}}_{\texttt{coord\_src1\_j}}
= S_{\rm geom}.
\tag{3.3}
$$

An identity. The source exists precisely to cancel the lateral pressure flux.

**Gnomonic cubed sphere.** After `flux2global2_`
(`gnomonic_equiangle.cpp:341-350`) the rest flux is
$\Phi=p\,\texttt{sine\_face2\_kj}$ — the $\cos\vartheta$ terms cancel between
the two covariant lines — so the lateral term is $p\,(f_{j+1}-f_j)/V$ with
$f=\texttt{face\_area2}\cdot\texttt{sine\_face2\_kj}$. And `:136-138` *defines*
`x_ov_rD_kji` as exactly $(f_{j+1}-f_j)/\texttt{cell\_volume()}$, which `:445-452`
then uses as the source. So here (3.3) is not merely true, it is the **same
floating-point expression** on both sides.

### 4A.3.2 The proof

Both sides of (3.2) are the same geometric coefficient times the same scalar
$p$, and at rest $p$ is a function of $r$ only, so $\Phi_j$ and $\Phi_{j+1}$
carry the *same* $p_i$. Replace $p\to p^\star$ in the flux **and in the source**:
the scalar factors out of both terms identically and

$$
\mathrm{div}[{\rm IVY}]\big|_{\rm rest}
= S_{\rm geom}\,p^\star - S_{\rm geom}\,p^\star = 0
\tag{3.4}
$$

**exactly, in floating point, for any metric** — the argument never touches the
value of $S_{\rm geom}$. Replace $p$ in the flux only, and the residual is
$-\,S_{\rm geom}\,\delta\,\partial_1p$, i.e. a relative error
$\delta|\partial_1p|/p$.

So the balance **holds**, on one condition:

> **The geometric pressure source must carry the same centroid-shifted pressure
> $p^\star$ as the lateral face flux.** This is the source weight Xi anticipated:
> the source inherits the measure mismatch because it is built from the cell
> pressure, and shifting it in lockstep restores exactness.

The same argument covers the $x_3$ row and the `use_x2_fluxes == false` branch
(`spherical_polar.cpp:285-288`), whose source term $\rho u_\phi u_\theta$ is
velocity-bearing and therefore zero at rest.

### 4A.3.3 Numeric check at machine precision

`rest_balance.py` transcribes the discrete operators from the cited lines,
builds an isothermal hydrostatic column ($R=7\times10^7$ m, depth $4\times10^4$ m,
$g=11$, $R_d=3700$, $T_0=100$ K), and evaluates (3.2) in three arms.

| grid | geometric coefficient identity (3.3) | base | flux only | flux + source |
|---|---|---|---|---|
| spherical-polar | $1.38\times10^{-22}$ abs, coef scale $5.35\times10^{-9}$ → rel $\sim\!2.6\times10^{-14}$ | $1.28\times10^{-17}$ (rel $2.39\times10^{-14}$) | $5.01\times10^{-11}$ (rel $\mathbf{9.359\times10^{-8}}$) | $1.24\times10^{-17}$ (rel $2.32\times10^{-14}$) |
| gnomonic panel | **exactly 0** (same expression) | $9.76\times10^{-19}$ (rel $8.39\times10^{-16}$) | $1.09\times10^{-10}$ (rel $\mathbf{9.359\times10^{-8}}$) | $9.76\times10^{-19}$ (rel $8.39\times10^{-16}$) |

At the column mid-point $\delta=3.306\times10^{-3}$ m,
$\partial_rp=-1.641$ Pa m$^{-1}$, so $\delta\,\partial_rp=-5.425\times10^{-3}$ Pa
against $p=5.518\times10^4$ Pa — a relative shift of $9.832\times10^{-8}$, which
is exactly the "flux only" break on both grids. Confirmed:

* **base** is balanced at machine precision on both grids;
* **flux only** breaks it by precisely the relative pressure shift;
* **flux + source** restores it to machine precision — on the cubed sphere
  bitwise identical to base.

**VERDICT: the balance proof HOLDS**, with the source-shift condition of §3.2.

---

## 4A.4 Telescoping, row by row

Every correction is added to the **face flux**, before the divergence
(`coordinate.cpp:492-496`, `:504`), so it telescopes in the $x_2$/$x_3$ sums
exactly like the flux it corrects. What that buys differs per row:

| row | telescopes? | discrete invariant |
|---|---|---|
| mass | yes | total mass **exact** to round-off; the correction cancels pairwise across interior faces, and $u_n=0$ at a wall so it vanishes there |
| tracer (each, incl. moist) | yes | total tracer mass **exact**; and a uniform $q$ stays uniform exactly, by (2.1) |
| energy | yes | $E+PE$ **exact** (§3.1 of the main file) |
| momentum, tangential | yes as a flux, **but** | the row is not a pure divergence — the geometric sources at `spherical_polar.cpp:266-288` are not fluxes — so no momentum component is a telescoping invariant *in main either*. The conserved quantity is angular momentum, a different combination; whether the corrected scheme conserves it discretely is **not** established here (§5, H5) |
| momentum, normal | same | same, plus the source-shift coupling of §3.2 |

So: **the three scalar-transport invariants stay exact; the momentum rows gain
no new invariant and lose none.** No limit sentence is needed or offered —
the corrections are exact to $O(h^4)$ on every row and every grid covered here.

---

## 4A.5 Assumptions and what is *not* established

Carried from the main file: A1 smoothness ($C^4$ across the face), A2 ideal gas
for naming $\ln(p/\rho)$ as $\ln T$, A3 enthalpy-only energy flux, A4 the
reconstructed face state read as the face average, A5 the transverse in-face
covariance dropped as perturbation-quadratic, A6 per-cell $\sigma_c^2$ and
$\delta$ so non-uniform $x_1$ needs no extra assumption.

Newly relevant, and labelled as such:

* **H5 (open).** Discrete *angular*-momentum conservation under the corrected
  momentum rows is not proved here. §4 establishes only that nothing is made
  worse: no momentum invariant exists in main to break.
* **H6 (open).** §3 proves the **rest** state exactly. A moving balanced state
  (e.g. geostrophic or cyclostrophic) is not covered; the cancellation of §3.2
  uses $u\equiv0$ to kill every other term.
* **H7 (open).** The source shift is derived for the *pressure* in the geometric
  source. `spherical_polar.cpp:262-265` also puts $\rho u_\phi^2$ into `m_pp`;
  that part is velocity-bearing so it is irrelevant at rest, but whether it needs
  its own shift away from rest is not settled here.
* **A7.** The frame transform `flux2global2_`/`flux2global3_` is linear in the
  flux vector, so the correction may equivalently be applied to the local-frame
  flux before it or transformed with the same matrix after it. The implementation
  must pick one and be consistent; the derivation is stated in the **local**
  frame.
* Nothing in this document has been **compiled or executed in snapy**. The
  evidence is symbolic plus the two quadrature/transcription scripts. No library
  has ever linked on this branch (62/104 objects, ~80/104, then a CMake configure
  that did not finish), and the build, `ctest -j1`, the lifted x2cov GREEN and the
  cell-volume test are owned by Xi's cluster worker.

# 4B. The implemented rows (density-weighted pairs, energy in h)

This section supersedes the covariance column of the 4A.2 table, and the
energy row of §5, for what is coded. The centroid rule of 4A.1.1 and the
balance proof of 4A.3 stand unchanged.

## 4B.1 Why no pair contains $\rho'$

A cell stores the conserved state. The solver's primitives are ratios of cell
averages: $q=\overline{\rho q}/\bar\rho$ and $u=\bar m/\bar\rho$
(`ideal_moist_impl.h:40,44`). A flux $\rho\,u_n\,q$ rebuilt from them is
$\bar m_n\,\overline{\rho q}/\bar\rho$, not $\bar\rho\,\bar u_n\,\bar q$.
Expanding that quotient about the face gives, for every such row,

$$
\langle\rho u_n q\rangle-\bar m_n\,\overline{\rho q}/\bar\rho
=\sigma_c^2\,\rho\,\partial_1u_n\,\partial_1q+O(h^4),
$$

and the total mass flux $\bar m_n$ is linear in the stored state, so its gap is
zero. The three-factor pairs of 4A.2 treat $\rho$, $u_n$ and $q$ as independent
cell averages, which they are not.

The same algebra with $p=\rho\,\Pi(e,y)$ and the W→I energy offsets gives the
energy row $\sigma_c^2\rho\,\partial_1h\,\partial_1u_n$ with
$h=(\text{W→I}+p)/\rho$ (no KE), to linear order in the velocity. Two things are
left out of it:
- the velocity-squared parts, which are the scheme's ordinary second-order
  truncation in nonlinear flow and vanish in the linear problem;
- the pressure term $\rho J$, where $J=\tfrac{\sigma^2}{2}\psi_1^{\mathsf T}\Pi_{\psi\psi}\psi_1$.
  It is about $10^{-3}$ of the energy term.

This is the independent symbolic and quadrature derivation in
`docs/derivations/issue289_moist_covariance_verifier.md` on zoeyzyhu/snapy
`study/289-moist-verifier` (afe9f6b827df51c6af5d3857e8cbd8d225db9b83).

## 4B.2 The rows as coded

All rows are in `src/hydro/hydro_forward.cpp`, `_flux_covariance()`, behind
`SNAP_FLUX_COVARIANCE`. $D_1$ is the centred $x_1$ difference, and
$\delta=$ `face_centroid_shift_x1()`.

| row | covariance part | centroid part |
|---|---|---|
| energy | $\sigma^2\rho\,D_1[h]\,D_1[u_n]$ | $-\delta\,D_1[\rho h\,u_n]$ |
| tracer $n$ | $\sigma^2\rho\,D_1[u_n]\,D_1[q_n]$ | $-\delta\,D_1[\rho u_n q_n]$ |
| dry mass (`IDN`) | $-\sum_n$ of the tracer parts | $-\delta\,D_1[\rho u_n q_d]$ |
| total mass (dry + tracers) | $0$ | $-\delta\,D_1[\rho u_n]$ |
| momentum, every component (face-local frame) | none | $-\delta\,D_1[\rho u_n u_v]$, plus $-\delta\,D_1^{\rm p}[p]$ on the normal one |

The momentum corrections are formed in the face-local frame and mapped with
`flux2global2_`/`flux2global3_`, as the Riemann flux is (`lmars.cpp`). The
pressure part uses $D_1^{\rm p}$, which is centred but one-sided at the first and
last interior cell (`d1_pressure`). The geometric pressure source is evaluated
with the same $p^\star=p-\delta\,D_1^{\rm p}[p]$ in step 5 of `forward()`, gated
per direction: the $x_2$ source (the `IVY` row) takes $p^\star$ exactly when the
$x_2$ faces carry the correction, and the $x_3$ source (`IVZ`) exactly when the
$x_3$ faces do. With every resolved direction corrected, and with the switch
off, this is the same single source evaluation as before.

**Why $D_1^{\rm p}$ is one-sided at the ends.** The balance argument of 4A.3.2
needs the flux and the source to see the *same* pressure difference. Next to a
cubed-sphere panel edge, the $x_2$/$x_3$ face states in the $x_1$ ghost rows do not
match the cell ghost values: the corner ghosts are not filled for that purpose.
A centred difference reaching into those rows broke the rest state there
(max $|v|/c_s$ $2.2\times10^{-7}$ after 20 steps, located at the panel edges in the
first and last interior $x_1$ cell). The one-sided ends read no ghost row and
restore it to the switch-off value. The cost is a first-order difference inside
an $O(h^2)$ term on one boundary cell row.

## 4B.3 What the checks test

`tests/test_flux_covariance_rows.py` runs each arm in its own process, because
the switch is read once per process.

- **Rest.** A balanced isothermal shell stays at rest with the switch on, on a
  spherical-polar block and on six gnomonic panels. This tests the
  $p^\star$-in-flux-and-source rule and the one-sided ends.
- **Uniform tracer.** On the moist six-panel shell under a solid-body wind, a
  uniform vapor stays uniform to round-off. With $D_1[q]=0$, the tracer row is
  $q$ times the total-mass row.
- **Offset invariance.** Shifting the vapor $u_0$ changes no primitive beyond
  round-off. The h-differenced energy row moves by exactly $c\,\cdot$ (tracer
  rows), while $(I+p)\,D_1\ln(p/\rho)$ does not.
- **Dry Cartesian limit.** One-step $\varepsilon_{\rm eff}n_z^2$ stays within
  $2\times10^{-3}$ of the covariance-only form at $n_z=16,32,64$. In a dry,
  zero-offset gas $\rho\,D_1[h]=(I+p)\,D_1\ln(p/\rho)+O(h^2)$, so the two agree
  to discretisation error, not bitwise.

---

# 4C. What §4B superseded, and what is excluded

Source of the correction: Xi Zhang's moist form, posted in Slack
(thread 1791520135.380369, ts 1791537083). Independent check: Zoey Hu's
derivation at `study/289-moist-verifier` @ `afe9f6b`, which reached the same
rows from the #289 formula without reading this file. The rows **as coded** are
§4B; this section is the record of what they replaced and why, because the
file is the PR's evidence that a defect was found.

**Second independent check.** `UCzhangxi/snapy` commit
`e2c3f57120d772e5e861d8bf9674e4f7ab447e35`, directory `study/289-allrows/`
(`derivation.md` plus `symbolic_rows.py`, `quad_source.py`, `rest_balance.py`
and their committed `.out` files). It reaches the same two-term form for every
$x_2$/$x_3$ row, $\sigma_c^2\rho\,\partial_1u_n\,\partial_1\varphi
-\delta\,\partial_1(\rho u_n\varphi)$ with $-\delta\,\partial_1p$ added in the
normal-momentum row and no covariance in the mass row (its §0, eqs. 2.3-2.9),
and its symbolic run reproduces the closed forms for $\delta$ and $\sigma_c^2$
used here with difference exactly $0$, and
$\sigma_c^2-\sigma_v^2=11h^4/(720\bar r^2)+O(h^6)$ -- the same value as §2A.5's
$h^4(1/45-1/144)/\bar r^2$.

**Numbers below are that study's, on its own deck ($R_0=1$ or $10$,
$H_p=0.1$), not ours.** Flux-only breaks the rest state by
$+\delta D_1p$ times the source coefficient, $7.97\times10^{-4}$ at
$h=0.0312$ falling to $1.26\times10^{-5}$ at $h=0.0039$, a factor $4$ per
halving, and matching that prediction to $2.6\times10^{-12}$; with
$p\to p-\delta D_1p$ in the geometric source the residual returns to round-off,
$\sim\!2\times10^{-15}$ in the spherical $\theta$ row and
$\sim\!1\times10^{-14}$ in the gnomonic $\alpha$ and $\beta$ rows. Correcting
the mass row without the tracer rows drifts a uniform $q$ at $O(\delta)$,
$6.34\times10^{-5}$; with both rows it holds at
$4.00\times10^{-16}$/$3.10\times10^{-16}$. Its corrections that carry $u_n$ are
bitwise zero at rest. Because its deck differs from the one in §4A.3.3, these
confirm the mechanism and the $h^2$ scaling, **not** the magnitudes measured
there.

**Three things it adds or sharpens, recorded rather than buried:**

1. **snapy's present geometric sources already carry the same $O(h^2)$
   mismatch** (its §4.5): the exact lateral pressure source wants
   $\langle p\rangle_A$ and snapy multiplies $\bar p_V$, so *today's* rest
   balance holds only because the flux and the source make the **same** error.
   This does not contradict §4A.3.2 -- the identity proved there is precisely
   that both sides carry the same cell pressure -- but it is a sharpening this
   file did not state, and it means the corrected pair is genuinely more
   accurate **in motion**, not merely equally balanced at rest. Its symbolic
   S3 makes the same point exactly: the exact $\theta$ source minus
   $\langle p\rangle_A(A_+-A_-)$ is $0$, while with $\bar p_V$ it is not.
2. **On the gnomonic grid `x1v` is the arithmetic mid-radius, not $r_v$**
   (`gnomonic_equiangle.cpp:36`), so the identification of $r_v$ with snapy's
   `x1v` in §2A.1 holds on **spherical-polar only**. $\delta$ is defined by the
   two measures, not by `x1v`, so (1.1)/(2A.3) stand on both grids and the
   rest-state result does not depend on $\delta$'s value at all; but a gnomonic
   initial condition sampled at `x1v` is a point value at $\bar r$, not a cell
   average, and the gnomonic $\delta$ is the right correction only under the
   cell-average reading of the stored state.
3. **A caution on §4C.3's velocity-squared exclusion.** Its symbolic S2e finds
   that when $p$ is recovered through `cons2prim` the face defect vanishes
   through $O(h^3)$ only if the kinetic term is kept; dropped, it leaves an
   $O(h^2)$ residual with coefficients as large as $-5h^2/6$ in its draws. The
   exclusion in §4C.3 still stands for the rest state (the term is exactly
   zero) and for the linear onset problem (it is quadratic in the
   perturbation), but it is **not** negligible in general, and §4C.3 should not
   be read as unconditional. Settling it is open.

## 4C.1 The superseded rows, marked CORRECTED not deleted

| row | §4B, CORRECTED | superseded (§2A/§4A of this file) | why it was wrong |
|---|---|---|---|
| total mass | $0$ | $\sigma_c^2\,\rho'u_n'$ | $\bar\rho\bar u=\langle\rho u\rangle$ identically (1A.2); the pair double-counted |
| tracer $n$ | $\sigma_c^2\,\rho\,\partial_1u_n\,\partial_1q_n$ | $\sigma_c^2(\rho'u_n'q+\rho'q'u_n+u_n'q'\rho)$ | the two $\rho'$ pairs double-counted; the Favre pair survives |
| dry mass | $-\sum_n$ (tracer covariances), plus its own centroid | the three-factor form in $\rho,r_d,u_n$ | so that dry $+\sum_n$ tracer leaves the total-mass row with **no** covariance |
| energy | $\sigma_c^2\,\rho\,\partial_1h\,\partial_1u_n$, $h=(I+p)/\rho$ | $\sigma_c^2\,(I+p)\,\partial_1\ln(p/\rho)\,\partial_1u_n$ | §4C.2: equal for constant $\kappa$, wrong once composition varies |
| momentum | centroid in the face-local frame, $p^\star$ in flux **and** source | derived in §4A.3, not implemented | now implemented |

**Unchanged by the correction:** the centroid term $-\delta\,\partial_1F$ on every
row, the universal centroid rule (§4A.1.1), the exact hydrostatic rest-balance
proof and its numbers (§4A.3), and the telescoping statements (§4A.4). The
centroid term is a property of the *measure*, not of how the factors are
averaged, so the Favre correction does not touch it.

## 4C.2 The energy row, settled

With $\kappa\equiv(I+p)/p$ and $h=\kappa p/\rho$,

$$
\boxed{\;\rho\,\partial_1h-(I+p)\,\partial_1\ln(p/\rho)=p\,\partial_1\kappa\;}
\tag{4C.1}
$$

identically. So the two forms **agree exactly** for a dry ideal gas and differ by
$\sigma_c^2\,p\,\partial_1\kappa\,\partial_1u_n$ once the composition varies.
Quadrature (`docs/derivations/energy_row.py`, 90-point Gauss–Legendre,
$\bar r=1.4$), relative residual $h=0.1\to0.0125$:

| case | geometry | $\rho\,\partial_1h$ (§4B) | $(I+p)\partial_1\ln(p/\rho)$ |
|---|---|---|---|
| $\kappa$ const | Cartesian | 6.582e-03 → 1.038e-04, ratio 4.00 | identical to every digit |
| $\kappa$ const | curved | 4.507e-02 → 7.226e-04, ratio 4.00 | identical |
| $\kappa$ varies | Cartesian | 6.678e-03 → 1.054e-04, ratio 4.00 | 2.574e-01 → 2.525e-01, **ratio 1.00, no convergence** |
| $\kappa$ varies | curved | 4.068e-02 → 6.505e-04, ratio 4.00 | 3.724e-01 → 3.426e-01, **no convergence** |

Measured gap at $h=0.0125$: $+3.176373\times10^{-2}$ against the predicted
$\sigma_c^2\,\partial_1u\,p\,\partial_1\kappa=+3.176373\times10^{-2}$, every
printed digit; for constant $\kappa$ the gap is $3.8\times10^{-10}$, the
finite-difference noise floor, against a predicted exact zero. §4B's form is
therefore correct and strictly more general.

## 4C.3 Known and excluded

Written here for the first time; no such section existed before.

* **Velocity-squared terms -- OUT OF SCOPE, by Xi's ruling** (Slack ts
  1791538028). $E+p=\rho h+\tfrac12\rho|\mathbf u|^2$ and §A3 keeps only the
  enthalpy part, so the dropped covariances are those of the kinetic term,
  smaller by $O(\mathrm{Ma}^2)$ -- a relative $10^{-8}$ to $10^{-6}$ in #289's
  $\mathrm{Ma}\sim10^{-4}\!-\!10^{-3}$ regime. The same ruling excludes the
  $\rho u_\phi^2$ part of the geometric source `m_pp`
  (`src/coord/spherical_polar.cpp:262-265`) away from rest; it is
  velocity-bearing and so exactly zero in the rest-balance proof.
* **The momentum centroid's velocity-squared part is not mirrored in the
  geometric source -- EXCLUDED, by Xi's ruling** (Slack ts 1791540809). The
  normal-momentum face flux is $\rho u_n^2+p$ and §4B shifts the whole flux,
  but only the pressure part $p^\star$ is mirrored into the source. The
  unmirrored remainder is quadratic in velocity: exactly zero at rest, and zero
  in the linear onset problem, where it is a product of two perturbations. It is
  left as it stands.
* **$J$ -- EXCLUDED, with its bound.** $J$ is the residual metric inconsistency
  after the exact solid angle and exact radial integral landed in `0486f2c`.
  **Provenance of the bound:** the *stratification equivalent* of the centroid
  shift, measured as the "flux only" arm of
  `docs/derivations/rest_balance.py` -- relative $9.359\times10^{-8}$ on both
  the spherical and the gnomonic grid, at $\delta=3.306\times10^{-3}$ m,
  $\partial_rp=-1.641$ Pa m$^{-1}$, $p=5.518\times10^4$ Pa, i.e.
  $\delta|\partial_rp|/p=9.832\times10^{-8}$. So $\sim6\times10^{-8}$ to
  $1\times10^{-7}$ at those deck parameters. Zoey Hu's §4B independently
  estimates the pressure Hessian part at $\sim10^{-3}$ of the energy term.
  **This is a transcription-based numeric estimate, not a measurement of
  snapy:** `rest_balance.py` evaluates a transcription of the discrete
  operators from the cited `file:line`.
* **The one-sided $x_1$ pressure difference at the first and last interior
  cell -- a deliberate stencil choice, not an exclusion.** Its reason is in
  §4B: next to a cubed-sphere panel edge the $x_2$/$x_3$ face states in the
  $x_1$ ghost rows are not filled for this purpose, so no ghost row is read.
  It is harmless for the rest balance because the balance argument of §4A.3.2
  never uses the stencil -- the same scalar multiplies two identical geometric
  coefficients, so any difference cancels provided the flux and the source use
  **the same** one. Measured, interior $x_1$ cells only, x2-momentum rest
  residual relative to $|S\,p|$: base $2.392\times10^{-14}$; flux only
  $9.592\times10^{-8}$; flux and source together $2.523\times10^{-14}$, and at
  the two cells where the stencil actually is one-sided
  $2.523\times10^{-14}$ (first) and $5.574\times10^{-15}$ (last) --
  machine precision, the same order as base
  (`docs/derivations/onesided.py`).

## 4C.4 Telescoping, restated

Mass, each tracer and energy are added to the face flux, so they telescope:
total mass, total tracer mass and $E+PE$ stay exact to round-off, and at a
closed $x_2$/$x_3$ wall every scalar-row correction is **exactly** zero, because
each carries either $u_n$ or $\partial_1u_n$ and $u_n\equiv0$ there. The
momentum row telescopes as a flux but is **not** a pure divergence -- its
geometric sources are not fluxes -- so no exact momentum invariant exists in
main either, and none is created or destroyed here. Discrete angular momentum
remains open (H5).

## 4C.5 Still not established

Carried from §4A.5 (A1-A7, H5-H7), plus:

* **H8.** The Favre argument of §1A is verified at the **cell** level, from
  `cons2prim`. `src/recon/reconstruct.cpp` reconstructs the primitive array
  without re-weighting (density `:133-134`, velocity/pressure `:141-142`,
  species `:151-152`) and the $x_2$/$x_3$ reconstruction does not act along
  $x_1$, so the value at an $x_1$ index on a horizontal face is still the
  cell's Favre value -- which is where (1A.2) is needed. Reconstructing $\rho$
  and $u$ separately **along $x_2$** is a separate, unquantified inconsistency,
  transverse and higher order in $\Delta x_2$.
* **H9 (closed by 668a647).** §4B first gated the source shift on *both*
  horizontal directions carrying the correction, so with one horizontal flux
  disabled the other's flux was shifted while its source was not. The gate is
  now per direction (§4B.2), and `test_flux_covariance_seams` checks the case:
  a spherical-polar rest state with the $x_3$ flux disabled stays at rest.
* Nothing in §1A or §4C has been compiled or executed in snapy.
