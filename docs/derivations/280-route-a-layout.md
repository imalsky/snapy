# Stretched x1 reconstruction layout

The Cartesian stage 1 path builds one coefficient row per output face from
this block's global face slice. Each row uses the five cells bounded by
`x1f[m..m+5]`. `cm[m]` targets face `m+2`; `cp[m]` targets face `m+3`.

WENO5 rows have 48 entries: three five-entry candidate rows, six five-entry
smoothness rows, and three linear weights. The smoothness rows retain the
kernel's `13/12` and `1/4` factors. cp5 rows have five entries and combine
the WENO5 candidates with their linear weights.

The CPU iterator carries the coefficient row as a zero-stride third operand;
the iterator's offsets identify the output face under chunking. CUDA launches
one block per line and uses `threadIdx.x` as the face index. The row remains
in global memory so the tile contains only state values.

## 4.7 Supported reconstruction dimensions and types

Only weno5 and cp5 accept per-face x1 coefficients. A stretched deck refuses
other reconstruction types. The per-face path requires the last state
dimension, x1; x2 and x3 continue through the unmodified uniform path.
