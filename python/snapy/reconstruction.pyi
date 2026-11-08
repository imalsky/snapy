"""
Stub file for snapy.reconstruction module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Reconstruction
class InterpOptions:
    """
    Interpolation options.

    This class manages interpolation parameters for reconstruction.
    """

    @overload
    def __init__(self) -> None:
        """Initialize InterpOptions with default values."""
        ...

    @overload
    def __init__(self, type: str) -> None:
        """
        Initialize InterpOptions with type.

        Args:
            type: Interpolation type
        """
        ...

    def __repr__(self) -> str: ...

    @overload
    def type(self) -> str:
        """Get interpolation type."""
        ...

    @overload
    def type(self, value: str) -> "InterpOptions":
        """Set interpolation type."""
        ...

    @overload
    def scale(self) -> bool:
        """Get scaling flag."""
        ...

    @overload
    def scale(self, value: bool) -> "InterpOptions":
        """Set scaling flag."""
        ...

class ReconstructOptions:
    """
    Reconstruction options.

    This class manages reconstruction parameters.
    """

    def __init__(self) -> None:
        """Initialize ReconstructOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def shock(self) -> bool:
        """Get shock detection flag."""
        ...

    @overload
    def shock(self, value: bool) -> "ReconstructOptions":
        """Set shock detection flag."""
        ...

    @overload
    def interp(self) -> InterpOptions:
        """Get interpolation options."""
        ...

    @overload
    def interp(self, value: InterpOptions) -> "ReconstructOptions":
        """Set interpolation options."""
        ...

class Reconstruct:
    """
    Spatial reconstruction implementation.

    This module handles high-order spatial reconstruction.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: ReconstructOptions) -> None:
        """
        Construct a Reconstruct module.

        Args:
            options: Reconstruction configuration options
        """
        ...

    def __repr__(self) -> str: ...

    options: ReconstructOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def set_x1_faces(self, x1f: torch.Tensor) -> None:
        """
        Install route A per-face x1 coefficients from a face vector (#280).

        Args:
            x1f: strictly increasing face vector, nc1 + 1 entries; face k is
                the lower face of cell k.

        The x1 sweep then RUNS these rows (readable as the buffers
        "interp1.cm_x1", "interp1.cp_x1", "interp2.cm_x1", "interp2.cp_x1",
        which exist only after this call), x2/x3 are refused, and so is a
        module whose type is not weno5 or cp5 (note Sec. 4.7, bc29594f).
        """
        ...

def weno5_x1_coeffs(x1f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-face WENO5 coefficient rows for a face vector (#280 route A).

    Args:
        x1f: strictly increasing face vector with at least 6 entries.

    Returns:
        (cm, cp), each of shape (x1f.numel() - 5, 48), float64.  Row m belongs
        to the five cells bounded by x1f[m:m+6]; cm targets face x1f[m+2] (the
        IRT state) and cp targets face x1f[m+3] (the ILT state).  Columns 0-44
        are the nine 5-entry rows c1..c9 the kernel walks, columns 45-47 are
        the linear weights d0, d1, d2.
    """
    ...

def cp5_x1_coeffs(x1f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-face cp5 coefficient rows for a face vector (#280 route A).

    Args:
        x1f: strictly increasing face vector with at least 6 entries.

    Returns:
        (cm, cp), each of shape (x1f.numel() - 5, 5), float64, with the same
        face convention as weno5_x1_coeffs.
    """
    ...
