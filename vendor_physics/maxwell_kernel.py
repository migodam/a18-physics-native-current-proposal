"""Memory-conscious FFT kernel for the finite A8 dyadic Maxwell grid.

It has the same lag convention and pointwise Green tensor as A6's
``fast_kernel_fft``.  The implementation intentionally constructs only six
unique symmetric tensor components and avoids a padded ``[..., 3, 3]`` real
array or a padded complex dyadic tensor.  This is an algebraic/layout change,
not an acceleration or physics claim.
"""
from __future__ import annotations

import time
from collections.abc import Callable

# Import common before NumPy/SciPy so the A6 isolated vendor route is selected.
from common import Costs  # noqa: F401
import numpy as np
from scipy.fft import fftn, next_fast_len


ProgressCallback = Callable[[int, int, float], None]


def kernel_memory_estimate(shape: tuple[int, int, int] | list[int] | np.ndarray) -> dict[str, int | list[int] | str]:
    """Payload estimate with the six retained symmetric arrays counted once."""
    shape = tuple(int(v) for v in shape)
    if len(shape) != 3 or any(v < 1 for v in shape):
        raise ValueError("shape must have three positive entries")
    pad = tuple(next_fast_len(2 * v - 1) for v in shape)
    points = int(np.prod(pad))
    real = np.dtype(float).itemsize
    complex_bytes = np.dtype(complex).itemsize
    # During one component FFT: r2/safe (two real), pref/A/B/component
    # (four complex). The six Fourier arrays persist after each temporary is
    # released.  The exact allocator/FFT workspace remains implementation
    # dependent and is deliberately not claimed as measured RSS.
    retained = 6 * points * complex_bytes
    work = 2 * points * real + 4 * points * complex_bytes
    return {"scope": "six unique symmetric Fourier components plus one-component construction payload; excludes FFT-library allocator workspace",
            "shape": list(shape), "pad": list(pad), "padded_points": points,
            "retained_unique_fourier_arrays": 6,
            "retained_fourier_bytes": retained,
            "one_component_working_bytes": work,
            "estimated_payload_peak_bytes": retained + work}


def _lag_axis(n: int, padded: int, h: float) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(padded)
    positive = index < n
    negative = index >= padded - (n - 1)
    valid = positive | negative
    lag = np.where(positive, index, index - padded).astype(float) * float(h)
    return lag, valid


def maxwell_kernel_fft(shape: tuple[int, int, int] | list[int] | np.ndarray, h: float, k: float,
                       *, progress_callback: ProgressCallback | None = None) -> tuple[list[list[np.ndarray]], tuple[int, int, int]]:
    """Return the exact finite-grid dyadic convolution kernel in Fourier space.

    ``progress_callback(i, j, elapsed_seconds)`` is called after every one of
    the six unique component FFTs in ``(00, 01, 02, 11, 12, 22)`` order.
    """
    shape = tuple(int(v) for v in shape)
    if len(shape) != 3 or any(v < 1 for v in shape) or not np.isfinite(h) or h <= 0 or not np.isfinite(k):
        raise ValueError("shape must be positive and h,k finite with h>0")
    pad = tuple(next_fast_len(2 * v - 1) for v in shape)
    axes = [_lag_axis(n, p, h) for n, p in zip(shape, pad)]
    x, y, z = (axes[0][0][:, None, None], axes[1][0][None, :, None], axes[2][0][None, None, :])
    valid = axes[0][1][:, None, None] & axes[1][1][None, :, None] & axes[2][1][None, None, :]
    r2 = x * x + y * y + z * z
    nonself = r2 > 0
    safe_r = np.sqrt(np.where(nonself, r2, 1.0))
    active = valid & nonself
    started = time.perf_counter()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pref = np.exp(1j * float(k) * safe_r) / (4.0 * np.pi * safe_r)
        A = pref * (float(k) ** 2 + 1j * float(k) / safe_r - 1.0 / r2)
        B = pref * (float(k) ** 2 + 3j * float(k) / safe_r - 3.0 / r2) / r2
    # Self and padded-wrap cells must be exact zero; calculating them with
    # safe_r only avoids a singular evaluation from entering a retained array.
    A = np.where(active, A, 0.0)
    B = np.where(active, B, 0.0)
    coordinates = (x, y, z)
    components: dict[tuple[int, int], np.ndarray] = {}
    for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        if i == j:
            component = np.ascontiguousarray(A - B * coordinates[i] * coordinates[j])
        else:
            component = np.ascontiguousarray(-B * coordinates[i] * coordinates[j])
        components[i, j] = fftn(component)
        if progress_callback is not None:
            progress_callback(i, j, time.perf_counter() - started)
    xx, xy, xz = components[0, 0], components[0, 1], components[0, 2]
    yy, yz, zz = components[1, 1], components[1, 2], components[2, 2]
    return [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], pad
