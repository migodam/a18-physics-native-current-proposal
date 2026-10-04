"""Dense reference algebra for A18 proposal research.

This module deliberately uses small explicit matrices for tests. It is NOT a
Maxwell solver, NN implementation, numerical certificate, or deployment backend.
Production code must replace matrix products with metered, adjoint-consistent
A17 backend actions and import the original acceptance policy unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import numpy as np
from numpy.typing import NDArray

CArray = NDArray[np.complex128]
RArray = NDArray[np.float64]


def adj(a: CArray) -> CArray:
    return a.conj().T


def project(v: CArray, x: CArray) -> CArray:
    """Euclidean projection; use only after declared current-metric whitening."""
    return x - v @ (adj(v) @ x)


def orth_columns(a: CArray, base: CArray | None = None,
                 rtol: float = 1e-12) -> CArray:
    """Ordered two-pass MGS, preserving the caller's degree/layer order."""
    a = np.asarray(a, dtype=np.complex128)
    if a.ndim != 2:
        raise ValueError('Expected a matrix of columns.')
    n = a.shape[0]
    accepted: list[CArray] = []
    qbase = np.empty((n, 0), complex) if base is None else base
    for col in a.T:
        scale = float(np.linalg.norm(col))
        if not np.isfinite(scale) or scale == 0.0:
            continue
        w = col.copy()
        for _ in range(2):
            w = project(qbase, w)
            if accepted:
                q = np.column_stack(accepted)
                w = project(q, w)
        nw = float(np.linalg.norm(w))
        if nw > rtol * scale:
            accepted.append(w / nw)
    return np.column_stack(accepted) if accepted else np.empty((n, 0), complex)


@dataclass
class SchurFeedback:
    """Retained-core elimination in whitened current coordinates."""
    L: CArray
    U: CArray
    relative_cutoff: float = 1e-10

    def __post_init__(self) -> None:
        self.L = np.asarray(self.L, dtype=np.complex128)
        self.U = np.asarray(self.U, dtype=np.complex128)
        n = self.L.shape[0]
        if self.L.shape != (n, n) or self.U.shape[0] != n:
            raise ValueError('Incompatible operator and retained basis shapes.')
        if np.linalg.norm(adj(self.U) @ self.U - np.eye(self.U.shape[1])) > 1e-10:
            raise ValueError('Retained basis must be orthonormal after whitening.')
        self.LU = self.L @ self.U
        self.D = adj(self.U) @ self.LU
        if self.D.size:
            s = np.linalg.svd(self.D, compute_uv=False)
            if s[-1] <= self.relative_cutoff * s[0]:
                raise np.linalg.LinAlgError('Unstable retained core: no Schur proposal.')

    def R(self, x: CArray) -> CArray:
        if not self.D.size:
            return np.zeros_like(x)
        return self.U @ np.linalg.solve(self.D, adj(self.U) @ x)

    def dress(self, x: CArray) -> CArray:
        x = project(self.U, x)
        return x - self.R(self.L @ x)

    def A(self, x: CArray) -> CArray:
        x = project(self.U, x)
        lx = self.L @ x
        return project(self.U, lx - self.LU @ np.linalg.solve(self.D, adj(self.U) @ lx))

    def A_adj(self, x: CArray) -> CArray:
        x = project(self.U, x)
        h = np.linalg.solve(adj(self.D), adj(self.LU) @ x)
        return project(self.U, adj(self.L) @ (x - self.U @ h))

    def F(self, x: CArray) -> CArray:
        return project(self.U, x) - self.A(x)

    def F_adj(self, x: CArray) -> CArray:
        return project(self.U, x) - self.A_adj(x)

    def effective_injection(self, x: CArray) -> CArray:
        return project(self.U, x - self.L @ self.R(x))

    def observation_seed(self, S: CArray, residual: CArray) -> CArray:
        v = adj(S) @ residual
        h = np.linalg.solve(adj(self.D), adj(self.U) @ v)
        return project(self.U, v - adj(self.L) @ (self.U @ h))


def galerkin_complex(L: CArray, S: CArray, B: CArray, W: CArray,
                     relative_cutoff: float = 1e-10) -> CArray:
    """One illumination; direct endpoint works even with a singular shared core."""
    A = adj(W) @ L @ W
    sv = np.linalg.svd(A, compute_uv=False)
    if not len(sv) or sv[-1] <= relative_cutoff * sv[0]:
        raise np.linalg.LinAlgError('Unstable Galerkin endpoint.')
    return S @ W @ np.linalg.solve(A, adj(W) @ B)


def realify_jacobians(jacobians: Sequence[CArray]) -> RArray:
    """Shared REAL material variables; per-illumination [Re; Im] packing."""
    if not jacobians:
        raise ValueError('At least one illumination is required.')
    return np.vstack([part for J in jacobians for part in (J.real, J.imag)])


def realify_data(data: Sequence[CArray]) -> RArray:
    return np.concatenate([part for y in data for part in (y.real, y.imag)])


def frozen_step(J: RArray, r: RArray, damping: RArray, ell: RArray) -> RArray:
    H = J.T @ J + damping
    return -np.linalg.solve(H, J.T @ r + ell)


def phi(J: RArray, r: RArray, damping: RArray, ell: RArray, s: RArray) -> float:
    y = r + J @ s
    return float(0.5 * y @ y + 0.5 * s @ damping @ s + ell @ s)


def finite_gain(Jfull: RArray, r: RArray, damping: RArray, ell: RArray,
                old: RArray, new: RArray) -> float:
    d = new - old
    u, z = r + Jfull @ old, Jfull @ d
    return float(-u @ z - (ell + damping @ old) @ d
                 - 0.5 * (z @ z + d @ damping @ d))


def compressed_endpoint(AZ: CArray, OZ: CArray, NZ: CArray, E: CArray) -> CArray:
    """Teacher-only exact projected endpoint identity, not an online feature."""
    A = adj(E) @ AZ @ E
    return OZ @ E @ np.linalg.solve(A, adj(E) @ NZ)


def single_chain(op: Callable[[CArray], CArray], b: CArray, depth: int,
                 rtol: float = 1e-12) -> list[CArray]:
    scale = np.linalg.norm(b)
    if scale == 0 or not np.isfinite(scale):
        return []
    chain = [b / scale]
    for _ in range(depth):
        w0 = op(chain[-1])
        w = w0.copy()
        for _ in range(2):
            w = project(np.column_stack(chain), w)
        if np.linalg.norm(w) <= rtol * max(np.linalg.norm(w0), np.finfo(float).tiny):
            break
        chain.append(w / np.linalg.norm(w))
    return chain


def joint_flag(feedback: SchurFeedback, seeds: Sequence[tuple[str, CArray]],
               depth: int = 2) -> tuple[CArray, list[int]]:
    """Independent stable chains, then degree-ordered joint orthogonalization."""
    chains = [single_chain(feedback.F_adj if family == 'O' else feedback.F,
                           project(feedback.U, b), depth)
              for family, b in seeds]
    n = feedback.L.shape[0]
    Q = np.empty((n, 0), complex)
    layers: list[int] = []
    for degree in range(depth + 1):
        cols = [chain[degree] for chain in chains if len(chain) > degree]
        if not cols:
            continue
        qd = orth_columns(np.column_stack(cols), np.column_stack([feedback.U, Q]))
        Q = np.column_stack([Q, qd])
        layers.extend([degree] * qd.shape[1])
    return Q, layers


def order_penalty(a: CArray, layers: Sequence[int]) -> float:
    if len(a) != len(layers):
        raise ValueError('Coefficient and layer dimensions must agree.')
    weights = (np.asarray(layers) + 1.0) ** 2 - 1.0
    return float(np.sum(weights * np.abs(a) ** 2))


def projective_distance(a: CArray, b: CArray) -> float:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(max(0.0, 1.0 - abs(np.vdot(a, b)) ** 2))


def expected_best_gain(gains: RArray, probabilities: RArray, k: int) -> float:
    """Exact iid-with-replacement expected max positive gain; not top-K utility."""
    g, p = np.asarray(gains, float), np.asarray(probabilities, float)
    if k < 1 or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError('Require k>=1 and a probability vector.')
    g = np.maximum(g, 0.0)
    levels = np.unique(np.r_[0.0, g])
    value = 0.0
    for lo, hi in zip(levels[:-1], levels[1:]):
        tail = float(p[g >= hi].sum())
        value += (hi - lo) * (1.0 - (1.0 - tail) ** k)
    return float(value)
