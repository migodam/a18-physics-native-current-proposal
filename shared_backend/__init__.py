"""A18 common online capability and selected-action verifier.

Teacher/full-J arrays are never members of FrozenState. This API preserves the
original real material coordinates and illumination-major data packing.
"""
from .core import (BackendProtocol, FrozenState, SeedBundle, ActionSpec,
                   CostReceipt, VerifiedExchange, OperatorBackend, pack, unpack,
                   endpoint_from_maps, directional_check, array_hash)

__all__ = ["BackendProtocol", "FrozenState", "SeedBundle", "ActionSpec",
           "CostReceipt", "VerifiedExchange", "OperatorBackend"]
