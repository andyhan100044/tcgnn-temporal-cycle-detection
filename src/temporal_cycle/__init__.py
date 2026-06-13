"""temporal_cycle: Temporal Cycle Detection algorithms and TC-GNN model."""
from .indexing import TemporalIndex, build_index, Edge

__all__ = ["TemporalIndex", "build_index", "Edge"]


def __getattr__(name):
    """Lazy import for submodules to allow partial-package testing."""
    import importlib
    if name in {"temporal_dfs", "temporal_sketch", "tc_gnn", "losses"}:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")