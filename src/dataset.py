"""Backward-compatible import surface for the original training scripts."""
try:  # Repository-root import: ``from src.dataset import ...``.
    from .data import ISBIDataset, CephalometricDataLoader
except ImportError:  # Notebook import after adding ``src`` to sys.path.
    from data import ISBIDataset, CephalometricDataLoader

__all__ = ["ISBIDataset", "CephalometricDataLoader"]
