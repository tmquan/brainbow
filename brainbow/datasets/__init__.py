"""
Dataset classes for connectomics research.

All datasets inherit from CircuitDataset and implement:
- paper: Reference/citation metadata
- resolution: Voxel/spatial resolution specification
- labels: List of segmentation class labels
- data_files: Dictionary with volume and segmentation paths/arrays
"""

from brainbow.datasets.base import CircuitDataset
from brainbow.datasets.lazy import LazyVolDataset
from brainbow.datasets.snemi3d import SNEMI3DDataset
from brainbow.datasets.microns import MICRONSDataset
from brainbow.datasets.neurons import NeuronsDataset

__all__ = [
    "CircuitDataset",
    "LazyVolDataset",
    "SNEMI3DDataset",
    "MICRONSDataset",
    "NeuronsDataset",
]
