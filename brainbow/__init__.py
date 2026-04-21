"""
Brainbow: a PyTorch Lightning infrastructure for connectomics research
with spatially-coloured (brainbow-style) instance segmentation targets.

Provides:
- MONAI-compatible dataset classes with a standardised interface
- Preprocessors for common data formats (TIFF, HDF5, NRRD, NIfTI)
- Cosmos-Transfer3D model wrapper (DiT + VAE backbone) for volumetric
  segmentation, together with a Vista3D reference implementation
- A ``BoundaryLoss`` that turns each connected-component instance into
  a 16-channel per-voxel target:
    * 1 channel   -- *raw*, the raw image intensity at that voxel
    * 3 channels  -- RGB of the instance's bounding-box *min* location
    * 3 channels  -- RGB of the instance's centroid (*avg* location)
    * 3 channels  -- RGB of the instance's bounding-box *max* location
    * 6 channels  -- *aff*, binary face-affinity to the 6 neighbours
                    in Z-Y-X order (T, B, U, D, L, R) with SAME /
                    replicate padding; supervised via soft-Dice on the
                    sigmoid of the logits.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="The cuda.cudart module is deprecated",
    category=FutureWarning,
)

__version__ = "0.1.0"

from brainbow.datasets import (
    CircuitDataset,
    SNEMI3DDataset,
    MICRONSDataset,
    NeuronsDataset,
)
from brainbow.preprocessors import (
    BasePreprocessor,
    TIFFPreprocessor,
    HDF5Preprocessor,
    NRRDPreprocessor,
    NFTYPreprocessor,
)
from brainbow.datamodules import (
    CircuitDataModule,
    SNEMI3DDataModule,
    MICRONSDataModule,
    NeuronsDataModule,
)
from brainbow.losses import (
    BoundaryLoss,
    CombinedLoss,
    GeometryLoss,
    InstanceLoss,
    SemanticLoss,
    build_boundary_target,
)
from brainbow.modules import (
    BaseCircuitModule,
    BaseCosmosModule,
    BaseVistaModule,
    CosmosTransfer3DModule,
    Vista3DModule,
)

__all__ = [
    # Data
    "CircuitDataset",
    "SNEMI3DDataset",
    "MICRONSDataset",
    "NeuronsDataset",
    "BasePreprocessor",
    "TIFFPreprocessor",
    "HDF5Preprocessor",
    "NRRDPreprocessor",
    "NFTYPreprocessor",
    "CircuitDataModule",
    "SNEMI3DDataModule",
    "MICRONSDataModule",
    "NeuronsDataModule",
    # Losses
    "BoundaryLoss",
    "CombinedLoss",
    "GeometryLoss",
    "InstanceLoss",
    "SemanticLoss",
    "build_boundary_target",
    # Modules
    "BaseCircuitModule",
    "BaseCosmosModule",
    "BaseVistaModule",
    "CosmosTransfer3DModule",
    "Vista3DModule",
]
