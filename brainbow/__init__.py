"""
Brainbow: a PyTorch Lightning infrastructure for connectomics research
with spatially-coloured (brainbow-style) instance segmentation targets.

Provides:
- MONAI-compatible dataset classes with a standardised interface
- Preprocessors for common data formats (TIFF, HDF5, NRRD, NIfTI)
- Cosmos-Transfer3D model wrapper (DiT + VAE backbone) for volumetric
  segmentation, together with a Vista3D reference implementation
- A ``BrainbowLoss`` that turns each connected-component instance into
  a 10-channel per-voxel target:
    * 1 channel   -- *rawval*, the raw image intensity at that voxel
    * 3 channels  -- RGB of the instance's bounding-box *minloc*
    * 3 channels  -- RGB of the instance's centroid / *avgloc*
    * 3 channels  -- RGB of the instance's bounding-box *maxloc*
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
    BrainbowLoss,
    CombinedLoss,
    GeometryLoss,
    InstanceLoss,
    SemanticLoss,
    build_brainbow_target,
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
    "BrainbowLoss",
    "CombinedLoss",
    "GeometryLoss",
    "InstanceLoss",
    "SemanticLoss",
    "build_brainbow_target",
    # Modules
    "BaseCircuitModule",
    "BaseCosmosModule",
    "BaseVistaModule",
    "CosmosTransfer3DModule",
    "Vista3DModule",
]
