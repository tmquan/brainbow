"""
Brainbow: a PyTorch Lightning infrastructure for connectomics research
with spatially-coloured (brainbow-style) instance segmentation targets.

Provides:
- MONAI-compatible dataset classes with a standardised interface
- Preprocessors for common data formats (TIFF, HDF5, NRRD, NIfTI)
- Cosmos-Transfer3D model wrapper (DiT + VAE backbone) for volumetric
  segmentation, together with a Vista3D reference implementation
- A unified 30-channel head supervised by ``CombinedLoss``:
  raw(1), sem(1), dir(3), cov(6), avg(3), emb(16), plus derived
  12-direction affinity losses on avg and emb.
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
from brainbow.losses import CombinedLoss, HEAD_CHANNELS, HEAD_LAYOUT, slice_head
from brainbow.models import (
    BaseModel,
    CosmosTransfer3DWrapper,
    Vista3DWrapper,
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
    "CombinedLoss",
    "HEAD_CHANNELS",
    "HEAD_LAYOUT",
    "slice_head",
    # Models (backbone wrappers)
    "BaseModel",
    "CosmosTransfer3DWrapper",
    "Vista3DWrapper",
    # Modules (Lightning)
    "BaseCircuitModule",
    "BaseCosmosModule",
    "BaseVistaModule",
    "CosmosTransfer3DModule",
    "Vista3DModule",
]
