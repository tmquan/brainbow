"""
Brainbow: a PyTorch Lightning infrastructure for connectomics research
with spatially-coloured (brainbow-style) instance segmentation targets.

Provides:
- MONAI-compatible dataset classes with a standardised interface
- Preprocessors for common data formats (TIFF, HDF5, NRRD, NIfTI)
- Cosmos-Transfer3D model wrapper (DiT + VAE backbone) for volumetric
  segmentation, together with a Vista3D reference implementation
- A ``BoundaryLoss`` whose head emits 10 per-voxel channels:
    * 1 channel   -- *raw*, the raw image intensity at that voxel
    * 3 channels  -- RGB of the instance's centroid (*avg* location)
    * 6 channels  -- *aff_pred*, the model's direct face-affinity
                    prediction in Z-Y-X order (T, B, U, D, L, R) with
                    SAME / replicate padding; supervised via BCE +
                    soft-Dice + soft-Jaccard.
  In addition, the loss derives a soft 6-face affinity from the
  predicted avgloc via
  ``aff_avg[c] = exp(-tau * sum_i |avg[i] - shift(avg[i], dir_c)|)``
  and supervises that derived signal against the same binary aff
  target (``weight_aff_pred`` / ``weight_aff_avg`` scale the two
  paths separately).
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
    "BoundaryLoss",
    "CombinedLoss",
    "GeometryLoss",
    "InstanceLoss",
    "SemanticLoss",
    "build_boundary_target",
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
