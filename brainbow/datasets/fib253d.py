"""
FIB-25 (3D) dataset for *Drosophila* medulla neuron segmentation.

FIB-25 (Takemura et al. 2015) is the Janelia FlyEM 7-column *Drosophila*
medulla FIB-SEM reconstruction, served as Neuroglancer ``precomputed``
volumes from a public Google bucket.  Crops are fetched with
``scripts/download_fib253d.py`` and written as HDF5 (dataset key ``main``,
axis order ``[Z, Y, X]``) -- byte-for-byte the layout
:class:`brainbow.datasets.MICRONSDataset` consumes.

This leaf is therefore a thin **metadata override** of
:class:`MICRONSDataset`: the loading / patching / normalisation logic is
shared verbatim, and only the citation, resolution (8x8x8 nm
*isotropic*, vs MICrONS' 8x8x40 nm anisotropic), and label names differ.
"""

from typing import Dict, List

from brainbow.datasets.microns import MICRONSDataset


class FIB253DDataset(MICRONSDataset):
    """FIB-25 FIB-SEM dataset (8x8x8 nm isotropic *Drosophila* medulla).

    Identical loading + patching to :class:`MICRONSDataset` (HDF5 crops in
    ``[Z, Y, X]`` order, per-volume ``{vol, seg, root, find_boundaries}``
    specs); only the dataset metadata differs.  Download crops with
    ``scripts/download_fib253d.py``.
    """

    _paper = (
        "Takemura, S. et al. (2015). Synaptic circuits and their variations "
        "within different columns in the visual system of Drosophila. "
        "PNAS, 112(44), 13711-13716. doi:10.1073/pnas.1509820112"
    )
    # FIB-SEM is isotropic: 8 nm in z, y, and x (unlike MICrONS' 8x8x40 nm).
    _resolution: Dict[str, float] = {"x": 8.0, "y": 8.0, "z": 8.0}
    _labels_base: List[str] = ["background", "neuron"]


__all__ = ["FIB253DDataset"]
