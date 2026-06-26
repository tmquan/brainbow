"""
CREMI (3D) dataset for *Drosophila* neuron segmentation.

CREMI (Circuit Reconstruction from Electron Microscopy Images, MICCAI
2016) ships three ssTEM training volumes (samples A / B / C) of adult
*Drosophila* brain at **4 x 4 x 40 nm** with dense neuron-id labels.
The native distribution packs raw + labels into a single nested ``.hdf``
(``volumes/raw`` + ``volumes/labels/neuron_ids``), which ``LazyVolDataset``
cannot read directly.  ``scripts/download_cremi3d.py`` therefore converts
each sample into the brainbow convention -- separate
``cremi3d_sample_<X>_volume.h5`` / ``..._segmentation.h5`` (dataset key
``main``, axis order ``[Z, Y, X]``) -- exactly what
:class:`brainbow.datasets.MICRONSDataset` consumes.

This leaf is thus a thin **metadata override** of :class:`MICRONSDataset`
(loading / patching / normalisation shared verbatim); only the citation,
resolution (4x4x40 nm), and label names differ.
"""

from typing import Dict, List

from brainbow.datasets.microns import MICRONSDataset


class CREMI3DDataset(MICRONSDataset):
    """CREMI ssTEM dataset (4x4x40 nm anisotropic *Drosophila* brain).

    Identical loading + patching to :class:`MICRONSDataset` (HDF5 crops in
    ``[Z, Y, X]`` order, per-volume ``{vol, seg, root, find_boundaries}``
    specs); only the dataset metadata differs.  Convert the official CREMI
    ``.hdf`` files with ``scripts/download_cremi3d.py`` first.
    """

    _paper = (
        "Funke, J., Saalfeld, S., Bock, D., Turaga, S., Perlman, E. (2016). "
        "CREMI: MICCAI Challenge on Circuit Reconstruction from Electron "
        "Microscopy Images. https://cremi.org/"
    )
    # ssTEM: 4 nm in-plane (x, y), 40 nm section thickness (z).
    _resolution: Dict[str, float] = {"x": 4.0, "y": 4.0, "z": 40.0}
    _labels_base: List[str] = ["background", "neuron"]


__all__ = ["CREMI3DDataset"]
