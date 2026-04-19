"""
Tests for dataset base classes and common functionality.
"""

import pytest
import numpy as np
from typing import Any, Dict, List, Union

from brainbow.datasets import CircuitDataset


class TestCircuitDataset:
    """Tests for CircuitDataset abstract class."""

    def test_abstract_class_cannot_instantiate(self) -> None:
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CircuitDataset(root_dir=".")  # type: ignore[abstract]

    def test_volumes_param_accepted(self) -> None:
        """Test that volumes parameter is stored correctly."""

        class MinimalDataset(CircuitDataset):
            @property
            def paper(self) -> str:
                return "Test"

            @property
            def resolution(self) -> Dict[str, float]:
                return {"x": 1.0, "y": 1.0, "z": 1.0}

            @property
            def labels(self) -> List[str]:
                return ["bg", "fg"]

            @property
            def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
                return {"vol": "test.h5"}

            def _prepare_data(self) -> List[Dict[str, Any]]:
                return [{"image": np.zeros((10, 10)), "label": np.zeros((10, 10))}]

        vols = [{"vol": "test", "seg": "test"}]
        dataset = MinimalDataset(root_dir=".", volumes=vols, cache_rate=0.0)
        assert dataset.volumes == vols

    def test_required_properties(self) -> None:
        """Test that subclasses must implement required properties."""

        class MissingPaper(CircuitDataset):
            @property
            def resolution(self) -> Dict[str, float]:
                return {"x": 1.0, "y": 1.0, "z": 1.0}

            @property
            def labels(self) -> List[str]:
                return ["bg"]

            @property
            def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
                return {}

            def _prepare_data(self) -> List[Dict[str, Any]]:
                return []

        with pytest.raises(TypeError):
            MissingPaper(root_dir=".")  # type: ignore[abstract]

    def test_resolution_tuple_method(self) -> None:
        """Test get_resolution_tuple method."""

        class TestDataset(CircuitDataset):
            @property
            def paper(self) -> str:
                return "Test"

            @property
            def resolution(self) -> Dict[str, float]:
                return {"x": 4.0, "y": 4.0, "z": 40.0}

            @property
            def labels(self) -> List[str]:
                return ["bg"]

            @property
            def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
                return {}

            def _prepare_data(self) -> List[Dict[str, Any]]:
                return [{"image": np.zeros((10, 10))}]

        dataset = TestDataset(root_dir=".", cache_rate=0.0)
        res_tuple = dataset.get_resolution_tuple()
        assert res_tuple == (40.0, 4.0, 4.0)

    def test_anisotropy_factor(self) -> None:
        """Test get_anisotropy_factor method."""

        class TestDataset(CircuitDataset):
            @property
            def paper(self) -> str:
                return "Test"

            @property
            def resolution(self) -> Dict[str, float]:
                return {"x": 6.0, "y": 6.0, "z": 30.0}

            @property
            def labels(self) -> List[str]:
                return ["bg"]

            @property
            def data_files(self) -> Dict[str, Union[str, np.ndarray]]:
                return {}

            def _prepare_data(self) -> List[Dict[str, Any]]:
                return [{"image": np.zeros((10, 10))}]

        dataset = TestDataset(root_dir=".", cache_rate=0.0)
        anisotropy = dataset.get_anisotropy_factor()
        assert anisotropy == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
