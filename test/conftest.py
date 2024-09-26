"""Fixtures for unit tests."""

import pytest
import pathlib


@pytest.fixture()
def resource_dir():
    """Get a path to the resource directory for tests."""
    return pathlib.Path(__file__).absolute().parent / "resources"


@pytest.fixture()
def suppress_torch():
    """Disable torch scientific notation."""
    import torch

    torch.set_printoptions(sci_mode=False, precision=3)
    yield
    torch.set_printoptions("default")


@pytest.fixture()
def cleanup_torch():
    """Cleanup torch memory."""
    import torch

    yield
    torch.cuda.empty_cache()


@pytest.fixture()
def image_factory():
    """Get images from resource directory."""
    import cv2
    import torch

    resource_path = pathlib.Path(__file__).absolute().parent / "resources"

    def factory(name):
        """Get image."""
        img_path = resource_path / name
        img = cv2.imread(str(img_path))
        assert img is not None
        # convert to RGB order and torch
        return torch.from_numpy(img[:, :, ::-1].copy())

    return factory
