"""Fixtures for unit tests."""

import pytest
import pathlib


@pytest.fixture()
def resources():
    """Get a path to the resource directory for tests."""
    return pathlib.Path(__file__).absolute().parent / "resources"
