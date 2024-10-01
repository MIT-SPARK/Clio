"""Unit tests for cluster helpers."""

import clio_batch.helpers as helpers
import numpy as np


def test_compute_cosine_sim():
    """Test compute cosine sim."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 0, 1])
    v3 = np.array([-1, 0, 0])
    assert 1 == helpers.compute_cosine_sim(v1, v1)
    assert 0 == helpers.compute_cosine_sim(v1, v2)
    assert -1 == helpers.compute_cosine_sim(v1, v3)


def test_compute_cosine_sim_multi_single():
    """Test compute cosine sim with single and multiple columns."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0]])
    sims = helpers.compute_cosine_sim(v1, v2)
    assert sims.shape == (1, 3)
    assert 1 == sims[0, 0]
    assert 0 == sims[0, 1]
    assert -1 == sims[0, 2]


def test_compute_cosine_sim_multi_multi():
    """Test compute cosine sim with multiple and multiple columns."""
    # In this case we take the max wrt v1
    v1 = np.array([[1, 0, 0], [0, 0, 1]])
    v2 = np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0]])
    sims = helpers.compute_cosine_sim(v1, v2)
    assert sims.shape == (1, 3)
    assert 1 == sims[0, 0]
    assert 1 == sims[0, 1]
    assert 0 == sims[0, 2]


def test_compute_sim_to_tasks():
    """Test compute cosine sim to multiple tasks."""
    # In this case we preserve all similarities
    v1 = np.array([[1, 0, 0], [0, 0, 1]])
    v2 = np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0]])
    sims = helpers.compute_sim_to_tasks(v1, v2)
    assert sims.shape == (2, 3)
    assert 1 == sims[0, 0]
    assert 0 == sims[0, 1]
    assert -1 == sims[0, 2]
    assert 0 == sims[1, 0]
    assert 1 == sims[1, 1]
    assert 0 == sims[1, 2]


def test_parse_tasks_from_yaml(resources):
    """Test compute cosine sim to multiple tasks."""
    yaml_path = resources / "test_tasks.yaml"
    prompts = helpers.parse_tasks_from_yaml(yaml_path)
    assert 3 == len(prompts)
    assert "bring me a pillow" == prompts[0]
    assert "clean toaster" == prompts[1]
    assert "find deck of cards" == prompts[2]
