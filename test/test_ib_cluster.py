"""Unit tests for ib clustering."""

import clio_batch.ib_cluster as ib_cluster
import pytest
import os
import numpy as np


def initialize_cluster_module():
    cwd = os.path.abspath(os.path.dirname(__file__))
    yaml_path = os.path.join(os.path.join(cwd, "resources"), "test_cluster_config.yaml")
    cluster_config = ib_cluster.ClusterIBConfig(yaml_path)
    return ib_cluster.ClusterIB(cluster_config)


def test_setup_py_x():
    """Test py_x initialization for top-k equal to 2."""
    region_features = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
    task_features = np.array([[1, 0, 0], [0, 1, 0]])
    clusterer = initialize_cluster_module()
    clusterer.setup_py_x(region_features, task_features)

    assert (3,) == clusterer.px.shape
    assert (3, 3) == clusterer.py_x.shape
    assert pytest.approx(0.0909, 1e-3) == clusterer.py_x[0, 0]
    assert pytest.approx(0.9091, 1e-3) == clusterer.py_x[1, 0]
    assert pytest.approx(0) == clusterer.py_x[2, 0]
    assert pytest.approx(0) == clusterer.py_x[0, 1]
    assert pytest.approx(0.3333, 1e-3) == clusterer.py_x[1, 1]
    assert pytest.approx(0.6667, 1e-3) == clusterer.py_x[2, 1]
    assert pytest.approx(0.0909, 1e-3) == clusterer.py_x[0, 2]
    assert pytest.approx(0) == clusterer.py_x[1, 2]
    assert pytest.approx(0.9091, 1e-3) == clusterer.py_x[2, 2]


def test_get_clusters_from_pc_x():
    """Test that cluster extraction works as expected."""
    pc_x = np.array([[0.3, 0, 1], [0.2, 1, 0], [0.5, 0, 0]])
    clusterer = initialize_cluster_module()
    clusters = clusterer.get_clusters_from_pc_x(pc_x)
    assert 3 == len(clusters)

    pc_x = np.array([[0.5, 0, 1], [0.2, 1, 0], [0.3, 0, 0]])
    clusters = clusterer.get_clusters_from_pc_x(pc_x)
    assert 2 == len(clusters)
    assert 2 == len(clusters[0])
