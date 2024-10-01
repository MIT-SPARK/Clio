"""Unit tests for mask refinement code."""

import clio_batch.information_metrics as metrics
import pytest
import numpy as np


def test_compute_shannon_entropy():
    """Test that Shannon entropy computation is correct."""
    p1 = np.array([1.0])
    assert 0 == metrics.shannon_entropy(p1)

    p2 = np.array([0.5, 0.5])
    assert -np.log2(0.5) == metrics.shannon_entropy(p2)


def test_compute_js_divergence():
    """Test that Jensen-Shannon divergence is correct."""
    pc = np.array([0.5, 0.5])
    py_c = 0.5 * np.ones([2, 2])
    assert 0 == metrics.js_divergence(py_c, pc)

    pc = np.array([1, 0])
    py_c = 0.5 * np.ones([2, 2])
    assert 0 == metrics.js_divergence(py_c, pc)

    pc = np.array([0.5, 0.5])
    py_c = np.eye(2) + 1e-8
    assert -np.log2(0.5) == pytest.approx(metrics.js_divergence(py_c, pc))


def test_compute_mutual_info():
    """Test that mutual information implementation is correct."""
    px = np.array([0.5, 0.5])
    pc = np.array([0.5, 0.5])
    pc_x = np.eye(2)
    assert -np.log2(0.5) == metrics.mutual_information(px, pc, pc_x)

    px = np.array([1, 0])
    pc = np.array([0.5, 0.5])
    pc_x = 0.5 * np.ones([2, 2])
    assert 0 == metrics.mutual_information(px, pc, pc_x)

    px = np.array([1, 0])
    pc = np.array([0.5, 0.5])
    pc_x = np.eye(2)
    assert -np.log2(0.5) == metrics.mutual_information(px, pc, pc_x)
