"""Unit tests for eval utils."""

import clio_eval.utils as eval_utils
import pytest
import open3d as o3d
import numpy as np


def _generate_test_eval_objs():
    bbox_1 = o3d.geometry.OrientedBoundingBox()
    bbox_1.center = [0, 0, 0]
    bbox_1.extent = [1, 1, 1]
    obj_1 = eval_utils.EvalObjects(
        np.array([0, 0, 1]), o3d.geometry.TriangleMesh(), bbox_1
    )

    bbox_2 = o3d.geometry.OrientedBoundingBox()
    bbox_2.center = [0, 0, 0]
    bbox_2.extent = [2, 2, 2]
    obj_2 = eval_utils.EvalObjects(
        np.array([0, 1, 0]), o3d.geometry.TriangleMesh(), bbox_2
    )

    return obj_1, obj_2


def test_compute_similarity():
    """Test that cosine similarity works as expected."""
    obj_1, obj_2 = _generate_test_eval_objs()

    assert 1.0 == obj_1.compute_similarity(obj_1.feature)
    assert 1.0 == obj_2.compute_similarity(obj_2.feature)
    assert 0.0 == obj_1.compute_similarity(obj_2.feature)


def test_compute_iou():
    """Test that IoU works as expected."""
    obj_1, obj_2 = _generate_test_eval_objs()

    assert 1.0 == pytest.approx(obj_1.compute_iou(obj_1.oriented_bbox))
    assert 1 / 8 == pytest.approx(obj_1.compute_iou(obj_2.oriented_bbox))
    assert 1 / 8 == pytest.approx(obj_2.compute_iou(obj_1.oriented_bbox))

    bbox_test = o3d.geometry.OrientedBoundingBox()
    bbox_test.extent = [1, 1, 1]

    bbox_test.center = [1.1, 1, 1]  # avoids qhull issues
    assert 0.0 == obj_1.compute_iou(bbox_test)

    bbox_test.center = [1, 1, 1]
    assert 1 / 71 == pytest.approx(obj_2.compute_iou(bbox_test))
