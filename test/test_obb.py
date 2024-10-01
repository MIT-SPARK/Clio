"""Test that IoU works as expected."""

import numpy as np
import open3d as o3d
import pytest
from clio_eval.iou import compute_iou_oriented_bbox


def _get_rotation(angle):
    theta = np.radians(angle)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return R


def test_boxes_disjoint_rotated():
    """Test that two oriented boxes have 0 IoU when (strictly) separated."""
    bbox1 = o3d.geometry.OrientedBoundingBox(
        np.array([0.0, 0.0, 0.0]), _get_rotation(45), np.array([1.0, 1.0, 1.0])
    )
    bbox2 = o3d.geometry.OrientedBoundingBox(
        np.array([0.9, 0.9, 0.0]), _get_rotation(0), np.array([1.0, 1.0, 1.0])
    )

    est_iou = compute_iou_oriented_bbox(bbox1, bbox2)
    assert est_iou == pytest.approx(0.0)
