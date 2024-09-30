"""Test that IoU works as expected."""

import numpy as np
import open3d as o3d
from clio_eval.iou import compute_iou_oriented_bbox
import pytest

def compute_iou_new(bbox_1, bbox_2):
    return compute_iou_oriented_bbox(bbox_1, bbox_2)

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

def visualize(bbox1, bbox2):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add OBBs to the visualizer
    vis.add_geometry(bbox1)
    vis.add_geometry(bbox2)

    # Set view control
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=-90)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

def test_boxes_disjoint_rotated():
    """Test that two oriented boxes have 0 IoU when (strictly) separated."""
    bbox1 = o3d.geometry.OrientedBoundingBox(
        np.array([0.0, 0.0, 0.0]), _get_rotation(45), np.array([1.0, 1.0, 1.0])
    )
    bbox2 = o3d.geometry.OrientedBoundingBox(
        np.array([0.9, 0.9, 0.0]), _get_rotation(0), np.array([1.0, 1.0, 1.0])
    )

    visualize(bbox1, bbox2)
    est_iou = compute_iou_new(bbox1, bbox2)
    print('est_iou', est_iou)
    assert est_iou == pytest.approx(0.0)

#test_boxes_disjoint_rotated()
