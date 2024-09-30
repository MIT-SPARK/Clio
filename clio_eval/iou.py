# Get IoU for 3D Oriented Bounding Boxes

import os
import sys
import numpy as np
import scipy.spatial as sp

cwd = os.path.abspath(os.path.dirname(__file__))
objectron_path = os.path.abspath(os.path.join(cwd, "../thirdparty/Objectron/"))
sys.path.insert(0, objectron_path)

from objectron.dataset.iou import IoU
from objectron.dataset.box import Box

def get_box_verticies(obb):
    center = obb.get_center()
    verticies = np.take(obb.get_box_points(), [0,3,2,5,1,6,7,4], axis=0)
    stack_points =  np.vstack((center, verticies))
    return stack_points

def compute_iou_oriented_bbox(bbox_1, bbox_2):
  box1 = Box(get_box_verticies(bbox_1))
  box2 = Box(get_box_verticies(bbox_2))
  iou = IoU(box1, box2)
  return iou.iou()