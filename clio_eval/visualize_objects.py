import clio_eval.evaluate_helpers as eval_helpers
import clio_eval.utils as eval_utils
from clio_eval.viz_scene import visualize_objects_on_mesh
import spark_dsg as sdsg
import open3d as o3d
import numpy as np
import argparse
import os
import yaml

def main():
    parser = argparse.ArgumentParser(
        description="Visualize. ")
    parser.add_argument('dsg_file', type=str, default=None)
    parser.add_argument('tasks_yaml', type=str, default=None)
    parser.add_argument('mesh_file', type=str, default=None)
    parser.add_argument('scene_name', type=str, default=None, help="office, apartment, cubicle")
    args = parser.parse_args()

    layer = sdsg.DsgLayers.OBJECTS

    dsg = sdsg.DynamicSceneGraph.load(args.dsg_file)

    offset_to_lower_corner = False
    version = ['' + i for i in eval_helpers.get_dsg_version(args.dsg_file)]
    print('dsg version: ', version)
    if version == ['1', '0', '0']:
        offset_to_lower_corner = True
    est_objects = eval_utils.get_objects_from_dsg(dsg, offset_to_lower_corner, layer=layer)

    print("num objects: ", len(est_objects))

    # colors to visualize estimated and ground truth bounding boxes
    green = (0.0, 1.0, 0.0)
    black = (0.0, 0.0, 0.0)

    viz_est_bboxes = []
    for i in range(len(est_objects)):
        viz_est_bboxes += eval_helpers.generate_est_bboxes_viz(
            [est_objects[i]], green, add_mesh=False)

    viz_gt_bboxes = []
    gt_data = eval_utils.GtData(args.tasks_yaml).task_objects.copy()
    for task in gt_data:
        bboxes = gt_data[task]
        viz_gt_bboxes.extend(eval_helpers.generate_gt_bboxes_viz(bboxes, black))

    # To visualize the COLMAP mesh we have to tranform it to metric scale in the correct coordinate frame.
    # Get correct transform from yaml file for the scene.
    cwd = os.path.abspath(os.path.dirname(__file__))
    scene_transforms_path = os.path.abspath(os.path.join(cwd, "scene_transforms.yaml"))
    transform, scale = eval_utils.get_transform_from_yaml(scene_transforms_path, args.scene_name)

    visualize_objects_on_mesh(args.mesh_file, viz_est_bboxes, viz_gt_bboxes, transform, scale, mesh_size=0.01)


if __name__ == "__main__":
    main()

