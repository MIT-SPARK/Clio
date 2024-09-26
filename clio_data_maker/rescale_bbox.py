# Transform from Colmap frame to world frame at metric scale.
import argparse
import yaml
import numpy as np
import open3d as o3d
import clio_eval.evaluate_helpers as eval_helpers
from scipy.spatial.transform import Rotation as R

def create_obb_from_dict(bbox_dict, transform_mat):
    # extract box pose
    quat = bbox_dict['rotation']
    w, x, y, z = quat['w'], quat['x'], quat['y'], quat['z']
    quat = [x, y, z, w]
    rot = R.from_quat(quat).as_matrix().astype(np.float64)
    center = np.array(bbox_dict['center'])
    box_pose = np.eye(4)
    box_pose[0:3,0:3] = rot
    box_pose[0:3,3] = center

    # apply transform to box pose
    box_tranformed_pose = transform_mat @ box_pose

    # repackage bounding box into dictionary
    box_tranformed_rot = box_tranformed_pose[0:3,0:3]
    box_tranformed_center = box_tranformed_pose[0:3,3].tolist()
    quat = R.from_matrix(box_tranformed_rot).as_quat().astype(np.float64)
    x, y, z, w = [float(i) for i in quat]
    print(x,y,z,w)
    bbox_dict_transformed = {
        'center': box_tranformed_center,
        'extents': bbox_dict['extents'],
        'rotation': {'w':w, 'x':x, 'y':y, 'z':z}
    }

    return bbox_dict_transformed


def rescale(detection, scale):
        return {'center':[scale * i for i in detection['center']],
                'extents':[scale * i for i in detection['extents']],
                'rotation':detection['rotation']}

def main():
    parser = argparse.ArgumentParser(description="Rescale Coordinates in yaml. ")

    parser.add_argument('--yaml_in', type=str, default=None)
    parser.add_argument('--yaml_out', type=str, default=None)
    parser.add_argument('--scene_name', type=str, default=None, help="office,cubicle,apartment")

    args = parser.parse_args()

    transform_mat, scale = eval_utils.get_transform_from_yaml(scene_transforms_path, args.scene_name)

    with open(args.yaml_in) as f:
        old_data = yaml.safe_load(f)

    scaled_dict = dict()
    for task, detections in old_data.items():
        scaled_dict[task] = []
        for detection in detections:
            scaled_box = rescale(detection, scale)
            transformed_box = (create_obb_from_dict(scaled_box. transform_mat))
            scaled_dict[task].append(transformed_box)


    with open(args.yaml_out, 'w') as f:
        yaml.dump(scaled_dict, f)

if __name__ == "__main__":
    main()
