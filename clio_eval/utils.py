import spark_dsg as sdsg
import yaml
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as sprot

import clio_batch.helpers as helpers
from clio_eval.iou import compute_iou_oriented_bbox


class EvalObjects:
    def __init__(self, feature, o3d_mesh, oriented_bbox):
        self.feature = feature
        self.o3d_mesh = o3d_mesh
        self.oriented_bbox = oriented_bbox

    def compute_similarity(self, other_feature):
        return helpers.compute_cosine_sim(other_feature, self.feature)

    def compute_iou(self, other_oriented_bbox):
        return compute_iou_oriented_bbox(self.oriented_bbox, other_oriented_bbox)

    def contains_centroid(self, other_oriented_bbox):
        # Extract centroids
        centroid = np.asarray(self.oriented_bbox.get_center())
        other_centroid = np.asarray(
            other_oriented_bbox.get_center())

        # Extract half-lengths (extent) of the bounding boxes
        extent = np.asarray(self.oriented_bbox.extent)

        # Compute bounds for each box
        bounds = np.concatenate([centroid - extent, centroid + extent])

        return is_point_inside_box(other_centroid, bounds)

    def centroid_contained_in(self, other_oriented_bbox):
        # Extract centroids
        other_centroid = np.asarray(other_oriented_bbox.get_center())
        centroid = np.asarray(
            self.oriented_bbox.get_center())

        # Extract half-lengths (extent) of the bounding boxes
        other_extent = np.asarray(other_oriented_bbox.extent)

        # Compute bounds for each box
        other_bounds = np.concatenate(
            [other_centroid - other_extent, other_centroid + other_extent])

        return is_point_inside_box(centroid, other_bounds)


class ExperimentGroup:
    def __init__(self, param):
        self.task_yaml = param["task_yaml"]
        self.experiments = []
        for ab in param["experiments"]:
            self.experiments.append([ab["name"]])


class Ablations:
    def __init__(self):
        self.experiments = {}  # name (str) :  setup (ExperimentGroup)
        self.env = {}
        self.eval = {}

    def parse(self, yaml_file):
        with open(yaml_file, "r") as stream:
            params = yaml.safe_load(stream)
            self.num_trials = int(params["num_trials"])
            self.env["log_path"] = params["log_path"]

            for group in params["datasets"]:
                eg = ExperimentGroup(group)
                self.experiments[group["name"]] = eg

            self.eval["semantic_threshold"] = params["semantic_threshold"]

class GtData:
    def __init__(self, gt_yaml):
        gt_data = yaml.safe_load(open(gt_yaml))
        self.task_objects = {}
        for task, gt_detections in yaml.safe_load(open(gt_yaml)).items():
            self.task_objects[task] = []
            for gt_detection in gt_detections:
                gt_bbox = get_open3d_bbox_from_dict(gt_detection)
                self.task_objects[task].append(gt_bbox)


class Results:
    def __init__(self, num_objects, weak_recall, strict_recall, avg_iou, weak_precision, strict_precision, f1=None, tpf=-1):
        self.num_objects = num_objects
        self.weak_recall = weak_recall
        self.strict_recall = strict_recall
        self.avg_iou = avg_iou
        self.weak_precision = weak_precision
        self.strict_precision = strict_precision
        self.f1 = f1
        if self.f1 is None:
            if self.weak_precision == 0 and self.weak_recall == 0:
                self.f1 = 0
            else:
                self.f1 = 2 * self.weak_precision * self.weak_recall / \
                    (self.weak_precision + self.weak_recall)
        self.tpf = tpf

    def to_dict(self, decimals=3):
        output = {}
        output["avg_iou"] = round(self.avg_iou, decimals)
        output["strict_recall"] = round(self.strict_recall, decimals)
        output["weak_recall"] = round(self.weak_recall, decimals)
        output["strict_precision"] = round(self.strict_precision, decimals)
        output["weak_precision"] = round(self.weak_precision, decimals)
        output["num_objects"] = round(self.num_objects)
        output["f1"] = round(self.f1, decimals)
        output["tpf"] = round(self.tpf, decimals)
        return output

    def to_list(self, fields, decimals=3):
        mapping = self.to_dict(decimals=decimals)
        return [mapping[f] for f in fields]

    def __str__(self):
        return "Results:\n\
        Num Objects: {}\n\
        Recall (weak): {}\n\
        Recall (strict): {}\n\
        Avg IOU: {}\n\
        Precision (weak): {}\n\
        Precision (strict): {}\n\
        F1: {}\n\
        TPF(s): {}".format(
            self.num_objects, self.weak_recall, self.strict_recall, self.avg_iou,
            self.weak_precision, self.strict_precision, self.f1, self.tpf)

def min_from_box_position(node, offset_to_lower_corner = False):
    # Older DSG version has mesh defined wrt lower corner of bbox.
    # New version is wrt center. Apply offset depending on version.
    if offset_to_lower_corner:
        offset = (node.bounding_box.dimensions / 2)[:, np.newaxis]
        return node.position[:, np.newaxis] - offset
    else:
        return node.position[:, np.newaxis]

def max_from_box_position(node, offset_to_lower_corner = False):
    # Older DSG version has mesh defined wrt lower corner of bbox.
    # New version is wrt center. Apply offset depending on version.
    if offset_to_lower_corner:
        offset = (node.bounding_box.dimensions / 2)[:, np.newaxis]
        return node.position[:, np.newaxis] + offset
    else:
        return node.position[:, np.newaxis]

def is_point_inside_box(point, bounds):
    return (bounds[0] <= point[0] <= bounds[3] and
            bounds[1] <= point[1] <= bounds[4] and
            bounds[2] <= point[2] <= bounds[5])


def obb_to_corners(obb):
    # Extract the eight corners of the OBB
    corners = np.asarray(obb.get_box_points())
    return corners


def get_open3d_bbox_from_dict(bbox_dict):
    center = np.array(bbox_dict['center'], dtype=np.float64).reshape(3, 1)
    extents = np.array(bbox_dict['extents'], dtype=np.float64).reshape(3, 1)
    quat = bbox_dict['rotation']
    if type(quat) is not list:
        w, x, y, z = quat['w'], quat['x'], quat['y'], quat['z']
        quat = [x, y, z, w]
    else:
        quat.append(quat.pop(0))
    rot = sprot.from_quat(quat).as_matrix().astype(np.float64)

    return o3d.geometry.OrientedBoundingBox(center, rot, extents)


def dsg_object_to_o3d(object_node, offset_to_lower_corner):
    world_t_object = min_from_box_position(object_node.attributes, offset_to_lower_corner)
    obj_faces = object_node.attributes.mesh().get_faces()
    obj_verts = object_node.attributes.mesh().get_vertices()

    object_vertices = obj_verts[:3, :] + world_t_object
    object_vertices = o3d.utility.Vector3dVector(np.transpose(object_vertices))

    oriented_object_bbox = None
    if obj_verts.shape[1] > 3:
        try:
            oriented_object_bbox = o3d.geometry.PointCloud(
                object_vertices).get_minimal_oriented_bounding_box()
        except RuntimeError:
            pass

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = object_vertices
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.transpose(obj_verts[3:, :]))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(
        np.transpose(obj_faces))

    return o3d_mesh, oriented_object_bbox

def get_objects_from_dsg(dsg, offset_to_lower_corner, visualize=False, layer=sdsg.DsgLayers.OBJECTS):
    # return list of bboxes and list of features
    eval_objects = []
    vis_meshes = []
    vis_bboxes = []
    object_layer = dsg.get_layer(layer)
    for node in object_layer.nodes:
        obj_mesh, oriented_bbox = dsg_object_to_o3d(node, offset_to_lower_corner)

        if oriented_bbox is None:
            continue

        if visualize:
            vis_meshes.append(obj_mesh)
            oriented_bbox.color = [1, 0, 0]
            vis_bboxes.append(oriented_bbox)

        avg_feature = np.mean(np.asarray(
            node.attributes.semantic_feature), axis=1)
        eval_objects.append(EvalObjects(avg_feature, obj_mesh, oriented_bbox))

    if len(vis_meshes) > 0:
        o3d.visualization.draw_geometries(vis_meshes + vis_bboxes)
    return eval_objects


def get_k_relevant_objects(eval_objects, feature, k):
    if k > len(eval_objects):
        k = len(eval_objects)
    t = []
    for object in eval_objects:
        temp = object.compute_similarity(feature)
        t.append(temp)
        # print('t', temp)
    # print('max:', max(t))
    sorted_objs = sorted(
        eval_objects, key=lambda x: x.compute_similarity(feature), reverse=True)
    return sorted_objs[:k]

def get_transform_from_yaml(transforms_file, scene_name):
    with open(transforms_file, "r") as file:
        data = yaml.safe_load(file)

    T = np.array(data[scene_name]['T'])
    scale = float(data[scene_name]['scale'])

    return T, scale