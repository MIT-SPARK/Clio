import clio_eval.utils as eval_utils
import clio_batch.helpers as helpers
import numpy as np
import spark_dsg as sdsg
import distinctipy
import copy
import json
import open3d as o3d

def get_dsg_version(dsg_file):
    with open(dsg_file) as file:
        d = json.load(file)
        if 'SPARK_DSG_header' not in d:
            version = ['1', '0', '0']
            return version
        elif "SPARK_ORIGIN_header" in d:
            version_dictionary = d['SPARK_ORIGIN_header']['version']
        else:
            version_dictionary = d['SPARK_DSG_header']['version']
        version = [str(version_dictionary['major']), str(version_dictionary['minor']), str(version_dictionary['patch'])]
    return version

def results_from_files(dsg_file, task_yaml, clip_model, clustered, visualize=False, thresh=0.0):
    print(dsg_file)
    layer = sdsg.DsgLayers.SEGMENTS
    if clustered:
        layer = sdsg.DsgLayers.OBJECTS

    dsg = sdsg.DynamicSceneGraph.load(dsg_file)
    offset_to_lower_corner = False
    version = ['' + i for i in get_dsg_version(dsg_file)]
    print('dsg version: ', version)
    if version == ['1', '0', '0']:
        offset_to_lower_corner = True
    est_objects = eval_utils.get_objects_from_dsg(dsg, offset_to_lower_corner, layer=layer)
    gt_data = eval_utils.GtData(task_yaml)
    if thresh > 0:
        est_objects = prune_objects(est_objects, clip_model, gt_data, thresh)
    return compute_results(est_objects, gt_data, clip_model, visualize=visualize)


def prune_objects(est_objects, clip_model, gt_data, thres):
    cliphandler = helpers.ClipHandler(clip_model)
    task_list = list(gt_data.task_objects.keys())
    task_features = cliphandler.get_text_clip_features(task_list)

    pruned_est_objects = []
    for object in est_objects:
        t = []
        for feature in task_features:
            temp = object.compute_similarity(feature)
            t.append(temp)
        max_sim = max(t)
        if max_sim > thres:
            pruned_est_objects.append(object)
    return pruned_est_objects


def results_from_cg(cg_file, task_yaml, clip_model, visualize=False, thres=0.0):
    est_objects = get_objects_from_cg(cg_file)
    gt_data = eval_utils.GtData(task_yaml)
    if thres > 0:
        est_objects = prune_objects(est_objects, clip_model, gt_data, thres)
    return compute_results(est_objects, gt_data, clip_model, visualize=visualize)


def compute_results(est_objects, gt_data, clip_model, min_sim_ratio=0.9, visualize=False):
    weak_recall, strict_recall, avg_iou = compute_recall_and_iou(
        est_objects, gt_data, clip_model, visualize)
    weak_precision, strict_precision = compute_precision(
        est_objects, gt_data, clip_model, min_sim_ratio, visualize)

    return eval_utils.Results(len(est_objects), weak_recall, strict_recall, avg_iou, weak_precision, strict_precision)


def generate_gt_bboxes_viz(gt_bboxes, color):
    viz_gt_bboxes = []
    for gt_bbox in gt_bboxes:
        outer = copy.deepcopy(gt_bbox)
        inner = copy.deepcopy(gt_bbox)
        inner.extent = 0.95 * inner.extent
        outer.color = color
        inner.color = [0, 0, 0]  # clip inner to black
        viz_gt_bboxes = viz_gt_bboxes + [inner, outer]
    return viz_gt_bboxes


def generate_est_bboxes_viz(est_objs, color, add_mesh=True):
    viz_est_bboxes = []
    for est_obj in est_objs:
        est_bbox = copy.deepcopy(est_obj.oriented_bbox)
        est_bbox.color = color
        viz_est_bboxes.append(est_bbox)
        # add object mesh
        if add_mesh:
            viz_est_bboxes.append(est_obj.o3d_mesh)
    return viz_est_bboxes


def compute_recall_and_iou(est_objects, gt_data, clip_model, visualize):
    # est_objects is a list of EvalObjects
    # gt_data is of GtData type
    cliphandler = helpers.ClipHandler(clip_model)
    task_list = list(gt_data.task_objects.keys())
    task_features = cliphandler.get_text_clip_features(task_list)
    # Find associated est bboxes
    total_gt_objects = 0
    total_strict_matches = 0
    total_weak_matches = 0
    sum_iou = 0
    viz_gt_bboxes = []
    viz_est_bboxes = []
    est_boxes = []
    colors = distinctipy.get_colors(len(task_list))
    for i in range(len(task_list)):
        gt_bboxes = gt_data.task_objects[task_list[i]].copy()

        if visualize:
            viz_gt_bboxes.extend(generate_gt_bboxes_viz(gt_bboxes, colors[i]))

        num_objs = len(gt_bboxes)
        rel_objects = eval_utils.get_k_relevant_objects(
            est_objects, task_features[i], num_objs)
        total_gt_objects += len(gt_bboxes)
        if visualize:
            viz_est_bboxes.extend(
                generate_est_bboxes_viz(rel_objects, colors[i]))
            est_boxes.append(rel_objects[0].oriented_bbox)

        while len(gt_bboxes) > 0 and len(rel_objects) > 0:
            best_iou = 0
            best_gt_est_pair = None
            for gt_id, gt_bbox in enumerate(gt_bboxes):
                for est_id, est_obj in enumerate(rel_objects):
                    iou = est_obj.compute_iou(gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_est_pair = (gt_id, est_id)

            if best_iou == 0:
                break
            
            sum_iou += best_iou
            best_gt_bbox = gt_bboxes[best_gt_est_pair[0]]
            best_est_obj = rel_objects[best_gt_est_pair[1]]

            if best_est_obj.contains_centroid(best_gt_bbox):
                total_weak_matches += 1
                if best_est_obj.centroid_contained_in(best_gt_bbox):
                    total_strict_matches += 1

            # delete
            del gt_bboxes[best_gt_est_pair[0]]
            del rel_objects[best_gt_est_pair[1]]

    if visualize:
        print("Visualizing recall computation...")
        o3d.visualization.draw_geometries(viz_gt_bboxes + viz_est_bboxes)
        merged_line_set = o3d.geometry.LineSet()
        for est_box in est_boxes:
            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(est_box)
            merged_line_set += line_set
        o3d.io.write_line_set('/home/dominic/bounding_boxes.ply', merged_line_set)

    return total_weak_matches / total_gt_objects, total_strict_matches / total_gt_objects, sum_iou / total_gt_objects


def compute_precision(est_objects, gt_data, clip_model, min_sim_ratio, visualize):
    # est_objects is a list of EvalObjects
    # gt_data is of GtData type
    cliphandler = helpers.ClipHandler(clip_model)

    # first sort all est objects into the different tasks
    task_list = list(gt_data.task_objects.keys())
    est_task_objects = {task: [] for task in task_list}
    max_sims = {task: 0 for task in task_list}
    task_features = cliphandler.get_text_clip_features(task_list)

    total_est_objects = 0
    total_strict_matches = 0
    total_weak_matches = 0
    for est_obj in est_objects:
        sims = helpers.compute_sim_to_tasks(task_features, est_obj.feature)
        task_idx = np.argmax(sims)
        est_task_objects[task_list[task_idx]].append(est_obj)
        if sims[task_idx] > max_sims[task_list[task_idx]]:
            max_sims[task_list[task_idx]] = sims[task_idx]

    viz_gt_bboxes = []
    viz_est_bboxes = []
    colors = distinctipy.get_colors(len(task_list))
    for i in range(len(task_list)):
        gt_bboxes = gt_data.task_objects[task_list[i]].copy()
        est_objects = est_task_objects[task_list[i]]
        task_feature = task_features[i]
        rel_objects = [obj for obj in est_objects if obj.compute_similarity(
            task_feature) > min_sim_ratio*max_sims[task_list[i]]]
        if visualize:
            viz_est_bboxes.extend(
                generate_est_bboxes_viz(rel_objects, colors[i]))
        if visualize:
            viz_gt_bboxes.extend(
                generate_gt_bboxes_viz(gt_bboxes, colors[i]))
        total_est_objects += len(rel_objects)
        while len(gt_bboxes) > 0 and len(rel_objects) > 0:
            best_iou = 0
            best_gt_est_pair = None
            for gt_id, gt_bbox in enumerate(gt_bboxes):
                for est_id, est_obj in enumerate(rel_objects):
                    iou = est_obj.compute_iou(gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_est_pair = (gt_id, est_id)

            if best_iou == 0:
                break

            best_gt_bbox = gt_bboxes[best_gt_est_pair[0]]
            best_est_obj = rel_objects[best_gt_est_pair[1]]
            if best_est_obj.contains_centroid(best_gt_bbox):
                total_weak_matches += 1
                if best_est_obj.centroid_contained_in(best_gt_bbox):
                    total_strict_matches += 1

            # delete
            del gt_bboxes[best_gt_est_pair[0]]
            del rel_objects[best_gt_est_pair[1]]

    if visualize:
        print("Visualizing precision computation...")
        o3d.visualization.draw_geometries(viz_gt_bboxes + viz_est_bboxes)

    if total_est_objects == 0:
        return 0, 0

    return total_weak_matches / total_est_objects, total_strict_matches / total_est_objects
