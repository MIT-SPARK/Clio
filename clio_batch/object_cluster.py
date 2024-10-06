"""Cluster objects in 3D."""
import spark_dsg as dsg
import spark_dsg.networkx as dsg_nx
import pathlib
import numpy as np
import networkx as nx
import click
import time
import clio_batch.helpers as helpers
import clio_batch.cluster_utilities as cluster_utilities
# import clio_batch.remote_visualizer as remote_visualizer

import clio_batch.ib_cluster as cluster
from clio_eval.utils import dsg_object_to_o3d
from clio_eval.evaluate_helpers import get_dsg_version

def bboxes_intersect(bbox_1, bbox_2, epsilon=0.0):
    bboxes = np.vstack((bbox_1.min, bbox_2.min, bbox_1.max, bbox_2.max))
    bboxes = bboxes - np.min(bboxes, axis=0)
    bboxes[:2, :] = bboxes[:2, :] - epsilon * (bboxes[2:, :] - bboxes[:2, :])
    bboxes[2:, :] = bboxes[2:, :] + epsilon * (bboxes[2:, :] - bboxes[:2, :])
    return (bboxes[0, :] <= bboxes[3, :]).all()\
        and (bboxes[2, :] >= bboxes[1, :]).all()


def contains_object(node, other_node, G_obj_nx):
    bbox = G_obj_nx.nodes[node]["bounding_box"]
    other_bbox = G_obj_nx.nodes[other_node]["bounding_box"]

    return bbox.is_inside(other_bbox.min) and bbox.is_inside(other_bbox.max)


def create_cluster_graphs(G_obj_nx, partition, bbox_dilation):
    nodes = [x for x in G_obj_nx]
    bboxes = [G_obj_nx.nodes[x]["bounding_box"] for x in G_obj_nx]
    if G_obj_nx.number_of_edges() == 0:
        for i in range(len(bboxes)):
            for j in range(i+1, len(bboxes)):
                # check overlap
                bbox_i = bboxes[i]
                bbox_j = bboxes[j]
                if bboxes_intersect(bbox_i, bbox_j, epsilon=bbox_dilation):
                    G_obj_nx.add_edge(nodes[i], nodes[j])

    cluster_graphs = [G_obj_nx]
    if partition:
        cluster_graphs = [G_obj_nx.subgraph(
            c).copy() for c in nx.connected_components(G_obj_nx)]

    return cluster_graphs


def cluster_objects(full_nx, cluster_nx, task_features, cluster_config):
    if cluster_nx.number_of_nodes() <= 1 or cluster_nx.number_of_edges() == 0:
        return [list(cluster_nx.nodes)]

    ib_cluster_config = cluster.ClusterIBConfig(cluster_config)
    ib_solver = cluster.ClusterIB(ib_cluster_config)

    region_features = np.array(
        [np.average(cluster_nx.nodes[x]["semantic_feature"], axis=1) for x in cluster_nx])
    full_region_features = np.array(
        [np.average(full_nx.nodes[x]["semantic_feature"], axis=1) for x in full_nx])
    ib_solver.setup_py_x(region_features, task_features)
    ib_solver.update_delta_as_part(full_region_features, task_features)
    ib_solver.initialize_nx_graph(cluster_nx)
    print("initial number of objects", ib_solver.nx_graph.number_of_nodes())
    cluster_assignments = ib_solver.find_clusters()
    print("number of objects after clustering",
          ib_solver.nx_graph.number_of_nodes())
    return cluster_assignments


def update_dsg(G_dsg, cluster_assignments, task_features, threshold):
    segments_layer = G_dsg.get_layer(dsg.DsgLayers.SEGMENTS)
    node_clusters = {}
    objects_layer = G_dsg.get_layer(dsg.DsgLayers.OBJECTS)
    id_to_remove = []
    for n in objects_layer.nodes:
        id_to_remove.append(n.id.value)
    for id in id_to_remove:
        G_dsg.remove_node(id)

    for idx, cluster_assignment in enumerate(cluster_assignments):
        # get corresponding dsg indices
        dsg_cluster = []
        for c in cluster_assignment:
            dsg_cluster.append(c)
        node_clusters[idx] = [G_dsg.get_node(n) for n in dsg_cluster]

    for new_node_id in node_clusters:
        merged_attrs = cluster_utilities.merge_object_attributes(
            node_clusters[new_node_id])
        assert merged_attrs.semantic_feature.shape[1] == 1
        sim = helpers.compute_cosine_sim(
            task_features, merged_attrs.semantic_feature, None, False)

        if sim < threshold:
            continue

        node_id = dsg.NodeSymbol('O', new_node_id).value
        G_dsg.add_node(dsg.DsgLayers.OBJECTS, node_id, merged_attrs)
        # print('num in cluster', len(node_clusters[new_node_id]))
        for child in node_clusters[new_node_id]:
            G_dsg.insert_edge(node_id, child.id.value)
    return G_dsg


def get_bboxes(dsg, layer=dsg.DsgLayers.OBJECTS):
    object_layer = dsg.get_layer(layer)
    boxes = []
    for node in object_layer.nodes:
        obj_mesh, oriented_bbox = dsg_object_to_o3d(node)
        boxes.append(oriented_bbox)
    return boxes


def cluster_3d(scene_graph_path, experiment_file, output_folder, cluster_config, model_name,
               thres, partition, bbox_dilation, recompute_edges=True):
    print(scene_graph_path)
    G = dsg.DynamicSceneGraph.load(str(scene_graph_path))
    G_nx = dsg_nx.layer_to_networkx(G.get_layer(dsg.DsgLayers.SEGMENTS))
    # remove all edges
    if recompute_edges:
        G_nx.remove_edges_from(list(G_nx.edges()))
    click.secho("loaded graph", fg="green")

    print(f"loading {model_name}...")
    cliphandler = helpers.ClipHandler(model_name)
    print(f"finished loading {model_name}")


    list_of_tasks = helpers.parse_tasks_from_yaml(experiment_file)
    tasks = "*".join(list_of_tasks)
    print(tasks)

    start_t = time.time()

    # create IB graph
    nx_graphs = create_cluster_graphs(G_nx, partition, bbox_dilation)
    print(f"partitioned into {len(nx_graphs)} subgraphs")

    task_phrases = [x for x in tasks.split("*") if x != ""]
    task_features = cliphandler.get_text_clip_features(task_phrases)

    cluster_assignments = []
    for nx_graph in nx_graphs:
        cluster_assignments += cluster_objects(
            G_nx, nx_graph, task_features, cluster_config)

    G = update_dsg(G, cluster_assignments, task_features, thres)
    print(scene_graph_path)
    print("clustering took", time.time()-start_t)
    return G


@click.command()
@click.argument("scene_graph_path", type=click.Path(exists=True))
@click.argument("task_file", type=str)
@click.argument("output_folder", type=str)
@click.argument("cluster_config", type=str)
@click.option("--model-name", "-m", default="ViT-L/14")
@click.option("--thres", "-t", type=float, default=0.23)
@click.option("--partition", "-p", is_flag=False)
@click.option("--dilation", "-d", type=float, default=0.0)
@click.option("--recompute-edges", is_flag=False)
def main(
    scene_graph_path,
    task_file,
    output_folder,
    cluster_config,
    model_name,
    thres,
    partition,
    dilation,
    recompute_edges,
):
    """Set up clustering and run."""
    # url = "tcp://127.0.0.1:8001"
    # viz = remote_visualizer.RemoteVisualizer(url=url)

    click.secho(f"loading graph from '{scene_graph_path}'...", fg="green")
    scene_graph_path = pathlib.Path(scene_graph_path).expanduser().absolute()

    clustered_G = cluster_3d(str(scene_graph_path), task_file, output_folder,
                             cluster_config, model_name, thres, partition, dilation, recompute_edges)
    # viz.visualize(clustered_G)

    final_num_objs = clustered_G.get_layer(dsg.DsgLayers.PLACES).num_nodes()
    print("{} objects in the final scene graph.".format(final_num_objs))

    output_dsg = pathlib.Path(output_folder) / "clio_dsg.json"
    clustered_G.save(output_dsg)

    # Keep output DSG version the same as input version.
    version = get_dsg_version(scene_graph_path)
    print(version)
    with open(output_dsg) as file:
        d = json.load(file)
    version = [int(i) for i in version]
    d['SPARK_ORIGIN_header'] = {'version':{'major':version[0], 'minor':version[1], 'patch':version[2]}, 'project_name':'main'}

    with open(output_dsg, 'w') as file:
        json.dump(d, file) 

if __name__ == "__main__":
    main()
