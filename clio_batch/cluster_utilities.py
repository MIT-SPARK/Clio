"""Various utility classes for clustering."""

import spark_dsg as dsg
import clio_batch.helpers as helpers
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

from clio_eval.utils import min_from_box_position, max_from_box_position

def get_last_feature(features):
    """Get the average of features in a matrix of features."""
    x = np.squeeze(features)
    if len(x.shape) == 1:
        return x
    return np.average(x,axis=1)

def _get_field_average(attrs, f, g=np.mean):
    X = np.array([f(x) for x in attrs])
    return g(X, axis=0)

def _get_mesh(attrs):
    mesh_faces = attrs.mesh().get_faces()
    mesh_vertices = attrs.mesh().get_vertices()
    mesh_vertices[:3, :] += min_from_box_position(attrs)
    mesh_vertices = np.transpose(mesh_vertices)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices[:, :3])
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_vertices[:, 3:])
    mesh.triangles = o3d.utility.Vector3iVector(np.transpose(mesh_faces))
    return mesh

def merge_object_attributes(node_list):
    """Combine object segment attributes into the final object node."""
    all_attrs = [node.attributes for node in node_list]

    merged = dsg.KhronosObjectAttributes()
    # NodeAttributes fields
    merged.position = _get_field_average(all_attrs, lambda x: x.position)
    merged.last_update_time_ns = max([x.last_update_time_ns for x in all_attrs])
    # SemanticNodeAttributes fields
    
    bbox_min = _get_field_average(all_attrs, lambda x: min_from_box_position(x), np.min)
    bbox_max = _get_field_average(all_attrs, lambda x: max_from_box_position(x), np.max)
    center = (bbox_min + bbox_max) / 2.0
    extent = bbox_max - bbox_min
    merged.bounding_box = dsg.BoundingBox(extent, center)
    merged.semantic_feature = _get_field_average(
        all_attrs, lambda x: get_last_feature(x.semantic_feature)
    )
    # KhronosObjectAttributes fields
    combined_mesh = o3d.geometry.TriangleMesh()
    for attrs in all_attrs:
        combined_mesh += _get_mesh(attrs)

    merged_mesh_vertices = np.zeros((6, np.asarray(combined_mesh.vertices).shape[0]))
    merged_mesh_vertices[:3, :] = (
        np.transpose(np.asarray(combined_mesh.vertices)) - bbox_min
    )
    merged_mesh_vertices[3:, :] = np.transpose(np.asarray(combined_mesh.vertex_colors))
    merged_mesh_faces = np.transpose(np.asarray(combined_mesh.triangles))
    merged.mesh().set_vertices(merged_mesh_vertices)
    merged.mesh().set_faces(merged_mesh_faces)

    return merged


def rewrite_objects(G, cluster_ws):
    """Rewrite object layer using clustering results."""
    G_new = G.clone()
    for n in G.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        G_new.remove_node(n.id.value)

    for cluster_idx, cluster in enumerate(cluster_ws.cluster_nodes):
        segments = [G.get_node(n) for n in cluster]
        attrs = merge_object_attributes(segments)
        node_id = dsg.NodeSymbol("O", cluster_idx).value
        G_new.add_node(dsg.DsgLayers.OBJECTS, node_id, attrs)

    return G_new


class Score:
    """Wrapper around cosine similiarity."""

    def __init__(
        self,
        tasks,
        model_name="ViT-B/32",
        use_lerf_loss=False,
        cannonical_phrases="object*stuff*things*texture",
    ):
        """Set up a score function."""
        self.use_lerf_loss = use_lerf_loss

        print(f"loading {model_name}...")
        clip_handler = helpers.ClipHandler(model_name)
        print(f"finished loading {model_name}")

        task_phrases = [x for x in tasks.split("*") if x != ""]
        self.tasks = task_phrases
        N_tasks = len(task_phrases)
        print(f"computing {N_tasks} task embeddings...")
        self.x_t = clip_handler.get_text_clip_features(task_phrases)

        if not use_lerf_loss:
            self.x_c = None
        else:
            cannonical_phrases = [x for x in cannonical_phrases.split("*") if x != ""]
            N_cannonical = len(cannonical_phrases)
            print(f"computing {N_cannonical} cannonical embeddings...")
            self.x_c = clip_handler.get_text_clip_features(cannonical_phrases)

        del clip_handler
        print("finished computing embeddings")

    def get_scores(self, X):
        """Get score matrix."""
        return helpers.compute_sim_to_tasks(self.x_t, X, self.x_c, self.use_lerf_loss)

    def get_best_tasks(self, X, show=False):
        """Get best task index."""
        scores = helpers.compute_sim_to_tasks(self.x_t, X, self.x_c, self.use_lerf_loss)
        if show:
            fig, ax = plt.subplots()
            im = ax.imshow(scores, aspect="auto", interpolation="none")
            ax.figure.colorbar(im, ax=ax)
            plt.show()

        return np.argmax(scores, axis=0)

    def draw_scores(self, ax, X):
        """Get best task index."""
        scores = helpers.compute_sim_to_tasks(self.x_t, X, self.x_c, self.use_lerf_loss)
        im = ax.imshow(scores, aspect="auto", interpolation="none")
        ax.figure.colorbar(im, ax=ax)

    def __call__(self, X):
        """Score features."""
        s = helpers.compute_cosine_sim(self.x_t, X, self.x_c, self.use_lerf_loss)
        if X.shape[0] > 1:
            return np.squeeze(s)
        else:
            return s


class Workspace:
    """Class to hold useful datamembers."""

    def __init__(self, G):
        """Initialize workspace with features and scores."""
        self.N = G.number_of_nodes()
        self.clusters = np.arange(G.number_of_nodes())
        self.order = {x: i for i, x in enumerate(G)}
        self.node_lookup = {i: x for x, i in self.order.items()}
        self.history = []

    def record_merge(self, edge, weight):
        """Save history for merges."""
        # contraction order is target -> source
        orig = self.order[edge[1]]
        self.clusters[self.clusters == orig] = self.order[edge[0]]
        self.history.append((edge, weight))

    @property
    def cluster_ids(self):
        """Get map between cluster ID and index."""
        return {x: i for x, i in enumerate(np.unique(self.clusters))}

    @property
    def cluster_map(self):
        """Map between node ID and parent cluster."""
        return {x: self.clusters[idx] for x, idx in self.order.items()}

    @property
    def cluster_nodes(self):
        """Map between cluster ID and nodes."""
        cluster_ids = np.unique(self.clusters)
        clusters = []
        for c in cluster_ids:
            indices = np.argwhere(self.clusters == c)
            clusters.append([self.node_lookup[int(x)] for x in indices])

        return clusters

    @property
    def num_clusters(self):
        """Get number of clusters."""
        return len(np.unique(self.clusters))

    def get_merged_features(self, G, f_score=None):
        """Get averaged features of each cluster."""
        result = []
        clusters = self.cluster_nodes
        for idx, cluster in enumerate(clusters):
            if len(cluster) == 1:
                result.append(G.nodes[cluster[0]]["semantic_feature"])
                continue

            X_cluster = np.array([G.nodes[idx]["semantic_feature"] for idx in cluster])
            weights = f_score(X_cluster) if f_score is not None else None
            result.append(np.average(X_cluster, axis=0, weights=weights))

        return result

    def __len__(self):
        """Get number of features in workspace."""
        return self.N
