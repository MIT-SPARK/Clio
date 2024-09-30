"""Agglomerative clustering method."""
import yaml
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import json
import os
import logging
import distinctipy

import clio_batch.information_metrics as metrics
import clio_batch.helpers as helpers


class ClusterIBConfig():
    def __init__(self, config_file_path):
        with open(config_file_path, 'r') as stream:
            config = yaml.safe_load(stream)

        self.debug = config.get('debug', False)
        self.debug_folder = config.get('debug_folder')
        self.sims_thres = config.get('sims_thres')
        self.delta = config.get('delta')
        self.top_k = config.get('top_k_tasks')
        self.cumulative = config.get('cumulative')
        self.lerf_loss_cannonical_phrases = config.get(
            'lerf_loss_cannonical_phrases')
        self.use_lerf_loss = config.get('use_lerf_loss')

    def dump(self, output):
        param_dict = {}
        param_dict["name"] = "Modified-AIB"
        param_dict["sims_thres"] = self.sims_thres
        param_dict["delta"] = self.delta
        param_dict["top_k_tasks"] = self.top_k
        param_dict["cumulative"] = self.cumulative
        param_dict["lerf_loss_cannonical_phrases"] = self.lerf_loss_cannonical_phrases
        param_dict["use_lerf_loss"] = self.use_lerf_loss
        param_json = json.dumps(param_dict, indent=4)

        filename = "{}.json".format("cluster_config")
        param_log = os.path.join(output, filename)
        with open(param_log, "w") as outfile:
            outfile.write(param_json)


class ClusterIB():
    def __init__(self, config):
        self.config = config
        self.n = None
        self.px = None
        self.py = None
        self.py_x = None
        self.num_clusters = []
        self.deltas = []
        self.nx_graph = None
        self.current_clusters = {}
        self.lerf_loss_cannonical_features = None
        self.colors = None
        self.pos = None
        self.Ixy = None
        self.dIcy_weight = 1.0

        self.idx_mapping = {}

    def setup_py_x(self, region_features, task_features):
        self.px, self.py_x, self.py = self.compute_initial_probabilities(
            region_features, task_features)
        self.n = len(self.px)
        self.Ixy = metrics.mutual_information(self.px, self.py, self.py_x)

    def compute_initial_probabilities(self, region_features, task_features):
        num_tasks = task_features.shape[0]
        m = num_tasks + 1
        n = int(len(region_features))
        k = min(num_tasks + 1, self.config.top_k)
        py_x_tmp = np.zeros((m, n))
        py_x_tmp[0, :] = self.config.sims_thres
        logging.debug("running CLIP")
        cos_sim = helpers.compute_sim_to_tasks(
            task_features, region_features, self.lerf_loss_cannonical_features, self.config.use_lerf_loss)
        cos_sim = np.clip(cos_sim, 0, 1000)  # make sure no negative values
        py_x_tmp[1:, :] = cos_sim
        py_x = 1e-12 * np.ones((m, n))
        if self.config.cumulative:
            l = 1
        else:
            l = k
        # get top task
        top_inds = np.argpartition(py_x_tmp, -1, axis=0)[-1:]
        null_tasks = np.where(top_inds==0)[1]
        while l <= k:
            scale = 1.0
            if l == 1:
                scale = 1.0
            top_inds = np.argpartition(py_x_tmp, -l, axis=0)[-l:]
            py_x[top_inds, np.arange(n)] += scale*py_x_tmp[top_inds, np.arange(n)]
            l += 1
        
        py_x[:, null_tasks] = 1e-12
        py_x[0, null_tasks] = 1.0
        px = np.ones(n) / n
        py = np.ones(m) / m
        py_x = py_x / np.sum(py_x, axis=0)
        return px, py_x, py

    def update_delta_as_part(self, full_region_features, task_features):
        # For the case of factorized clustering.
        px_full, py_x_full, py = self.compute_initial_probabilities(
            full_region_features, task_features)
        self.Ixy = metrics.mutual_information(px_full, self.py, py_x_full)
        self.dIcy_weight = (self.n / len(full_region_features))

    def initialize_nx_graph(self, nx_graph):
        node_idx = {}
        idx = 0
        self.nx_graph = nx.Graph()
        self.pos = {}
        for node in nx_graph.nodes:
            self.idx_mapping[idx] = node
            self.nx_graph.add_node(idx)
            self.pos[idx] = nx_graph.nodes[node]["position"][:2]
            node_idx[node] = idx
            idx += 1

        for edge in nx_graph.edges:
            self.nx_graph.add_edge(node_idx[edge[0]], node_idx[edge[1]])

    def find_clusters(self):
        # p_y_gn_x from normalizing min_sim and task-cosine-sim

        # initialize pc_x
        pc_x = np.eye(self.n)
        pc = pc_x @ self.px

        py_c = self.py_x * self.px @ np.transpose(pc_x) / pc

        prev_Icy = metrics.mutual_information(pc, self.py, py_c)

        last_merged_node = None

        if self.config.debug:
            logging.debug("clusters: \n", self.current_clusters)
            self.log_info(pc_x, np.inf)

        t = 0
        delta = 0
        while True:
            if self.nx_graph.number_of_edges() == 0:
                break

            if self.config.debug:
                logging.debug("clusters: \n", self.current_clusters)
                self.log_info(pc_x, delta)

                plt_path = self.config.debug_folder + \
                    "/graph_" + str(t) + ".png"
                self.plot_current_graph(plt_path, py_c)

            update_edges = []
            if last_merged_node is None:
                update_edges = list(self.nx_graph.edges())
            else:
                update_edges = [(last_merged_node, nb)
                                for nb in self.nx_graph.neighbors(last_merged_node)]

            for e in update_edges:
                d = self.compute_edge_weight(pc, py_c, e[0], e[1])
                self.nx_graph[e[0]][e[1]]['weight'] = d

            min_edge = min(self.nx_graph.edges(),
                           key=lambda x: self.nx_graph[x[0]][x[1]]["weight"])

            # in nx contract edge, right node merge to left node
            # we want to stop the merge if it goes over to preserve identical results between batch and factorized
            py_c_temp = np.copy(py_c)
            pc_temp = np.copy(pc)
            pc_x_temp = np.copy(pc_x)

            py_c_temp[:, min_edge[0]] = (py_c[:, min_edge[0]] * pc[min_edge[0]] +
                                         py_c[:, min_edge[1]] * pc[min_edge[1]]) / (pc[min_edge[0]] + pc[min_edge[1]])
            py_c_temp[:, min_edge[1]] = 0

            pc_temp[min_edge[0]] = pc[min_edge[0]] + pc[min_edge[1]]
            pc_temp[min_edge[1]] = 0

            pc_x_temp[min_edge[0], :] = pc_x[min_edge[0], :] + \
                pc_x[min_edge[1], :]
            pc_x_temp[min_edge[1], :] = 0

            Icy = metrics.mutual_information(pc_temp, self.py, py_c_temp)
            dIcy = prev_Icy - Icy
            prev_Icy = Icy
            delta = self.dIcy_weight * dIcy / self.Ixy
            logging.debug("delta: ", delta)
            if delta > self.config.delta:
                break

            py_c = py_c_temp
            pc = pc_temp
            pc_x = pc_x_temp

            last_merged_node = min_edge[0]
            self.nx_graph = nx.contracted_edge(
                self.nx_graph, min_edge, self_loops=False)

            t += 1

        if self.config.debug:
            plt_path = self.config.debug_folder + "/cluster_delta.png"
            self.generate_plot(plt_path)

        internal_clusters = self.get_clusters_from_pc_x(pc_x)
        return self.get_node_clusters(internal_clusters)

    def get_clusters_from_pc_x(self, pc_x):
        cluster_mapping = np.argmax(pc_x, axis=0)
        clusters = {}
        for i in range(pc_x.shape[1]):
            if cluster_mapping[i] not in clusters.keys():
                clusters[cluster_mapping[i]] = []
            clusters[cluster_mapping[i]].append(i)

        return list(clusters.values())

    def get_node_clusters(self, idx_clusters):
        node_clusters = []
        for cluster in idx_clusters:
            node_cluster = []
            for idx in cluster:
                node_cluster.append(self.idx_mapping[idx])
            node_clusters.append(node_cluster)
        return node_clusters

    def compute_edge_weight(self, pc, py_c, i, j):
        prior = pc[[i, j]] / np.sum(pc[[i, j]])
        weight = (pc[i] + pc[j]) * metrics.js_divergence(py_c[:, [i, j]], prior)
        return weight

    def log_info(self, pc_x, delta):
        obj_clusters = self.get_clusters_from_pc_x(pc_x)
        self.num_clusters.append(len(obj_clusters))
        self.deltas.append(delta)

    def get_plot_pos(self):
        if self.pos is None:
            return nx.spring_layout(self.nx_graph)
        return self.pos

    def plot_current_graph(self, output, py_c):
        if self.colors is None:
            self.colors = distinctipy.get_colors(py_c.shape[0])

        plt.figure(figsize=(12, 12))
        relevant_task_idx = np.argmax(py_c, axis=0)
        color_map = [self.colors[relevant_task_idx[idx]]
                     for idx in self.nx_graph.nodes]
        nx.draw(self.nx_graph, node_color=color_map,
                pos=self.get_plot_pos(), node_size=5, alpha=0.7)
        plt.savefig(output)
        plt.close()

    def generate_plot(self, output):
        fig = plt.figure()
        plt.plot(self.num_clusters, self.deltas)
        plt.xlabel("Number of Clusters")
        plt.ylabel("delta")
        plt.yscale('log')
        plt.savefig(output)
        plt.close()
