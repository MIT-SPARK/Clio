/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hydra_llm_ros/llm_layer_plugin.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <glog/logging.h>
#include <hydra_ros/visualizer/visualizer_utilities.h>
#include <visualization_msgs/MarkerArray.h>

namespace hydra {

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

void declare_config(LLMLayerPlugin::Config& config) {
  using namespace config;
  name("LLMLayerPlugin::Config");
  field(config.layer, "layer");
  field(config.visualization_layer, "visualization_layer");
}

LLMLayerPlugin::LLMLayerPlugin(const ros::NodeHandle& nh, const std::string& name)
    : DsgVisualizerPlugin(nh, name),
      config(config::fromRos<Config>(nh_)),
      need_redraw_(true) {
  std::string marker_ns =
      config.marker_ns.empty() ? nh_.getNamespace() : config.marker_ns;

  node_ns_ = marker_ns + "_nodes";
  edge_ns_ = marker_ns + "_edges";
  label_ns_ = marker_ns + "_labels";
  bbox_ns_ = marker_ns + "_bbox";
  bbox_edge_ns_ = marker_ns + "_bbox_edges";

  pub_ = nh_.advertise<visualization_msgs::MarkerArray>("", 1, true);
  colors_ = std::make_shared<llm::LayerColorFunctor>(nh_.getNamespace());
}

LLMLayerPlugin::~LLMLayerPlugin() {}

void LLMLayerPlugin::drawLabels(const ConfigManager& manager,
                                const std_msgs::Header& header,
                                const SceneGraphLayer& layer,
                                MarkerArray& msg) {
  const auto& layer_config =
      *CHECK_NOTNULL(manager.getLayerConfig(config.visualization_layer));
  if (!layer_config.use_label) {
    return;
  }

  const auto& graph_config = manager.getVisualizerConfig();
  for (auto&& [node_id, node] : layer.nodes()) {
    if (filter_ && !filter_(*node)) {
      continue;
    }

    auto label = makeTextMarker(header, layer_config, *node, graph_config, label_ns_);
    tracker_.addMarker(label, msg);
  }
}

void LLMLayerPlugin::drawBoxes(const ConfigManager& manager,
                               const std_msgs::Header& header,
                               const DynamicSceneGraph& graph,
                               MarkerArray& msg) {
  const auto& layer = graph.getLayer(config.layer);
  const auto& layer_config =
      *CHECK_NOTNULL(manager.getLayerConfig(config.visualization_layer));
  if (!layer_config.use_bounding_box) {
    return;
  }

  const auto& graph_config = manager.getVisualizerConfig();
  const auto bbox = makeLayerWireframeBoundingBoxes(
      header,
      layer_config,
      layer,
      graph_config,
      bbox_ns_,
      [&](const auto& n) { return getNodeColor(graph, n); },
      filter_);
  tracker_.addMarker(bbox, msg);

  if (layer_config.collapse_bounding_box) {
    const auto bbox_edges = makeEdgesToBoundingBoxes(
        header,
        layer_config,
        layer,
        graph_config,
        bbox_edge_ns_,
        [&](const auto& n) { return getNodeColor(graph, n); },
        filter_);
    tracker_.addMarker(bbox_edges, msg);
  }
}

NodeColor LLMLayerPlugin::getNodeColor(const DynamicSceneGraph& graph,
                                       const SceneGraphNode& node) const {
  return colors_->getColor(graph, node);
}

void LLMLayerPlugin::draw(const ConfigManager& manager,
                          const std_msgs::Header& header,
                          const DynamicSceneGraph& graph) {
  colors_->setLayer(config.layer);
  colors_->setGraph(graph);
  const auto& layer = graph.getLayer(config.layer);
  const auto& layer_config =
      *CHECK_NOTNULL(manager.getLayerConfig(config.visualization_layer));
  const auto& graph_config = manager.getVisualizerConfig();

  MarkerArray msg;
  if (!layer_config.visualize) {
    tracker_.clearPrevious(header, msg);
    pub_.publish(msg);
    return;
  }

  auto nodes = makeCentroidMarkers(
      header,
      layer_config,
      layer,
      graph_config,
      node_ns_,
      [&](const auto& n) { return getNodeColor(graph, n); },
      filter_);
  tracker_.addMarker(nodes, msg);

  auto edges = makeLayerEdgeMarkers(
      header, layer_config, layer, graph_config, NodeColor::Zero(), edge_ns_, filter_);
  tracker_.addMarker(edges, msg);

  drawLabels(manager, header, layer, msg);
  try {
    drawBoxes(manager, header, graph, msg);
  } catch (const std::bad_cast&) {
    LOG(WARNING) << "Failed to draw bounding boxes for layer";
    return;
  }

  tracker_.clearPrevious(header, msg);
  pub_.publish(msg);
}

void LLMLayerPlugin::reset(const std_msgs::Header& header, const DynamicSceneGraph&) {
  MarkerArray msg;
  tracker_.clearPrevious(header, msg);
  pub_.publish(msg);
}

bool LLMLayerPlugin::hasChange() const { return need_redraw_ || colors_->hasChange(); }

void LLMLayerPlugin::clearChangeFlag() {
  need_redraw_ = false;
  colors_->clearChangeFlag();
}

void LLMLayerPlugin::setTasks(const llm::TaskInformation::Ptr& tasks) {
  colors_->setTasks(tasks);
  need_redraw_ = true;
}

}  // namespace hydra
