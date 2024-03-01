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
#include "hydra_llm_ros/interlayer_plugin.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <glog/logging.h>
#include <hydra_ros/visualizer/visualizer_utilities.h>
#include <visualization_msgs/MarkerArray.h>

namespace hydra {

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

void declare_config(InterlayerPlugin::Config& config) {
  using namespace config;
  name("InterlayerPlugin::Config");
  field(config.marker_ns, "marker_ns");
  field(config.skiplist, "skiplist");
}

InterlayerPlugin::InterlayerPlugin(const ros::NodeHandle& nh, const std::string& name)
    : DsgVisualizerPlugin(nh, name),
      config(config::fromRos<Config>(nh_)),
      need_redraw_(true) {
  std::string marker_ns =
      config.marker_ns.empty() ? nh_.getNamespace() : config.marker_ns;

  edge_ns_ = marker_ns + "_edges";
  pub_ = nh_.advertise<visualization_msgs::MarkerArray>("", 1, true);
}

InterlayerPlugin::~InterlayerPlugin() {}

void InterlayerPlugin::draw(const ConfigManager& manager,
                            const std_msgs::Header& header,
                            const DynamicSceneGraph& graph) {
  const auto& graph_config = manager.getVisualizerConfig();

  MarkerArray msg;

  std::map<LayerId, LayerConfig> layer_configs;
  for (const auto& layer_id : graph.layer_ids) {
    if (config.skiplist.count(layer_id)) {
      continue;
    }

    layer_configs[layer_id] = *CHECK_NOTNULL(manager.getLayerConfig(layer_id));
  }

  auto edges = makeGraphEdgeMarkers(
      header, graph, layer_configs, graph_config, edge_ns_, filter_);
  for (const auto& marker : edges.markers) {
    tracker_.addMarker(marker, msg);
  }

  tracker_.clearPrevious(header, msg);
  pub_.publish(msg);
}

void InterlayerPlugin::reset(const std_msgs::Header& header, const DynamicSceneGraph&) {
  MarkerArray msg;
  tracker_.clearPrevious(header, msg);
}

bool InterlayerPlugin::hasChange() const { return need_redraw_; }

void InterlayerPlugin::clearChangeFlag() { need_redraw_ = false; }

}  // namespace hydra
