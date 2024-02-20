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
#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/ros.h>
#include <glog/logging.h>
#include <hydra_ros/visualizer/dsg_visualizer_plugin.h>
#include <hydra_ros/visualizer/hydra_visualizer.h>
#include <khronos_ros/visualization/dsg_visualizer_plugins/dsg_visualizer_plugin.h>

#include "hydra_llm_ros/llm_layer_color_functor.h"

struct LLMPluginConfig {
  std::map<std::string, std::string> plugins;
  std::vector<config::VirtualConfig<khronos::DsgVisualizerPlugin>> khronos_plugins;
};

using KhronosPluginList =
    std::vector<config::VirtualConfig<khronos::DsgVisualizerPlugin>>;

void declare_config(LLMPluginConfig& config) {
  using namespace config;
  name("LLMPluginConfig");
  field(config.plugins, "llm_plugins");
  field(config.khronos_plugins, "khronos_plugins");
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "dsg_visualizer_node");

  FLAGS_minloglevel = 0;
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  ros::NodeHandle nh("~");
  hydra::HydraVisualizer node(nh);
  node.clearPlugins();

  const auto plugin_config = config::fromRos<LLMPluginConfig>(nh);
  for (auto&& [name, plugin_type] : plugin_config.plugins) {
    node.addPlugin(config::create<hydra::DsgVisualizerPlugin>(plugin_type, nh, name));
  }

  for (const auto& conf : plugin_config.khronos_plugins) {
    node.addPlugin(conf.create(nh, node.visualizer_));
  }

  std::shared_ptr<hydra::llm::LayerColorFunctor> room_colors;
  std::shared_ptr<hydra::llm::LayerColorFunctor> object_colors;

  bool color_objects_by_task = false;
  nh.getParam("color_objects_by_task", color_objects_by_task);
  bool color_rooms_by_task = false;
  nh.getParam("color_rooms_by_task", color_rooms_by_task);
  bool color_places_by_task = true;
  nh.getParam("color_places_by_task", color_places_by_task);

  auto& viz = node.getVisualizer();

  if (color_objects_by_task) {
    object_colors = std::make_shared<hydra::llm::LayerColorFunctor>(
        ros::NodeHandle(nh, "room_tasks_visualizer"));
    viz.addUpdateCallback([&](const auto& graph) { object_colors->setGraph(graph); });
    viz.setLayerColorFunction(spark_dsg::DsgLayers::OBJECTS,
                              [&](const spark_dsg::SceneGraphNode& node)
                                  -> spark_dsg::SemanticNodeAttributes::ColorVector {
                                return object_colors->getNodeColor(node);
                              });
  }

  if (color_rooms_by_task) {
    room_colors = std::make_shared<hydra::llm::LayerColorFunctor>(
        ros::NodeHandle(nh, "room_tasks_visualizer"));
    viz.addUpdateCallback([&](const auto& graph) { room_colors->setGraph(graph); });
    viz.setLayerColorFunction(spark_dsg::DsgLayers::ROOMS,
                              [&](const spark_dsg::SceneGraphNode& node)
                                  -> spark_dsg::SemanticNodeAttributes::ColorVector {
                                return room_colors->getNodeColor(node);
                              });
    if (color_places_by_task) {
      viz.setLayerColorFunction(spark_dsg::DsgLayers::PLACES,
                                [&](const spark_dsg::SceneGraphNode& node)
                                    -> spark_dsg::SemanticNodeAttributes::ColorVector {
                                  return room_colors->getNodeColor(node);
                                });
    }
  }

  node.spin();

  return 0;
}
