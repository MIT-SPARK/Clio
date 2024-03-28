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
#include "hydra_llm_ros/task_information.h"

using KhronosPluginList =
    std::vector<config::VirtualConfig<khronos::DsgVisualizerPlugin>>;
using NodeColor = spark_dsg::SemanticNodeAttributes::ColorVector;

struct LLMPluginConfig {
  std::map<std::string, std::string> plugins;
  KhronosPluginList khronos_plugins;
  bool color_objects_by_task = false;
  bool color_rooms_by_task = false;
  bool color_places_by_task = false;
  std::string object_ns = "object_task_visualizer";
  std::string room_ns = "room_tasks_visualizer";
  std::vector<std::string> tasks;
  std::vector<std::string> room_tasks;
  hydra::llm::TaskInformation::Config tasks_config;
};

void declare_config(LLMPluginConfig& config) {
  using namespace config;
  name("LLMPluginConfig");
  field(config.plugins, "llm_plugins");
  field(config.khronos_plugins, "khronos_plugins");
  field(config.color_objects_by_task, "color_objects_by_task");
  field(config.color_rooms_by_task, "color_rooms_by_task");
  field(config.color_places_by_task, "color_places_by_task");
  field(config.object_ns, "object_ns");
  field(config.room_ns, "room_ns");
  field(config.tasks, "tasks");
  field(config.room_tasks, "room_tasks");
  field(config.tasks_config, "tasks_config");
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

  const auto config = config::fromRos<LLMPluginConfig>(nh);
  for (auto&& [name, plugin_type] : config.plugins) {
    node.addPlugin(config::create<hydra::DsgVisualizerPlugin>(plugin_type, nh, name));
  }

  for (const auto& conf : config.khronos_plugins) {
    node.addPlugin(conf.create(nh, node.visualizer_));
  }

  std::shared_ptr<hydra::llm::LayerColorFunctor> room_colors;
  std::shared_ptr<hydra::llm::LayerColorFunctor> object_colors;

  auto& viz = node.getVisualizer();
  auto object_task_info =
      std::make_shared<hydra::llm::TaskInformation>(config.tasks_config, config.tasks);

  hydra::llm::TaskInformation::Ptr room_task_info;
  if (config.room_tasks.empty()) {
    room_task_info = object_task_info;
  } else {
    room_task_info = std::make_shared<hydra::llm::TaskInformation>(config.tasks_config,
                                                                   config.room_tasks);
  }

  if (config.color_objects_by_task) {
    const ros::NodeHandle onh(nh, config.object_ns);
    object_colors = std::make_shared<hydra::llm::LayerColorFunctor>(onh);
    object_colors->setTasks(object_task_info);
    viz.addUpdateCallback([&](const auto& graph) {
      if (graph) {
        object_colors->setGraph(*graph, spark_dsg::DsgLayers::OBJECTS);
      }
    });
    viz.setLayerColorFunction(spark_dsg::DsgLayers::OBJECTS,
                              [&](const spark_dsg::SceneGraphNode& node) -> NodeColor {
                                return object_colors->getNodeColor(node);
                              });
  }

  if (config.color_rooms_by_task) {
    const ros::NodeHandle rnh(nh, config.room_ns);
    room_colors = std::make_shared<hydra::llm::LayerColorFunctor>(rnh);
    room_colors->setTasks(room_task_info);
    viz.addUpdateCallback([&](const auto& graph) {
      if (graph) {
        room_colors->setGraph(*graph, spark_dsg::DsgLayers::PLACES);
      }
    });
    viz.setLayerColorFunction(spark_dsg::DsgLayers::ROOMS,
                              [&](const spark_dsg::SceneGraphNode& node) -> NodeColor {
                                return room_colors->getNodeColor(node);
                              });
  }

  if (config.color_rooms_by_task && config.color_places_by_task) {
    viz.setLayerColorFunction(spark_dsg::DsgLayers::PLACES,
                              [&](const spark_dsg::SceneGraphNode& node) -> NodeColor {
                                return room_colors->getNodeColor(node);
                              });
  }

  node.spin();

  return 0;
}
