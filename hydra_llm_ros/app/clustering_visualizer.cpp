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
#include <hydra_ros/visualizer/config_manager.h>
#include <hydra_ros/visualizer/dsg_visualizer_plugin.h>

#include "hydra_llm_ros/graph_wrappers.h"
#include "hydra_llm_ros/gt_publisher.h"
#include "hydra_llm_ros/interlayer_plugin.h"
#include "hydra_llm_ros/llm_layer_plugin.h"
#include "hydra_llm_ros/node_filter.h"

namespace hydra {

class GraphVisualizer {
 public:
  using Ptr = std::unique_ptr<GraphVisualizer>;

  struct Config {
    std::map<std::string, std::string> plugins;
    config::VirtualConfig<GraphWrapper> graph;
    llm::NodeFilter::Config node_filter;
  } const config;

  explicit GraphVisualizer(const ros::NodeHandle& nh,
                           const Config& config,
                           const std::string& frame_id)
      : config(config), nh_(nh), frame_id_(frame_id) {
    graph_wrapper_ = config.graph.create();
    for (auto&& [name, plugin_type] : config.plugins) {
      plugins_.push_back(config::create<DsgVisualizerPlugin>(plugin_type, nh_, name));
    }

    node_filter_ = std::make_shared<llm::NodeFilter>(config.node_filter, nh);
    manager_ = std::make_shared<ConfigManager>(nh);
    // TODO(nathan) make it so this isn't required
    manager_->reset();
  }

  void setTaskInformation(const llm::TaskInformation::Ptr& tasks) {
    if (!tasks) {
      return;
    }

    for (const auto& plugin : plugins_) {
      auto derived = dynamic_cast<LLMLayerPlugin*>(plugin.get());
      if (!derived) {
        continue;
      }

      derived->setTasks(tasks);
    }
  }

  void spinOnce() {
    // TODO(nathan) ugly
    if (node_filter_->hasChange()) {
      node_filter_->clearChangeFlag();
      for (const auto& plugin : plugins_) {
        {
          auto derived = dynamic_cast<LLMLayerPlugin*>(plugin.get());
          if (derived) {
            derived->setFilter(node_filter_->getFilter());
          }
        }

        {
          auto derived = dynamic_cast<InterlayerPlugin*>(plugin.get());
          if (derived) {
            derived->setFilter(node_filter_->getFilter());
          }
        }
      }
    }

    bool has_change = graph_wrapper_->hasChange() || manager_->hasChange();
    for (const auto& plugin : plugins_) {
      has_change |= plugin->hasChange();
    }

    if (!has_change) {
      return;
    } else {
      VLOG(1) << "change @ " << nh_.getNamespace() << std::boolalpha
              << ": graph=" << graph_wrapper_->hasChange()
              << ", manager=" << manager_->hasChange();
    }

    auto graph = graph_wrapper_->graph();
    if (!graph) {
      return;
    }

    std_msgs::Header header;
    header.frame_id = frame_id_;
    header.stamp = ros::Time::now();
    for (const auto& plugin : plugins_) {
      plugin->draw(*manager_, header, *graph);
    }

    graph_wrapper_->clearChangeFlag();
    manager_->clearChangeFlags();
    for (auto& plugin : plugins_) {
      plugin->clearChangeFlag();
    }
  }

 private:
  ros::NodeHandle nh_;
  std::string frame_id_;

  std::list<DsgVisualizerPlugin::Ptr> plugins_;
  GraphWrapper::Ptr graph_wrapper_;
  ConfigManager::Ptr manager_;
  llm::NodeFilter::Ptr node_filter_;
};

void declare_config(GraphVisualizer::Config& config) {
  using namespace config;
  name("GraphVisualizer::Config");
  field(config.graph, "graph");
  field(config.plugins, "plugins");
}

class ClusteringVisualizer {
 public:
  struct Config {
    std::map<std::string, GraphVisualizer::Config> graphs;
    GtPublisher::Config gt_publisher;
  } const config;

  explicit ClusteringVisualizer(const ros::NodeHandle& nh);

  void spin() const {
    ros::WallRate rate(10);
    while (ros::ok()) {
      ros::spinOnce();

      llm::TaskInformation::Ptr tasks;
      if (gt_publisher_->hasChange()) {
        tasks = gt_publisher_->getTasks();
        gt_publisher_->clearChangeFlag();
      }

      for (const auto& graph : graphs_) {
        graph->setTaskInformation(tasks);
        graph->spinOnce();
      }

      rate.sleep();
    }
  }

 private:
  GtPublisher::Ptr gt_publisher_;
  std::vector<GraphVisualizer::Ptr> graphs_;
};

void declare_config(ClusteringVisualizer::Config& config) {
  using namespace config;
  name("ClusteringVisualizer::Config");
  field(config.graphs, "graphs");
  field(config.gt_publisher, "gt_publisher");
}

ClusteringVisualizer::ClusteringVisualizer(const ros::NodeHandle& nh)
    : config(config::checkValid(config::fromRos<Config>(nh))) {
  LOG(INFO) << std::endl << config::toString(config);
  gt_publisher_ = std::make_shared<GtPublisher>(config.gt_publisher);
  for (auto&& [ns, gconfig] : config.graphs) {
    graphs_.push_back(std::make_unique<GraphVisualizer>(
        ros::NodeHandle(nh, ns), gconfig, config.gt_publisher.frame_id));
  }
}

}  // namespace hydra

int main(int argc, char** argv) {
  ros::init(argc, argv, "clustering_visualizer");

  FLAGS_minloglevel = 0;
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  ros::NodeHandle nh("~");
  hydra::ClusteringVisualizer node(nh);
  node.spin();

  return 0;
}
