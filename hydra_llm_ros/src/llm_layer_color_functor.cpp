#include "hydra_llm_ros/llm_layer_color_functor.h"

#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/common.h>
#include <hydra/common/hydra_config.h>
#include <hydra_llm/embedding_distances.h>
#include <hydra_llm/embedding_group.h>
#include <hydra_ros/visualizer/colormap_utilities.h>
#include <hydra_ros/visualizer/visualizer_utilities.h>

#include "hydra_llm_ros/ros_embedding_group.h"

namespace hydra::llm {

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

void declare_config(ColorFunctorConfig& config) {
  using namespace config;
  name("ColorFunctorConfig");
  field(config.color_by_task, "color_by_task");
  field(config.metric, "metric");
  field(config.colormap_filepath, "colormap_filepath");
  field(config.min_score, "min_score");
  field(config.max_score, "max_score");
  field(config.use_fixed_range, "use_fixed_range");
  field(config.layer_to_use, "layer_to_use");
  field(config.min_hue, "min_hue");
  field(config.max_hue, "max_hue");
  field(config.min_saturation, "min_saturation");
  field(config.max_saturation, "max_saturation");
  field(config.min_luminance, "min_luminance");
  field(config.max_luminance, "max_luminance");
}

LayerColorFunctor::LayerColorFunctor(const ros::NodeHandle& nh)
    : nh_(nh), color_by_task_(false), has_current_task_feature_(false) {
  config_ = config::fromRos<ColorFunctorConfig>(nh_);
  VLOG(2) << std::endl << config::toString(config_);
  config::checkValid(config_);
  colormap_ = SemanticColorMap::fromCsv(config_.colormap_filepath);
  color_by_task_ = config_.color_by_task;
  if (color_by_task_) {
    CHECK(colormap_) << "Invalid colormap: '" << config_.colormap_filepath << "'";
  }
  metric_ = config_.metric.create();
  CHECK(metric_);

  score_cmap_.min_hue = config_.min_hue;
  score_cmap_.max_hue = config_.max_hue;
  score_cmap_.min_saturation = config_.min_saturation;
  score_cmap_.max_saturation = config_.max_saturation;
  score_cmap_.min_luminance = config_.min_luminance;
  score_cmap_.max_luminance = config_.max_luminance;

  sub_ = nh_.subscribe("current_task", 1, &LayerColorFunctor::handleNewTask, this);
  srv_ = nh_.advertiseService("color_by_task", &LayerColorFunctor::handleService, this);
  if (color_by_task_) {
    resetTasks();
  }
}

LayerColorFunctor::~LayerColorFunctor() {}

void LayerColorFunctor::setGraph(const DynamicSceneGraph::Ptr& graph) {
  graph_ = graph;

  if (!graph_) {
    LOG(WARNING) << "graph not set!";
    return;
  }

  if (color_by_task_) {
    resetTasks();
    return;
  }

  const auto& layer = graph_->getLayer(config_.layer_to_use);
  if (has_current_task_feature_) {
    double min_score = 1.0;
    double max_score = 0.0;
    double average_score = 0.0;
    for (const auto& id_node_pair : layer.nodes()) {
      const auto& attrs = id_node_pair.second->attributes<SemanticNodeAttributes>();
      const auto score = metric_->score(current_task_feature_, attrs.semantic_feature);
      average_score += score;
      min_score = std::min(min_score, score);
      max_score = std::max(max_score, score);
    }
    average_score /= layer.numNodes();
    VLOG(VLEVEL_TRACE) << "Scores: " << average_score << " average score (range: ["
                       << min_score << ", " << max_score << "])";
    curr_score_range_ = {min_score, max_score};
  }
}

NodeColor LayerColorFunctor::getNodeColor(const SceneGraphNode& node) const {
  const auto& attrs = node.attributes<SemanticNodeAttributes>();
  if (color_by_task_) {
    if (!tasks_) {
      resetTasks();
      CHECK(tasks_);
    }

    const auto result = tasks_->getBestScore(*metric_, attrs.semantic_feature);
    LOG_IF(WARNING, result.index >= colormap_->getNumLabels())
        << "Colormap too small for number of tasks";

    // avoid bad colors near 0
    const auto index = (result.index + 1) % colormap_->getNumLabels();
    const auto color = colormap_->getColorFromLabel(index);
    NodeColor color_vec;
    color_vec.x() = color.r;
    color_vec.y() = color.g;
    color_vec.z() = color.b;
    return color_vec;
  }

  if (!has_current_task_feature_) {
    return NodeColor::Zero();
  }

  const auto score = metric_->score(current_task_feature_, attrs.semantic_feature);
  double ratio;
  if (config_.use_fixed_range) {
    ratio = (score - config_.min_score) / (config_.max_score - config_.min_score);
  } else {
    ratio = (score - curr_score_range_.first) /
            (curr_score_range_.second - curr_score_range_.first);
  }

  ratio = std::clamp(ratio, 0.0, 1.0);
  return dsg_utils::interpolateColorMap(score_cmap_, ratio);
}

void LayerColorFunctor::handleNewTask(const ::llm::ClipVector& msg) {
  LOG(INFO) << "Got new task!";
  current_task_feature_ =
      Eigen::Map<const Eigen::VectorXd>(msg.elements.data(), msg.elements.size());
  has_current_task_feature_ = true;
}

bool LayerColorFunctor::handleService(std_srvs::SetBool::Request& req,
                                      std_srvs::SetBool::Response& res) {
  color_by_task_ = req.data;
  res.success = true;
  return true;
}

void LayerColorFunctor::resetTasks() const {
  RosEmbeddingGroup::Config config;
  config.silent_wait = true;
  tasks_ = std::make_shared<RosEmbeddingGroup>(config);
}

}  // namespace hydra::llm
