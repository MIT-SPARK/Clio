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

void declare_config(LayerColorFunctor::Config& config) {
  using namespace config;
  name("ColorFunctorConfig");
  field(config.layer_to_use, "layer_to_use");
  field(config.color_by_task, "color_by_task");
  field(config.min_score, "min_score");
  field(config.max_score, "max_score");
  field(config.use_fixed_range, "use_fixed_range");
  field(config.min_hue, "min_hue");
  field(config.max_hue, "max_hue");
  field(config.min_saturation, "min_saturation");
  field(config.max_saturation, "max_saturation");
  field(config.min_luminance, "min_luminance");
  field(config.max_luminance, "max_luminance");
}

namespace {

inline ColormapConfig fromLayerConfig(const LayerColorFunctor::Config& config) {
  ColormapConfig cmap;
  cmap.min_hue = config.min_hue;
  cmap.max_hue = config.max_hue;
  cmap.min_saturation = config.min_saturation;
  cmap.max_saturation = config.max_saturation;
  cmap.min_luminance = config.min_luminance;
  cmap.max_luminance = config.max_luminance;
  return cmap;
}

inline double getRatio(double value, double lo, double hi) {
  return std::clamp((value - lo) / (hi - lo), 0.0, 1.0);
}

}  // namespace

LayerColorFunctor::LayerColorFunctor(const ros::NodeHandle& nh)
    : config(config::fromRos<Config>(nh)),
      nh_(nh),
      has_change_(false),
      color_by_task_(config.color_by_task),
      label_by_task_(config.label_by_task),
      has_current_task_feature_(false),
      score_cmap_(fromLayerConfig(config)) {
  VLOG(2) << "Loaded color functor from: '" << nh_.getNamespace() << "'";
  VLOG(2) << std::endl << config::toString(config);
  config::checkValid(config);

  sub_ = nh_.subscribe("current_task", 1, &LayerColorFunctor::handleNewTask, this);
  srv_ = nh_.advertiseService("color_by_task", &LayerColorFunctor::handleService, this);
  lsrv_ = nh_.advertiseService("label_by_task", &LayerColorFunctor::handleLabel, this);
}

void LayerColorFunctor::setGraph(const DynamicSceneGraph& graph) {
  if (!tasks_) {
    return;
  }

  const auto& layer = graph.getLayer(config.layer_to_use);
  for (auto&& [node_id, node] : layer.nodes()) {
    auto& attrs = node->attributes<SemanticNodeAttributes>();
    if (label_by_task_) {
      attrs.name = tasks_->getNearestTask(attrs.semantic_feature);
    } else {
      attrs.name = NodeSymbol(node_id).getLabel();
    }
  }

  if (!has_current_task_feature_) {
    return;
  }

  double min_score = 1.0;
  double max_score = 0.0;
  for (const auto& id_node_pair : layer.nodes()) {
    const auto& attrs = id_node_pair.second->attributes<SemanticNodeAttributes>();
    const auto score =
        tasks_->metric().score(current_task_feature_, attrs.semantic_feature);
    min_score = std::min(min_score, score);
    max_score = std::max(max_score, score);
  }

  VLOG(VLEVEL_TRACE) << "score range: [" << min_score << ", " << max_score << "]";
  curr_score_range_ = {min_score, max_score};

  ColormapLegend::ColorEndpoint cmap_start;
  cmap_start.color_hls << score_cmap_.min_hue, score_cmap_.min_luminance,
      score_cmap_.min_saturation;
  cmap_start.value = config.use_fixed_range ? config.min_score : min_score;

  ColormapLegend::ColorEndpoint cmap_end;
  cmap_end.color_hls << score_cmap_.max_hue, score_cmap_.max_luminance,
      score_cmap_.max_saturation;
  cmap_end.value = config.use_fixed_range ? config.max_score : max_score;

  legend_ = ColormapLegend::fromEndpoints(nh_, cmap_start, cmap_end);
}

NodeColor LayerColorFunctor::getNodeColor(const SceneGraphNode& node) const {
  if (!tasks_) {
    LOG_FIRST_N(WARNING, 1) << "No tasks set!";
    return NodeColor::Zero();
  }

  const auto& attrs = node.attributes<SemanticNodeAttributes>();
  return color_by_task_ ? getTaskColor(attrs) : getScoreColor(attrs);
}

NodeColor LayerColorFunctor::getTaskColor(const SemanticNodeAttributes& attrs) const {
  return tasks_->getNearestTaskColor(attrs.semantic_feature);
}

NodeColor LayerColorFunctor::getScoreColor(const SemanticNodeAttributes& attrs) const {
  if (!has_current_task_feature_) {
    return NodeColor::Zero();
  }

  const auto score =
      tasks_->metric().score(current_task_feature_, attrs.semantic_feature);

  double ratio;
  if (config.use_fixed_range) {
    ratio = getRatio(score, config.min_score, config.max_score);
  } else {
    ratio = getRatio(score, curr_score_range_.first, curr_score_range_.second);
  }

  return dsg_utils::interpolateColorMap(score_cmap_, ratio);
}

void LayerColorFunctor::handleNewTask(const ::llm::ClipVector& msg) {
  VLOG(1) << "Got new task!";
  const auto& vec = msg.elements;
  current_task_feature_ = Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size());
  has_current_task_feature_ = true;
  has_change_ = true;
}

bool LayerColorFunctor::handleService(std_srvs::SetBool::Request& req,
                                      std_srvs::SetBool::Response& res) {
  color_by_task_ = req.data;
  res.success = true;
  has_change_ = true;
  return true;
}

bool LayerColorFunctor::handleLabel(std_srvs::SetBool::Request& req,
                                    std_srvs::SetBool::Response& res) {
  label_by_task_ = req.data;
  res.success = true;
  has_change_ = true;
  return true;
}

}  // namespace hydra::llm
