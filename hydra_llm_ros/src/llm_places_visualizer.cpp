#include "hydra_llm_ros/llm_places_visualizer.h"

#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/hydra_config.h>
#include <hydra_llm/embedding_distances.h>
#include <hydra_llm/embedding_group.h>
#include <hydra_ros/visualizer/colormap_utilities.h>
#include <hydra_ros/visualizer/visualizer_utilities.h>

#include "hydra_llm_ros/ros_embedding_group.h"

namespace hydra::llm {

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

void declare_config(LLMPlacesConfig& config) {
  using namespace config;
  name("LLMPlacesConfig");
  field(config.metric, "metric");
  field(config.colormap_filepath, "colormap_filepath");
  field(config.min_score, "min_score");
  field(config.max_score, "max_score");
  field(config.layer_to_use, "layer_to_use");
}

LLMPlacesVisualizer::LLMPlacesVisualizer(const ros::NodeHandle& nh,
                                         const std::string& name)
    : DsgVisualizerPlugin(nh, name),
      ns_("llm_places"),
      color_by_task_(false),
      has_current_task_feature_(false) {
  pub_ = nh_.advertise<MarkerArray>("markers", 1, true);
  sub_ = nh_.subscribe("current_task", 1, &LLMPlacesVisualizer::handleNewTask, this);
  srv_ =
      nh_.advertiseService("color_by_task", &LLMPlacesVisualizer::handleService, this);

  config_ = config::checkValid(config::fromRos<LLMPlacesConfig>(nh_));
  VLOG(2) << std::endl << config::toString(config_);
  colormap_ = SemanticColorMap::fromCsv(config_.colormap_filepath);
  metric_ = config_.metric.create();
  CHECK(metric_);
}

LLMPlacesVisualizer::~LLMPlacesVisualizer() {}

void LLMPlacesVisualizer::resetTasks() {
  RosEmbeddingGroup::Config config;
  config.silent_wait = true;
  tasks_ = std::make_shared<RosEmbeddingGroup>(config);
}

NodeColor LLMPlacesVisualizer::getNodeColor(const ConfigManager& configs,
                                            const SceneGraphNode& node) const {
  const auto& attrs = node.attributes<SemanticNodeAttributes>();
  if (color_by_task_) {
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
  auto ratio = (score - config_.min_score) / (config_.max_score - config_.min_score);
  ratio = std::clamp(ratio, 0.0, 1.0);
  return dsg_utils::interpolateColorMap(configs.getColormapConfig("place_tasks"),
                                        ratio);
}

void LLMPlacesVisualizer::draw(const ConfigManager& configs,
                               const std_msgs::Header& header,
                               const DynamicSceneGraph& graph) {
  if (color_by_task_) {
    resetTasks();
  }

  const auto& places = graph.getLayer(config_.layer_to_use);
  if (has_current_task_feature_) {
    double average_score = 0.0;
    for (const auto& id_node_pair : places.nodes()) {
      const auto& attrs = id_node_pair.second->attributes<SemanticNodeAttributes>();
      average_score += metric_->score(current_task_feature_, attrs.semantic_feature);
    }
    average_score /= places.numNodes();
    LOG(ERROR) << "Average score: " << average_score;
  }

  MarkerArray msg;

  auto viz_config = configs.getVisualizerConfig();
  // TODO(nathan) make this flexible
  const auto layer_config = configs.getLayerConfig(DsgLayers::PLACES);
  const auto node_ns = ns_ + "_nodes";
  Marker nodes = makeCentroidMarkers(header,
                                     *CHECK_NOTNULL(layer_config),
                                     places,
                                     viz_config,
                                     node_ns,
                                     [&](const SceneGraphNode& node) -> NodeColor {
                                       return getNodeColor(configs, node);
                                     });
  addMarkerToArray(nodes, msg);

  const auto edge_ns = ns_ + "_edges";
  Marker edges = makeLayerEdgeMarkers(
      header, *layer_config, places, viz_config, NodeColor::Zero(), edge_ns);
  addMarkerToArray(edges, msg);

  if (!msg.markers.empty()) {
    pub_.publish(msg);
  }
}

void LLMPlacesVisualizer::reset(const std_msgs::Header& header,
                                const DynamicSceneGraph&) {
  MarkerArray msg;
  for (const auto& ns : published_markers_) {
    msg.markers.push_back(makeDeleteMarker(header, 0, ns));
  }

  published_markers_.clear();

  if (!msg.markers.empty()) {
    pub_.publish(msg);
  }
}

bool LLMPlacesVisualizer::hasChange() const { return need_redraw_; }

void LLMPlacesVisualizer::clearChangeFlag() { need_redraw_ = false; }

void LLMPlacesVisualizer::addMarkerToArray(const Marker& marker, MarkerArray& msg) {
  if (!marker.points.empty()) {
    msg.markers.push_back(marker);
    published_markers_.insert(marker.ns);
    return;
  }

  if (!published_markers_.count(marker.ns)) {
    return;
  }

  msg.markers.push_back(makeDeleteMarker(marker.header, 0, marker.ns));
  published_markers_.erase(marker.ns);
}

void LLMPlacesVisualizer::handleNewTask(const ::llm::ClipVector& msg) {
  LOG(INFO) << "Got new task!";
  current_task_feature_ =
      Eigen::Map<const Eigen::VectorXd>(msg.elements.data(), msg.elements.size());
  has_current_task_feature_ = true;
  need_redraw_ = true;
}

bool LLMPlacesVisualizer::handleService(std_srvs::SetBool::Request& req,
                                        std_srvs::SetBool::Response& res) {
  color_by_task_ = req.data;
  res.success = true;
  return true;
}

}  // namespace hydra::llm
