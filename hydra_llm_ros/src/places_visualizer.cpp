#include <config_utilities/virtual_config.h>
#include <glog/logging.h>
#include <hydra_llm/task_embeddings.h>
#include <hydra_ros/visualizer/dsg_visualizer_plugin.h>
#include <hydra_ros/visualizer/visualizer_utilities.h>
#include <visualization_msgs/MarkerArray.h>

#include "hydra_llm_ros/ros_task_embeddings.h"

namespace hydra::llm {

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

class PlaceVisualizerPlugin : public DsgVisualizerPlugin {
 public:
  PlaceVisualizerPlugin(const ros::NodeHandle& nh, const std::string& name);

  virtual ~PlaceVisualizerPlugin();

  void draw(const ConfigManager& configs,
            const std_msgs::Header& header,
            const DynamicSceneGraph& graph) override;

  void reset(const std_msgs::Header& header, const DynamicSceneGraph& graph) override;

  SemanticNodeAttributes::ColorVector getNodeColor(const SceneGraphNode& node) {
    const auto& attrs = node.attributes<SemanticNodeAttributes>();
    if (color_by_task_) {
      const auto result = tasks_->getBestScore(*norm_, attrs.semantic_feature);
    }

    // grab score and return appropriate color
    const auto color_idx =
        color_by_task_ ? clusters[i]->best_task_index : region_id_.categoryId();
    const auto color = HydraConfig::instance().getRoomColor(color_idx);
    attrs->color = Eigen::Map<const SemanticNodeAttributes::ColorVector>(color.data());
  }

  void addMarkerToArray(const Marker& marker, MarkerArray& msg);

  void resetTasks();

  bool color_by_task_;
  std::string ns_;
  ros::Publisher pub_;
  TaskEmbeddings::Ptr tasks_;
  EmbeddingNorm::Ptr norm_;
  std::set<std::string> published_markers_;
};

PlaceVisualizerPlugin::PlaceVisualizerPlugin(const ros::NodeHandle& nh,
                                             const std::string& name)
    : DsgVisualizerPlugin(nh, name), ns_("llm_places") {
  pub_ = nh_.advertise<MarkerArray>("markers", 1, true);
  norm_ = std::make_shared<CosineDistance>(CosineDistance::Config());
}

void PlaceVisualizerPlugin::resetTasks() {
  tasks_ = std::make_shared<RosTaskEmbeddings>(RosTaskEmbeddings::Config());
}

void PlaceVisualizerPlugin::draw(const ConfigManager& configs,
                                 const std_msgs::Header& header,
                                 const DynamicSceneGraph& graph) {
  MarkerArray msg;

  auto viz_config = configs.getVisualizerConfig();
  // TODO(nathan) make this flexible
  const auto layer_config = configs.getLayerConfig(DsgLayers::PLACES);
  const auto node_ns = ns_ + "_nodes";
  const auto& places = graph.getLayer(DsgLayers::PLACES);
  Marker nodes =
      makeCentroidMarkers(header,
                          *CHECK_NOTNULL(layer_config),
                          places,
                          viz_config,
                          node_ns,
                          [this](const SceneGraphNode& node) { getNodeColor(node); });
  addMarkerToArray(nodes, msg);

  const auto edge_ns = ns_ + "_edges";
  Marker edges = makeLayerEdgeMarkers(
      header, *layer_config, places, viz_config, NodeColor::Zero(), edge_ns);
  addMarkerToArray(nodes, msg);

  if (!msg.markers.empty()) {
    pub_.publish(msg);
  }
}

void PlaceVisualizerPlugin::reset(const std_msgs::Header& header,
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

void PlaceVisualizerPlugin::addMarkerToArray(const Marker& marker, MarkerArray& msg) {
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

}  // namespace hydra::llm
