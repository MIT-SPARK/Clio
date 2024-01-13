#include <config_utilities/virtual_config.h>
#include <glog/logging.h>
#include <hydra/common/hydra_config.h>
#include <hydra_llm/task_embeddings.h>
#include <hydra_ros/visualizer/colormap_utilities.h>
#include <hydra_ros/visualizer/dsg_visualizer_plugin.h>
#include <hydra_ros/visualizer/visualizer_utilities.h>
#include <llm/ClipVector.h>
#include <std_srvs/SetBool.h>
#include <visualization_msgs/MarkerArray.h>

#include "hydra_llm_ros/ros_task_embeddings.h"

namespace hydra::llm {

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

using NodeColor = SemanticNodeAttributes::ColorVector;

class PlaceVisualizerPlugin : public DsgVisualizerPlugin {
 public:
  PlaceVisualizerPlugin(const ros::NodeHandle& nh, const std::string& name);

  virtual ~PlaceVisualizerPlugin();

  void draw(const ConfigManager& configs,
            const std_msgs::Header& header,
            const DynamicSceneGraph& graph) override;

  void reset(const std_msgs::Header& header, const DynamicSceneGraph& graph) override;

  bool hasChange() const override;

  void clearChangeFlag() override;

  NodeColor getNodeColor(const ConfigManager& configs,
                         const SceneGraphNode& node) const;

  void addMarkerToArray(const Marker& marker, MarkerArray& msg);

  void handleNewTask(const ::llm::ClipVector& msg);

  bool handleService(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res);

  void resetTasks();

  std::string ns_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
  ros::ServiceServer srv_;

  bool need_redraw_;
  bool color_by_task_;
  TaskEmbeddings::Ptr tasks_;
  std::unique_ptr<EmbeddingNorm> norm_;
  Eigen::VectorXd current_task_feature_;
  std::set<std::string> published_markers_;
};

PlaceVisualizerPlugin::PlaceVisualizerPlugin(const ros::NodeHandle& nh,
                                             const std::string& name)
    : DsgVisualizerPlugin(nh, name), ns_("llm_places"), color_by_task_(false) {
  pub_ = nh_.advertise<MarkerArray>("markers", 1, true);
  sub_ = nh_.subscribe("current_task", 1, &PlaceVisualizerPlugin::handleNewTask, this);
  srv_ = nh_.advertiseService(
      "color_by_task", &PlaceVisualizerPlugin::handleService, this);
  // TODO(nathan) this would actually be better as a virtual config
  norm_ = std::make_unique<CosineDistance>(CosineDistance::Config());
}

PlaceVisualizerPlugin::~PlaceVisualizerPlugin() {}

void PlaceVisualizerPlugin::resetTasks() {
  tasks_ = std::make_shared<RosTaskEmbeddings>(RosTaskEmbeddings::Config());
}

NodeColor PlaceVisualizerPlugin::getNodeColor(const ConfigManager& configs,
                                              const SceneGraphNode& node) const {
  const auto& attrs = node.attributes<SemanticNodeAttributes>();
  if (color_by_task_) {
    const auto result = tasks_->getBestScore(*norm_, attrs.semantic_feature);
    const auto color = HydraConfig::instance().getRoomColor(result.index);
    return Eigen::Map<const SemanticNodeAttributes::ColorVector>(color.data());
  }

  const auto score = norm_->score(current_task_feature_, attrs.semantic_feature);
  return dsg_utils::interpolateColorMap(configs.getColormapConfig("place_tasks"),
                                        score);
}

void PlaceVisualizerPlugin::draw(const ConfigManager& configs,
                                 const std_msgs::Header& header,
                                 const DynamicSceneGraph& graph) {
  resetTasks();

  MarkerArray msg;

  auto viz_config = configs.getVisualizerConfig();
  // TODO(nathan) make this flexible
  const auto layer_config = configs.getLayerConfig(DsgLayers::PLACES);
  const auto node_ns = ns_ + "_nodes";
  const auto& places = graph.getLayer(DsgLayers::PLACES);
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

bool PlaceVisualizerPlugin::hasChange() const { return need_redraw_; }

void PlaceVisualizerPlugin::clearChangeFlag() { need_redraw_ = false; }

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

void PlaceVisualizerPlugin::handleNewTask(const ::llm::ClipVector& msg) {
  current_task_feature_ =
      Eigen::Map<const Eigen::VectorXd>(msg.elements.data(), msg.elements.size());
}

bool PlaceVisualizerPlugin::handleService(std_srvs::SetBool::Request& req,
                                          std_srvs::SetBool::Response& res) {
  color_by_task_ = req.data;
  res.success = true;
  return true;
}

}  // namespace hydra::llm
