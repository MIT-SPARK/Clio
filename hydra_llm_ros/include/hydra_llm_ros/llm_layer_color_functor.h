#include <hydra/common/semantic_color_map.h>
#include <hydra_llm/embedding_distances.h>
#include <hydra_llm/embedding_group.h>
#include <hydra_ros/visualizer/visualizer_types.h>
#include <llm/ClipVector.h>
#include <ros/ros.h>
#include <std_srvs/SetBool.h>

namespace hydra::llm {

using NodeColor = SemanticNodeAttributes::ColorVector;

struct ColorFunctorConfig {
  bool color_by_task = true;
  config::VirtualConfig<EmbeddingDistance> metric{CosineDistance::Config(), "cosine"};
  std::string colormap_filepath = "";
  double min_score = 0.0;
  double max_score = 1.0;
  bool use_fixed_range = false;
  LayerId layer_to_use = DsgLayers::PLACES;
  double min_hue = 0.0;
  double max_hue = 0.13;
  double min_saturation = 0.7;
  double max_saturation = 0.9;
  double min_luminance = 0.5;
  double max_luminance = 0.8;
};

class LayerColorFunctor {
 public:
  LayerColorFunctor(const ros::NodeHandle& nh);

  virtual ~LayerColorFunctor();

  void setGraph(const DynamicSceneGraph::Ptr& graph);

  NodeColor getNodeColor(const SceneGraphNode& node) const;

  void handleNewTask(const ::llm::ClipVector& msg);

  bool handleService(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res);

  void resetTasks() const;

  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::ServiceServer srv_;

  DynamicSceneGraph::Ptr graph_;

  bool color_by_task_;
  ColormapConfig score_cmap_;
  mutable EmbeddingGroup::Ptr tasks_;
  std::unique_ptr<EmbeddingDistance> metric_;
  bool has_current_task_feature_;
  Eigen::VectorXd current_task_feature_;
  std::set<std::string> published_markers_;
  SemanticColorMap::Ptr colormap_;
  std::pair<double, double> curr_score_range_;

  ColorFunctorConfig config_;
};

}  // namespace hydra::llm
