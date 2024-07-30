#pragma once
#include <hydra_ros/visualizer/color_adaptors.h>
#include <hydra_ros/visualizer/visualizer_types.h>
#include <llm/ClipVector.h>
#include <ros/ros.h>
#include <std_srvs/SetBool.h>

#include "hydra_llm_ros/colormap_legend.h"
#include "hydra_llm_ros/marker_tracker.h"
#include "hydra_llm_ros/task_information.h"

namespace hydra::llm {

using NodeColor = SemanticNodeAttributes::ColorVector;

class LayerColorFunctor : public ColorAdaptor {
 public:
  struct Config {
    bool color_by_task = true;
    bool label_by_task = true;
    double min_score = 0.0;
    double max_score = 1.0;
    bool use_fixed_range = false;
    double min_hue = 0.0;
    double max_hue = 0.13;
    double min_saturation = 0.7;
    double max_saturation = 0.9;
    double min_luminance = 0.5;
    double max_luminance = 0.8;
  } const config;

  explicit LayerColorFunctor(const std::string& ns);

  LayerColorFunctor(const Config& config, const std::string& ns);

  virtual ~LayerColorFunctor() = default;

  void setGraph(const DynamicSceneGraph& graph) override;

  NodeColor getColor(const DynamicSceneGraph& graph,
                     const SceneGraphNode& node) const override;

  bool hasChange() const { return has_change_; }

  void clearChangeFlag() { has_change_ = false; }

  void setTasks(const llm::TaskInformation::Ptr& tasks) { tasks_ = tasks; }

  void setLayer(LayerId layer_to_use);

 private:
  NodeColor getTaskColor(const SemanticNodeAttributes& attrs) const;

  NodeColor getScoreColor(const SemanticNodeAttributes& attrs) const;

  void handleNewTask(const ::llm::ClipVector& msg);

  bool handleService(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res);

  bool handleLabel(std_srvs::SetBool::Request& req, std_srvs::SetBool::Response& res);

  ros::NodeHandle nh_;
  bool has_change_;

  std::optional<LayerId> layer_to_use_;
  bool color_by_task_;
  bool label_by_task_;
  bool has_current_task_feature_;
  std::pair<double, double> curr_score_range_;
  ColormapConfig score_cmap_;
  mutable ColormapLegend::Ptr legend_;

  Eigen::VectorXd current_task_feature_;
  TaskInformation::Ptr tasks_;

  ros::Subscriber sub_;
  ros::ServiceServer srv_;
  ros::ServiceServer lsrv_;

  inline static const auto registration_ = config::
      RegistrationWithConfig<ColorAdaptor, LayerColorFunctor, Config, std::string>(
          "LayerColorFunctor");
};

void declare_config(LayerColorFunctor::Config& config);

}  // namespace hydra::llm