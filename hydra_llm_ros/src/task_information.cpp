#include "hydra_llm_ros/task_information.h"

#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/path.h>
#include <config_utilities/validation.h>

#include "hydra_llm_ros/ros_embedding_group.h"

namespace hydra::llm {

using NodeColor = SemanticNodeAttributes::ColorVector;

void declare_config(TaskInformation::Config& config) {
  using namespace config;
  name("TaskInformation::Config");
  field(config.ns, "ns");
  field<Path>(config.colormap_filepath, "colormap_filepath");
  field(config.task_service_ns, "task_service_ns");
  field(config.make_legend, "make_legend");
  config.metric.setOptional();
  field(config.metric, "metric");
}

SemanticColorMap::Ptr getColormap(const std::filesystem::path& filepath,
                                  size_t num_tasks) {
  if (std::filesystem::exists(filepath)) {
    return SemanticColorMap::fromCsv(filepath, voxblox::Color(255, 255, 255));
  }

  return SemanticColorMap::randomColors(num_tasks);
}

std::map<std::string, size_t> getIndices(const std::vector<std::string>& tasks) {
  std::map<std::string, size_t> mapping;
  size_t index = 1;
  for (const auto& task : tasks) {
    mapping[task] = index;
    ++index;
  }

  return mapping;
}

CategoryLegend::Ptr getLegend(const std::string& ns,
                              const SemanticColorMap& cmap,
                              const std::map<std::string, size_t>& mapping) {
  return CategoryLegend::fromColormap(ros::NodeHandle(ns), cmap, mapping);
}

EmbeddingGroup::Ptr getTasks(const std::string& service,
                             const std::vector<std::string>& tasks) {
  RosEmbeddingGroup::Config config{service, true, tasks};
  return std::make_shared<RosEmbeddingGroup>(config);
}

TaskInformation::TaskInformation(const Config& config,
                                 const std::vector<std::string>& tasks)
    : config(config::checkValid(config)),
      tasks_(getTasks(config.task_service_ns, tasks)),
      task_indices_(getIndices(tasks_->tasks)),
      colormap_(getColormap(config.colormap_filepath, tasks_->tasks.size())),
      legend_(config.make_legend ? getLegend(config.ns, *colormap_, task_indices_)
                                 : nullptr),
      metric_(config.metric.create()) {
  VLOG(1) << std::endl << config::toString(config);
  DCHECK(colormap_);
  DCHECK(legend_);
  DCHECK(tasks_);
  DCHECK(metric_);
  LOG_IF(WARNING, tasks.size() > colormap_->getNumLabels())
      << "Colormap too small for number of tasks";
}

NodeColor TaskInformation::getColor(const std::string& task) const {
  const auto index = task_indices_.at(task) % colormap_->getNumLabels();
  const auto color = colormap_->getColorFromLabel(index);

  NodeColor vec;
  vec << color.r, color.g, color.b;
  return vec;
}

std::string TaskInformation::getNearestTask(const Eigen::MatrixXd& feature) const {
  const auto result = tasks_->getBestScore(*metric_, feature);
  return tasks_->tasks.at(result.index);
}

NodeColor TaskInformation::getNearestTaskColor(const Eigen::MatrixXd& feature) const {
  return getColor(getNearestTask(feature));
}

const EmbeddingDistance& TaskInformation::metric() const { return *metric_; }

const EmbeddingGroup& TaskInformation::embeddings() const { return *tasks_; }

}  // namespace hydra::llm
