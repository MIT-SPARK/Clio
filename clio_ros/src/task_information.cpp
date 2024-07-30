#include "clio_ros/task_information.h"

#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/path.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra_ros/openset/ros_embedding_group.h>

namespace clio {

using spark_dsg::Color;

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

hydra::SemanticColorMap::Ptr getColormap(const std::filesystem::path& filepath,
                                         size_t num_tasks) {
  if (std::filesystem::exists(filepath)) {
    return hydra::SemanticColorMap::fromCsv(filepath);
  }

  return hydra::SemanticColorMap::randomColors(num_tasks);
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
                              const hydra::SemanticColorMap& cmap,
                              const std::map<std::string, size_t>& mapping) {
  return CategoryLegend::fromColormap(ros::NodeHandle(ns), cmap, mapping);
}

hydra::EmbeddingGroup::Ptr getTasks(const std::string& service,
                                    const std::vector<std::string>& tasks) {
  hydra::RosEmbeddingGroup::Config config{service, true, tasks};
  return std::make_shared<hydra::RosEmbeddingGroup>(config);
}

TaskInformation::TaskInformation(const Config& config,
                                 const std::vector<std::string>& tasks)
    : config(config::checkValid(config)),
      tasks_(getTasks(config.task_service_ns, tasks)),
      task_indices_(getIndices(tasks_->names)),
      colormap_(getColormap(config.colormap_filepath, tasks_->size())),
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

Color TaskInformation::getColor(const std::string& task) const {
  const auto index = task_indices_.at(task) % colormap_->getNumLabels();
  return colormap_->getColorFromLabel(index);
}

std::string TaskInformation::getNearestTask(const Eigen::MatrixXf& feature) const {
  const auto result = tasks_->getBestScore(*metric_, feature);
  return tasks_->names.at(result.index);
}

Color TaskInformation::getNearestTaskColor(const Eigen::MatrixXf& feature) const {
  return getColor(getNearestTask(feature));
}

const hydra::EmbeddingDistance& TaskInformation::metric() const { return *metric_; }

const hydra::EmbeddingGroup& TaskInformation::embeddings() const { return *tasks_; }

}  // namespace clio
