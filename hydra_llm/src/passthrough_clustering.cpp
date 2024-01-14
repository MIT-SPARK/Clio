#include "hydra_llm/passthrough_clustering.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

namespace hydra::llm {

using Clusters = std::vector<Cluster::Ptr>;

void declare_config(PassthroughClustering::Config& config) {
  using namespace config;
  name("PassthroughClustering::Config");
  base<Clustering::Config>(config);
  field(config.metric, "metric");
}

PassthroughClustering::PassthroughClustering(const Config& config)
    : Clustering(config),
      config(config::checkValid(config)),
      metric_(config.metric.create()) {}

Clusters PassthroughClustering::cluster(const SceneGraphLayer&,
                                        const NodeEmbeddingMap& embeddings) const {
  if (tasks_->empty()) {
    LOG(ERROR) << "No tasks present: cannot cluster";
    return {};
  }

  Clusters to_return;
  for (auto&& [node, clip] : embeddings) {
    if (!clip) {
      LOG(WARNING) << "Node '" << NodeSymbol(node).getLabel()
                   << "' missing clip feature";
      continue;
    }

    const auto result = tasks_->getBestScore(*metric_, clip->embedding);

    auto new_cluster = std::make_shared<Cluster>();
    new_cluster->clip = std::make_unique<ClipEmbedding>(clip->embedding);
    new_cluster->score = result.score;
    new_cluster->best_task_index = result.index;
    new_cluster->best_task_name = tasks_->tasks.at(result.index);
    new_cluster->nodes.insert(node);
    to_return.push_back(new_cluster);
  }

  return to_return;
}

}  // namespace hydra::llm
