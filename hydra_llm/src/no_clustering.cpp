#include "hydra_llm/no_clustering.h"

namespace hydra::llm {

NoClustering::NoClustering(const Config& config) : Clustering(config) {}

std::vector<Cluster::Ptr> NoClustering::cluster(const SceneGraphLayer&,
                                                const NodeEmbeddingMap&) const {
  return {};
}

void declare_config(NoClustering::Config& config) { config.tasks.setOptional(); }

}  // namespace hydra::llm
