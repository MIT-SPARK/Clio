#pragma once
#include "hydra_llm/clustering.h"
#include "hydra_llm/edge_selector.h"
#include "hydra_llm/embedding_distances.h"

namespace hydra::llm {

using Clusters = std::vector<Cluster::Ptr>;

class AgglomerativeClustering : public Clustering {
 public:
  struct Config : Clustering::Config {
    config::VirtualConfig<EmbeddingDistance> metric;
    config::VirtualConfig<EdgeSelector> selector;
  };

  AgglomerativeClustering(const Config& config);

  Clusters cluster(const SceneGraphLayer& layer,
                   const NodeEmbeddingMap& embeddings) const override;

  Clusters getClusters(const std::vector<size_t>& assignments,
                       const std::map<size_t, NodeId>& node_lookup,
                       const NodeEmbeddingMap& features) const;

  const Config config;

 private:
  std::unique_ptr<EmbeddingDistance> metric_;
  EdgeSelector::Ptr edge_selector_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<Clustering,
                                     AgglomerativeClustering,
                                     AgglomerativeClustering::Config>(
          "AgglomerativeClustering");
};

void declare_config(Clustering::Config& config);

}  // namespace hydra::llm
