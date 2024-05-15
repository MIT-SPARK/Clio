#pragma once
#include "hydra_llm/clustering.h"
#include "hydra_llm/clustering_workspace.h"
#include "hydra_llm/ib_edge_selector.h"
#include "hydra_llm/embedding_distances.h"

namespace hydra::llm {

using Clusters = std::vector<Cluster::Ptr>;

void clusterAgglomerative(ClusteringWorkspace& ws,
                          const EmbeddingGroup& tasks,
                          EdgeSelector& edge_selector,
                          const EmbeddingDistance& metric,
                          bool reweight = false,
                          double I_xy = -1,
                          double delta_weight = 1,
                          int verbosity = 5);

class AgglomerativeClustering : public Clustering {
 public:
  struct Config : Clustering::Config {
    config::VirtualConfig<EmbeddingDistance> metric{CosineDistance::Config()};
    IBEdgeSelector::Config selector;
    bool filter_regions = false;
  };

  AgglomerativeClustering(const Config& config);

  Clusters cluster(const SceneGraphLayer& layer,
                   const NodeEmbeddingMap& embeddings) const override;

  Clusters getClusters(const ClusteringWorkspace& workspace,
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

void declare_config(AgglomerativeClustering::Config& config);

}  // namespace hydra::llm
