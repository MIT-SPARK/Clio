#pragma once
#include "hydra_llm/clustering.h"
#include "hydra_llm/embedding_norms.h"
#include "hydra_llm/merge_utilities.h"

namespace hydra::llm {

class GreedyClustering : public Clustering {
 public:
  struct Config : Clustering::Config {
    config::VirtualConfig<EmbeddingNorm> norm;
    config::VirtualConfig<EmbeddingMerger> merge;
    double stop_value = 0.0;
    double min_score = 0.022;
  };

  GreedyClustering(const Config& config);

  std::vector<Cluster::Ptr> cluster(const SceneGraphLayer& layer,
                                    const NodeEmbeddingMap& embeddings) const override;

  const Config config;

 private:
  std::unique_ptr<EmbeddingNorm> norm_;
  EmbeddingMerger::Ptr embedding_merge_;

 public:
  const EmbeddingNorm& norm;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<Clustering,
                                     GreedyClustering,
                                     GreedyClustering::Config>("GreedyClustering");
};

void declare_config(Clustering::Config& config);

}  // namespace hydra::llm
