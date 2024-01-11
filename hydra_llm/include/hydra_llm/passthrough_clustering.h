#pragma once
#include "hydra_llm/clustering.h"
#include "hydra_llm/embedding_norms.h"

namespace hydra::llm {

using NodeEmbeddingMap = std::map<NodeId, const ClipEmbedding*>;

class PassthroughClustering : public Clustering {
 public:
  struct Config : Clustering::Config {
    config::VirtualConfig<EmbeddingNorm> norm;
  };

  PassthroughClustering(const Config& config);

  std::vector<Cluster::Ptr> cluster(const SceneGraphLayer& layer,
                                    const NodeEmbeddingMap& embeddings) const override;

  const Config config;

 private:
  std::unique_ptr<EmbeddingNorm> norm_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<Clustering,
                                     PassthroughClustering,
                                     PassthroughClustering::Config>(
          "PassthroughClustering");
};

void declare_config(PassthroughClustering::Config& config);

}  // namespace hydra::llm
