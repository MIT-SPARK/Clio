#pragma once
#include "hydra_llm/clustering.h"

namespace hydra::llm {

class NoClustering : public Clustering {
 public:
  struct Config : public Clustering::Config {};

  NoClustering(const Config& config);

  virtual ~NoClustering() = default;

  std::vector<Cluster::Ptr> cluster(const SceneGraphLayer& layer,
                                    const NodeEmbeddingMap& embeddings) const override;

  inline static const auto registration_ =
      config::RegistrationWithConfig<Clustering, NoClustering, NoClustering::Config>(
          "NoClustering");
};

void declare_config(NoClustering::Config& config);

}  // namespace hydra::llm
