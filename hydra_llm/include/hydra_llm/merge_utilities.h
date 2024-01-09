#pragma once
#include <config_utilities/factory.h>

#include "hydra_llm/clip_types.h"

namespace hydra::llm {

struct EmbeddingMerger {
  using Ptr = std::unique_ptr<EmbeddingMerger>;

  virtual ClipEmbedding::Ptr merge(const ClipEmbedding& lhs,
                                   double lhs_score,
                                   const ClipEmbedding& rhs,
                                   double rhs_score) const = 0;
};

struct MeanMerger : EmbeddingMerger {
  ClipEmbedding::Ptr merge(const ClipEmbedding& lhs,
                           double lhs_score,
                           const ClipEmbedding& rhs,
                           double rhs_score) const override;

 private:
  inline static const auto registration_ =
      config::Registration<EmbeddingMerger, MeanMerger>("mean");
};

struct WeightedMeanMerger : EmbeddingMerger {
  ClipEmbedding::Ptr merge(const ClipEmbedding& lhs,
                           double lhs_score,
                           const ClipEmbedding& rhs,
                           double rhs_score) const override;

 private:
  inline static const auto registration_ =
      config::Registration<EmbeddingMerger, WeightedMeanMerger>("weighted_mean");
};

struct MaxMerger : EmbeddingMerger {
  ClipEmbedding::Ptr merge(const ClipEmbedding& lhs,
                           double lhs_score,
                           const ClipEmbedding& rhs,
                           double rhs_score) const override;

 private:
  inline static const auto registration_ =
      config::Registration<EmbeddingMerger, MaxMerger>("max");
};

}  // namespace hydra::llm
