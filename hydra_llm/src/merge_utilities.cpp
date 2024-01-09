#include "hydra_llm/merge_utilities.h"

namespace hydra::llm {

ClipEmbedding::Ptr MeanMerger::merge(const ClipEmbedding& lhs,
                                     double,
                                     const ClipEmbedding& rhs,
                                     double) const {
  return std::make_unique<ClipEmbedding>((lhs.embedding + rhs.embedding) / 2.0);
}

ClipEmbedding::Ptr WeightedMeanMerger::merge(const ClipEmbedding& lhs,
                                             double lhs_score,
                                             const ClipEmbedding& rhs,
                                             double rhs_score) const {
  return std::make_unique<ClipEmbedding>(
      (lhs_score * lhs.embedding + rhs_score * rhs.embedding) /
      (lhs_score + rhs_score));
}

ClipEmbedding::Ptr MaxMerger::merge(const ClipEmbedding& lhs,
                                    double lhs_score,
                                    const ClipEmbedding& rhs,
                                    double rhs_score) const {
  return lhs_score >= rhs_score ? std::make_unique<ClipEmbedding>(lhs.embedding)
                                : std::make_unique<ClipEmbedding>(rhs.embedding);
}

}  // namespace hydra::llm
