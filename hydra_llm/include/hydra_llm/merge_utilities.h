#pragma once
#include <config_utilities/config.h>
#include <config_utilities/factory.h>

#include "hydra_llm/clip_types.h"

namespace hydra::llm {

struct EmbeddingMerger {
  using Ptr = std::unique_ptr<EmbeddingMerger>;

  // TODO(nathan) refactor to do mutliple merges
  virtual Eigen::VectorXd merge(const Eigen::VectorXd& lhs,
                                double lhs_score,
                                const Eigen::VectorXd& rhs,
                                double rhs_score) const = 0;
};

struct MeanMerger : EmbeddingMerger {
  struct Config {};

  explicit MeanMerger(const Config& = {}) {}

  Eigen::VectorXd merge(const Eigen::VectorXd& lhs,
                        double lhs_score,
                        const Eigen::VectorXd& rhs,
                        double rhs_score) const override;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingMerger, MeanMerger, Config>("mean");
};

inline void declare_config(MeanMerger::Config&) { config::name("MeanMerger::Config"); }

struct WeightedMeanMerger : EmbeddingMerger {
  struct Config {};

  explicit WeightedMeanMerger(const Config& = {}) {}

  Eigen::VectorXd merge(const Eigen::VectorXd& lhs,
                        double lhs_score,
                        const Eigen::VectorXd& rhs,
                        double rhs_score) const override;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingMerger, WeightedMeanMerger, Config>(
          "weighted_mean");
};

inline void declare_config(WeightedMeanMerger::Config&) {
  config::name("WeightedMeanMerger::Config");
}

struct MaxMerger : EmbeddingMerger {
  struct Config {};

  explicit MaxMerger(const Config& = {}) {}

  Eigen::VectorXd merge(const Eigen::VectorXd& lhs,
                        double lhs_score,
                        const Eigen::VectorXd& rhs,
                        double rhs_score) const override;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingMerger, MaxMerger, Config>("max");
};

inline void declare_config(MaxMerger::Config&) { config::name("MaxMerger::Config"); }

}  // namespace hydra::llm
