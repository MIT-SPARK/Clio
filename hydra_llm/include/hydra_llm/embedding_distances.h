#pragma once
#include <config_utilities/config.h>
#include <config_utilities/virtual_config.h>

#include <Eigen/Dense>

#include "hydra_llm/embedding_group.h"

namespace hydra::llm {

struct EmbeddingDistance {
  virtual ~EmbeddingDistance() = default;

  inline double operator()(const Eigen::VectorXd& lhs,
                           const Eigen::VectorXd& rhs) const {
    return score(lhs, rhs);
  }

  virtual double dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const = 0;

  virtual double score(const Eigen::VectorXd& lhs,
                       const Eigen::VectorXd& rhs) const = 0;
};

struct CosineDistance : EmbeddingDistance {
  struct Config {};

  explicit CosineDistance(const Config& = {}) {}

  double dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

  double score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingDistance, CosineDistance, Config>(
          "cosine");
};

inline void declare_config(CosineDistance::Config&) {
  config::name("CosineDistance::Config");
}

struct L1Norm : public EmbeddingDistance {
  struct Config {};

  explicit L1Norm(const Config& = {}) {}

  double dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

  double score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingDistance, L1Norm, Config>("l1");
};

inline void declare_config(L1Norm::Config&) { config::name("L1Norm::Config"); }

struct L2Norm : public EmbeddingDistance {
  struct Config {};

  explicit L2Norm(const Config& = {}) {}

  double dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

  double score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingDistance, L2Norm, Config>("l2");
};

inline void declare_config(L2Norm::Config&) { config::name("L2Norm::Config"); }

struct LerfScore : public EmbeddingDistance {
  struct Config {
    config::VirtualConfig<EmbeddingGroup> cannonical_features;
  };

  explicit LerfScore(const Config& config);

  double dist(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

  double score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

 private:
  std::unique_ptr<EmbeddingGroup> cannonical_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingDistance, LerfScore, Config>("lerf");
};

void declare_config(LerfScore::Config& config);

}  // namespace hydra::llm
