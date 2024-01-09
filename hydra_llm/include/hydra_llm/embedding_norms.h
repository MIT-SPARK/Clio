#pragma once
#include <config_utilities/config.h>
#include <config_utilities/factory.h>

#include <Eigen/Dense>

namespace hydra::llm {

struct EmbeddingNorm {
  virtual ~EmbeddingNorm() = default;

  inline double operator()(const Eigen::VectorXd& lhs,
                           const Eigen::VectorXd& rhs) const {
    return score(lhs, rhs);
  }

  virtual double norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const = 0;

  virtual double score(const Eigen::VectorXd& lhs,
                       const Eigen::VectorXd& rhs) const = 0;
};

struct CosineDistance : EmbeddingNorm {
  struct Config {};

  explicit CosineDistance(const Config& = {}) {}

  double norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

  double score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingNorm, CosineDistance, Config>("cosine");
};

inline void declare_config(CosineDistance::Config&) {
  config::name("CosineDistance::Config");
}

struct L1Norm : public EmbeddingNorm {
  struct Config {};

  explicit L1Norm(const Config& = {}) {}

  double norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

  double score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingNorm, L1Norm, Config>("l1");
};

inline void declare_config(L1Norm::Config&) { config::name("L1Norm::Config"); }

struct L2Norm : public EmbeddingNorm {
  struct Config {};

  explicit L2Norm(const Config& = {}) {}

  double norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

  double score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingNorm, L2Norm, Config>("l2");
};

inline void declare_config(L2Norm::Config&) { config::name("L2Norm::Config"); }

}  // namespace hydra::llm
