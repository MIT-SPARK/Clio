#pragma once

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
  double norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

  double score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

 private:
  inline static const auto registration_ =
      config::Registration<EmbeddingNorm, CosineDistance>("cosine");
};

struct L2Norm : public EmbeddingNorm {
  double norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

  double score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

 private:
  inline static const auto registration_ =
      config::Registration<EmbeddingNorm, L2Norm>("l2");
};

struct L1Norm : public EmbeddingNorm {
  double norm(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

  double score(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) const override;

 private:
  inline static const auto registration_ =
      config::Registration<EmbeddingNorm, L1Norm>("l1");
};

}  // namespace hydra::llm
