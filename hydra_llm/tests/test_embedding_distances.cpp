#include <gtest/gtest.h>
#include <hydra_llm/embedding_distances.h>

namespace hydra::llm {

TEST(EmbeddingDistances, TestCosineCorrect) {
  Eigen::VectorXd a = Eigen::VectorXd::Zero(10);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(10);
  a(0) = 1.0;
  b(0) = -1.0;
  CosineDistance dist;
  EXPECT_NEAR(dist.score(a, b), -1.0, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 2.0, 1.0e-9);

  b(0) = 2.0;
  EXPECT_NEAR(dist.score(a, b), 1.0, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 0.0, 1.0e-9);

  b(0) = 0.0;
  EXPECT_NEAR(dist.score(a, b), 0.0, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 1.0, 1.0e-9);

  a(0) = 0.0;
  EXPECT_NEAR(dist.score(a, b), 0.0, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 1.0, 1.0e-9);
}

TEST(EmbeddingDistances, TestL1Norm) {
  Eigen::VectorXd a = Eigen::VectorXd::Zero(10);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(10);
  a(0) = 1.0;
  b(0) = -1.0;
  L1Norm dist;
  EXPECT_NEAR(dist.score(a, b), 0.0, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 2.0, 1.0e-9);

  b(0) = 2.0;
  EXPECT_NEAR(dist.score(a, b), 1.0, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 0.0, 1.0e-9);

  b(0) = 0.0;
  EXPECT_NEAR(dist.score(a, b), 0.5, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 1.0, 1.0e-9);

  a(0) = 0.0;
  EXPECT_NEAR(dist.score(a, b), 1.0, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 0.0, 1.0e-9);
}

TEST(EmbeddingDistances, TestL2Norm) {
  Eigen::VectorXd a = Eigen::VectorXd::Zero(10);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(10);
  a(0) = 1.0;
  b(0) = -1.0;
  L2Norm dist;
  EXPECT_NEAR(dist.score(a, b), 0.0, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 2.0, 1.0e-9);

  b(0) = 2.0;
  EXPECT_NEAR(dist.score(a, b), 1.0, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 0.0, 1.0e-9);

  b(0) = 0.0;
  EXPECT_NEAR(dist.score(a, b), 0.5, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 1.0, 1.0e-9);

  a(0) = 0.0;
  EXPECT_NEAR(dist.score(a, b), 1.0, 1.0e-9);
  EXPECT_NEAR(dist.dist(a, b), 0.0, 1.0e-9);
}

struct TestEmbeddingGroup : public EmbeddingGroup {
  struct Config {};

  explicit TestEmbeddingGroup(const Config&) {
    embeddings.push_back(Eigen::VectorXd::Zero(10));
    embeddings.push_back(Eigen::VectorXd::Zero(10));
    embeddings[1](0) = 1.0;
  }

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingGroup, TestEmbeddingGroup, Config>(
          "test_group");
};

void declare_config(TestEmbeddingGroup::Config&) {}

TEST(EmbeddingDistances, TestLerf) {
  LerfScore::Config config;
  config.cannonical_features =
      config::VirtualConfig<EmbeddingGroup>(TestEmbeddingGroup::Config(), "test_group");
  LerfScore dist(config);

  Eigen::VectorXd a = Eigen::VectorXd::Zero(10);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(10);
  a(0) = 1.0;
  b(0) = -1.0;
  {
    const double expected = std::exp(-1.0) / (std::exp(-1.0) + std::exp(1.0));
    EXPECT_NEAR(dist.score(a, b), expected, 1.0e-9);
  }

  b(0) = 2.0;
  EXPECT_NEAR(dist.score(a, b), 0.5, 1.0e-9);

  b(0) = 0.0;
  {
    const double expected = 1 / (1 + std::exp(1.0));
    EXPECT_NEAR(dist.score(a, b), expected, 1.0e-9);
  }

  a(0) = 0.0;
  EXPECT_NEAR(dist.score(a, b), 0.5, 1.0e-9);
}

}  // namespace hydra::llm
