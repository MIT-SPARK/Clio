#include <gtest/gtest.h>
#include <hydra_llm/common.h>
#include <hydra_llm/embedding_distances.h>
#include <hydra_llm/ib_edge_selector.h>

namespace hydra::llm {

struct TestableIBEdgeSelector : public IBEdgeSelector {
  explicit TestableIBEdgeSelector(const IBEdgeSelector::Config& config)
      : IBEdgeSelector(config) {}

  using IBEdgeSelector::px_;
  using IBEdgeSelector::py_;
  using IBEdgeSelector::py_x_;
  using IBEdgeSelector::py_z_;
  using IBEdgeSelector::pz_;
  using IBEdgeSelector::pz_x_;
};

namespace {

inline std::unique_ptr<Eigen::VectorXd> getOneHot(size_t i, size_t dim) {
  Eigen::VectorXd p = Eigen::VectorXd::Zero(dim);
  p(i) = 1.0;
  return std::make_unique<Eigen::VectorXd>(p);
}

}  // namespace

TEST(IBEdgeSelector, SetupCorrect) {
  IsolatedSceneGraphLayer layer(2);
  std::vector<std::unique_ptr<Eigen::VectorXd>> features;
  NodeEmbeddingMap map;
  for (size_t i = 0; i < 5; ++i) {
    layer.emplaceNode(2 * i, std::make_unique<NodeAttributes>());
    features.emplace_back(getOneHot(i, 10));
    map[2 * i] = features[i].get();
  }

  for (size_t i = 0; i < 5; ++i) {
    layer.emplaceNode(2 * i + 1, std::make_unique<NodeAttributes>());
  }

  for (size_t i = 0; i < 4; ++i) {
    layer.insertEdge(2 * i, 2 * (i + 1));
  }

  ClusteringWorkspace ws(layer, map);
  CosineDistance dist;
  // results in nice distribution
  auto ref = Eigen::VectorXd(10);
  ref << 1, 2, 3, 4, 5, 0, 0, 0, 0, 0;
  const auto score_func = [&](const Eigen::VectorXd& x) { return dist(x, ref); };

  IBEdgeSelector::Config config;
  // makes the norms easy
  config.score_threshold = 1.0 / std::sqrt(55);
  TestableIBEdgeSelector selector(config);
  selector.setup(ws, score_func);

  ASSERT_EQ(selector.px_.rows(), 5);
  ASSERT_EQ(selector.pz_.rows(), 5);
  ASSERT_EQ(selector.py_.rows(), 2);
  ASSERT_EQ(selector.pz_x_.rows(), 5);
  ASSERT_EQ(selector.pz_x_.cols(), 5);
  ASSERT_EQ(selector.py_x_.rows(), 2);
  ASSERT_EQ(selector.py_x_.cols(), 5);
  ASSERT_EQ(selector.py_z_.rows(), 2);
  ASSERT_EQ(selector.py_z_.cols(), 5);
  EXPECT_TRUE(selector.px_.isApprox(selector.pz_));
  EXPECT_TRUE(selector.py_x_.isApprox(selector.py_z_));

  const auto fmt = getDefaultFormat();
  Eigen::MatrixXd expected_py_x(2, 5);
  expected_py_x << 0.5, 1.0 / 3.0, 0.25, 0.2, 1.0 / 6.0, 0.5, 2.0 / 3.0, 0.75, 0.8,
      5.0 / 6.0;
  EXPECT_TRUE(selector.py_x_.isApprox(expected_py_x))
      << "expected: " << expected_py_x.format(fmt)
      << ", result: " << selector.py_x_.format(fmt);

  Eigen::VectorXd expected_py(2);
  expected_py << 0.29, 0.71;  // row-wise sum normalized via l1-norm
  EXPECT_TRUE(selector.py_.isApprox(expected_py))
      << "expected: " << expected_py.format(fmt)
      << ", result: " << selector.py_.format(fmt);
}

TEST(IBEdgeSelector, UpdateCorrect) {
  IsolatedSceneGraphLayer layer(2);
  std::vector<std::unique_ptr<Eigen::VectorXd>> features;
  NodeEmbeddingMap map;
  for (size_t i = 0; i < 5; ++i) {
    layer.emplaceNode(2 * i, std::make_unique<NodeAttributes>());
    features.emplace_back(getOneHot(i, 10));
    map[2 * i] = features[i].get();
  }

  for (size_t i = 0; i < 5; ++i) {
    layer.emplaceNode(2 * i + 1, std::make_unique<NodeAttributes>());
  }

  for (size_t i = 0; i < 4; ++i) {
    layer.insertEdge(2 * i, 2 * (i + 1));
  }

  ClusteringWorkspace ws(layer, map);
  CosineDistance dist;
  // results in nice distribution
  auto ref = Eigen::VectorXd(10);
  ref << 1, 2, 3, 4, 5, 0, 0, 0, 0, 0;
  const auto score_func = [&](const Eigen::VectorXd& x) { return dist(x, ref); };

  IBEdgeSelector::Config config;
  config.score_threshold = 1.0 / std::sqrt(55);  // makes the norms easy
  TestableIBEdgeSelector selector(config);
  selector.setup(ws, score_func);
  selector.updateFromEdge(ws, score_func, EdgeKey(0, 1));

  const auto fmt = getDefaultFormat();

  // check that hard-cluster assumptions hold
  EXPECT_EQ(selector.pz_(1), 0.0);
  EXPECT_EQ(selector.pz_(0), 0.4);
  EXPECT_EQ(selector.pz_x_(1, 1), 0.0);
  EXPECT_EQ(selector.pz_x_(0, 0), 1.0);
  EXPECT_EQ(selector.pz_x_(1, 0), 1.0) << "p(z|x): " << selector.pz_x_.format(fmt);

  Eigen::VectorXd py_0(2);
  py_0 << 5.0 / 12.0, 7.0 / 12.0;
  EXPECT_TRUE(selector.py_z_.col(0).isApprox(py_0))
      << "p(y|z=0): " << selector.py_z_.col(0).format(fmt);
  Eigen::VectorXd py_1(2);
  py_1 << 0.0, 0.0;
  EXPECT_TRUE(selector.py_z_.col(1).isApprox(py_1))
      << "p(y|z=1): " << selector.py_z_.col(1).format(fmt);
}

TEST(IBEdgeSelector, CompareEdgesCorrect) {
  IBEdgeSelector::Config config;
  TestableIBEdgeSelector selector(config);
  std::pair<EdgeKey, double> e1{{0, 1}, 0.0};
  std::pair<EdgeKey, double> e2{{0, 1}, 0.1};
  EXPECT_TRUE(selector.compareEdges(e1, e2));
  EXPECT_FALSE(selector.compareEdges(e1, e1));
  EXPECT_FALSE(selector.compareEdges(e2, e1));
}

}  // namespace hydra::llm
