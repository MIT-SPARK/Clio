#include <clio/object_update_functor.h>
#include <gtest/gtest.h>

#include "clio_tests/utilities.h"

namespace clio {

TEST(IntersectionPolicy, TestOverlap) {
  OverlapIntersection::Config config;
  config.tolerance = 0.0;
  OverlapIntersection checker(config);

  KhronosObjectAttributes attrs1;
  attrs1.bounding_box.min << 1.0f, 1.0f, 1.0f;
  attrs1.bounding_box.max << 2.0f, 2.0f, 2.0f;

  KhronosObjectAttributes attrs2;
  attrs2.bounding_box.min << 3.0f, 3.0f, 3.0f;
  attrs2.bounding_box.max << 4.0f, 4.0f, 4.0f;

  EXPECT_FALSE(checker(attrs1, attrs2));

  // make sure the two bounding boxes overlap
  attrs1.bounding_box.max << 3.5f, 3.5f, 3.5f;
  EXPECT_TRUE(checker(attrs1, attrs2));
}

struct ObjectUpdateFunctorTests : public ::testing::Test {
  ObjectUpdateFunctorTests()
      : graph_info({{DsgLayers::SEGMENTS, 's'},
                    {DsgLayers::OBJECTS, 'o'},
                    {DsgLayers::PLACES, 'p'}}) {
    config.tasks = test::TestEmbeddingGroup::getDefault(2);
  }

  void SetUp() override {}

  void addSegment(size_t index,
                  double min,
                  double max,
                  std::optional<size_t> onehot_index = std::nullopt) {
    auto attrs = std::make_unique<KhronosObjectAttributes>();
    attrs->position << (min + max) / 2.0, 0.0, 0.0;

    Eigen::Vector3f x_min(min, -1.0, -1.0);
    Eigen::Vector3f x_max(max, 1.0, 1.0);
    attrs->bounding_box = BoundingBox(x_min, x_max);
    attrs->semantic_feature =
        test::TestEmbeddingGroup::getEmbedding(onehot_index.value_or(index));

    graph().emplaceNode(DsgLayers::SEGMENTS, NodeSymbol('s', index), std::move(attrs));
  }

  DynamicSceneGraph& graph() { return *graph_info.graph; }

  SharedDsgInfo graph_info;
  ObjectUpdateFunctor::Config config;
};

TEST_F(ObjectUpdateFunctorTests, AddEdges) {
  ObjectUpdateFunctor functor(config);
  addSegment(0, -1.0, 1.0, 1);
  addSegment(1, 0.5, 1.5, 1);
  addSegment(2, 2.0, 3.0, 0);

  functor.call(graph_info, {});
  EXPECT_TRUE(graph().hasEdge("s0"_id, "s1"_id));
  EXPECT_FALSE(graph().hasEdge("s0"_id, "s2"_id));
  EXPECT_FALSE(graph().hasEdge("s1"_id, "s2"_id));

  // bridge two components
  addSegment(3, 1.2, 2.5, 0);
  functor.call(graph_info, {});

  EXPECT_FALSE(graph().hasEdge("s3"_id, "s0"_id));
  EXPECT_TRUE(graph().hasEdge("s3"_id, "s1"_id));
  EXPECT_TRUE(graph().hasEdge("s3"_id, "s2"_id));
}

}  // namespace clio
