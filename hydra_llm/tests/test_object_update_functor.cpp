#include <gtest/gtest.h>
#include <hydra_llm/object_update_functor.h>

namespace hydra::llm {

TEST(ObjectUpdateFunctor, TestOverlap) {
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

}  // namespace hydra::llm
