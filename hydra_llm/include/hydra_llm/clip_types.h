#pragma once
#include <hydra/reconstruction/sensor.h>

#include <Eigen/Geometry>
#include <memory>

namespace hydra::llm {

struct ClipViewEmbedding {
  using Ptr = std::unique_ptr<ClipViewEmbedding>;
  uint64_t timestamp_ns;
  Eigen::VectorXd embedding;
};

struct ClipView {
  using Ptr = std::shared_ptr<ClipView>;
  Eigen::Isometry3d world_T_sensor;
  ClipViewEmbedding::Ptr clip;
  Sensor::Ptr sensor;
};

}  // namespace hydra::llm
