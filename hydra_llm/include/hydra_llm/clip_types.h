#pragma once
#include <hydra/reconstruction/sensor.h>

#include <Eigen/Geometry>
#include <memory>

namespace hydra::llm {

struct ClipEmbedding {
  using Ptr = std::unique_ptr<ClipEmbedding>;

  explicit ClipEmbedding(const Eigen::VectorXd& vec) : embedding(vec) {}

  explicit ClipEmbedding(const std::vector<double>& vec)
      : embedding(Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size())) {}

  const Eigen::VectorXd embedding;
};

struct ClipView {
  using Ptr = std::shared_ptr<ClipView>;
  uint64_t timestamp_ns;
  std::shared_ptr<Sensor> sensor;
  Eigen::Isometry3d sensor_T_world;
  ClipEmbedding::Ptr clip;
};

}  // namespace hydra::llm
