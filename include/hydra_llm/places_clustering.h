#pragma once
#include <hydra/common/dsg_types.h>
#include <hydra/reconstruction/sensor.h>

#include <set>

namespace hydra::llm {

struct Problem {
  std::set<NodeId> nodes;
};

double getOverlap(const Sensor& sensor,
                  const Eigen::Isometry3d& world_T_sensor,
                  const PlaceNodeAttributes& attrs);

void clusterPlaces(const SceneGraphLayer& places, const Problem& problem);

}  // namespace hydra::llm
