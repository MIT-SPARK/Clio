#pragma once
#include <hydra/backend/update_functions.h>
#include <hydra_llm/places_clustering.h>
#include <hydra_ros/common/hydra_ros_pipeline.h>

#include <mutex>

namespace hydra::llm {

struct RegionUpdateFunctor : dsg_updates::UpdateFunctor {
  struct Config {
    config::VirtualConfig<PlaceClustering> extractor;
  };

  explicit RegionUpdateFunctor(const Config& config);

  MergeMap call(SharedDsgInfo& dsg, const UpdateInfo& info) const override;

  mutable std::mutex feature_mutex;
  mutable NodeEmbeddingMap latest_features;
  mutable NodeEmbeddingMap place_features;
  PlaceClustering::Ptr places_clustering;
};

class HydraLLMPipeline : public HydraRosPipeline {
 public:
  HydraLLMPipeline(const ros::NodeHandle& nh, int robot_id);

  virtual ~HydraLLMPipeline();

  void init() override;

 protected:
  std::shared_ptr<RegionUpdateFunctor> region_clustering_;
};

}  // namespace hydra::llm
