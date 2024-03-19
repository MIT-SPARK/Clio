#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/frontend/frontend_module.h>
#include <hydra_llm/embedding_distances.h>
#include <hydra_llm/places_clustering.h>
#include <hydra_llm/view_database.h>
#include <llm/ClipVectorStamped.h>

#include "hydra_llm_ros/ros_embedding_group.h"

namespace hydra::llm {

struct LLMFrontendConfig : public FrontendModule::Config {
  bool enable_object_clustering = false;
  double spatial_window_radius_m = 8.0;
  bool override_active_window = false;
  double min_object_merge_similiarity = 0.3;
  ViewDatabase::Config view_database;
  config::VirtualConfig<EmbeddingGroup> tasks{RosEmbeddingGroup::Config(),
                                              "RosEmbeddingGroup"};
  config::VirtualConfig<EmbeddingDistance> metric{CosineDistance::Config(), "cosine"};
  double min_object_score = 0.0;
};

void declare_config(LLMFrontendConfig& config);

class LLMFrontend : public FrontendModule {
 public:
  LLMFrontend(const LLMFrontendConfig& config,
              const SharedDsgInfo::Ptr& dsg,
              const SharedModuleState::Ptr& state,
              const LogSetup::Ptr& logs = nullptr);

  virtual ~LLMFrontend();

  const LLMFrontendConfig config;

 protected:
  void initCallbacks() override;

  void handleClipFeatures(const ::llm::ClipVectorStamped& msg);

  void updateActiveWindowViews(uint64_t curr_timestamp_ns);

  void updateImpl(const ReconstructionOutput::Ptr& msg) override;

  void updateKhronosObjects(const ReconstructionOutput& base_msg);

  void updateLLmPlaces(const ReconstructionOutput& msg);

  void archiveObjects();

  void connectNewObjects();

  void updateBestViews();

  void updateMap(ReconstructionOutput& msg);

  void pruneMap(const ReconstructionOutput& msg);

 protected:
  ros::NodeHandle nh_;
  ros::Subscriber clip_sub_;

  std::mutex clip_mutex_;
  std::map<uint64_t, Eigen::VectorXd> keyframe_clip_vectors_;
  ViewDatabase::Ptr views_database_;
  std::set<NodeId> new_objects_;
  std::shared_ptr<VolumetricMap> map_;
  voxblox::BlockIndexList archived_blocks_;

  EmbeddingGroup::Ptr tasks_;
  std::unique_ptr<EmbeddingDistance> metric_;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<FrontendModule,
                                     LLMFrontend,
                                     LLMFrontendConfig,
                                     SharedDsgInfo::Ptr,
                                     SharedModuleState::Ptr,
                                     LogSetup::Ptr>("LLMFrontend");
};

}  // namespace hydra::llm
