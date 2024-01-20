#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/frontend/frontend_module.h>
#include <hydra_llm/places_clustering.h>
#include <hydra_llm/view_database.h>
#include <llm/ClipVectorStamped.h>

namespace hydra::llm {

struct LLMFrontendConfig : public FrontendConfig {
  double spatial_window_radius_m = 8.0;
  bool override_active_window = false;
  double min_object_merge_similiarity = 0.3;
};

void declare_config(LLMFrontendConfig& config);

class LLMFrontend : public FrontendModule {
 public:
  using ViewCallback =
      std::function<void(const ViewDatabase&, const std::map<NodeId, NodeId>&)>;

  LLMFrontend(const LLMFrontendConfig& config,
              const SharedDsgInfo::Ptr& dsg,
              const SharedModuleState::Ptr& state,
              const LogSetup::Ptr& logs = nullptr);

  virtual ~LLMFrontend();

  const LLMFrontendConfig config;

  void addViewCallback(const ViewCallback& func);

  void setSensor(const std::shared_ptr<Sensor>& sensor);

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
  std::map<uint64_t, ClipEmbedding::Ptr> keyframe_clip_vectors_;
  ViewDatabase::Ptr views_database_;
  std::list<ViewCallback> view_callbacks_;
  std::set<NodeId> new_objects_;
  std::shared_ptr<VolumetricMap> map_;
  voxblox::BlockIndexList archived_blocks_;

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
