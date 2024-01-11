#include "hydra_llm_ros/hydra_llm_pipeline.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/validation.h>
#include <config_utilities/virtual_config.h>
#include <hydra/backend/backend_module.h>
#include <hydra/backend/update_functions.h>
#include <hydra/utils/timing_utilities.h>
#include <hydra_llm/places_clustering.h>
#include <hydra_llm/view_database.h>

#include "hydra_llm_ros/llm_frontend.h"

namespace hydra::llm {

using timing::ScopedTimer;

RegionUpdateFunctor::RegionUpdateFunctor(const Config& config) {
  places_clustering = config.extractor.create();
}

MergeMap RegionUpdateFunctor::call(SharedDsgInfo& dsg, const UpdateInfo& info) const {
  ScopedTimer timer("backend/region_clustering", info.timestamp_ns);
  {  // start critical section to update features
    std::lock_guard<std::mutex> lock(feature_mutex);
    for (auto&& [id, clip] : latest_features) {
      place_features[id] = clip;
    }
    latest_features.clear();
  }

  places_clustering->clusterPlaces(*dsg.graph, place_features);
  return {};
}

void declare_config(RegionUpdateFunctor::Config& config) {
  using namespace config;
  name("RegionUpdateFunctorConfig::Config");
  field(config.extractor, "extractor");
}

HydraLLMPipeline::HydraLLMPipeline(const ros::NodeHandle& nh, int robot_id)
    : HydraRosPipeline(nh, robot_id) {}

HydraLLMPipeline::~HydraLLMPipeline() {}

void HydraLLMPipeline::init() {
  HydraRosPipeline::init();
  const auto conf = config::checkValid(config::fromRos<RegionUpdateFunctor::Config>(
      ros::NodeHandle(nh_, "backend/regions")));
  region_clustering_ = std::make_unique<RegionUpdateFunctor>(conf);

  auto backend = getModule<BackendModule>("backend");
  CHECK(backend);
  backend->setUpdateFunctor(DsgLayers::ROOMS, region_clustering_);

  auto frontend = getModule<LLMFrontend>("frontend");
  CHECK(frontend);
  frontend->addViewCallback(
      [this](const ViewDatabase& db, const std::map<NodeId, NodeId>& best_views) {
        std::lock_guard<std::mutex> lock(region_clustering_->feature_mutex);
        for (auto&& [place, agent] : best_views) {
          const auto entry = db.getView(agent);
          if (!entry) {
            continue;
          }

          region_clustering_->latest_features[place] = entry->clip.get();
        }
      });
}

}  // namespace hydra::llm
