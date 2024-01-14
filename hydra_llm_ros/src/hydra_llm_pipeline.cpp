#include "hydra_llm_ros/hydra_llm_pipeline.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/validation.h>
#include <hydra/backend/backend_module.h>
#include <hydra_llm/view_database.h>

#include "hydra_llm_ros/llm_frontend.h"

namespace hydra::llm {

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
        region_clustering_->updateFromViewDb(db, best_views);
      });
}

}  // namespace hydra::llm
