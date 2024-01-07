#include "hydra_llm_ros/llm_frontend.h"

#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include <hydra/utils/timing_utilities.h>

namespace hydra::llm {

using hydra::timing::ScopedTimer;

LLMFrontend::LLMFrontend(const LLMFrontendConfig& config,
                         const SharedDsgInfo::Ptr& dsg,
                         const SharedModuleState::Ptr& state,
                         const LogSetup::Ptr& logs)
    : FrontendModule(config, dsg, state, logs), config(config::checkValid(config)) {}

LLMFrontend::~LLMFrontend() {}

void LLMFrontend::initCallbacks() {
  initialized_ = true;
  input_callbacks_.clear();
  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updateMesh, this, std::placeholders::_1));
  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updateDeformationGraph, this, std::placeholders::_1));
  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updatePoseGraph, this, std::placeholders::_1));

  post_mesh_callbacks_.clear();
  post_mesh_callbacks_.push_back(
      std::bind(&LLMFrontend::updateObjects, this, std::placeholders::_1));

  if (!place_extractor_) {
    return;
  }

  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updatePlaces, this, std::placeholders::_1));
}

void LLMFrontend::updatePlaces(const ReconstructionOutput& input) {
  if (!place_extractor_) {
    return;
  }

  NodeIdSet active_nodes;
  place_extractor_->detect(input);
  {  // start graph critical section
    std::unique_lock<std::mutex> graph_lock(dsg_->mutex);
    place_extractor_->updateGraph(input.timestamp_ns, *dsg_->graph);

    active_nodes = place_extractor_->getActiveNodes();
    const auto& places = dsg_->graph->getLayer(DsgLayers::PLACES);
    places_nn_finder_.reset(new NearestNodeFinder(places, active_nodes));
    addPlaceAgentEdges(input.timestamp_ns);
    addPlaceObjectEdges(input.timestamp_ns);
    state_->latest_places = active_nodes;
  }  // end graph update critical section

  archivePlaces(active_nodes);
  previous_active_places_ = active_nodes;
}

void LLMFrontend::handleClipFeatures(const ::llm::ClipVectorStamped& msg) {
  auto view = std::make_unique<ClipView>();
  view->timestamp_ns = msg.header.stamp.toNSec();
  view->embedding = Eigen::Map<const Eigen::VectorXd>(msg.embedding.elements.data(),
                                                      msg.embedding.elements.size());
  // start critical section to push clip vector from ROS thread
  std::lock_guard<std::mutex> lock(clip_mutex_);
  keyframe_clip_vectors_.push_back(std::move(view));
}

}  // namespace hydra::llm
