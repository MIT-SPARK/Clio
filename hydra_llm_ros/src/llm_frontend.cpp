#include "hydra_llm_ros/llm_frontend.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/hydra_config.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include <hydra/utils/timing_utilities.h>

namespace hydra::llm {

using hydra::timing::ScopedTimer;

void declare_config(LLMFrontendConfig& config) {
  using namespace config;
  name("LLMFrontendConfig");
  base<FrontendConfig>(config);
  config.clustering.setOptional();
  field(config.clustering, "clustering");
}

LLMFrontend::LLMFrontend(const LLMFrontendConfig& config,
                         const SharedDsgInfo::Ptr& dsg,
                         const SharedModuleState::Ptr& state,
                         const LogSetup::Ptr& logs)
    : FrontendModule(config, dsg, state, logs),
      config(config::checkValid(config)),
      nh_("~") {
  places_clustering_ = config.clustering.create();
  if (places_clustering_) {
    clip_sub_ =
        nh_.subscribe("input/clip_vector", 10, &LLMFrontend::handleClipFeatures, this);
  }
}

LLMFrontend::~LLMFrontend() {}

void LLMFrontend::handleClipFeatures(const ::llm::ClipVectorStamped& msg) {
  auto view = std::make_unique<ClipViewEmbedding>();
  view->timestamp_ns = msg.header.stamp.toNSec();
  view->embedding = Eigen::Map<const Eigen::VectorXd>(msg.embedding.elements.data(),
                                                      msg.embedding.elements.size());
  // start critical section to push clip vector from ROS thread
  std::lock_guard<std::mutex> lock(clip_mutex_);
  keyframe_clip_vectors_.push_back(std::move(view));
}

void LLMFrontend::updateActiveWindowViews(uint64_t curr_timestamp_ns) {
  ScopedTimer timer("frontend/update_active_views", curr_timestamp_ns);
  const auto& prefix = HydraConfig::instance().getRobotPrefix();
  const auto& active_nodes = active_agent_nodes_.at(prefix.key);
  if (active_nodes.empty()) {
    return;
  }

  auto window_iter = active_window_views_.begin();
  while (window_iter != active_window_views_.end()) {
    if (!active_nodes.count(window_iter->first)) {
      window_iter = active_window_views_.erase(window_iter);
    } else {
      ++window_iter;
    }
  }

  // assumes active nodes remains sorted
  const auto& agents = dsg_->graph->getLayer(DsgLayers::AGENTS, prefix.key);
  std::map<uint64_t, size_t> timestamp_map;
  for (const auto index : active_nodes) {
    // TODO(nathan) think about subsampling the timestamps to include more than
    // just keyframes
    const DynamicSceneGraphNode& node = agents.getNodeByIndex(index).value();
    timestamp_map[index] = node.timestamp.count();
  }

  auto iter = keyframe_clip_vectors_.begin();
  while (iter != keyframe_clip_vectors_.end()) {
    const auto clip_stamp_ns = (*iter)->timestamp_ns;
    if (clip_stamp_ns > curr_timestamp_ns) {
      // skip any observations outside the time horizon of the active window
      continue;
    }

    auto stamp_iter = timestamp_map.find(clip_stamp_ns);
    if (stamp_iter == timestamp_map.end()) {
      // this frame is inside the time horizon of the active window (latest timestamp
      // defines horizon) and not observed. Assumes lastest timestamp is keyframe
      iter = keyframe_clip_vectors_.erase(iter);
      continue;
    }

    const auto keyframe_idx = stamp_iter->second;
    auto view = std::make_shared<ClipView>();
    view->clip = std::move(*iter);
    iter = keyframe_clip_vectors_.erase(iter);

    view->sensor = keyframe_sensor_map_.at(keyframe_idx);
    const auto& attrs =
        agents.getNodeByIndex(keyframe_idx)->get().attributes<AgentNodeAttributes>();
    Eigen::Isometry3d world_T_body =
        Eigen::Translation3d(attrs.position) * attrs.world_R_body;
    view->world_T_sensor = world_T_body * view->sensor->body_T_sensor();

    active_window_views_[keyframe_idx] = view;
  }
}

void LLMFrontend::updateImpl(const ReconstructionOutput& msg) {
  FrontendModule::updateImpl(msg);
  if (places_clustering_) {
    // okay without locking: we're not modifying the graph
    updateActiveWindowViews(msg.timestamp_ns);

    // start critical section for modifying graph
    std::lock_guard<std::mutex> lock(dsg_->mutex);
    places_clustering_->clusterPlaces(
        *dsg_->graph, active_window_views_, previous_active_places_);
  }
}

}  // namespace hydra::llm
