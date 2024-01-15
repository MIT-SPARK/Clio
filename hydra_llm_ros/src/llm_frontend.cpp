#include "hydra_llm_ros/llm_frontend.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/hydra_config.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include <hydra/utils/timing_utilities.h>
#include <khronos/active_window/data/output_data.h>
#include <khronos/common/utils/khronos_attribute_utils.h>

namespace hydra::llm {

using hydra::timing::ScopedTimer;

void declare_config(LLMFrontendConfig& config) {
  using namespace config;
  name("LLMFrontendConfig");
  base<FrontendConfig>(config);
}

LLMFrontend::LLMFrontend(const LLMFrontendConfig& config,
                         const SharedDsgInfo::Ptr& dsg,
                         const SharedModuleState::Ptr& state,
                         const LogSetup::Ptr& logs)
    : FrontendModule(config, dsg, state, logs),
      config(config::checkValid(config)),
      nh_("~") {
  views_database_ = std::make_shared<ViewDatabase>();
  clip_sub_ =
      nh_.subscribe("input/clip_vector", 10, &LLMFrontend::handleClipFeatures, this);
}

LLMFrontend::~LLMFrontend() {}

void LLMFrontend::setSensor(const std::shared_ptr<Sensor>& sensor) {
  view_database_->setSensor(sensor);
}

void LLMFrontend::initCallbacks() {
  initialized_ = true;
  input_callbacks_.clear();
  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updateMesh, this, std::placeholders::_1));
  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updateDeformationGraph, this, std::placeholders::_1));
  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updatePoseGraph, this, std::placeholders::_1));
  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updateKhronosObjects, this, std::placeholders::_1));

  post_mesh_callbacks_.clear();

  if (!place_extractor_) {
    return;
  }

  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updatePlaces, this, std::placeholders::_1));
}

void LLMFrontend::updateKhronosObjects(const ReconstructionOutput& base_msg) {
  // this is janky, but the try-catch is uglier
  const auto derived = dynamic_cast<const khronos::OutputData*>(&base_msg);
  CHECK(derived) << "ActiveWindow output required!";
  const auto& msg = *derived;

  std::lock_guard<std::mutex> lock(dsg_->mutex);
  ScopedTimer timer("frontend/update_objects", msg.timestamp_ns);

  for (const auto& object : msg.objects) {
    const NodeSymbol node_id(config.object_config.prefix, object.id);
    CHECK(!dsg_->graph->hasNode(node_id))
        << "Found duplicate node " << node_id.getLabel();

    auto attrs = khronos::fromOutputObject(object);
    dsg_->graph->emplaceNode(DsgLayers::OBJECTS, node_id, std::move(attrs));
  }
}

void LLMFrontend::handleClipFeatures(const ::llm::ClipVectorStamped& msg) {
  const auto timestamp_ns = msg.header.stamp.toNSec();
  auto clip = std::make_unique<ClipEmbedding>(msg.embedding.elements);
  // start critical section to push clip vector from ROS thread
  std::lock_guard<std::mutex> lock(clip_mutex_);
  keyframe_clip_vectors_.emplace(timestamp_ns, std::move(clip));
}

void LLMFrontend::updateActiveWindowViews(uint64_t curr_timestamp_ns) {
  ScopedTimer timer("frontend/update_active_views", curr_timestamp_ns);
  const auto& prefix = HydraConfig::instance().getRobotPrefix();
  const auto& active_nodes = active_agent_nodes_.at(prefix.key);
  if (active_nodes.empty()) {
    return;
  }

  // assumes active nodes remains sorted
  const auto& agents = dsg_->graph->getLayer(DsgLayers::AGENTS, prefix.key);
  std::map<uint64_t, NodeId> timestamp_map;
  for (const auto index : active_nodes) {
    // TODO(nathan) think about subsampling the timestamps to include more than
    // just keyframes
    const DynamicSceneGraphNode& node = agents.getNodeByIndex(index).value();
    timestamp_map[node.timestamp.count()] = node.id;
  }

  auto iter = keyframe_clip_vectors_.begin();
  while (iter != keyframe_clip_vectors_.end()) {
    const auto clip_stamp_ns = iter->first;
    if (clip_stamp_ns > curr_timestamp_ns) {
      // skip any observations outside the time horizon of the active window
      ++iter;
      continue;
    }

    auto stamp_iter = timestamp_map.find(clip_stamp_ns);
    if (stamp_iter == timestamp_map.end()) {
      // this frame is inside the time horizon of the active window (latest timestamp
      // defines horizon) and not observed. Assumes lastest timestamp is keyframe
      iter = keyframe_clip_vectors_.erase(iter);
      continue;
    }

    const auto node_id = stamp_iter->second;
    views_database_->addView(node_id, std::move(iter->second));
    iter = keyframe_clip_vectors_.erase(iter);
  }
}

void LLMFrontend::addViewCallback(const ViewCallback& func) {
  view_callbacks_.push_back(func);
}

void LLMFrontend::updateImpl(const ReconstructionOutput& msg) {
  FrontendModule::updateImpl(msg);
  // okay without locking: we're not modifying the graph
  updateActiveWindowViews(msg.timestamp_ns);

  const auto& prefix = HydraConfig::instance().getRobotPrefix();
  const auto& active_nodes = active_agent_nodes_.at(prefix.key);
  if (active_nodes.empty()) {
    return;
  }

  std::vector<NodeId> active_ids;
  for (const auto index : active_nodes) {
    active_ids.push_back(NodeSymbol(prefix.key, index));
  }

  std::map<NodeId, NodeId> best_views;
  views_database_->updateAssignments(
      *dsg_->graph, active_ids, previous_active_places_, best_views);
  for (const auto& cb : view_callbacks_) {
    cb(*views_database_, best_views);
  }
}

}  // namespace hydra::llm
