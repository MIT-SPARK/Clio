#include "hydra_llm_ros/llm_frontend.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/hydra_config.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include <hydra/utils/timing_utilities.h>
#include <hydra_llm/embedding_distances.h>
#include <khronos/active_window/data/output_data.h>
#include <khronos/common/utils/globals.h>
#include <khronos/common/utils/khronos_attribute_utils.h>
#include <kimera_pgmo/compression/DeltaCompression.h>

namespace hydra::llm {

using hydra::timing::ScopedTimer;

void declare_config(LLMFrontendConfig& config) {
  using namespace config;
  name("LLMFrontendConfig");
  base<FrontendModule::Config>(config);
  field(config.enable_object_clustering, "enable_object_clustering");
  field(config.spatial_window_radius_m, "spatial_window_radius_m");
  field(config.override_active_window, "override_active_window");
  field(config.min_object_merge_similiarity, "min_object_merge_similiarity");
  field(config.view_database, "view_database");
  config.tasks.setOptional();
  field(config.tasks, "tasks");
  config.metric.setOptional();
  field(config.metric, "metric");
  field(config.min_object_score, "min_object_score");
}

LLMFrontend::LLMFrontend(const LLMFrontendConfig& config,
                         const SharedDsgInfo::Ptr& dsg,
                         const SharedModuleState::Ptr& state,
                         const LogSetup::Ptr& logs)
    : FrontendModule(config, dsg, state, logs),
      config(config::checkValid(config)),
      nh_("~") {
  views_database_ = std::make_shared<ViewDatabase>(config.view_database);
  clip_sub_ =
      nh_.subscribe("input/clip_vector", 10, &LLMFrontend::handleClipFeatures, this);
  if (config.min_object_score > 0.0) {
    tasks_ = config.tasks.create();
    metric_ = config.metric.create();
  }
}

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
  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updateKhronosObjects, this, std::placeholders::_1));

  post_mesh_callbacks_.clear();

  if (!place_extractor_) {
    return;
  }

  input_callbacks_.push_back(
      std::bind(&LLMFrontend::updateLLmPlaces, this, std::placeholders::_1));
}

bool isSameClass(const khronos::OutputObject& object,
                 const SemanticNodeAttributes& attrs,
                 double min_similiarity) {
  // note: eigen default vectors are not size 0, but we likely will never
  // have scalar semantic features
  if (object.semantic_feature.size() <= 1) {
    return static_cast<uint32_t>(object.semantic_id) == attrs.semantic_label;
  }

  const auto score =
      CosineDistance().dist(object.semantic_feature, attrs.semantic_feature);
  return score >= min_similiarity;
}

bool objectsOverlap(const spark_dsg::BoundingBox& bbox,
                    const Eigen::Vector3f& pos,
                    const SemanticNodeAttributes& attrs) {
  return bbox.isInside(attrs.position) || attrs.bounding_box.isInside(pos);
}

void LLMFrontend::updateKhronosObjects(const ReconstructionOutput& base_msg) {
  // this is janky, but the try-catch is uglier
  const auto derived = dynamic_cast<const khronos::OutputData*>(&base_msg);
  CHECK(derived) << "ActiveWindow output required!";
  const auto& msg = *derived;

  std::lock_guard<std::mutex> lock(dsg_->mutex);
  ScopedTimer timer("frontend/update_objects", msg.timestamp_ns);

  const auto layer_id =
      config.enable_object_clustering ? DsgLayers::SEGMENTS : DsgLayers::OBJECTS;

  VLOG(VLEVEL_TRACE) << "Got " << msg.objects.size() << " new objects ("
                     << dsg_->graph->getLayer(layer_id).numNodes() << " total nodes)";
  for (const auto& object : msg.objects) {
    if (metric_ && tasks_ && !tasks_->empty() && object.semantic_feature.size() > 1) {
      const Eigen::VectorXd feature = object.semantic_feature.rowwise().mean();
      const auto result = tasks_->getBestScore(*metric_, feature);
      if (result.score < config.min_object_score) {
        LOG(ERROR) << "Skipping object with score: " << result.score;
        continue;
      }
    }

    const NodeSymbol node_id(config.object_config.prefix, object.id);
    CHECK(!dsg_->graph->hasNode(node_id))
        << "Found duplicate node " << node_id.getLabel();

    auto attrs = khronos::fromOutputObject(object);
    attrs->is_active = true;
    CHECK(dsg_->graph->emplaceNode(layer_id, node_id, std::move(attrs)));
    new_objects_.insert(node_id);
  }
}

inline Eigen::VectorXd msgToVec(const ::llm::ClipVector& msg) {
  return Eigen::Map<const Eigen::VectorXd>(msg.elements.data(), msg.elements.size());
}

void LLMFrontend::handleClipFeatures(const ::llm::ClipVectorStamped& msg) {
  const auto timestamp_ns = msg.header.stamp.toNSec();
  // start critical section to push clip vector from ROS thread
  std::lock_guard<std::mutex> lock(clip_mutex_);
  keyframe_clip_vectors_.emplace(timestamp_ns, msgToVec(msg.embedding));
}

void LLMFrontend::updateActiveWindowViews(uint64_t curr_timestamp_ns) {
  ScopedTimer timer("frontend/update_active_views", curr_timestamp_ns);
  const auto& prefix = HydraConfig::instance().getRobotPrefix();
  if (!active_agent_nodes_.count(prefix.key)) {
    return;
  }

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
    // TODO(nathan) handle the clip view assumption better
    const auto sensor = CHECK_NOTNULL(khronos::Globals::getSensor(0));
    views_database_->addView(node_id, std::move(iter->second), sensor);
    iter = keyframe_clip_vectors_.erase(iter);
  }
}

void LLMFrontend::archiveObjects() {
  for (const auto prev : previous_active_places_) {
    const auto has_prev_node = dsg_->graph->getNode(prev);
    if (!has_prev_node) {
      continue;
    }

    const SceneGraphNode& node = *has_prev_node;
    if (node.attributes().is_active) {
      continue;
    }

    for (const auto& child : node.children()) {
      if (!new_objects_.count(child)) {
        continue;
      }

      auto& attrs = dsg_->graph->getNode(child)->get().attributes<NodeAttributes>();
      attrs.is_active = false;
      new_objects_.erase(child);
    }
  }
}

void LLMFrontend::connectNewObjects() {
  if (!places_nn_finder_) {
    return;
  }

  for (const auto& object_id : new_objects_) {
    const auto object_opt = dsg_->graph->getNode(object_id);
    if (!object_opt) {
      continue;
    }

    const SceneGraphNode& object_node = *object_opt;
    const auto parent_opt = object_node.getParent();
    if (parent_opt) {
      dsg_->graph->removeEdge(object_id, *parent_opt);
    }

    const Eigen::Vector3d object_position = dsg_->graph->getPosition(object_id);
    places_nn_finder_->find(
        object_position, 1, false, [&](NodeId place_id, size_t, double) {
          dsg_->graph->insertEdge(place_id, object_id);
        });
  }
}

void LLMFrontend::updateBestViews() {
  views_database_->updateAssignments(*dsg_->graph, previous_active_places_);
}

void LLMFrontend::updateLLmPlaces(const ReconstructionOutput& input) {
  if (!place_extractor_ || !map_) {
    return;
  }

  NodeIdSet active_nodes;
  place_extractor_->detectImpl(
      input.timestamp_ns, input.world_T_body<float>(), archived_blocks_, *map_);
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

void LLMFrontend::updateMap(ReconstructionOutput& msg) {
  if (!map_) {
    map_ = msg.getMapPointer();
    return;
  }

  // copy updates into larger map
  // TODO(nathan) filter empty blocks?
  map_->updateFrom(msg.map());

  const auto& tsdf = map_->getTsdfLayer();

  archived_blocks_.clear();
  voxblox::BlockIndexList blocks;
  tsdf.getAllAllocatedBlocks(&blocks);
  for (const auto& idx : blocks) {
    auto block = tsdf.getBlockPtrByIndex(idx);
    const auto dist_m = (msg.world_t_body.cast<float>() - block->origin()).norm();
    if (dist_m < config.spatial_window_radius_m) {
      continue;
    }

    archived_blocks_.push_back(idx);
  }

  // we override the map to use a spatial active window for
  // the places and other processes
  if (config.override_active_window) {
    msg.setMap(map_);
    msg.archived_blocks = archived_blocks_;
  }
}

void LLMFrontend::pruneMap(const ReconstructionOutput&) {
  if (!map_) {
    return;
  }

  map_->removeBlocks(archived_blocks_);

  auto& tsdf = map_->getTsdfLayer();
  voxblox::BlockIndexList blocks;
  tsdf.getAllAllocatedBlocks(&blocks);
  for (const auto& idx : blocks) {
    tsdf.getBlockPtrByIndex(idx)->updated().reset();
  }
}

void LLMFrontend::updateImpl(const ReconstructionOutput::Ptr& msg) {
  updateMap(*msg);

  FrontendModule::updateImpl(msg);

  // okay without locking: no one else is modifying the graph
  archiveObjects();
  connectNewObjects();
  updateActiveWindowViews(msg->timestamp_ns);
  updateBestViews();

  pruneMap(*msg);
}

}  // namespace hydra::llm
