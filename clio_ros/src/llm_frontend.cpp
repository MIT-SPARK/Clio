#include "clio_ros/llm_frontend.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/global_info.h>
#include <hydra/openset/embedding_distances.h>
#include <hydra/utils/nearest_neighbor_utilities.h>
#include <hydra/utils/timing_utilities.h>
#include <khronos/active_window/data/output_data.h>
#include <kimera_pgmo/compression/delta_compression.h>

namespace clio {

using hydra::timing::ScopedTimer;
using namespace spark_dsg;

void declare_config(LLMFrontendConfig& config) {
  using namespace config;
  name("LLMFrontendConfig");
  base<hydra::FrontendModule::Config>(config);
  field(config.spatial_window_radius_m, "spatial_window_radius_m");
  field(config.override_active_window, "override_active_window");
}

LLMFrontend::LLMFrontend(const LLMFrontendConfig& config,
                         const hydra::SharedDsgInfo::Ptr& dsg,
                         const hydra::SharedModuleState::Ptr& state,
                         const hydra::LogSetup::Ptr& logs)
    : FrontendModule(config, dsg, state, logs), config(config::checkValid(config)) {}

LLMFrontend::~LLMFrontend() {}

std::string LLMFrontend::printInfo() const {
  std::stringstream ss;
  ss << config::toString(config);
  return ss.str();
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
}

bool objectsOverlap(const spark_dsg::BoundingBox& bbox,
                    const Eigen::Vector3f& pos,
                    const SemanticNodeAttributes& attrs) {
  return bbox.contains(attrs.position) || attrs.bounding_box.contains(pos);
}

void LLMFrontend::updateKhronosObjects(const hydra::ReconstructionOutput& base_msg) {
  // this is janky, but the try-catch is uglier
  const auto derived = dynamic_cast<const khronos::OutputData*>(&base_msg);
  CHECK(derived) << "ActiveWindow output required!";
  const auto& msg = *derived;

  std::lock_guard<std::mutex> lock(dsg_->mutex);
  ScopedTimer timer("frontend/update_objects", msg.timestamp_ns);

  const auto layer_id = DsgLayers::SEGMENTS;

  VLOG(2) << "Got " << msg.objects.size() << " new objects ("
          << dsg_->graph->getLayer(layer_id).numNodes() << " total nodes)";
  const auto prefix = dsg_->layer_prefix_map.at(DsgLayers::SEGMENTS);
  for (auto& object : msg.objects) {
    if (!object) {
      continue;
    }

    auto attrs = object->clone();
    attrs->is_active = true;
    const NodeSymbol node_id(prefix, object_id_);
    CHECK(!dsg_->graph->hasNode(node_id))
        << "Found duplicate node " << node_id.getLabel();

    CHECK(dsg_->graph->emplaceNode(layer_id, node_id, std::move(attrs)));
    new_objects_.insert(node_id);
    ++object_id_;
  }
}

void LLMFrontend::archiveObjects() {
  for (const auto prev : previous_active_places_) {
    const auto node = dsg_->graph->findNode(prev);
    if (!node) {
      continue;
    }

    if (node->attributes().is_active) {
      continue;
    }

    for (const auto& child : node->children()) {
      if (!new_objects_.count(child)) {
        continue;
      }

      auto& attrs = dsg_->graph->getNode(child).attributes<NodeAttributes>();
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
    const auto node = dsg_->graph->findNode(object_id);
    if (!node) {
      continue;
    }

    const auto parent_opt = node->getParent();
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

void LLMFrontend::updateImpl(const hydra::ReconstructionOutput::Ptr& msg) {
  FrontendModule::updateImpl(msg);

  // okay without locking: no one else is modifying the graph
  archiveObjects();
  connectNewObjects();
}

}  // namespace clio
