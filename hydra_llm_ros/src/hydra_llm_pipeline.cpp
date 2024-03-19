#include "hydra_llm_ros/hydra_llm_pipeline.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <hydra/backend/backend_module.h>
#include <hydra/common/hydra_config.h>
#include <hydra_llm/view_database.h>
#include <khronos/common/utils/globals.h>
#include <khronos/common/utils/khronos_attribute_utils.h>

#include "hydra_llm_ros/active_window_module.h"
#include "hydra_llm_ros/khronos_input_module.h"
#include "hydra_llm_ros/llm_frontend.h"

namespace hydra::llm {

using RegionConfig = RegionUpdateFunctor::Config;

struct PipelineConfig {
  std::vector<InputSensorConfig> inputs;
  config::VirtualConfig<khronos::LabelHandler> label_info;
};

void declare_config(PipelineConfig& config) {
  using namespace config;
  name("PipelineSensorConfig");
  field(config.inputs, "inputs");
  config.label_info.setOptional();
  field(config.label_info, "label_info");
}

HydraLLMPipeline::HydraLLMPipeline(const ros::NodeHandle& nh, int robot_id)
    : HydraRosPipeline(nh, robot_id) {
  // recreate scene graph with new bottom layer
  auto layer_map = HydraConfig::instance().getConfig().layer_id_map;
  layer_map[DsgLayers::SEGMENTS] = 's';
  frontend_dsg_ = std::make_shared<SharedDsgInfo>(layer_map);
  backend_dsg_ = std::make_shared<SharedDsgInfo>(layer_map);
  shared_state_->lcd_graph = std::make_shared<SharedDsgInfo>(layer_map);
  shared_state_->backend_graph = std::make_shared<SharedDsgInfo>(layer_map);
}

HydraLLMPipeline::~HydraLLMPipeline() {}

void HydraLLMPipeline::init() {
  const auto& pipeline_config = HydraConfig::instance().getConfig();

  const auto conf = config::fromRos<PipelineConfig>(nh_);
  if (conf.label_info) {
    khronos::Globals::setLabelHandler(conf.label_info.create());
  }

  initFrontend();
  initBackend();
  initReconstruction();
  if (pipeline_config.enable_lcd) {
    initLCD();
  }

  configureRegions();

  const auto module = getModule<ActiveWindowModule>("active_window");
  CHECK(module);

  input_module_ =
      std::make_unique<KhronosInputModule>(nh_, conf.inputs, module->getInputQueue());
}

void HydraLLMPipeline::stop() {
  const auto aw_module = getModule<ActiveWindowModule>("active_window");
  // Add all objects that are currently in the active window.
  const auto aw_objects = aw_module->extractActiveObjects();

  HydraPipeline::stop();

  for (const auto& object : aw_objects) {
    auto attrs = khronos::fromOutputObject(object);
    NodeSymbol object_symbol('O', object.id);
    backend_dsg_->graph->emplaceNode(
        DsgLayers::OBJECTS, object_symbol, std::move(attrs));
  }
}

void HydraLLMPipeline::initReconstruction() {
  const auto frontend = getModule<LLMFrontend>("frontend");
  CHECK(frontend) << "LLMFrontendModule required!";

  const ros::NodeHandle nh(nh_, "reconstruction");
  auto conf = config::fromRos<ActiveWindowModule::Config>(nh);
  auto mod = std::make_shared<ActiveWindowModule>(conf, frontend->getQueue());
  modules_["active_window"] = mod;
}

void HydraLLMPipeline::configureRegions() {
  const ros::NodeHandle nh(nh_, "backend/regions");
  const auto conf = config::checkValid(config::fromRos<RegionConfig>(nh));
  region_clustering_ = std::make_unique<RegionUpdateFunctor>(conf);

  auto backend = getModule<BackendModule>("backend");
  CHECK(backend);
  backend->setUpdateFunctor(DsgLayers::ROOMS, region_clustering_);
}

}  // namespace hydra::llm
