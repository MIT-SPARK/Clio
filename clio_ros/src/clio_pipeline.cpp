#include "clio_ros/clio_pipeline.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <hydra/common/global_info.h>
#include <khronos/active_window/active_window.h>

#include "clio_ros/llm_frontend.h"

namespace clio {

using namespace spark_dsg;

ClioPipeline::ClioPipeline(const ros::NodeHandle& nh, int robot_id)
    : hydra::HydraRosPipeline(nh, robot_id) {
  // recreate scene graph with new bottom layer
  auto layer_map = hydra::GlobalInfo::instance().getConfig().layer_id_map;
  layer_map[DsgLayers::SEGMENTS] = 's';
  frontend_dsg_ = std::make_shared<hydra::SharedDsgInfo>(layer_map);
  backend_dsg_ = std::make_shared<hydra::SharedDsgInfo>(layer_map);
  shared_state_->lcd_graph = std::make_shared<hydra::SharedDsgInfo>(layer_map);
  shared_state_->backend_graph = std::make_shared<hydra::SharedDsgInfo>(layer_map);
}

ClioPipeline::~ClioPipeline() {}

void ClioPipeline::init() {
  const auto& pipeline_config = hydra::GlobalInfo::instance().getConfig();

  initFrontend();
  initBackend();
  initReconstruction();
  if (pipeline_config.enable_lcd) {
    initLCD();
  }
}

void ClioPipeline::stop() {
  const auto aw_module = getModule<khronos::ActiveWindow>("active_window");
  // Add all objects that are currently in the active window.
  auto aw_objects = aw_module->extractObjects();

  HydraPipeline::stop();

  auto curr_index = backend_dsg_->graph->getLayer(DsgLayers::OBJECTS).numNodes();
  for (const auto& attrs : aw_objects) {
    if (!attrs) {
      continue;
    }

    NodeSymbol node_id('O', curr_index);
    backend_dsg_->graph->emplaceNode(DsgLayers::OBJECTS, node_id, attrs->clone());
    ++curr_index;
  }
}

void ClioPipeline::initReconstruction() {
  const auto frontend = getModule<LLMFrontend>("frontend");
  CHECK(frontend) << "LLMFrontendModule required!";

  const ros::NodeHandle nh(nh_, "reconstruction");
  auto conf = config::fromRos<khronos::ActiveWindow::Config>(nh);
  auto mod = std::make_shared<khronos::ActiveWindow>(conf);
  mod->setOutputQueue(frontend->getQueue());
  modules_["active_window"] = mod;
}

}  // namespace clio
