#include "hydra_llm_ros/hydra_llm_pipeline.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <hydra/backend/backend_module.h>
#include <hydra/common/hydra_config.h>
#include <hydra_llm/view_database.h>
#include <khronos/common/utils/globals.h>

#include "hydra_llm_ros/active_window_module.h"
#include "hydra_llm_ros/llm_frontend.h"

namespace hydra::llm {

class KhronosInputModule : public Module, public khronos::InputSynchronizer {
 public:
  using InputQueue = khronos::InputSynchronizer::InputQueue;

  KhronosInputModule(const ros::NodeHandle& nh, InputQueue::Ptr data_queue)
      : InputSynchronizer(nh, data_queue) {}

  virtual ~KhronosInputModule() = default;

  void start() override { khronos::InputSynchronizer::start(); }

  void stop() override { khronos::InputSynchronizer::stop(); }

  void save(const LogSetup&) override {}

  std::string printInfo() const override {
    std::stringstream ss;
    ss << std::endl << config::toString(config_);
    return ss.str();
  }
};

using RegionConfig = RegionUpdateFunctor::Config;

struct PipelineSensorConfig {
  config::VirtualConfig<Sensor> sensor;
};

void declare_config(PipelineSensorConfig& config) {
  using namespace config;
  name("PipelineSensorConfig");
  field(config.sensor, "sensor");
}

HydraLLMPipeline::HydraLLMPipeline(const ros::NodeHandle& nh, int robot_id)
    : HydraRosPipeline(nh, robot_id) {}

HydraLLMPipeline::~HydraLLMPipeline() {}

void HydraLLMPipeline::init() {
  const auto& pipeline_config = HydraConfig::instance().getConfig();
  initFrontend();
  initBackend();
  initReconstruction();
  if (pipeline_config.enable_lcd) {
    initLCD();
  }

  configureRegions();
  initInput();

  const auto conf = config::fromRos<PipelineSensorConfig>(nh_);
  std::shared_ptr<Sensor> sensor(conf.sensor.create());
  khronos::Globals::setSensor(sensor);
}

void HydraLLMPipeline::initReconstruction() {
  const auto frontend = getModule<LLMFrontend>("frontend");
  CHECK(frontend) << "LLMFrontendModule required!";

  const ros::NodeHandle nh(nh_, "reconstruction");
  auto conf = config::fromRos<ActiveWindowModule::Config>(nh);
  auto mod = std::make_shared<ActiveWindowModule>(conf, frontend->getQueue());
  modules_["active_window"] = mod;
}

void HydraLLMPipeline::initInput() {
  const auto module = getModule<ActiveWindowModule>("active_window");
  CHECK(module);

  input_module_ = std::make_unique<KhronosInputModule>(ros::NodeHandle(nh_, "input"),
                                                       module->getInputQueue());
}

void HydraLLMPipeline::configureRegions() {
  const ros::NodeHandle nh(nh_, "backend/regions");
  const auto conf = config::checkValid(config::fromRos<RegionConfig>(nh));
  region_clustering_ = std::make_unique<RegionUpdateFunctor>(conf);

  auto backend = getModule<BackendModule>("backend");
  CHECK(backend);
  backend->setUpdateFunctor(DsgLayers::ROOMS, region_clustering_);

  auto frontend = getModule<LLMFrontend>("frontend");
  CHECK(frontend);
  frontend->addViewCallback([this](const auto& db, const auto& views) {
    region_clustering_->updateFromViewDb(db, views);
  });
}

}  // namespace hydra::llm
