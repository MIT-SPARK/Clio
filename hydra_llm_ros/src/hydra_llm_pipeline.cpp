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

using RegionConfig = RegionUpdateFunctor::Config;

struct InputSensorConfig {
  std::string ns = "input";
  config::VirtualConfig<Sensor> sensor;
};

struct PipelineConfig {
  std::vector<InputSensorConfig> inputs;
  config::VirtualConfig<khronos::LabelHandler> label_info;
};

void declare_config(InputSensorConfig& config) {
  using namespace config;
  name("InputSensorConfig");
  field(config.ns, "ns");
  field(config.sensor, "sensor");
}

void declare_config(PipelineConfig& config) {
  using namespace config;
  name("PipelineSensorConfig");
  field(config.inputs, "inputs");
  config.label_info.setOptional();
  field(config.label_info, "label_info");
}

class KhronosInputModule : public Module {
 public:
  using InputQueue = khronos::InputSynchronizer::InputQueue;

  KhronosInputModule(const ros::NodeHandle& nh,
                     const std::vector<InputSensorConfig>& inputs,
                     const InputQueue::Ptr& data_queue) {
    for (const auto& conf : inputs) {
      std::shared_ptr<Sensor> sensor(config::checkValid(conf).sensor.create());
      const size_t index = khronos::Globals::addSensor(sensor);
      inputs_.push_back(std::make_shared<khronos::InputSynchronizer>(
          ros::NodeHandle(nh, conf.ns), data_queue, index));
    }
  }

  virtual ~KhronosInputModule() = default;

  void start() override {
    for (const auto& input : inputs_) {
      input->start();
    }
  }

  void stop() override {
    for (const auto& input : inputs_) {
      input->stop();
    }
  }

  void save(const LogSetup&) override {}

  std::string printInfo() const override {
    std::stringstream ss;
    size_t index = 0;
    for (const auto& input : inputs_) {
      ss << "input " << index << ": " << std::endl << config::toString(input->config());
      ++index;
    }
    return ss.str();
  }

  std::vector<std::shared_ptr<khronos::InputSynchronizer>> inputs_;
};

HydraLLMPipeline::HydraLLMPipeline(const ros::NodeHandle& nh, int robot_id)
    : HydraRosPipeline(nh, robot_id) {}

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
