#include <config_utilities/config_utilities.h>
#include <config_utilities/formatting/asl.h>
#include <config_utilities/logging/log_to_glog.h>
#include <config_utilities/parsing/ros.h>
#include <hydra/common/hydra_config.h>
#include <hydra_ros/utils/node_utilities.h>

#include "hydra_llm_ros/hydra_llm_pipeline.h"

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "hydra_llm_node");

  FLAGS_minloglevel = 3;
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  config::Settings().setLogger("glog");
  config::Settings().print_width = 100;
  config::Settings().print_indent = 45;

  ros::NodeHandle nh("~");
  const int robot_id = nh.param<int>("robot_id", 0);
  hydra::llm::HydraLLMPipeline hydra(nh, robot_id);

  hydra.start();
  hydra::spinAndWait(nh);
  hydra.stop();
  hydra.save();
  hydra::HydraConfig::exit();

  return 0;
}
