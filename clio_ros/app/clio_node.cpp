#include <config_utilities/config_utilities.h>
#include <config_utilities/formatting/asl.h>
#include <config_utilities/logging/log_to_glog.h>
#include <config_utilities/parsing/ros.h>
#include <hydra/common/global_info.h>
#include <hydra_ros/utils/node_utilities.h>

#include "clio_ros/clio_pipeline.h"

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "clio_node");

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
  clio::ClioPipeline clio(nh, robot_id);
  clio.init();

  clio.start();
  hydra::spinAndWait(nh);
  clio.stop();
  clio.save();
  hydra::GlobalInfo::exit();

  return 0;
}
