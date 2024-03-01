/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#pragma once
#include <config_utilities/factory.h>
#include <hydra/common/dsg_types.h>
#include <hydra_ros/utils/dsg_streaming_interface.h>
#include <ros/ros.h>
#include <spark_dsg/zmq_interface.h>
#include <std_srvs/Empty.h>
#include <voxblox_msgs/FilePath.h>

#include <filesystem>
#include <mutex>
#include <thread>

namespace hydra {

struct GraphWrapper {
  using Ptr = std::shared_ptr<GraphWrapper>;
  virtual ~GraphWrapper() = default;
  virtual bool hasChange() const = 0;
  virtual void clearChangeFlag() = 0;
  virtual DynamicSceneGraph::Ptr graph() const = 0;
};

class GraphFileWrapper : public GraphWrapper {
 public:
  struct Config {
    std::filesystem::path filepath;
    std::string wrapper_ns = "~";
  } const config;

  explicit GraphFileWrapper(const Config& config);

  bool hasChange() const override;

  void clearChangeFlag() override;

  DynamicSceneGraph::Ptr graph() const override;

  bool reload(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

  bool load(voxblox_msgs::FilePath::Request& req,
            voxblox_msgs::FilePath::Response& res);

 private:
  ros::NodeHandle nh_;
  bool has_change_;
  std::filesystem::path filepath_;
  DynamicSceneGraph::Ptr graph_;
  ros::ServiceServer service_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<GraphWrapper,
                                     GraphFileWrapper,
                                     GraphFileWrapper::Config>("GraphFromFile");
};

class GraphRosWrapper : public GraphWrapper {
 public:
  struct Config {
    std::string wrapper_ns = "~";
  } const config;

  explicit GraphRosWrapper(const Config& config);

  bool hasChange() const override;

  void clearChangeFlag() override;

  DynamicSceneGraph::Ptr graph() const override;

 private:
  void graphCallback(const ros::Time& time, size_t num_bytes);

  ros::NodeHandle nh_;
  bool has_change_;
  DynamicSceneGraph::Ptr graph_;
  std::unique_ptr<DsgReceiver> receiver_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<GraphWrapper,
                                     GraphRosWrapper,
                                     GraphRosWrapper::Config>("GraphFromRos");
};

class GraphZmqWrapper : public GraphWrapper {
 public:
  struct Config {
    std::string url = "tcp://127.0.0.1:8001";
    size_t num_threads = 2;
    size_t poll_time_ms = 10;
  } const config;

  explicit GraphZmqWrapper(const Config& config);

  virtual ~GraphZmqWrapper();

  bool hasChange() const override;

  void clearChangeFlag() override;

  DynamicSceneGraph::Ptr graph() const override;

 private:
  void spin();

  bool has_change_;
  std::atomic<bool> should_shutdown_;
  mutable std::mutex graph_mutex_;
  std::unique_ptr<std::thread> recv_thread_;
  std::unique_ptr<spark_dsg::ZmqReceiver> receiver_;
  DynamicSceneGraph::Ptr graph_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<GraphWrapper,
                                     GraphZmqWrapper,
                                     GraphZmqWrapper::Config>("GraphFromZmq");
};

void declare_config(GraphFileWrapper::Config& config);
void declare_config(GraphRosWrapper::Config& config);
void declare_config(GraphZmqWrapper::Config& config);

}  // namespace hydra
