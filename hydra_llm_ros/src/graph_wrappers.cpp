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
#include "hydra_llm_ros/graph_wrappers.h"

#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/types/path.h>
#include <glog/logging.h>

namespace hydra {

GraphFileWrapper::GraphFileWrapper(const Config& config)
    : config(config::checkValid(config)), nh_(config.wrapper_ns), has_change_(true) {
  filepath_ = config.filepath;
  graph_ = DynamicSceneGraph::load(filepath_);
  service_ = nh_.advertiseService("reload", &GraphFileWrapper::reload, this);
}

bool GraphFileWrapper::hasChange() const { return has_change_; }

void GraphFileWrapper::clearChangeFlag() { has_change_ = false; }

DynamicSceneGraph::Ptr GraphFileWrapper::graph() const { return graph_; }

bool GraphFileWrapper::reload(std_srvs::Empty::Request&, std_srvs::Empty::Response&) {
  graph_ = DynamicSceneGraph::load(filepath_);
  has_change_ = true;
  return true;
}

bool GraphFileWrapper::load(voxblox_msgs::FilePath::Request& req,
                            voxblox_msgs::FilePath::Response&) {
  std::filesystem::path req_path(req.file_path);
  if (!std::filesystem::exists(req_path)) {
    LOG(ERROR) << "Graph does not exist at '" << req_path.string() << "'";
    return false;
  }

  filepath_ = req_path;
  graph_ = DynamicSceneGraph::load(filepath_);
  has_change_ = true;
  return true;
}

GraphRosWrapper::GraphRosWrapper(const Config& config)
    : config(config::checkValid(config)), nh_(config.wrapper_ns), has_change_(false) {
  receiver_.reset(new DsgReceiver(nh_, [this](const ros::Time& time, size_t bytes) {
    graphCallback(time, bytes);
  }));
}

bool GraphRosWrapper::hasChange() const { return has_change_; }

void GraphRosWrapper::clearChangeFlag() { has_change_ = false; }

DynamicSceneGraph::Ptr GraphRosWrapper::graph() const { return graph_; }

void GraphRosWrapper::graphCallback(const ros::Time&, size_t) {
  has_change_ = true;
  graph_ = receiver_->graph();
}

GraphZmqWrapper::GraphZmqWrapper(const Config& config)
    : config(config::checkValid(config)), has_change_(false), should_shutdown_(false) {
  receiver_.reset(new spark_dsg::ZmqReceiver(config.url, config.num_threads));
  recv_thread_.reset(new std::thread(&GraphZmqWrapper::spin, this));
}

GraphZmqWrapper::~GraphZmqWrapper() {
  should_shutdown_ = true;
  if (recv_thread_) {
    LOG(INFO) << "joining receive thread...";
    recv_thread_->join();
    LOG(INFO) << "joined receive thread";
    recv_thread_.reset();
  }
  receiver_.reset();
}

bool GraphZmqWrapper::hasChange() const {
  std::lock_guard<std::mutex> lock(graph_mutex_);
  return has_change_;
}

void GraphZmqWrapper::clearChangeFlag() {
  std::lock_guard<std::mutex> lock(graph_mutex_);
  has_change_ = false;
}

DynamicSceneGraph::Ptr GraphZmqWrapper::graph() const {
  std::lock_guard<std::mutex> lock(graph_mutex_);
  if (!graph_) {
    return graph_;
  }

  return graph_->clone();
}

void GraphZmqWrapper::spin() {
  while (!should_shutdown_) {
    // we always receive all messages
    if (!receiver_->recv(config.poll_time_ms, true)) {
      continue;
    }

    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (receiver_->graph()) {
      graph_ = receiver_->graph()->clone();
    }
    has_change_ = true;
    VLOG(1) << "Got graph!";
  }
}

void declare_config(GraphFileWrapper::Config& config) {
  using namespace config;
  name("GraphFileWrapper::Config");
  field<Path>(config.filepath, "filepath");
  field(config.wrapper_ns, "wrapper_ns");

  check<Path::Exists>(config.filepath, "filepath");
}

void declare_config(GraphRosWrapper::Config& config) {
  using namespace config;
  name("GraphRosWrapper::Config");
  field(config.wrapper_ns, "wrapper_ns");
}

void declare_config(GraphZmqWrapper::Config& config) {
  using namespace config;
  name("GraphZmqWrapper::Config");
  field(config.url, "url");
  field(config.num_threads, "num_threads");
  field(config.poll_time_ms, "poll_time_ms");
}

}  // namespace hydra
