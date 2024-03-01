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
#include "hydra_llm_ros/gt_publisher.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/yaml.h>
#include <config_utilities/printing.h>
#include <config_utilities/types/eigen_matrix.h>
#include <config_utilities/types/path.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <hydra/common/config_utilities.h>
#include <hydra_ros/visualizer/colormap_utilities.h>
#include <hydra_ros/visualizer/visualizer_utilities.h>
#include <tf2_eigen/tf2_eigen.h>

#include <algorithm>

namespace hydra {

using visualization_msgs::Marker;
using visualization_msgs::MarkerArray;

struct TaskBoundingBox {
  Eigen::Vector3f center;
  Eigen::Vector3f extents;
  Eigen::Quaterniond rotation;

  BoundingBox convert() const {
    const Eigen::Matrix3f rot = rotation.toRotationMatrix().cast<float>();
    return BoundingBox(
        BoundingBox::Type::OBB, -extents / 2.0f, extents / 2.0f, center, rot);
  }
};

void declare_config(TaskBoundingBox& box) {
  using namespace config;
  name("TaskBoundingBox");
  field(box.center, "center");
  field(box.extents, "extents");
  field<QuaternionConverter>(box.rotation, "rotation");
}

struct TaskBoundingBoxes {
  std::vector<TaskBoundingBox> boxes;

  GtPublisher::BoundingBoxList convert() const {
    GtPublisher::BoundingBoxList converted;
    for (const auto& box : boxes) {
      converted.push_back(box.convert());
    }
    return converted;
  }
};

void declare_config(TaskBoundingBoxes& boxes) {
  using namespace config;
  name("TaskBoundingBoxes");
  field(boxes.boxes, "");
}

struct TaskAnnotations {
  std::map<std::string, TaskBoundingBoxes> tasks;

  GtPublisher::BoundingBoxCollection convert() const {
    GtPublisher::BoundingBoxCollection converted;
    for (auto&& [task, boxes] : tasks) {
      converted.emplace(task, boxes.convert());
    }

    return converted;
  }
};

void declare_config(TaskAnnotations& annotations) {
  using namespace config;
  name("TaskAnnotations");
  field(annotations.tasks, "");
}

Eigen::Vector3d getBoundingBoxCentroid(const BoundingBox& box) {
  switch (box.type) {
    case BoundingBox::Type::AABB:
      return ((box.min + box.max) / 2.0).eval().cast<double>();
    case BoundingBox::Type::RAABB:
    case BoundingBox::Type::OBB:
      return box.world_P_center.cast<double>();
    case BoundingBox::Type::INVALID:
    default:
      LOG(ERROR) << "Invalid bounding box";
      return Eigen::Vector3d::Zero();
  }
}

double getMaxHeight(const GtPublisher::BoundingBoxCollection& boxes) {
  double max_z = 0.0;
  for (const auto& task_boxes_pair : boxes) {
    for (const auto& box : task_boxes_pair.second) {
      const auto curr_z = static_cast<double>(box.min.z() + box.world_P_center.z());
      max_z = std::max(curr_z, max_z);
    }
  }
  return max_z;
}

Eigen::Vector3d getCentroidOfBoxes(const std::vector<BoundingBox>& boxes) {
  Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  if (boxes.empty()) {
    return pos;
  }

  for (const auto& box : boxes) {
    pos += getBoundingBoxCentroid(box);
  }

  pos /= boxes.size();
  return pos;
}

void declare_config(GtPublisher::Config& config) {
  using namespace config;
  name("GtPublisher::Config");
  field<Path>(config.filepath, "filepath");
  field(config.ns, "ns");
  field(config.marker_ns, "marker_ns");
  field(config.draw_labels, "draw_labels");
  field(config.scale, "scale");
  field(config.text_scale, "text_scale");
  field(config.text_line_scale, "text_line_scale");
  field(config.text_height, "text_height");
  field(config.alpha, "alpha");
  field(config.frame_id, "frame_id");
  field(config.tasks, "tasks");
}

GtPublisher::GtPublisher(const Config& config)
    : config(config::checkValid(config)), nh_(config.ns), has_change_(false) {
  pub_ = nh_.advertise<MarkerArray>("gt_markers", 1, true);
  marker_ns_ = config.marker_ns == "" ? nh_.getNamespace() : config.marker_ns;

  voxblox_msgs::FilePath srv;
  srv.request.file_path = config.filepath.string();
  load(srv.request, srv.response);
  service_ = nh_.advertiseService("load", &GtPublisher::load, this);
}

void GtPublisher::fillBoundingBoxes(const std_msgs::Header& header,
                                    const BoundingBoxCollection& boxes,
                                    MarkerArray& msg) const {
  Marker marker;
  marker.header = header;
  marker.type = Marker::LINE_LIST;
  marker.id = 0;
  marker.ns = marker_ns_;
  marker.scale.x = config.scale;
  marker.pose.orientation.w = 1.0;

  Marker lines;
  lines.header = header;
  lines.type = Marker::LINE_LIST;
  lines.id = 0;
  lines.ns = marker_ns_ + "_edges";
  lines.scale.x = config.text_line_scale;
  lines.pose.orientation.w = 1.0;

  const auto max_z = getMaxHeight(boxes);
  Eigen::MatrixXf corners(3, 8);
  for (auto&& [task, task_boxes] : boxes) {
    auto pos = getCentroidOfBoxes(task_boxes);
    const auto color = tasks_->getColor(task);

    for (const auto& box : task_boxes) {
      fillCornersFromBbox(box, corners);

      const auto color_msg = dsg_utils::makeColorMsg(color, config.alpha);
      addWireframeToMarker(corners, color_msg, marker);
      if (config.draw_labels && config.text_height > 0.0) {
        geometry_msgs::Point center_point;
        tf2::convert(pos, center_point);
        center_point.z += max_z + config.text_height;
        addEdgesToCorners(corners, center_point, color_msg, lines);
      }
    }
  }

  tracker_.addMarker(marker, msg);
  tracker_.addMarker(lines, msg);
}

void GtPublisher::fillLabels(const std_msgs::Header& header,
                             const BoundingBoxCollection& boxes,
                             MarkerArray& msg) const {
  const auto max_z = getMaxHeight(boxes);
  size_t task_index = 0;
  for (auto&& [task, task_boxes] : boxes) {
    if (task_boxes.empty()) {
      continue;
    }

    const auto pos = getCentroidOfBoxes(task_boxes);

    Marker marker;
    marker.header = header;
    marker.ns = marker_ns_ + "_labels";
    marker.id = task_index;
    marker.type = Marker::TEXT_VIEW_FACING;
    marker.action = Marker::ADD;
    marker.pose.position.x = pos.x();
    marker.pose.position.y = pos.y();
    marker.pose.position.z = pos.z() + config.text_height;
    if (config.text_height > 0.0) {
      marker.pose.position.z += max_z;
    }
    marker.pose.orientation.w = 1.0;
    marker.scale.z = config.text_scale;
    marker.color.a = 1.0;
    marker.text = task;
    tracker_.addMarker(marker, msg);
    ++task_index;
  }
}

std::vector<std::string> getTaskPrompts(
    const GtPublisher::BoundingBoxCollection& boxes) {
  std::vector<std::string> tasks;
  for (const auto& task_boxes_pair : boxes) {
    tasks.push_back(task_boxes_pair.first);
  }

  return tasks;
}

bool GtPublisher::load(voxblox_msgs::FilePath::Request& req,
                       voxblox_msgs::FilePath::Response&) {
  if (req.file_path.empty()) {
    LOG(INFO) << "Clearing published GT";
    return true;
  }

  const std::filesystem::path filepath(req.file_path);
  if (!std::filesystem::exists(filepath)) {
    LOG(WARNING) << "Filepath '" << filepath.string() << "' does not exist!";
    return false;
  }

  std_msgs::Header header;
  header.stamp = ros::Time::now();
  header.frame_id = config.frame_id;

  MarkerArray clear_msg;
  tracker_.clearPrevious(header, clear_msg);
  pub_.publish(clear_msg);

  const auto gt_tasks = config::fromYamlFile<TaskAnnotations>(req.file_path);
  VLOG(3) << "Loaded GT bounding boxes from '" << req.file_path << "':\n"
          << config::toString(gt_tasks);

  if (gt_tasks.tasks.empty()) {
    LOG(ERROR) << "TBD";
    return false;
  }

  const auto converted_tasks = gt_tasks.convert();
  const auto task_prompts = getTaskPrompts(converted_tasks);
  tasks_ = std::make_shared<llm::TaskInformation>(config.tasks, task_prompts);
  has_change_ = true;

  MarkerArray msg;
  fillBoundingBoxes(header, converted_tasks, msg);
  if (config.draw_labels) {
    fillLabels(header, converted_tasks, msg);
  }
  pub_.publish(msg);

  return true;
}

bool GtPublisher::hasChange() const { return has_change_; }

void GtPublisher::clearChangeFlag() { has_change_ = false; }

llm::TaskInformation::Ptr GtPublisher::getTasks() const { return tasks_; }

}  // namespace hydra
