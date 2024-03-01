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
#include <hydra/common/semantic_color_map.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

namespace hydra {

class ColormapLegend {
 public:
  using Ptr = std::shared_ptr<ColormapLegend>;
  using ColorMapping = std::map<std::string, std::vector<double>>;

  struct Config {
    double hue_start = 0.0;
    double hue_end = 0.0;
    double luminance_start = 0.0;
    double luminance_end = 0.0;
    double saturation_start = 0.0;
    double saturation_end = 0.0;
    double vmin = 0.0;
    double vmax = 0.0;
    std::string label = "";
    struct RenderSettings {
      double height_inches = 10.0;
      double width_inches = 2.0;
      double dpi = 300.0;
    } render;
  };

  struct ColorEndpoint {
    Eigen::Vector3d color_hls;
    double value;
  };

  static Ptr fromEndpoints(const ros::NodeHandle& nh,
                           const ColorEndpoint& start,
                           const ColorEndpoint& end);

  void publishLegend() const;

 private:
  ColormapLegend(const ros::NodeHandle& nh, const Config& config);

  void handleSettings(const std_msgs::String& msg) const;

 private:
  mutable Config config_;

  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
};

void declare_config(ColormapLegend::Config::RenderSettings& config);
void declare_config(ColormapLegend::Config& config);

}  // namespace hydra
