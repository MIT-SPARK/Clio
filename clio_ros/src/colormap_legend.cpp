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
#include "clio_ros/colormap_legend.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/parsing/yaml.h>
#include <hydra_visualizer/color/color_parsing.h>

namespace clio {

void declare_config(ColormapLegend::Config& config) {
  using namespace config;
  name("ColormapLegend::Config");
  field(config.colors, "colors");
  field(config.vmin, "vmin");
  field(config.vmax, "vmax");
  field(config.height_inches, "height_inches");
  field(config.width_inches, "width_inches");
  field(config.dpi, "dpi");
}

ColormapLegend::ColormapLegend(const Config& config, const ros::NodeHandle& nh)
    : config_(config),
      nh_(nh),
      pub_(nh_.advertise<std_msgs::String>("/legend_config", 10, true)),
      sub_(nh_.subscribe("settings", 10, &ColormapLegend::handleSettings, this)) {}

void ColormapLegend::publishLegend() const {
  YAML::Node node;
  node["topic"] = nh_.resolveName("legend");
  node["type"] = "cmap";
  node["kwargs"] = config::toYaml(config_);

  std_msgs::String msg;
  std::stringstream ss;
  ss << node;
  msg.data = ss.str();
  pub_.publish(msg);
}

void ColormapLegend::handleSettings(const std_msgs::String& msg) const {
  const auto node = YAML::Load(msg.data);
  config::internal::Visitor::setValues(config_, node);
  publishLegend();
}

}  // namespace clio
