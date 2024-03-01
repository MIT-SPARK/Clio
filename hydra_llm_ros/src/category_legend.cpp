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
#include "hydra_llm_ros/category_legend.h"

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/parsing/yaml.h>

namespace hydra {

using CategoryMap = std::map<std::string, size_t>;

void declare_config(CategoryLegend::Config& config) {
  using namespace config;
  name("CategoryLegend::Config");
  field(config.ncols, "ncols");
  field(config.cell_width, "cell_width");
  field(config.cell_height, "cell_height");
  field(config.swatch_width, "swatch_width");
  field(config.swatch_height, "swatch_height");
  field(config.margin, "margin");
  field(config.text_margin, "text_margin");
  field(config.dpi, "dpi");
  field(config.fontsize, "fontsize");
  field(config.wrapsize, "wrapsize");
}

CategoryLegend::Ptr CategoryLegend::fromColormap(const ros::NodeHandle& nh,
                                                 const SemanticColorMap& cmap,
                                                 const CategoryMap& categories) {
  ColorMapping colors;
  for (auto&& [label, index] : categories) {
    const auto color = cmap.getColorFromLabel(index % cmap.getNumLabels());
    colors[label] = {color.r / 255.0, color.g / 255.0, color.b / 255.0};
  }

  CategoryLegend::Ptr legend(new CategoryLegend(nh, colors));
  legend->publishLegend();
  return legend;
}

CategoryLegend::CategoryLegend(const ros::NodeHandle& nh, const ColorMapping& colors)
    : config_(config::fromRos<Config>(nh)),
      colors_(colors),
      nh_(nh),
      pub_(nh_.advertise<std_msgs::String>("/legend_config", 10, true)),
      sub_(nh_.subscribe(
          "legend_settings", 10, &CategoryLegend::handleSettings, this)) {}

void CategoryLegend::publishLegend() const {
  YAML::Node node;
  node["topic"] = nh_.resolveName("legend");
  node["type"] = "grid";

  auto args = config::toYaml(config_);
  args["color_dict"] = colors_;

  node["kwargs"] = args;

  std_msgs::String msg;
  std::stringstream ss;
  ss << node;
  msg.data = ss.str();
  pub_.publish(msg);
}

void CategoryLegend::handleSettings(const std_msgs::String& msg) const {
  const auto node = YAML::Load(msg.data);
  config::internal::Visitor::setValues(config_, node);
  publishLegend();
}

}  // namespace hydra
