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

class CategoryLegend {
 public:
  using Ptr = std::shared_ptr<CategoryLegend>;
  using ColorMapping = std::map<std::string, std::vector<double>>;

  struct Config {
    size_t ncols = 10;
    size_t cell_width = 500;
    size_t cell_height = 100;
    size_t swatch_width = 120;
    size_t swatch_height = 90;
    size_t margin = 12;
    size_t text_margin = 10;
    double dpi = 300.0;
    size_t fontsize = 8;
    size_t wrapsize = 20;
  };

  static Ptr fromColormap(const ros::NodeHandle& nh,
                          const SemanticColorMap& map,
                          const std::map<std::string, size_t>& categories);

  void publishLegend() const;

 private:
  CategoryLegend(const ros::NodeHandle& nh, const ColorMapping& colors);

  void handleSettings(const std_msgs::String& msg) const;

 private:
  mutable Config config_;
  ColorMapping colors_;

  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;
};

void declare_config(CategoryLegend::Config& config);

}  // namespace hydra
