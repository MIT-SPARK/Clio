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
#include <hydra/common/dsg_types.h>
#include <hydra/common/semantic_color_map.h>
#include <hydra_llm/embedding_distances.h>
#include <hydra_llm/embedding_group.h>

#include <filesystem>

#include "hydra_llm_ros/category_legend.h"

namespace hydra::llm {

class TaskInformation {
 public:
  using Ptr = std::shared_ptr<TaskInformation>;
  using NodeColor = SemanticNodeAttributes::ColorVector;

  struct Config {
    std::string ns = "~";
    std::filesystem::path colormap_filepath;
    std::string task_service_name = "/get_embedding";
    bool make_legend = true;
    config::VirtualConfig<EmbeddingDistance> metric{CosineDistance::Config(), "cosine"};
  } const config;

  TaskInformation(const Config& config, const std::vector<std::string>& tasks);

  NodeColor getColor(const std::string& task) const;

  std::string getNearestTask(const Eigen::MatrixXd& feature) const;

  NodeColor getNearestTaskColor(const Eigen::MatrixXd& feature) const;

  const EmbeddingDistance& metric() const;

  const EmbeddingGroup& embeddings() const;

 private:
  SemanticColorMap::Ptr colormap_;
  std::map<std::string, size_t> task_indices_;
  CategoryLegend::Ptr legend_;
  EmbeddingGroup::Ptr tasks_;
  std::unique_ptr<EmbeddingDistance> metric_;
};

void declare_config(TaskInformation::Config& config);

}  // namespace hydra::llm
