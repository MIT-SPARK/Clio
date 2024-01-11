#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/common/dsg_types.h>

#include <unordered_set>

#include "hydra_llm/clip_types.h"
#include "hydra_llm/clustering.h"

namespace hydra::llm {

struct PlaceClustering {
  using Ptr = std::unique_ptr<PlaceClustering>;
  struct Config {
    config::VirtualConfig<Clustering> clustering;
    bool is_batch = true;
    double min_assocation_iou = 1.0;
    bool color_by_task = true;
  };

  explicit PlaceClustering(const Config& config);

  ~PlaceClustering();

  void clusterPlaces(DynamicSceneGraph& graph, const NodeEmbeddingMap& views);

  const Config config;

 protected:
  void updateGraphBatch(DynamicSceneGraph& graph,
                        const std::vector<Cluster::Ptr>& clusters) const;

  void updateGraphIncremental(DynamicSceneGraph& graph,
                              const NodeEmbeddingMap& views,
                              const std::vector<Cluster::Ptr>& clusters) const;

 private:
  std::unique_ptr<Clustering> clustering_;
  mutable NodeSymbol region_id_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<PlaceClustering,
                                     PlaceClustering,
                                     PlaceClustering::Config>("PlaceClustering");
};

void declare_config(PlaceClustering::Config& config);

}  // namespace hydra::llm
