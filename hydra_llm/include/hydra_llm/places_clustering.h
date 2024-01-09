#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/common/dsg_types.h>

#include <unordered_set>

#include "hydra_llm/clip_types.h"
#include "hydra_llm/clustering.h"

namespace hydra::llm {

const ClipView* getBestView(const std::map<size_t, ClipView::Ptr>& views,
                            const PlaceNodeAttributes& attrs);

struct PlaceClustering {
  using Ptr = std::unique_ptr<PlaceClustering>;
  struct Config {
    config::VirtualConfig<Clustering> clustering;
    double min_assocation_iou = 0.3;
  };

  explicit PlaceClustering(const Config& config);

  ~PlaceClustering();

  void clusterPlaces(DynamicSceneGraph& graph,
                     const std::map<size_t, ClipView::Ptr>& views,
                     const std::unordered_set<NodeId>& nodes);

  const Config config;

 private:
  std::unique_ptr<Clustering> clustering_;
  NodeSymbol region_id_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<PlaceClustering,
                                     PlaceClustering,
                                     PlaceClustering::Config>("PlaceClustering");
};

void declare_config(PlaceClustering::Config& config);

}  // namespace hydra::llm
