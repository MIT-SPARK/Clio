#pragma once
#include <config_utilities/factory.h>
#include <hydra/common/dsg_types.h>

#include <unordered_set>

#include "hydra_llm/clip_types.h"

namespace hydra::llm {

const ClipView* getBestView(const std::map<size_t, ClipView::Ptr>& views,
                            const PlaceNodeAttributes& attrs);

struct PlaceClustering {
  using Ptr = std::unique_ptr<PlaceClustering>;
  struct Config {
    double similarity_threshold = 0.022;
    bool run_preprune = false;
  };

  explicit PlaceClustering(const Config& config);

  ~PlaceClustering();

  void clusterPlaces(DynamicSceneGraph& graph,
                     const std::map<size_t, ClipView::Ptr>& views,
                     const std::unordered_set<NodeId>& nodes);

  const Config config;

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<PlaceClustering,
                                     PlaceClustering,
                                     PlaceClustering::Config>("PlaceClustering");
};

void declare_config(PlaceClustering::Config& config);

}  // namespace hydra::llm
