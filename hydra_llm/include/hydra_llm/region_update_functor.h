#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/backend/update_functions.h>

#include "hydra_llm/places_clustering.h"
#include "hydra_llm/view_database.h"

namespace hydra::llm {

struct RegionUpdateFunctor : dsg_updates::UpdateFunctor {
  struct Config {
    config::VirtualConfig<PlaceClustering> extractor;
  };

  explicit RegionUpdateFunctor(const Config& config);

  MergeMap call(SharedDsgInfo& dsg, const UpdateInfo& info) const override;

  void updateFromViewDb(const ViewDatabase& db,
                        const std::map<NodeId, NodeId>& best_views);

  mutable std::mutex feature_mutex;
  mutable NodeEmbeddingMap latest_features;
  mutable NodeEmbeddingMap place_features;
  PlaceClustering::Ptr places_clustering;
};

void declare_config(RegionUpdateFunctor::Config& config);

}  // namespace hydra::llm
