#include "hydra_llm/region_update_functor.h"

#include <config_utilities/config.h>
#include <hydra/utils/timing_utilities.h>

namespace hydra::llm {

using timing::ScopedTimer;

RegionUpdateFunctor::RegionUpdateFunctor(const Config& config) {
  places_clustering = config.extractor.create();
}

MergeMap RegionUpdateFunctor::call(SharedDsgInfo& dsg, const UpdateInfo& info) const {
  ScopedTimer timer("backend/region_clustering", info.timestamp_ns);
  {  // start critical section to update features
    std::lock_guard<std::mutex> lock(feature_mutex);
    for (auto&& [id, clip] : latest_features) {
      place_features[id] = clip;
    }
    latest_features.clear();
  }

  places_clustering->clusterPlaces(*dsg.graph, place_features);
  return {};
}

void RegionUpdateFunctor::updateFromViewDb(const ViewDatabase& db,
                                           const std::map<NodeId, NodeId>& best_views) {
  std::lock_guard<std::mutex> lock(feature_mutex);
  for (auto&& [place, agent] : best_views) {
    const auto entry = db.getView(agent);
    if (!entry) {
      continue;
    }

    latest_features[place] = entry->clip.get();
  }
}

void declare_config(RegionUpdateFunctor::Config& config) {
  using namespace config;
  name("RegionUpdateFunctorConfig::Config");
  field(config.extractor, "extractor");
}

}  // namespace hydra::llm
