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
  places_clustering->clusterPlaces(*dsg.graph);
  return {};
}

void declare_config(RegionUpdateFunctor::Config& config) {
  using namespace config;
  name("RegionUpdateFunctorConfig::Config");
  field(config.extractor, "extractor");
}

}  // namespace hydra::llm
