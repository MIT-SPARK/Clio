#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/backend/update_functions.h>

#include "hydra_llm/places_clustering.h"

namespace hydra::llm {

struct RegionUpdateFunctor : dsg_updates::UpdateFunctor {
  struct Config {
    config::VirtualConfig<PlaceClustering> extractor;
  };

  explicit RegionUpdateFunctor(const Config& config);

  MergeMap call(SharedDsgInfo& dsg, const UpdateInfo& info) const override;

  PlaceClustering::Ptr places_clustering;
};

void declare_config(RegionUpdateFunctor::Config& config);

}  // namespace hydra::llm
