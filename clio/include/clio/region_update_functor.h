#pragma once
#include <hydra/backend/update_functions.h>

#include "clio/agglomerative_clustering.h"

namespace clio {

struct RegionUpdateFunctor : public hydra::UpdateFunctor {
  struct Config {
    AgglomerativeClustering::Config clustering;
  } const config;

  explicit RegionUpdateFunctor(const Config& config);

  hydra::MergeList call(const spark_dsg::DynamicSceneGraph& unmmerged,
                        hydra::SharedDsgInfo& dsg,
                        const hydra::UpdateInfo::ConstPtr& info) const override;

  void updateGraphBatch(spark_dsg::DynamicSceneGraph& graph,
                        const std::vector<Cluster::Ptr>& clusters) const;

 private:
  mutable NodeSymbol region_id_;
  AgglomerativeClustering clustering_;
};

void declare_config(RegionUpdateFunctor::Config& config);

}  // namespace clio
