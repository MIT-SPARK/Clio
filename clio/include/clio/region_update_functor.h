#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/backend/update_functions.h>

#include "clio/agglomerative_clustering.h"

namespace clio {

struct RegionUpdateFunctor : public hydra::UpdateFunctor {
  struct Config {
    bool color_by_task = true;
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

  inline static const auto registration_ =
      config::RegistrationWithConfig<UpdateFunctor, RegionUpdateFunctor, Config>(
          "RegionUpdateFunctor");
};

void declare_config(RegionUpdateFunctor::Config& config);

}  // namespace clio
