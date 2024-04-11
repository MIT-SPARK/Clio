#pragma once
#include <config_utilities/virtual_config.h>
#include <hydra/common/dsg_types.h>
#include <hydra/rooms/room_finder.h>

#include <unordered_set>

#include "hydra_llm/agglomerative_clustering.h"
#include "hydra_llm/clip_types.h"

namespace hydra::llm {

class PlaceClustering {
 public:
  using Ptr = std::unique_ptr<PlaceClustering>;
  struct Config {
    bool color_by_task = true;
  } const config;

  explicit PlaceClustering(const Config& config);

  virtual ~PlaceClustering() = default;

  virtual void clusterPlaces(DynamicSceneGraph& graph) = 0;

 protected:
  void updateGraphBatch(DynamicSceneGraph& graph,
                        const std::vector<Cluster::Ptr>& clusters) const;

  mutable NodeSymbol region_id_;
};

void declare_config(PlaceClustering::Config& config);

class SemanticClustering : public PlaceClustering {
 public:
  using Ptr = std::unique_ptr<PlaceClustering>;
  struct Config : PlaceClustering::Config {
    config::VirtualConfig<Clustering> clustering;
  } const config;

  explicit SemanticClustering(const Config& config);

  ~SemanticClustering() = default;

  void clusterPlaces(DynamicSceneGraph& graph) override;

 private:
  std::unique_ptr<Clustering> clustering_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<PlaceClustering,
                                     SemanticClustering,
                                     SemanticClustering::Config>("SemanticClustering");
};

void declare_config(SemanticClustering::Config& config);

struct GeometricClustering : public PlaceClustering {
 public:
  struct Config : PlaceClustering::Config {
    RoomFinderConfig rooms;
    AgglomerativeClustering::Config clustering;
  } const config;

  explicit GeometricClustering(const Config& config);

  ~GeometricClustering() = default;

  void clusterPlaces(DynamicSceneGraph& graph) override;

 private:
  std::unique_ptr<RoomFinder> room_finder_;
  EmbeddingGroup::Ptr tasks_;
  std::unique_ptr<EmbeddingDistance> metric_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<PlaceClustering,
                                     GeometricClustering,
                                     GeometricClustering::Config>(
          "GeometricClustering");
};

void declare_config(GeometricClustering::Config& config);

}  // namespace hydra::llm
