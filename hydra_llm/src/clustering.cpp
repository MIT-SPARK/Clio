#include "hydra_llm/clustering.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>

namespace hydra::llm {

void declare_config(Clustering::Config& config) {
  using namespace config;
  name("Clustering::Config");
  field(config.tasks, "tasks");
}

Clustering::Clustering(const Config& config)
    : config(config::checkValid(config)), tasks_(config.tasks.create()) {}

}  // namespace hydra::llm
