#pragma once
#include <config_utilities/factory.h>
#include <hydra_llm/embedding_group.h>

namespace hydra::llm {

struct RosEmbeddingGroup : public EmbeddingGroup {
  struct Config {
    std::string service_name = "/get_tasks";
    bool silent_wait = false;
  };

  explicit RosEmbeddingGroup(const Config& config);

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<EmbeddingGroup, RosEmbeddingGroup, Config>(
          "RosEmbeddingGroup");
};

void declare_config(RosEmbeddingGroup::Config& config);

}  // namespace hydra::llm
