#pragma once
#include <hydra_llm/task_embeddings.h>

namespace hydra::llm {

// task_embeddings = [helpers.get_text_clip_feature(x) for x in tasks]
struct RosTaskEmbeddings : public TaskEmbeddings {
  struct Config {};

  explicit RosTaskEmbeddings(const Config& config);

 private:
  inline static const auto registration_ =
      config::RegistrationWithConfig<TaskEmbeddings, RosTaskEmbeddings, Config>("ros");
};

void declare_config(RosTaskEmbeddings::Config&);

}  // namespace hydra::llm
