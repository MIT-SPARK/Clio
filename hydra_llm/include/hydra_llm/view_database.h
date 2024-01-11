#pragma once
#include <hydra/common/dsg_types.h>

#include "hydra_llm/clip_types.h"

namespace hydra::llm {

const ClipView* getBestView(const std::map<size_t, ClipView::Ptr>& views,
                            const PlaceNodeAttributes& attrs);

struct ViewEntry {
  NodeId node_id;
  ClipEmbedding::Ptr clip;
};

class ViewDatabase {
 public:
  using Ptr = std::shared_ptr<ViewDatabase>;

  ViewDatabase();

  ~ViewDatabase();

  void addView(NodeId node, ClipEmbedding::Ptr&& embedding);

  const ViewEntry* getView(NodeId node) const;

 protected:
  std::mutex mutex_;
  std::map<NodeId, ViewEntry> entries_;
};

}  // namespace hydra::llm
