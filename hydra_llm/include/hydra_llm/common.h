#pragma once
#include <Eigen/Dense>

namespace hydra::llm {

inline Eigen::IOFormat getDefaultFormat(int precision = Eigen::StreamPrecision) {
  // julia-esque print fmt
  return Eigen::IOFormat(precision, Eigen::DontAlignCols, ", ", "; ", "", "", "[", "]");
}

}  // namespace hydra::llm
