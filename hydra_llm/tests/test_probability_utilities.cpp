#include <gtest/gtest.h>
#include <hydra_llm/common.h>
#include <hydra_llm/probability_utilities.h>

namespace hydra::llm {

Eigen::VectorXd getBinaryProbability(double p_true) {
  Eigen::VectorXd p(2);
  p << p_true, 1.0 - p_true;
  return p;
}

TEST(ProbabilityUtilities, TestShannonEntropy) {
  const auto fmt = getDefaultFormat();
  {  // test case 1: no entropy
    Eigen::VectorXd p = getBinaryProbability(0.0);
    EXPECT_NEAR(shannonEntropy(p), 0.0, 1.0e-9) << "input: " << p.format(fmt);
  }

  {  // test case 2: also no entropy
    Eigen::VectorXd p = getBinaryProbability(1.0);
    EXPECT_NEAR(shannonEntropy(p), 0.0, 1.0e-9) << "input: " << p.format(fmt);
  }

  {  // test case 3: max entropy (of 1 bit)
    Eigen::VectorXd p = getBinaryProbability(0.5);
    EXPECT_NEAR(shannonEntropy(p), 1.0, 1.0e-9) << "input: " << p.format(fmt);
  }
}

TEST(ProbabilityUtilities, TestJSDivergence) {
  {  // test case 1: no divergence
    Eigen::MatrixXd pa_b = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd p_b = getBinaryProbability(0.5);
    EXPECT_NEAR(jensenShannonDivergence(pa_b, p_b), 1.0, 1.0e-9);
  }

  {  // test case 2: complete divergence
    Eigen::MatrixXd pa_b(2, 2);
    pa_b << 0.5, 0.5, 0.5, 0.5;
    Eigen::MatrixXd p_b = getBinaryProbability(0.5);
    EXPECT_NEAR(jensenShannonDivergence(pa_b, p_b), 0.0, 1.0e-9);
  }
}

TEST(ProbabilityUtilities, TestMutualInformation) {
  const auto fmt = getDefaultFormat();
  {  // test case 1: mutual info of 1
    Eigen::MatrixXd pa_b = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd p_a = getBinaryProbability(0.5);
    Eigen::MatrixXd p_b = getBinaryProbability(0.5);
    EXPECT_NEAR(mutualInformation(p_a, p_b, pa_b), 1.0, 1.0e-9)
        << "p(a): " << p_a.format(fmt) << ", p(b): " << p_b.format(fmt)
        << ", p(a|b): " << pa_b.format(fmt);
  }

  {  // test case 2: mutual info of 0
    Eigen::MatrixXd pa_b(2, 2);
    pa_b << 0.5, 0.5, 0.5, 0.5;
    Eigen::MatrixXd p_a = getBinaryProbability(0.5);
    Eigen::MatrixXd p_b = getBinaryProbability(0.5);
    EXPECT_NEAR(mutualInformation(p_a, p_b, pa_b), 0.0, 1.0e-9)
        << "p(a): " << p_a.format(fmt) << ", p(b): " << p_b.format(fmt)
        << ", p(a|b): " << pa_b.format(fmt);
  }

  {  // test case 2: mutual info for bad distribution
    Eigen::MatrixXd pa_b(2, 2);
    pa_b << 0.5, 0.5, 0.5, 0.5;
    Eigen::MatrixXd p_a = getBinaryProbability(1.0);
    Eigen::MatrixXd p_b = getBinaryProbability(0.5);
    // TODO(nathan) technically, mutual information is non-negative, but the joint
    // distribution we're providing isn't valid
    EXPECT_NEAR(mutualInformation(p_a, p_b, pa_b), -0.5, 1.0e-9)
        << "p(a): " << p_a.format(fmt) << ", p(b): " << p_b.format(fmt)
        << ", p(a|b): " << pa_b.format(fmt);
  }
}

}  // namespace hydra::llm
