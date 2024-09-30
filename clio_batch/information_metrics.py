"""Functions dealing with entropy, information, and divergence."""

import numpy as np


def shannon_entropy(p):
    """Compute Shannon entropy of PMF."""
    return -np.sum(p * np.log2(p), axis=0)


def js_divergence(py_c, pc):
    """Compute Jensen-Shannon divergence between two PMFs."""
    assert py_c.shape[1] == pc.shape[0]
    joint_probs = py_c @ pc
    sum_entropies = np.sum(pc * shannon_entropy(py_c))
    return shannon_entropy(joint_probs) - sum_entropies


def mutual_information(px, pc, pc_x):
    """Get mutual information between two PMFs."""
    px_ = px[px > 0]
    pc_ = pc[pc > 0]
    pc_x_ = pc_x[pc > 0, :]
    pc_x_ = pc_x_[:, px > 0]
    log_term = pc_x_ / pc_[:, None]
    log_term[log_term == 0] = 1  # this is to avoid infs
    log_term = np.log2(log_term)
    return np.sum((pc_x_ * log_term) @ px_)
