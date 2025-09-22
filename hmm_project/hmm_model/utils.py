import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import rel_entr

def align_states(true_emissions: np.ndarray, learned_emissions: np.ndarray):
    """
    Align learned states to true states by minimizing a symmetric KL cost
    between emission probability rows.
    """
    eps = 1e-12
    P = np.clip(true_emissions, eps, 1.0)
    Q = np.clip(learned_emissions, eps, 1.0)

    cost = np.zeros((P.shape[0], Q.shape[0]), dtype=float)
    for i in range(P.shape[0]):
        for j in range(Q.shape[0]):
            kl_pq = np.sum(P[i] * (np.log(P[i]) - np.log(Q[j])))
            kl_qp = np.sum(Q[j] * (np.log(Q[j]) - np.log(P[i])))
            cost[i, j] = kl_pq + kl_qp

    _, col_ind = linear_sum_assignment(cost)
    return col_ind

def reorder_hmm_parameters(model, alignment_indices):
    """
    Reorder an HMM's parameters according to the given alignment so that
    state 0/1 of the learned model matches state 0/1 of the true model.
    """
    return {
        "emissionprob": model.emissionprob_[alignment_indices],
        "startprob":    model.startprob_[alignment_indices],
        "transmat":     model.transmat_[alignment_indices][:, alignment_indices],
    }

def evaluate_mae(true, recovered):
    return np.mean(np.abs(np.asarray(true) - np.asarray(recovered)))

def kl_divergence(p, q):
    """KL(p || q) for discrete distributions."""
    p = np.asarray(p); q = np.asarray(q)
    return np.sum(rel_entr(p, q))
