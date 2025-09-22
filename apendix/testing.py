import numpy as np
from hmmlearn import hmm
from scipy.optimize import linear_sum_assignment
import sys

# --- Choose HMM class robustly across hmmlearn versions ---
HMMClass = getattr(hmm, "CategoricalHMM", None)
if HMMClass is None:
    # Older hmmlearn: use MultinomialHMM (same API for our usage)
    HMMClass = hmm.MultinomialHMM

# --- Build a ground-truth 2-state categorical HMM with 8 symbols ---
rng = np.random.RandomState(42)

true_startprob = np.array([0.8, 0.2])
true_transmat  = np.array([[0.7, 0.3],
                           [0.4, 0.6]])

engaged    = np.array([0.25, 0.05, 0.02, 0.25, 0.05, 0.25, 0.10, 0.03])
frustrated = np.array([0.05, 0.30, 0.15, 0.05, 0.20, 0.05, 0.15, 0.05])
true_emissions = np.vstack([engaged/engaged.sum(), frustrated/frustrated.sum()])

true_model = HMMClass(n_components=2, init_params="", random_state=42)
true_model.startprob_  = true_startprob
true_model.transmat_   = true_transmat
# name differs but attribute is the same on Multinomial/Categorical
true_model.emissionprob_ = true_emissions

# --- Sample data ---
train_X, _     = true_model.sample(20000)   # more data -> more stable EM
test_X, states = true_model.sample(5000)

# --- Learner configured for your hmmlearn version ---
# params: use 't' (transitions) + 'e' (emissions). DO NOT include 's' so startprob_ stays fixed.
learner = HMMClass(
    n_components=2,
    n_iter=250,
    random_state=0,
    tol=1e-4,
    init_params="",   # don't overwrite our manual init
    params="te",      # learn Transitions + Emissions (not startprob)
)
# REQUIRED for Multinomial/Categorical HMM
learner.n_features = true_emissions.shape[1]

# Sensible initialization
learner.startprob_ = np.array([0.8, 0.2])            # fixed
learner.transmat_  = np.array([[0.6, 0.4],
                               [0.4, 0.6]])
learner.emissionprob_ = np.full((2, learner.n_features), 1.0/learner.n_features)

# Mild Dirichlet priors for stability (if supported by your version)
# These attributes exist on recent hmmlearn; if not present, this block is skipped.
for attr, val in [
    ("startprob_prior", 1.1),
    ("transmat_prior", 1.1),
    ("emissionprob_prior", 1.1),
]:
    if hasattr(learner, attr):
        setattr(learner, attr, val)

# --- Fit ---
learner.fit(train_X)

# --- Align learned states to true states using symmetric KL on emissions ---
def align_states(P, Q):
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    cost = np.zeros((P.shape[0], Q.shape[0]))
    for i in range(P.shape[0]):
        for j in range(Q.shape[0]):
            kl_pq = np.sum(P[i] * (np.log(P[i]) - np.log(Q[j])))
            kl_qp = np.sum(Q[j] * (np.log(Q[j]) - np.log(P[i])))
            cost[i, j] = kl_pq + kl_qp
    _, col_ind = linear_sum_assignment(cost)
    return col_ind

alignment = align_states(true_emissions, learner.emissionprob_)

# --- Predict and evaluate ---
pred = learner.predict(test_X)
remapped = np.zeros_like(pred)
for i, a in enumerate(alignment):
    remapped[pred == a] = i
acc = np.mean(remapped == states)

# --- Pretty print matrices aligned to truth ---
aligned_trans = learner.transmat_[alignment][:, alignment]
aligned_emiss = learner.emissionprob_[alignment]

np.set_printoptions(precision=3, suppress=True)
print("hmmlearn version:", getattr(hmm, "__version__", "unknown"))
print("HMM class used  :", HMMClass.__name__)
print("Accuracy        :", round(float(acc), 4))

print("\nTrue transmat:\n", true_transmat)
print("Learned transmat (aligned):\n", aligned_trans)

print("\nTrue emissions:\n", true_emissions)
print("Learned emissions (aligned):\n", aligned_emiss)
