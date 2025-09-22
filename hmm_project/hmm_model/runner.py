import numpy as np
from hmmlearn import hmm
from hmm_model.core import create_true_model
from hmm_model.utils import align_states, reorder_hmm_parameters, kl_divergence
from hmm_model.evaluation import evaluate_predictions, detailed_parameter_report, kl_divergence_states

def run_experiment(n_samples, seed, show_plot=False, n_iter=200):
    true_model = create_true_model()

    np.random.seed(seed)
    train_X, _ = true_model.sample(n_samples)

    # Use MultinomialHMM on older hmmlearn if CategoricalHMM is absent
    HMMClass = getattr(hmm, "CategoricalHMM", None) or hmm.MultinomialHMM
    learner = HMMClass(n_components=2, n_iter=n_iter, random_state=seed,
                       tol=1e-4, init_params="", params="te")  # learn transitions+emissions
    learner.n_features = true_model.emissionprob_.shape[1]

    # sensible init + mild priors (if supported)
    learner.startprob_ = np.array([0.8, 0.2])       # fixed (since 's' not in params)
    learner.transmat_  = np.array([[0.6, 0.4],[0.4, 0.6]])
    learner.emissionprob_ = np.full((2, learner.n_features), 1/learner.n_features)
    for attr, val in [("startprob_prior",1.1),("transmat_prior",1.1),("emissionprob_prior",1.1)]:
        if hasattr(learner, attr): setattr(learner, attr, val)

    learner.fit(train_X)

    np.random.seed(seed+1)
    test_X, test_states = true_model.sample(n_samples)

    alignment = align_states(true_model.emissionprob_, learner.emissionprob_)
    acc, cm, pred_states = evaluate_predictions(test_states, learner, test_X, alignment=alignment, show_plot=show_plot)

    reordered = reorder_hmm_parameters(learner, alignment)
    detailed_parameter_report(true_model.transmat_,  reordered["transmat"],  "Transition Matrix",   threshold=0.10)
    detailed_parameter_report(true_model.startprob_, reordered["startprob"], "Start Probabilities", threshold=0.10)

    return {
        "accuracy": acc,
        "kld_transmat": kl_divergence(true_model.transmat_,  reordered["transmat"]),
        "kld_startprob": kl_divergence(true_model.startprob_, reordered["startprob"]),
        "sample_size": n_samples,
        "seed": seed,
        "kld_state_distribution": kl_divergence_states(test_states, pred_states),
    }
