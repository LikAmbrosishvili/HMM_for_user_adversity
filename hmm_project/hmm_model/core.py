from hmmlearn import hmm
import numpy as np

def create_true_model(random_state: int = 4):
    model = hmm.CategoricalHMM(n_components=2, init_params="", random_state=random_state)
    model.startprob_ = np.array([0.5, 0.5])
    model.transmat_  = np.array([[0.7, 0.3],[0.4, 0.6]])
    engaged    = np.array([0.25, 0.05, 0.02, 0.25, 0.05, 0.25, 0.10, 0.03])
    frustrated = np.array([0.05, 0.30, 0.15, 0.05, 0.20, 0.05, 0.15, 0.05])
    model.emissionprob_ = np.vstack([engaged/engaged.sum(), frustrated/frustrated.sum()])
    return model
