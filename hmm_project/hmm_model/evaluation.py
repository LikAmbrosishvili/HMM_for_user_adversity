import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from hmm_model.plotting import plot_confusion_matrix
from scipy.special import rel_entr

def evaluate_predictions(true_states, model, observations, alignment=None, show_plot=True):
    preds = model.predict(observations)
    if alignment is not None:
        remapped = np.zeros_like(preds)
        for i, a in enumerate(alignment):
            remapped[preds == a] = i
        preds = remapped

    acc = accuracy_score(true_states, preds)
    cm  = confusion_matrix(true_states, preds, labels=[0, 1])
    if show_plot:
        plot_confusion_matrix(cm, title="Confusion Matrix: True vs Predicted")
    return acc, cm, preds

def detailed_parameter_report(true, recovered, name="Parameter", threshold=0.10):
    print(f"\n=== Detailed Recovery Report: {name} ===")
    T = np.array(true); R = np.array(recovered)
    if T.ndim == 1: T = T.reshape(1, -1); R = R.reshape(1, -1)
    D = np.abs(T - R)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            delta = D[i, j]
            flag  = "⚠" if delta >= threshold else "✓"
            print(f"[{i},{j}] true={T[i,j]:.3f} rec={R[i,j]:.3f} Δ={delta:.3f} {flag}")
    print(f"Mean Abs Error: {D.mean():.4f}")

def _state_dist(states, n_states=2):
    c = np.bincount(states, minlength=n_states)
    return c / c.sum()

def kl_divergence_states(true_states, pred_states, n_states=2):
    p = _state_dist(true_states, n_states)
    q = _state_dist(pred_states, n_states)
    return np.sum(rel_entr(p, q))

def print_overall_statistics(df):
    """Pretty summary across all runs."""
    metrics = ["accuracy", "kld_transmat", "kld_state_distribution", "kld_startprob"]
    print("\nOverall Summary Statistics Across All Seeds & Sample Sizes:")
    for metric in metrics:
        if metric in df.columns:
            s = df[metric].dropna()
            print(f"\n— {metric.upper()} —")
            print(f"  Mean : {s.mean():.4f}")
            print(f"  Std  : {s.std():.4f}")
            print(f"  Min  : {s.min():.4f}")
            print(f"  Max  : {s.max():.4f}")
