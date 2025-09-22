import os, random, pandas as pd
from hmm_model.runner import run_experiment
from hmm_model.plotting import plot_accuracy_per_seed, plot_kld_per_seed
from hmm_model.evaluation import print_overall_statistics

os.makedirs("Logs", exist_ok=True)

def run_multiple_seeds(sample_sizes, seeds, show_plot=False):
    rows = []
    for n in sample_sizes:
        for s in seeds:
            print(f"▶ sample_size={n}, seed={s}")
            r = run_experiment(n_samples=n, seed=s, show_plot=show_plot)
            rows.append(r)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Build sample sizes dynamically
    sample_sizes = [10] + list(range(50, 5001, 50))

    seeds = random.sample(range(1, 100000), 12)

    df = run_multiple_seeds(sample_sizes, seeds, show_plot=False)
    df.to_csv("Logs/multirun_results.csv", index=False)
    print(df.groupby("sample_size")["accuracy"].describe())

    plot_accuracy_per_seed(df)
    plot_kld_per_seed(df, "kld_transmat", "KLD (Transition Matrix) per Seed")
    plot_kld_per_seed(df, "kld_state_distribution", "KLD (State Dist) per Seed")
    print_overall_statistics(df)

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
