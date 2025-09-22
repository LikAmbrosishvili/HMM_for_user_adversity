## Project structure

```text
hmm_project/
│
├── main.py               ← Runs all experiments
│
└── hmm_model/
    ├── __init__.py
    ├── core.py           ← Defines the true HMM model (startprob, transmat, emissions)
    ├── runner.py         ← Runs 1 experiment: sampling, training, evaluation
    ├── utils.py          ← Helper functions (KL divergence, flatten/unflatten)
    ├── evaluation.py     ← Accuracy, confusion matrix, KL divergence for states
    └── plotting.py       ← Plots: accuracy vs. sample size, KL divergence, etc.
