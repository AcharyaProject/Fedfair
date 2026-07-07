# Continuous Fairness Assurance for Federated Learning — Reference Implementation

This is a runnable prototype of the experiment plan discussed for the paper:
*"Continuous Fairness Assurance for Federated Learning: Privacy-Preserving
Drift Monitoring and Provably Correct Recertification."*

It implements, end to end:

1. A federated (FedAvg) logistic-regression model, certified on a baseline
   population, with an explicit Lipschitz constant (needed for the theory).
2. A **deterministic** fairness-degradation bound (builds on the FedPF /
   Wasserstein-audit style of result — this part is *not* claimed as novel).
3. A **privacy-preserving drift estimator**: client histograms are summed
   via simulated secure aggregation (only the sum is ever touched) and then
   perturbed with calibrated Gaussian noise to satisfy (ε, δ)-DP, giving a
   noisy Total-Variation drift estimate `delta_hat` plus a high-probability
   error bound `eta`.
4. The **privacy-composed bound**: substitutes `delta_hat + eta` into the
   deterministic bound.
5. The **recertification trigger**: the paper's flagship claim — audit
   (expensive, cryptographic fairness verification) if and only if the
   privacy-composed bound would exceed the regulator's tolerance
   `epsilon_max`.
6. Two baselines for comparison: fixed-interval recertification, and
   always-audit.

## Quick start

```bash
cd fedfair_recert
pip install -r requirements.txt
python run_experiment.py                      # synthetic data (offline, default)
python run_experiment.py --data real           # real Folktables data (needs internet)
python run_experiment.py --epsilon_max 0.15 --epsilon 2.0 --n_rounds 40
```

Outputs (CSVs + PNG figures) are written to `outputs/`:
- `experiment_a_bound_validation.{csv,png}` — Tier 1: confirms the bound is
  a valid, non-vacuous upper envelope on true demographic parity across a
  sweep of injected drift levels.
- `experiment_b_trigger_vs_baselines.csv`, `experiment_b_summary.csv`,
  `experiment_b_trigger_trace.png`, `experiment_b_summary_bars.png` —
  Tier 2/3: our trigger vs. fixed-interval vs. always-audit, measured on
  (a) number of expensive audits performed and (b) detection lag.

## Datasets

### Default (used automatically in this sandbox): synthetic ACS-like generator

`src/data.py` contains a synthetic data generator that mimics the structure
of the ACS Income task: age, education, hours-worked, occupation code,
a binary protected attribute `A`, and a binary income label `Y`, with
**proxy discrimination built in** (features are generated conditionally on
`A`, so a model that excludes `A` still inherits disparate impact through
correlated features — exactly the real-world mechanism this paper is
about). Population drift is injected via an explicit, controllable
parameter, which is what lets Experiment A validate the bound against a
known ground truth.

This generator is fully offline and reproducible, which is why it's the
default here — this sandboxed environment cannot reach external data hosts
(census.gov, openml.org, PhysioNet, Kaggle, etc.), only a small allowlist
of package registries (PyPI, GitHub, etc.).

### For the actual paper: real datasets to switch in

| Dataset | Why it fits | Access |
|---|---|---|
| **Folktables / ACS PUMS** (`--data real` flag, already wired up in `src/data.py::load_folktables`) | Real US Census microdata, natively indexed by **state** and **year** — a genuine, non-synthetic drift axis (geographic and temporal). Already used in the fairness-under-shift literature you're positioning against. | `pip install folktables`; downloads directly from census.gov, no credentialing needed. |
| **FLamby** (`github.com/owkin/FLamby`) | 7 real healthcare datasets with **natural** (not synthetic) cross-silo client splits across real institutions — the closest thing to an authentic federated fairness testbed. Recommended tasks: `Fed-ISIC2019` (skin lesion classification, multi-site, real demographic skew) or `Fed-Heart-Disease` (small, tabular, easy to compute demographic parity on). | `pip install flamby` (or clone the repo); some sub-datasets require accepting data use terms on their original source. |
| **MIMIC-IV / eICU** | Multi-year, multi-ICU real temporal drift, if you want chronological rather than cross-sectional drift. | Requires PhysioNet credentialing (short training + data use agreement — start early). |
| **Lending Club / Give Me Some Credit** | Optional second domain (finance) with loan-vintage/year cohorts standing in for temporal drift. | Open download (Kaggle). |

To switch `run_experiment.py` from the synthetic generator to FLamby or
MIMIC, replace the call to `get_certified_baseline()` with a loader that
returns a `dict[client_id] -> DataFrame` with the same column contract
(`FEATURE_COLS` in `src/fairness.py`, plus `A` and `Y`) — everything
downstream (model training, drift estimator, bound, trigger) is agnostic
to the data source.

## File structure

```
fedfair_recert/
├── run_experiment.py       # main entry point, both experiments
├── requirements.txt
├── src/
│   ├── data.py              # real Folktables loader + synthetic generator
│   ├── fairness.py          # FedAvg training, demographic parity metric
│   ├── drift.py             # secure-aggregation-simulated, DP-noised TV drift estimator
│   ├── bound.py             # deterministic + privacy-composed fairness bound, trigger
│   └── baselines.py         # fixed-interval / always-audit comparison policies
└── outputs/                 # generated CSVs + figures
```

## Honest scope notes (carried over from the novelty audit)

- The **deterministic bound** (`bound.deterministic_bound`) is a
  Lipschitz/TV-based bound in the style of prior work (FedPF, Wasserstein
  fairness audits) — it is implemented here as a *foundation*, not
  presented as new.
- The **secure aggregation** in `drift.py` is *simulated* (we only ever
  operate on summed histograms, never per-client ones, matching what real
  SecAgg guarantees) rather than cryptographically implemented, since the
  cryptographic engineering is orthogonal to the statistical claim being
  tested. A real deployment would replace `secure_aggregate_histograms`
  with an actual SecAgg/MPC protocol call.
- The **recertification trigger** (`bound.recertification_trigger`) is the
  part of this codebase closest to the paper's actual theoretical
  contribution — the sufficiency argument implemented here is deliberately
  simple (a direct threshold check on the composed bound); strengthening
  it toward a necessity/optimality result, per the earlier discussion, is
  the main remaining theoretical work and is not yet reflected in this code.
