"""
run_experiment.py
==================
End-to-end experiment runner for:

  "Continuous Fairness Assurance for Federated Learning: Privacy-
   Preserving Drift Monitoring and Provably Correct Recertification"

Runs three things:

  EXPERIMENT A (Tier 1 - bound validation):
      Injects controlled, KNOWN drift levels and checks that the
      deterministic bound and the privacy-composed bound both stay
      >= the true measured fairness violation (i.e. the bound is a
      valid, non-vacuous upper envelope), across a sweep of drift
      magnitudes and privacy budgets (epsilon).

  EXPERIMENT B (Tier 2/3 - trigger vs. baselines):
      Simulates a federation drifting over many rounds (gradual drift
      in some states + a sudden shock in one state, mimicking a new
      hospital/site joining) and compares:
        - our privacy-preserving, drift-triggered recertification
        - fixed-interval recertification
        - always-audit (upper bound on cost)
      on (i) number of expensive cryptographic audits performed and
      (ii) detection lag (rounds between the TRUE fairness violation
      and the policy actually catching it).

USAGE
-----
    python run_experiment.py                # synthetic data (offline, default)
    python run_experiment.py --data real     # real Folktables data (needs internet)

Outputs are written to ./outputs/ as CSV + PNG figures.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data as D
from src import fairness as F
from src import drift as DR
from src import bound as B
from src import baselines as BL

OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

STATES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]  # 8 simulated cross-silo clients
SEED = 42


# ------------------------------------------------------------------
# Data loading (real Folktables if available & requested, else synthetic)
# ------------------------------------------------------------------
def get_certified_baseline(use_real=False, use_flamby=False):
    if use_flamby:
        try:
            print("Attempting to load FLamby Fed-Heart-Disease "
                  "(requires flamby installed + dataset already downloaded)...")
            fed = D.load_flamby_heart_disease()
            print("Loaded FLamby Fed-Heart-Disease for centers:", list(fed.keys()))
            return fed, list(fed.keys()), "flamby_heart"
        except Exception as e:
            print(f"[WARN] FLamby load failed ({e}). Falling back to synthetic ACS-like data.")
    elif use_real:
        try:
            print("Attempting to download real ACS/Folktables data "
                  "(requires internet access to census.gov)...")
            real_states = ["CA", "TX", "NY", "FL", "PA", "IL", "OH", "GA"]
            fed = D.load_folktables(real_states, year="2016", task="income")
            print("Loaded real Folktables data for:", list(fed.keys()))
            return fed, real_states, "real"
        except Exception as e:
            print(f"[WARN] Real data load failed ({e}). "
                  f"Falling back to synthetic ACS-like data.")
    fed = D.make_synthetic_federation(STATES, n_per_client=4000, seed=SEED)
    return fed, STATES, "synthetic"


# ------------------------------------------------------------------
# EXPERIMENT A: bound validation across a drift sweep
# ------------------------------------------------------------------
def experiment_a(baseline_fed, states, scaler, model, L, Bnorm, dp0, epsilon=1.0, delta_dp=1e-5):
    rows = []
    rng = np.random.default_rng(SEED + 1)
    drift_levels = np.linspace(0.0, 1.0, 11)

    for lvl in drift_levels:
        schedule = {st: lvl for st in states}  # uniform drift across all clients
        current_fed = D.make_drifted_snapshot(states, schedule, n_per_client=4000, seed=SEED + 100)

        dp_true = F.demographic_parity(current_fed, scaler, model)

        # true (non-private) TV distance, for reference / theory validation only
        all_age = np.concatenate([df["age"].values for df in baseline_fed.values()])
        all_educ = np.concatenate([df["education"].values for df in baseline_fed.values()])
        all_hours = np.concatenate([df["hours"].values for df in baseline_fed.values()])
        edges = (DR._bin_edges(all_age), DR._bin_edges(all_educ), DR._bin_edges(all_hours))
        hist_base = DR.secure_aggregate_histograms(baseline_fed, edges)
        hist_curr = DR.secure_aggregate_histograms(current_fed, edges)
        delta_true = DR.tv_distance_from_hists(hist_base, hist_curr)

        det_bound = B.deterministic_bound(dp0, delta_true, L, Bnorm)

        delta_hat, eta = DR.private_drift_estimate(baseline_fed, current_fed, epsilon, delta_dp, rng=rng)
        priv_bound = B.private_bound(dp0, delta_hat, eta, L, Bnorm)

        rows.append(dict(drift_level=lvl, dp_true=dp_true, delta_true=delta_true,
                          det_bound=det_bound, delta_hat=delta_hat, eta=eta,
                          priv_bound=priv_bound,
                          det_valid=det_bound >= dp_true - 1e-9,
                          priv_valid=priv_bound >= dp_true - 1e-9))
        print(f"  drift={lvl:.1f}  DP_true={dp_true:.4f}  det_bound={det_bound:.4f}"
              f"  priv_bound={priv_bound:.4f}  (delta_hat={delta_hat:.4f}, eta={eta:.4f})")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "experiment_a_bound_validation.csv"), index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(df.drift_level, df.dp_true, "o-", label="True DP(P_t, h)", color="black", linewidth=2)
    plt.plot(df.drift_level, df.det_bound, "s--", label="Deterministic bound (true drift)")
    plt.plot(df.drift_level, df.priv_bound, "^--", label=f"Privacy-composed bound (eps={epsilon})")
    plt.xlabel("Injected drift level")
    plt.ylabel("Demographic parity difference")
    plt.title("Experiment A: Bound validity across injected drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "experiment_a_bound_validation.png"), dpi=150)
    plt.close()

    print(f"\n  Deterministic bound valid (>= true) in {df.det_valid.mean()*100:.0f}% of settings")
    print(f"  Privacy-composed bound valid (>= true) in {df.priv_valid.mean()*100:.0f}% of settings")
    return df


# ------------------------------------------------------------------
# EXPERIMENT B: recertification trigger vs. baselines over many rounds
# ------------------------------------------------------------------
def build_round_schedule(states, n_rounds, shock_state, shock_round):
    """
    Builds a per-round drift schedule dict[state] -> drift_level in [0,1]:
      - most states drift slowly and linearly over time (realistic
        gradual demographic change)
      - one state has a sudden shock at `shock_round` (e.g. a new site
        with a very different population joins), simulating the
        "non-ideal, failure-prone" scenario the CFP calls out.
    """
    schedules = []
    for t in range(n_rounds):
        sched = {}
        for st in states:
            gradual = min(1.0, 0.35 * (t / n_rounds))
            sched[st] = gradual
        if t >= shock_round:
            sched[shock_state] = min(1.0, 0.9)
        schedules.append(sched)
    return schedules


def experiment_b(baseline_fed, states, scaler, model, L, Bnorm, dp0,
                  epsilon=1.0, delta_dp=1e-5, epsilon_max=0.10,
                  n_rounds=40, fixed_interval=5):
    rng = np.random.default_rng(SEED + 2)
    schedules = build_round_schedule(states, n_rounds, shock_state=states[-1], shock_round=n_rounds // 2)

    rows = []
    for t, sched in enumerate(schedules):
        current_fed = D.make_drifted_snapshot(states, sched, n_per_client=3000, seed=SEED + 200 + t)
        dp_true = F.demographic_parity(current_fed, scaler, model)
        violation = dp_true > epsilon_max

        delta_hat, eta = DR.private_drift_estimate(baseline_fed, current_fed, epsilon, delta_dp, rng=rng)
        our_trigger, our_bound_value = B.recertification_trigger(dp0, delta_hat, eta, L, Bnorm, epsilon_max)

        fixed_audit = BL.fixed_interval_policy(t, fixed_interval)
        always_audit = BL.always_audit_policy(t)

        rows.append(dict(round=t, dp_true=dp_true, violation=violation,
                          delta_hat=delta_hat, eta=eta, our_bound=our_bound_value,
                          our_audit=our_trigger, fixed_audit=fixed_audit, always_audit=always_audit))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "experiment_b_trigger_vs_baselines.csv"), index=False)

    # --- summary metrics ---
    def detection_lag(policy_col):
        """Rounds between the FIRST true violation and the first audit at/after it."""
        viol_idx = df.index[df["violation"]].tolist()
        if not viol_idx:
            return np.nan
        first_viol = viol_idx[0]
        audit_idx = df.index[(df[policy_col]) & (df.index >= first_viol)].tolist()
        if not audit_idx:
            return len(df) - first_viol  # never caught within horizon
        return audit_idx[0] - first_viol

    summary = pd.DataFrame({
        "policy": ["Ours (drift-triggered)", "Fixed-interval", "Always-audit"],
        "n_audits": [df.our_audit.sum(), df.fixed_audit.sum(), df.always_audit.sum()],
        "detection_lag_rounds": [detection_lag("our_audit"),
                                  detection_lag("fixed_audit"),
                                  detection_lag("always_audit")],
    })
    summary.to_csv(os.path.join(OUTDIR, "experiment_b_summary.csv"), index=False)
    print("\n  Policy comparison summary:")
    print(summary.to_string(index=False))

    # --- plots ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(df["round"], df.dp_true, color="black", linewidth=2, label="True DP(P_t, h)")
    axes[0].axhline(epsilon_max, color="red", linestyle=":", label="Tolerance epsilon_max")
    axes[0].scatter(df["round"][df.our_audit], df.dp_true[df.our_audit],
                     marker="^", color="tab:blue", s=60, label="Our trigger audits", zorder=5)
    axes[0].scatter(df["round"][df.fixed_audit], df.dp_true[df.fixed_audit] + 0.005,
                     marker="s", color="tab:orange", s=30, label="Fixed-interval audits", zorder=4)
    axes[0].set_ylabel("Demographic parity")
    axes[0].set_title("Experiment B: true fairness trajectory & when each policy audits")
    axes[0].legend(loc="upper left")

    axes[1].plot(df["round"], df.our_bound, color="tab:blue", label="Our privacy-composed bound")
    axes[1].axhline(epsilon_max, color="red", linestyle=":")
    axes[1].set_xlabel("Federated round")
    axes[1].set_ylabel("Bound value")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "experiment_b_trigger_trace.png"), dpi=150)
    plt.close()

    bar_fig, bax = plt.subplots(1, 2, figsize=(10, 4))
    bax[0].bar(summary.policy, summary.n_audits, color=["tab:blue", "tab:orange", "tab:gray"])
    bax[0].set_ylabel("# expensive cryptographic audits")
    bax[0].set_title("Audit cost")
    bax[0].tick_params(axis="x", rotation=15)

    bax[1].bar(summary.policy, summary.detection_lag_rounds, color=["tab:blue", "tab:orange", "tab:gray"])
    bax[1].set_ylabel("Detection lag (rounds)")
    bax[1].set_title("Violation detection speed")
    bax[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "experiment_b_summary_bars.png"), dpi=150)
    plt.close()

    return df, summary


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "real", "flamby_heart"], default="synthetic")
    parser.add_argument("--epsilon", type=float, default=2.0, help="DP epsilon for drift estimator")
    parser.add_argument("--epsilon_max", type=float, default=0.15, help="Regulator's fairness tolerance")
    parser.add_argument("--n_rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global SEED
    SEED = args.seed

    print("=" * 70)
    print("STEP 1: Load / generate certified baseline federation")
    print("=" * 70)
    baseline_fed, states, source = get_certified_baseline(
        use_real=(args.data == "real"), use_flamby=(args.data == "flamby_heart"))
    print(f"Data source used: {source}  |  clients: {states}")

    feature_cols = D.FLAMBY_HEART_FEATURE_COLS if source == "flamby_heart" else None

    print("\n" + "=" * 70)
    print("STEP 2: Train the certified federated model h (FedAvg, logistic reg.)")
    print("=" * 70)
    scaler, model, L = F.federated_average_train(baseline_fed, seed=SEED, feature_cols=feature_cols)
    Bnorm = B.estimate_feature_norm_bound(baseline_fed, scaler, feature_cols=feature_cols)
    dp0 = F.demographic_parity(baseline_fed, scaler, model, feature_cols=feature_cols)
    acc0 = F.accuracy(baseline_fed, scaler, model, feature_cols=feature_cols)
    print(f"Certified model: accuracy={acc0:.3f}, DP(P0,h)={dp0:.4f}, "
          f"Lipschitz L={L:.4f}, feature-norm bound B={Bnorm:.4f}")

    if source in ("real", "flamby_heart"):
        print("\n[NOTE] Experiments A and B below inject SYNTHETIC drift and are only "
              "meaningful for the synthetic generator, which has a controllable drift dial. "
              "Real data (Folktables/FLamby) has no such dial -- validating the bound on real "
              "drift means comparing two real snapshots directly (e.g. two different states, or "
              "two different years), not injecting a parameter. That is a different experiment "
              "script; see the note in the chat response for how to adapt this one.")
        print("All outputs written to:", OUTDIR)
        return

    print("\n" + "=" * 70)
    print("STEP 3: EXPERIMENT A — bound validation across injected drift")
    print("=" * 70)
    df_a = experiment_a(baseline_fed, states, scaler, model, L, Bnorm, dp0, epsilon=args.epsilon)

    print("\n" + "=" * 70)
    print("STEP 4: EXPERIMENT B — trigger vs. fixed-interval vs. always-audit")
    print("=" * 70)
    df_b, summary_b = experiment_b(baseline_fed, states, scaler, model, L, Bnorm, dp0,
                                    epsilon=args.epsilon, epsilon_max=args.epsilon_max,
                                    n_rounds=args.n_rounds)

    print("\nAll outputs written to:", OUTDIR)


if __name__ == "__main__":
    main()
