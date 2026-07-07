"""
drift.py
========
Privacy-preserving estimation of population drift between the certified
baseline population P0 and the current population P_t.

We use Total Variation (TV) distance on a binned joint distribution of
(A, feature-summary) as the drift statistic, because:
  (a) it composes cleanly with the FedPF-style TV-based fairness bound
      (see bound.py), and
  (b) it reduces to comparing HISTOGRAM COUNTS across clients, which is
      exactly what secure aggregation is good at (summing masked
      per-client count vectors) -- no client ever reveals its own
      histogram to the server or to other clients.

SECURE AGGREGATION SIMULATION
------------------------------
We do not implement real cryptography here (masking protocols /
homomorphic encryption) since that is an engineering artifact orthogonal
to the statistical claim being validated. Instead we simulate its
*privacy-utility effect*: each client's histogram is:
    1. summed exactly across clients (this is what secure aggregation
       guarantees: only the SUM is revealed, individual client
       histograms never leave the client) -- so we only ever operate
       on the summed histogram in this code, never on a per-client one;
    2. perturbed with calibrated noise to satisfy (eps, delta)-DP on
       top of the secure sum, matching the standard "SecAgg + distributed
       DP noise" utility profile used in the FL privacy literature
       (e.g. distributed discrete Gaussian mechanisms).

This gives us the estimator's (epsilon_est, delta_est)-DP guarantee and
its noise-induced estimation error, which is exactly the "eta" term the
theory composes into the fairness bound.
"""

import numpy as np


N_BINS_PER_DIM = 4  # coarse binning keeps histograms small -> less noise needed


def _bin_edges(values, n_bins=N_BINS_PER_DIM):
    qs = np.linspace(0, 100, n_bins + 1)
    return np.unique(np.percentile(values, qs))


def compute_joint_histogram(df, edges_age, edges_educ, edges_hours):
    """
    Bins (A, age, education, hours) into a joint histogram (a flat count
    vector). This is the per-client statistic that gets secure-aggregated.
    """
    a_bin = df["A"].values  # already binary: 0/1
    age_bin = np.digitize(df["age"].values, edges_age)
    educ_bin = np.digitize(df["education"].values, edges_educ)
    hours_bin = np.digitize(df["hours"].values, edges_hours)

    n_age = len(edges_age) + 1
    n_educ = len(edges_educ) + 1
    n_hours = len(edges_hours) + 1

    flat_idx = ((a_bin * n_age + age_bin) * n_educ + educ_bin) * n_hours + hours_bin
    n_total_bins = 2 * n_age * n_educ * n_hours
    hist = np.bincount(flat_idx, minlength=n_total_bins).astype(float)
    return hist


def secure_aggregate_histograms(client_dfs, edges):
    """
    Simulates SecAgg: returns ONLY the summed histogram across clients.
    No intermediate per-client histogram is exposed outside this function.
    """
    edges_age, edges_educ, edges_hours = edges
    total = None
    for df in client_dfs.values():
        h = compute_joint_histogram(df, edges_age, edges_educ, edges_hours)
        total = h if total is None else total + h
    return total


def dp_noised_histogram(hist_sum, epsilon, delta, sensitivity=1.0, rng=None):
    """
    Adds calibrated Gaussian noise to the securely-aggregated histogram
    to achieve (epsilon, delta)-DP for a single release (one client's
    departure/arrival changes the sum by at most `sensitivity` in each
    bin, i.e. L2 sensitivity ~= sensitivity for a single record).

    Returns: noised histogram (clipped at 0), and the per-bin noise std.
    """
    rng = rng or np.random.default_rng()
    sigma = (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
    noise = rng.normal(0, sigma, size=hist_sum.shape)
    noised = np.clip(hist_sum + noise, 0, None)
    return noised, sigma


def tv_distance_from_hists(hist_a, hist_b):
    """Total variation distance between two (unnormalized) histograms."""
    pa = hist_a / hist_a.sum()
    pb = hist_b / hist_b.sum()
    return 0.5 * np.abs(pa - pb).sum()


def private_drift_estimate(baseline_dfs, current_dfs, epsilon, delta, rng=None):
    """
    Full privacy-preserving drift estimation pipeline.

    Returns
    -------
    delta_hat : float   noisy TV-distance estimate between baseline and
                         current federation populations
    eta       : float   an analytic high-probability bound on the
                         estimation error |delta_hat - delta_true|,
                         derived from the noise added at each histogram
                         (used directly in bound.py to inflate the
                         fairness certificate).
    """
    rng = rng or np.random.default_rng()
    all_age = np.concatenate([df["age"].values for df in baseline_dfs.values()])
    all_educ = np.concatenate([df["education"].values for df in baseline_dfs.values()])
    all_hours = np.concatenate([df["hours"].values for df in baseline_dfs.values()])
    edges = (_bin_edges(all_age), _bin_edges(all_educ), _bin_edges(all_hours))

    hist_base = secure_aggregate_histograms(baseline_dfs, edges)
    hist_curr = secure_aggregate_histograms(current_dfs, edges)

    hist_base_noised, sigma_b = dp_noised_histogram(hist_base, epsilon, delta, rng=rng)
    hist_curr_noised, sigma_c = dp_noised_histogram(hist_curr, epsilon, delta, rng=rng)

    delta_hat = tv_distance_from_hists(hist_base_noised, hist_curr_noised)

    # High-probability error bound (union bound over both noised releases,
    # each Gaussian coordinate contributes to L1-normalized TV error;
    # this is a conservative closed-form envelope, not a tight one).
    n_bins = len(hist_base)
    alpha = 0.05
    z = np.sqrt(2 * np.log(2 * n_bins / alpha))  # per-coordinate high-prob factor
    err_base = z * sigma_b * np.sqrt(n_bins) / hist_base.sum()
    err_curr = z * sigma_c * np.sqrt(n_bins) / hist_curr.sum()
    eta = 0.5 * (err_base + err_curr)  # propagate through the 0.5 * L1 TV definition

    return delta_hat, eta
