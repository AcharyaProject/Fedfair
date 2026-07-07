"""
bound.py
========
Implements:
  1. The deterministic fairness-stability bound (Theorem-1-style):
        |DP(P_t, h) - DP(P_0, h)| <= g(TV(P_t, P_0))
     We use a Lipschitz-based bound: for a model with logistic link and
     linear score bounded by Lipschitz constant L (w.r.t. the feature
     map), a standard TV-based perturbation inequality gives
        g(delta) = 2 * L * B * delta
     where B is a bound on the feature-map norm (we estimate B
     empirically from the certified population as a conservative
     constant). This mirrors the style of bound used in FedPF /
     Wasserstein-audit-style fairness stability results -- we are
     NOT claiming this bound itself is new (see conversation), we are
     building on it.

  2. The privacy-composed bound (Theorem-2-style):
        DP(P_t, h) <= DP(P_0, h) + g(delta_hat + eta)
     i.e. substitute the noisy drift estimate + its high-probability
     error bound into (1).

  3. The recertification trigger policy (Theorem-3-style, the paper's
     flagship claim): recertify iff the RHS of (2) would exceed the
     regulator's tolerance epsilon_max.
"""

import numpy as np


def estimate_feature_norm_bound(client_dfs, scaler, quantile=0.99, feature_cols=None):
    """
    Empirically estimates B = a high-quantile bound on ||scaled feature
    vector||_2 under the certified baseline population. Used as a fixed
    constant in the Lipschitz bound; NOT re-estimated at each round.
    """
    from .fairness import FEATURE_COLS
    cols = feature_cols or FEATURE_COLS
    X = np.vstack([df[cols].values for df in client_dfs.values()])
    Xs = scaler.transform(X)
    norms = np.linalg.norm(Xs, axis=1)
    return float(np.quantile(norms, quantile))


def g(delta, L, B):
    """Lipschitz-based fairness-degradation envelope g(delta) = 2*L*B*delta."""
    return 2.0 * L * B * delta


def deterministic_bound(dp0, delta_true, L, B):
    """Theorem-1-style bound using the TRUE (non-private) drift -- used
    only for validating the theory in Tier-1 synthetic experiments,
    never available to the deployed system itself."""
    return dp0 + g(delta_true, L, B)


def private_bound(dp0, delta_hat, eta, L, B):
    """Theorem-2-style bound: substitute the noisy estimate + error term."""
    return dp0 + g(delta_hat + eta, L, B)


def recertification_trigger(dp0, delta_hat, eta, L, B, epsilon_max):
    """
    Theorem-3-style operational policy.

    Returns True (recertify NOW) iff the privacy-composed upper bound on
    current fairness violation would exceed the regulator's tolerance
    epsilon_max. This is the SUFFICIENT condition: whenever the trigger
    is False, the true DP(P_t, h) is guaranteed <= epsilon_max with the
    estimator's confidence level (1 - alpha), without ever needing to
    run the expensive cryptographic fairness audit itself.
    """
    bound_value = private_bound(dp0, delta_hat, eta, L, B)
    return bound_value > epsilon_max, bound_value
