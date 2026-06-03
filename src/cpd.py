"""Changepoint detection via Gaussian Processes.

Implements the Matern 3/2 kernel fit and the changepoint kernel from
Wood, Roberts & Zohren (2022), "Slow Momentum with Fast Reversion".

All GP computations use scipy and numpy only -- no GPflow / TensorFlow.

Reference equations from the paper:
    Eq. 4  -- Matern 3/2 kernel
    Eq. 7  -- Negative log marginal likelihood (NLML)
    Eq. 9  -- Sigmoid-blended changepoint kernel
    Eq. 10 -- Severity (nu) and location (gamma)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import ruptures as rpt

# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------


def _matern32_kernel(X: np.ndarray, sigma_f: float, lengthscale: float) -> np.ndarray:
    """Matern 3/2 covariance matrix.  (Paper Eq. 4)

    k(x, x') = sigma_f^2 * (1 + sqrt(3)*|x - x'| / l)
                          * exp(-sqrt(3)*|x - x'| / l)

    Parameters
    ----------
    X : (n,) array of input locations (time indices).
    sigma_f : output-scale standard deviation.
    lengthscale : length-scale lambda.

    Returns
    -------
    K : (n, n) covariance matrix.
    """
    dist = np.abs(X[:, None] - X[None, :])
    lengthscale = max(lengthscale, 1e-10)
    r = np.sqrt(3.0) * dist / lengthscale
    K = sigma_f ** 2 * (1.0 + r) * np.exp(-r)
    np.nan_to_num(K, copy=False, nan=0.0, posinf=1e10, neginf=0.0)
    return K


def _sigmoid(x: np.ndarray, c: float, s: float) -> np.ndarray:
    """Logistic sigmoid used for the changepoint blend.

    sigma(x) = 1 / (1 + exp(-s * (x - c)))

    where c is the changepoint location and s > 0 is the steepness.
    (Paper: sigma(x) = 1/(1 + e^{-s(x-c)}), see text below Eq. 8.)
    """
    z = np.asarray(s * (x - c), dtype=np.float64)
    z_clamped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clamped))


def _changepoint_kernel(
    X: np.ndarray,
    sigma_f1: float, l1: float,
    sigma_f2: float, l2: float,
    c: float, s: float,
) -> np.ndarray:
    """Changepoint kernel.  (Paper Eq. 9)

    k_cp(x, x') = k_{s1}(x, x') * sig(x) * sig(x')
                 + k_{s2}(x, x') * sig_bar(x) * sig_bar(x')

    where sig_bar(x) = 1 - sig(x).
    """
    K1 = _matern32_kernel(X, sigma_f1, l1)
    K2 = _matern32_kernel(X, sigma_f2, l2)

    sig = _sigmoid(X, c, s)
    sig_bar = 1.0 - sig

    S = sig[:, None] * sig[None, :]
    S_bar = sig_bar[:, None] * sig_bar[None, :]

    return K1 * S + K2 * S_bar


# ---------------------------------------------------------------------------
# Negative log marginal likelihood
# ---------------------------------------------------------------------------


def _nlml(K: np.ndarray, y: np.ndarray, sigma_n: float) -> float:
    """Negative log marginal likelihood of a GP.  (Paper Eq. 7)

    nlml = 0.5 * y^T V^{-1} y  +  0.5 * log|V|  +  n/2 * log(2*pi)

    where  V = K + sigma_n^2 * I.  Uses Cholesky for numerical stability.
    """
    n = len(y)
    V = K + sigma_n ** 2 * np.eye(n)
    V += 1e-6 * np.eye(n)  # jitter for Cholesky stability

    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return 1e10

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    data_fit  = 0.5 * y @ alpha
    complexity = np.sum(np.log(np.diag(L)))
    constant   = 0.5 * n * np.log(2.0 * np.pi)

    return data_fit + complexity + constant


# ---------------------------------------------------------------------------
# Fitting routines
# ---------------------------------------------------------------------------


def _fit_base_matern(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit a GP with a single Matern 3/2 kernel by minimizing NLML.

    Hyperparameters: theta = [log(sigma_f), log(lengthscale), log(sigma_n)]
    All initialized to 1 (log=0) as in the paper (page 9).

    Returns
    -------
    best_nlml : minimized NLML.
    best_params : [sigma_f, lengthscale, sigma_n] at the optimum.
    """

    def objective(log_theta):
        sigma_f, lengthscale, sigma_n = np.exp(log_theta)
        K = _matern32_kernel(X, sigma_f, lengthscale)
        return _nlml(K, y, sigma_n)

    log_theta0 = np.array([0.0, 0.0, 0.0])
    log_bounds = [(-10.0, 10.0)] * 3

    result = minimize(
        objective,
        log_theta0,
        method="L-BFGS-B",
        bounds=log_bounds,
        options={"maxiter": 200, "ftol": 1e-8},
    )

    return result.fun, np.exp(result.x)


def _fit_changepoint(
    X: np.ndarray,
    y: np.ndarray,
    base_params: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit a GP with the changepoint kernel by minimizing NLML.

    7 hyperparameters: [log(sf1), log(l1), log(sf2), log(l2), c, log(s), log(sn)]
    Initialized with base Matern params (paper page 9):
        c = (n-1)/2, s = 1, k_{s1} = k_{s2} = base Matern.

    Returns
    -------
    best_nlml : minimized NLML.
    best_params : [sigma_f1, l1, sigma_f2, l2, c, s, sigma_n] at optimum.
    """
    n = len(y)
    sigma_f_base, l_base, sigma_n_base = base_params

    theta0 = np.array([
        np.log(sigma_f_base),
        np.log(l_base),
        np.log(sigma_f_base),
        np.log(l_base),
        (n - 1) / 2.0,    # c = middle of window
        np.log(1.0),       # s = 1
        np.log(sigma_n_base),
    ])

    bounds = [
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0),
        (float(X[0] + 0.5), float(X[-1] - 0.5)),  # c stays inside window
        (-10.0, 10.0),
        (-10.0, 10.0),
    ]

    def objective(theta):
        sf1    = np.exp(theta[0]); l1  = np.exp(theta[1])
        sf2    = np.exp(theta[2]); l2  = np.exp(theta[3])
        c      = theta[4]
        s      = np.exp(theta[5]); sn  = np.exp(theta[6])
        K = _changepoint_kernel(X, sf1, l1, sf2, l2, c, s)
        return _nlml(K, y, sn)

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-8},
    )

    t = result.x
    best_params = np.array([
        np.exp(t[0]), np.exp(t[1]),
        np.exp(t[2]), np.exp(t[3]),
        t[4],
        np.exp(t[5]), np.exp(t[6]),
    ])

    return result.fun, best_params


def _fit_changepoint_with_retry(
    X: np.ndarray,
    y: np.ndarray,
    base_params: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit changepoint kernel, retrying with reset params if k_s1 == k_s2.

    From the paper (page 9):
        'In the rare case this process fails, we try again by reinitializing
         all changepoint kernel parameters to 1, except c = t - l/2.'
    """
    nlml_cp, cp_params = _fit_changepoint(X, y, base_params)

    sf1, l1, sf2, l2 = cp_params[0], cp_params[1], cp_params[2], cp_params[3]
    if np.isclose(sf1, sf2, rtol=1e-3) and np.isclose(l1, l2, rtol=1e-3):
        retry_params = np.array([1.0, 1.0, 1.0])
        nlml_cp, cp_params = _fit_changepoint(X, y, retry_params)

    return nlml_cp, cp_params


# ---------------------------------------------------------------------------
# Public API -- GP changepoint scores
# ---------------------------------------------------------------------------


def cpd_scores(returns, lbw: int) -> tuple[float, float]:
    """Return (severity, location) pair (nu, gamma) for a lookback window.

    Given a window of returns of length ``lbw``, the function:
    1. Standardizes returns to zero mean and unit variance.
    2. Fits a base Matern 3/2 GP → nlml_M.
    3. Fits a changepoint GP → nlml_cp.
    4. Computes nu and gamma per Paper Eq. 10:

        nu    = sigmoid(nlml_M - nlml_cp)
        gamma = c / (lbw - 1)           (changepoint location ∈ [0, 1])

    Parameters
    ----------
    returns : array-like, shape (lbw,)
        Raw returns over the lookback window.
    lbw : int
        Lookback window size in days.

    Returns
    -------
    nu : float in (0, 1). Close to 1 = strong changepoint signal.
    gamma : float in (0, 1). Close to 1 = changepoint near the end.
    """
    y = np.asarray(returns, dtype=np.float64).ravel()
    assert len(y) == lbw, f"len(returns)={len(y)} != lbw={lbw}"

    # Standardize (Paper Eq. 2)
    mu  = np.mean(y)
    std = np.std(y, ddof=0)
    if std < 1e-12:
        return 0.0, 0.5  # constant series -- no changepoint
    y_std = (y - mu) / std

    X = np.arange(lbw, dtype=np.float64)

    nlml_base, base_params = _fit_base_matern(X, y_std)
    nlml_cp, cp_params     = _fit_changepoint_with_retry(X, y_std, base_params)

    # nu = sigmoid(nlml_M - nlml_cp): nu → 1 when changepoint kernel fits better
    delta = nlml_base - nlml_cp
    nu    = 1.0 / (1.0 + np.exp(-delta))

    # gamma: normalized position of changepoint in the window
    c_opt = cp_params[4]
    gamma = c_opt / (lbw - 1) if lbw > 1 else 0.5

    return float(np.clip(nu, 0.0, 1.0)), float(np.clip(gamma, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def fit_matern(returns):
    """Fit a GP with a Matern 3/2 kernel. Returns (nlml, [sigma_f, l, sigma_n])."""
    y = np.asarray(returns, dtype=np.float64).ravel()
    X = np.arange(len(y), dtype=np.float64)
    return _fit_base_matern(X, y)


def fit_changepoint_kernel(returns):
    """Fit a GP with the changepoint kernel. Returns (nlml, [sf1,l1,sf2,l2,c,s,sn])."""
    y = np.asarray(returns, dtype=np.float64).ravel()
    X = np.arange(len(y), dtype=np.float64)
    _, base_params = _fit_base_matern(X, y)
    return _fit_changepoint_with_retry(X, y, base_params)


# ---------------------------------------------------------------------------
# Binary Segmentation (offline)
# ---------------------------------------------------------------------------


def binary_segmentation(
    returns: np.ndarray,
    penalty_mult: float = 0.25,
    model: str = "rbf",
) -> list[int]:
    """Offline CPD via Binary Segmentation (ruptures library).

    The penalty scales with the series variance to be asset-class agnostic::

        penalty = penalty_mult * log(n) * var(returns)

    Parameters
    ----------
    returns : (n,) array of returns.
    penalty_mult : multiplier on the BIC-style penalty.
    model : ruptures cost model.

    Returns
    -------
    breaks : list of detection indices (final index n excluded).
    """
    n = len(returns)
    sigma2 = returns.var()
    pen = penalty_mult * np.log(n) * sigma2
    algo = rpt.Binseg(model=model).fit(returns.reshape(-1, 1))
    breaks = algo.predict(pen=pen)
    return [b for b in breaks if b < n]


# ---------------------------------------------------------------------------
# CUSUM (online, combined mean + variance)
# ---------------------------------------------------------------------------


def cusum_combined(
    returns: np.ndarray,
    ref_window: int = 60,
    h_mean: float = 4.0,
    h_var: float = 4.0,
    k_mean: float = 0.5,
    k_var: float = 0.5,
    cooldown: int = 20,
) -> tuple[list[int], np.ndarray]:
    """Online CPD via combined mean + variance CUSUM.

    Maintains two-sided CUSUM statistics for the mean and a one-sided
    statistic for the variance, each referenced against a rolling
    ``ref_window`` estimate.  Statistics reset after each detection.

    Parameters
    ----------
    returns : (n,) array of returns.
    ref_window : observations used to estimate local mean and std.
    h_mean : alert threshold for mean CUSUM.
    h_var : alert threshold for variance CUSUM.
    k_mean : allowance (slack) for mean CUSUM.
    k_var : allowance (slack) for variance CUSUM.
    cooldown : minimum observations between consecutive detections.

    Returns
    -------
    dets : list of detection indices.
    score : (n,) array in [0, 1] via tanh normalisation (NaN for burn-in).
    """
    n = len(returns)
    dets, last = [], -cooldown - 1
    s_pos, s_neg, v_stat = 0.0, 0.0, 0.0
    score = np.full(n, np.nan)

    for t in range(ref_window, n):
        ref = returns[t - ref_window : t]
        mu, sigma = ref.mean(), ref.std(ddof=1)
        if sigma < 1e-12:
            continue
        z = (returns[t] - mu) / sigma

        s_pos = max(0.0, s_pos + z - k_mean)
        s_neg = max(0.0, s_neg - z - k_mean)
        v_stat = max(0.0, v_stat + (z * z - 1.0) - k_var)

        max_mean = max(s_pos, s_neg)
        score[t] = np.tanh(max(max_mean / h_mean, v_stat / h_var))

        if (max_mean > h_mean or v_stat > h_var) and (t - last) > cooldown:
            dets.append(t); last = t
            s_pos = s_neg = v_stat = 0.0

    return dets, score


# ---------------------------------------------------------------------------
# BOCPD (Bayesian Online Changepoint Detection)
# ---------------------------------------------------------------------------


def _log_gaussian_pdf(x: float, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Log-density of a Gaussian at scalar x for arrays of (mu, sigma2)."""
    return -0.5 * (np.log(2 * np.pi * sigma2) + (x - mu) ** 2 / sigma2)


def bocpd(
    returns: np.ndarray,
    hazard: float = 1 / 500,
    prior_mu: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1e-4,
    cooldown: int = 20,
    drop_threshold: int = 30,
    fresh_rl: int = 5,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

    Uses a Normal-Inverse-Gamma (Student-t predictive) observation model.
    Detections are triggered when the MAP run-length drops by at least
    ``drop_threshold`` in a single step.

    The continuous score is P(run_length <= fresh_rl).

    Parameters
    ----------
    returns : (n,) array of returns.
    hazard : constant hazard rate (expected changepoint every 1/hazard steps).
    prior_mu : prior mean for the NIG model.
    kappa0, alpha0, beta0 : NIG prior parameters.
    cooldown : minimum observations between consecutive detections.
    drop_threshold : MAP run-length drop required to register a detection.
    fresh_rl : threshold for the continuous score.

    Returns
    -------
    dets : list of detection indices.
    map_run_length : (n,) int array of MAP run-length estimates.
    score : (n,) float array of changepoint probability in [0, 1].
    """
    n = len(returns)
    log_R = np.zeros(1)
    mu_arr    = np.array([prior_mu]); kappa_arr = np.array([kappa0])
    alpha_arr = np.array([alpha0]);   beta_arr  = np.array([beta0])

    map_rl = np.zeros(n, dtype=int)
    score  = np.zeros(n)
    dets, last = [], -cooldown - 1

    log_haz = np.log(hazard)
    log_1mh = np.log(1 - hazard)
    MAX_RL  = 400

    for t in range(n):
        x = returns[t]
        pred_var = beta_arr * (kappa_arr + 1) / (alpha_arr * kappa_arr)
        log_pred = _log_gaussian_pdf(x, mu_arr, pred_var)

        log_growth = log_R + log_pred + log_1mh
        log_cp     = logsumexp(log_R + log_pred + log_haz)
        log_R      = np.concatenate([[log_cp], log_growth])
        log_R     -= logsumexp(log_R)

        mu_new    = (kappa_arr * mu_arr + x) / (kappa_arr + 1)
        kappa_new = kappa_arr + 1
        alpha_new = alpha_arr + 0.5
        beta_new  = beta_arr + 0.5 * kappa_arr * (x - mu_arr) ** 2 / (kappa_arr + 1)
        mu_arr    = np.concatenate([[prior_mu], mu_new])
        kappa_arr = np.concatenate([[kappa0],   kappa_new])
        alpha_arr = np.concatenate([[alpha0],   alpha_new])
        beta_arr  = np.concatenate([[beta0],    beta_new])

        if len(log_R) > MAX_RL:
            log_R     = log_R[:MAX_RL]; log_R -= logsumexp(log_R)
            mu_arr    = mu_arr[:MAX_RL]; kappa_arr = kappa_arr[:MAX_RL]
            alpha_arr = alpha_arr[:MAX_RL]; beta_arr  = beta_arr[:MAX_RL]

        map_rl[t] = int(np.argmax(log_R))
        score[t]  = float(np.exp(logsumexp(log_R[: fresh_rl + 1])))

        if t > 0:
            drop = map_rl[t - 1] - map_rl[t]
            if drop >= drop_threshold and (t - last) > cooldown:
                dets.append(t); last = t

    return dets, map_rl, score
