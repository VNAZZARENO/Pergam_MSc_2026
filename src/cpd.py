# Méthodes de détection de rupture (CPD) — pipeline STOXX 600
# Toutes les méthodes online produisent : nu ∈ [0,1] (sévérité) et gamma ∈ [0,1] (localisation)

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
import ruptures as rpt


# --- noyaux GP ---


def _matern32_kernel(X, sigma_f, lengthscale):
    dist = np.abs(X[:, None] - X[None, :])
    lengthscale = max(lengthscale, 1e-10)
    r = np.sqrt(3.0) * dist / lengthscale
    K = sigma_f ** 2 * (1.0 + r) * np.exp(-r)
    np.nan_to_num(K, copy=False, nan=0.0, posinf=1e10, neginf=0.0)
    return K


def _sigmoid(x, c, s):
    z = np.asarray(s * (x - c), dtype=np.float64)
    z_clamped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clamped))


def _changepoint_kernel(X, sigma_f1, l1, sigma_f2, l2, c, s):
    # noyau CP = k1*sig*sig + k2*(1-sig)*(1-sig)  (éq. 9)
    K1 = _matern32_kernel(X, sigma_f1, l1)
    K2 = _matern32_kernel(X, sigma_f2, l2)
    sig = _sigmoid(X, c, s)
    sig_bar = 1.0 - sig
    S = sig[:, None] * sig[None, :]
    S_bar = sig_bar[:, None] * sig_bar[None, :]
    return K1 * S + K2 * S_bar


# --- vraisemblance marginale ---


def _nlml(K, y, sigma_n):
    # log-vraisemblance marginale négative d'un GP  (éq. 7)
    n = len(y)
    V = K + sigma_n ** 2 * np.eye(n)
    V += 1e-6 * np.eye(n)  # jitter pour la stabilité de Cholesky
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return 1e10
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    data_fit  = 0.5 * y @ alpha
    complexity = np.sum(np.log(np.diag(L)))
    constant   = 0.5 * n * np.log(2.0 * np.pi)
    return data_fit + complexity + constant


# --- ajustement des hyperparamètres ---


def _fit_base_matern(X, y):
    def objective(log_theta):
        sigma_f, lengthscale, sigma_n = np.exp(log_theta)
        K = _matern32_kernel(X, sigma_f, lengthscale)
        return _nlml(K, y, sigma_n)

    log_theta0 = np.array([0.0, 0.0, 0.0])
    log_bounds = [(-10.0, 10.0)] * 3
    result = minimize(objective, log_theta0, method="L-BFGS-B",
                      bounds=log_bounds, options={"maxiter": 200, "ftol": 1e-8})
    return result.fun, np.exp(result.x)


def _fit_changepoint(X, y, base_params):
    n = len(y)
    sigma_f_base, l_base, sigma_n_base = base_params

    theta0 = np.array([
        np.log(sigma_f_base), np.log(l_base),
        np.log(sigma_f_base), np.log(l_base),
        (n - 1) / 2.0,
        np.log(1.0),
        np.log(sigma_n_base),
    ])
    bounds = [
        (-10.0, 10.0), (-10.0, 10.0),
        (-10.0, 10.0), (-10.0, 10.0),
        (float(X[0] + 0.5), float(X[-1] - 0.5)),
        (-10.0, 10.0), (-10.0, 10.0),
    ]

    def objective(theta):
        sf1 = np.exp(theta[0]); l1  = np.exp(theta[1])
        sf2 = np.exp(theta[2]); l2  = np.exp(theta[3])
        c   = theta[4]
        s   = np.exp(theta[5]); sn  = np.exp(theta[6])
        K = _changepoint_kernel(X, sf1, l1, sf2, l2, c, s)
        return _nlml(K, y, sn)

    result = minimize(objective, theta0, method="L-BFGS-B",
                      bounds=bounds, options={"maxiter": 300, "ftol": 1e-8})
    t = result.x
    best_params = np.array([
        np.exp(t[0]), np.exp(t[1]),
        np.exp(t[2]), np.exp(t[3]),
        t[4], np.exp(t[5]), np.exp(t[6]),
    ])
    return result.fun, best_params


def _fit_changepoint_with_retry(X, y, base_params):
    nlml_cp, cp_params = _fit_changepoint(X, y, base_params)
    sf1, l1, sf2, l2 = cp_params[0], cp_params[1], cp_params[2], cp_params[3]
    # si k_s1 ≈ k_s2 l'optimisation a échoué — on relance avec des params neutres
    if np.isclose(sf1, sf2, rtol=1e-3) and np.isclose(l1, l2, rtol=1e-3):
        retry_params = np.array([1.0, 1.0, 1.0])
        nlml_cp, cp_params = _fit_changepoint(X, y, retry_params)
    return nlml_cp, cp_params


# --- API publique GP ---


def cpd_scores(returns, lbw):
    y = np.asarray(returns, dtype=np.float64).ravel()
    assert len(y) == lbw, f"len(returns)={len(y)} != lbw={lbw}"

    # Standardisation (éq. 2)
    mu  = np.mean(y)
    std = np.std(y, ddof=0)
    if std < 1e-12:
        return 0.0, 0.5
    y_std = (y - mu) / std

    X = np.arange(lbw, dtype=np.float64)
    nlml_base, base_params = _fit_base_matern(X, y_std)
    nlml_cp, cp_params     = _fit_changepoint_with_retry(X, y_std, base_params)

    # nu = sigmoid(nlml_M - nlml_cp) → 1 si le noyau CP est meilleur
    delta = nlml_base - nlml_cp
    nu    = 1.0 / (1.0 + np.exp(-delta))

    # gamma : position normalisée du point de rupture dans la fenêtre
    c_opt = cp_params[4]
    gamma = c_opt / (lbw - 1) if lbw > 1 else 0.5

    return float(np.clip(nu, 0.0, 1.0)), float(np.clip(gamma, 0.0, 1.0))


def fit_matern(returns):
    y = np.asarray(returns, dtype=np.float64).ravel()
    X = np.arange(len(y), dtype=np.float64)
    return _fit_base_matern(X, y)


def fit_changepoint_kernel(returns):
    y = np.asarray(returns, dtype=np.float64).ravel()
    X = np.arange(len(y), dtype=np.float64)
    _, base_params = _fit_base_matern(X, y)
    return _fit_changepoint_with_retry(X, y, base_params)


# --- Méthode 1 : Binary Segmentation (offline) ---


def binary_segmentation(returns, penalty_mult=0.25, model="rbf"):
    n = len(returns)
    sigma2 = returns.var()
    pen = penalty_mult * np.log(n) * sigma2
    algo = rpt.Binseg(model=model).fit(returns.reshape(-1, 1))
    breaks = algo.predict(pen=pen)
    return [b for b in breaks if b < n]


# --- Méthode 2 : CUSUM combiné moyenne + variance (online) ---


def cusum_combined(returns, ref_window=60, h_mean=4.0, h_var=4.0,
                   k_mean=0.5, k_var=0.5, cooldown=20):
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


# --- Méthode 3 : BOCPD (online) ---


def _log_gaussian_pdf(x, mu, sigma2):
    return -0.5 * (np.log(2 * np.pi * sigma2) + (x - mu) ** 2 / sigma2)


def bocpd(returns, hazard=1/500, prior_mu=0.0, kappa0=1.0,
          alpha0=1.0, beta0=1e-4, cooldown=20, drop_threshold=30, fresh_rl=5):
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
