"""Changepoint detection tools.

The first block reproduces the idea from the paper:
compare a standard Gaussian Process with a changepoint Gaussian Process.

The second block contains faster student-friendly detectors used for our
experiments and presentation: CUSUM, jump score, t-test score, BOCPD,
moving-average crossing and an ensemble score.
"""

from __future__ import annotations

import sys
import math
import numpy as np
try:
    from scipy import stats
    from scipy.optimize import minimize
except ModuleNotFoundError:
    stats = None
    minimize = None

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

# Organisation du fichier :
# 1. outils GP proches du papier, precis mais lents ;
# 2. detecteurs rapides, plus pratiques pour les notebooks ;
# 3. fonctions de synthese pour produire des scores/features CPD.


# [UTIL] _as_array
# Convertit une serie/list en tableau numpy float propre.
def _as_array(values):
    """Return a clean float numpy array."""
    return np.asarray(values, dtype=float)


# [UTIL] sigmoid
# Transforme un signal numerique en score borne entre 0 et 1.
def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _normal_sf(x):
    """Normal survival function fallback used when scipy is unavailable."""
    arr = np.asarray(x, dtype=float)
    erfc_vec = np.vectorize(math.erfc)
    return 0.5 * erfc_vec(arr / np.sqrt(2.0))


def _student_t_pdf(x, df, loc=0.0, scale=1.0):
    """Student-t PDF with a scipy fallback replacement."""
    if stats is not None:
        return stats.t.pdf(x, df=df, loc=loc, scale=scale)

    x = np.asarray(x, dtype=float)
    df = np.asarray(df, dtype=float)
    loc = np.asarray(loc, dtype=float)
    scale = np.asarray(scale, dtype=float)
    z = (x - loc) / scale
    log_gamma = np.vectorize(math.lgamma)
    log_coef = (
        log_gamma((df + 1.0) / 2.0)
        - log_gamma(df / 2.0)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
    )
    log_kernel = -((df + 1.0) / 2.0) * np.log1p((z ** 2) / df)
    return np.exp(log_coef + log_kernel)


# ---------------------------------------------------------------------------
# Bloc GP : methode utile comme reference theorique mais lente, a utiliser sur de petits echantillons
# ---------------------------------------------------------------------------


# [GP] _matern32
# Construit le noyau de covariance Matern 3/2 sans rupture.
def _matern32(x1, x2, length_scale, variance):
    """Matern 3/2 covariance matrix."""
    r = np.abs(x1[:, None] - x2[None, :]) / length_scale
    s3r = np.sqrt(3.0) * r
    return variance * (1.0 + s3r) * np.exp(-s3r)


# [GP] _sigmoid_blend
# Cree une transition douce autour d'une position de rupture
def _sigmoid_blend(t, changepoint, steepness=5.0):
    """Smooth transition centered on the changepoint."""
    return sigmoid(steepness * (t - changepoint))


# [GP] _cp_kernel
# Melange deux noyaux Matern, un avant et un apres la rupture
def _cp_kernel(x1, x2, changepoint, ls1, var1, ls2, var2, steepness=5.0):
    """Changepoint kernel: smooth blend of two Matern kernels."""
    s1 = _sigmoid_blend(x1, changepoint, steepness)[:, None]
    s2 = _sigmoid_blend(x2, changepoint, steepness)[None, :]
    k1 = _matern32(x1, x2, ls1, var1)
    k2 = _matern32(x1, x2, ls2, var2)
    return (1.0 - s1) * k1 * (1.0 - s2) + s1 * k2 * s2


# [GP] _negative_log_marginal_likelihood
# Mesure la qualite d'ajustement d'un GP : plus bas = meilleur fit
def _negative_log_marginal_likelihood(kernel, y, noise=1e-4):
    """Negative log marginal likelihood of a zero-mean GP."""
    n = len(y)
    ky = kernel + noise * np.eye(n)
    try:
        chol = np.linalg.cholesky(ky)
    except np.linalg.LinAlgError:
        return 1e10

    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    fit = 0.5 * y @ alpha
    complexity = np.sum(np.log(np.diag(chol)))
    normalizer = 0.5 * n * np.log(2 * np.pi)
    return fit + complexity + normalizer


# [GP] fit_matern
# Ajuste le GP de base, sans rupture, sur une fenetre de rendements et retourne le meilleur NLML
def fit_matern(returns):
    """Fit a GP with one Matern 3/2 kernel and return the best NLML."""
    if minimize is None:
        raise ImportError("scipy is required for GP CPD. Use --method fast or install scipy.")
    y = _as_array(returns)
    x = np.arange(len(y), dtype=float)

    def objective(params):
        length_scale = np.exp(params[0])
        variance = np.exp(params[1])
        kernel = _matern32(x, x, length_scale, variance)
        return _negative_log_marginal_likelihood(kernel, y)

    start = [np.log(max(len(y) / 4, 1.0)), 0.0]
    result = minimize(objective, start, method="L-BFGS-B")
    return float(result.fun)


# [GP] fit_changepoint_kernel
# Ajuste le GP avec rupture imposee a une position donnee et retourne le meilleur NLML
def fit_changepoint_kernel(returns, changepoint):
    """Fit a GP with a changepoint kernel and return the best NLML."""
    if minimize is None:
        raise ImportError("scipy is required for GP CPD. Use --method fast or install scipy.")
    y = _as_array(returns)
    x = np.arange(len(y), dtype=float)

    def objective(params):
        kernel = _cp_kernel(
            x,
            x,
            changepoint,
            np.exp(params[0]),
            np.exp(params[1]),
            np.exp(params[2]),
            np.exp(params[3]),
        )
        return _negative_log_marginal_likelihood(kernel, y)

    base_length = np.log(max(len(y) / 4, 1.0))
    start = [base_length, 0.0, base_length, 0.0]
    result = minimize(objective, start, method="L-BFGS-B")
    return float(result.fun)


# [GP] cpd_scores
# Renvoie severity et location : intensite et position de la rupture
def cpd_scores(returns, lbw):
    """Return the paper-style CPD scores ``(severity, location)``.

    ``severity`` is high when the changepoint kernel improves the GP fit.
    ``location`` is the best changepoint position inside the lookback window.
    """
    y = _as_array(returns)
    y = y[~np.isnan(y)]
    if len(y) < lbw:
        return 0.0, 0.5

    y = y[-lbw:]
    std = y.std(ddof=1)
    if std < 1e-12:
        return 0.0, 0.5

    y = (y - y.mean()) / std
    base_nlml = fit_matern(y)

    best_nlml = np.inf
    best_c = lbw // 2
    for changepoint in range(3, lbw - 3):
        nlml = fit_changepoint_kernel(y, float(changepoint))
        if nlml < best_nlml:
            best_nlml = nlml
            best_c = changepoint

    severity = sigmoid(base_nlml - best_nlml)
    location = best_c / (lbw - 1)
    return float(severity), float(location)


# ---------------------------------------------------------------------------
# Bloc detecteurs rapides : methodes legeres pour tests et features.
# ---------------------------------------------------------------------------


# [SCORE] threshold_score
# Convertit un score continu en dates de rupture, en appliquant un seuil et un cooldown
def threshold_score(scores, threshold=0.5, cooldown=30):
    """Convert a continuous score into changepoint dates/indices."""
    scores = _as_array(scores)
    detections = []
    last_detection = -cooldown - 1
    for t, score in enumerate(scores):
        if not np.isnan(score) and score > threshold and (t - last_detection) > cooldown:
            detections.append(t)
            last_detection = t
    return detections


# [EVAL] evaluate_detections
# Compare nos detections a une liste d'evenements connus du marche
def evaluate_detections(detected, true_cps, tol=20):
    """Compare detected changepoints with known event indices.

    A detection is counted as true positive if it falls within ``tol`` trading
    days of a known event. The reported rate is a false discovery rate:
    ``FP / (TP + FP)``. The historical ``fpr`` key is kept as a backward-
    compatible alias for notebooks/scripts that already consume it.
    """
    detected = sorted(set(detected))
    true_cps = sorted(set(true_cps))
    if not detected:
        return {
            "fdr": 0.0,
            "fpr": 0.0,
            "recall": 0.0,
            "n_det": 0,
            "n_tp": 0,
            "n_fp": 0,
            "delays": [],
            "mean_delay": np.nan,
        }

    matched_events = {}
    true_positive_detections = set()
    for detection in detected:
        for event in true_cps:
            if abs(detection - event) <= tol:
                true_positive_detections.add(detection)
                if event not in matched_events or abs(detection - event) < abs(matched_events[event] - event):
                    matched_events[event] = detection
                break

    delays = [detection - event for event, detection in matched_events.items()]
    n_tp = len(true_positive_detections)
    n_fp = len(detected) - n_tp
    fdr = n_fp / len(detected)
    return {
        "fdr": fdr,
        "fpr": fdr,
        "recall": len(matched_events) / len(true_cps) if true_cps else 1.0,
        "n_det": len(detected),
        "n_tp": n_tp,
        "n_fp": n_fp,
        "delays": delays,
        "mean_delay": float(np.mean(delays)) if delays else np.nan,
    }


# [DETECTEUR] detect_cusum
# Detecte les ecarts persistants a la moyenne recente des rendements
def detect_cusum(returns, ref_window=60, threshold=4.0, cooldown=20):
    """Two-sided CUSUM detector for mean shifts."""
    returns = _as_array(returns)
    detections = []
    last_detection = -cooldown - 1
    s_pos = 0.0
    s_neg = 0.0
    k = 0.5

    for t in range(ref_window, len(returns)):
        ref = returns[t - ref_window:t]
        ref = ref[~np.isnan(ref)]
        if len(ref) < 5 or np.isnan(returns[t]):
            continue
        sigma = ref.std(ddof=1)
        if sigma < 1e-12:
            continue

        z = (returns[t] - ref.mean()) / sigma
        s_pos = max(0.0, s_pos + z - k)
        s_neg = max(0.0, s_neg - z - k)

        if max(s_pos, s_neg) > threshold and (t - last_detection) > cooldown:
            detections.append(t)
            last_detection = t
            s_pos = 0.0
            s_neg = 0.0
    return detections


# [SCORE] cusum_continuous
# Produit l'intensite CUSUM en continu, au lieu d'un simple oui/non
def cusum_continuous(returns, ref_window=60):
    """Continuous CUSUM score in [0, 1]."""
    returns = _as_array(returns)
    scores = np.full(len(returns), np.nan)
    s_pos = 0.0
    s_neg = 0.0
    k = 0.5

    for t in range(ref_window, len(returns)):
        ref = returns[t - ref_window:t]
        ref = ref[~np.isnan(ref)]
        if len(ref) < 5 or np.isnan(returns[t]):
            continue
        sigma = ref.std(ddof=1)
        if sigma < 1e-12:
            scores[t] = 0.5
            continue

        z = (returns[t] - ref.mean()) / sigma
        s_pos = max(0.0, s_pos + z - k)
        s_neg = max(0.0, s_neg - z - k)
        scores[t] = sigmoid(max(s_pos, s_neg) - 4.0)
    return scores


# [SCORE] jump_continuous
# Repere les journees anormales par rapport a la volatilite recente.
def jump_continuous(returns, window=40, threshold=2.5):
    """Score isolated jumps with a rolling z-score."""
    returns = _as_array(returns)
    scores = np.full(len(returns), np.nan)

    for t in range(window, len(returns)):
        ref = returns[t - window:t]
        ref = ref[~np.isnan(ref)]
        if len(ref) < 5 or np.isnan(returns[t]):
            continue
        sigma = ref.std(ddof=1)
        if sigma < 1e-12:
            scores[t] = 0.5
            continue
        z = abs(returns[t] - ref.mean()) / sigma
        scores[t] = sigmoid(z - threshold)
    return scores


# [DETECTEUR] detect_jump
# Transforme le score de saut en dates de detection espacees
def detect_jump(returns, window=60, threshold=3.0, cooldown=20):
    """Binary jump detector built from the continuous jump score."""
    scores = jump_continuous(returns, window=window, threshold=threshold)
    return threshold_score(scores, threshold=0.5, cooldown=cooldown)


# [SCORE] ttest_continuous
# Compare deux fenetres voisines pour scorer un changement de moyenne
def ttest_continuous(returns, window=20):
    """Rolling Welch t-test score for local mean changes."""
    returns = _as_array(returns)
    series = pd.Series(returns)

    post_mean = series.rolling(window, min_periods=window).mean()
    post_var = series.rolling(window, min_periods=window).var()
    pre_mean = post_mean.shift(window)
    pre_var = post_var.shift(window)

    denominator = np.sqrt(pre_var / window + post_var / window)
    t_stat = (post_mean - pre_mean) / denominator.replace(0, np.nan)
    return sigmoid(t_stat.abs().to_numpy() - 2.0)


# [DETECTEUR] detect_ttest
# Garde les dates ou le changement de moyenne est statistiquement fort
def detect_ttest(returns, window=30, alpha=0.001, cooldown=20):
    """Binary detector based on a rolling Welch t-test."""
    returns = _as_array(returns)
    series = pd.Series(returns)
    counts = series.rolling(window, min_periods=5).count()
    means = series.rolling(window, min_periods=5).mean()
    variances = series.rolling(window, min_periods=5).var()

    post_count = counts
    post_mean = means
    post_var = variances
    pre_count = counts.shift(window)
    pre_mean = means.shift(window)
    pre_var = variances.shift(window)

    se2 = pre_var / pre_count + post_var / post_count
    denom = np.sqrt(se2.replace(0, np.nan))
    t_stat = (post_mean - pre_mean) / denom
    df_num = se2 ** 2
    df_den = (
        (pre_var / pre_count) ** 2 / (pre_count - 1)
        + (post_var / post_count) ** 2 / (post_count - 1)
    )
    df = df_num / df_den.replace(0, np.nan)
    if stats is not None:
        p_values = 2.0 * stats.t.sf(np.abs(t_stat), df)
    else:
        p_values = 2.0 * _normal_sf(np.abs(t_stat))

    detections = []
    last_detection = -cooldown - 1

    for t, p_value in enumerate(p_values):
        if p_value < alpha and (t - last_detection) > cooldown:
            detections.append(t)
            last_detection = t
    return detections


# [SCORE] bocpd
# Estime en ligne la probabilite qu'une rupture soit recente
def bocpd(
    returns,
    hazard_lambda=250,
    mu0=0.0,
    kappa0=1.0,
    alpha0=1.0,
    beta0=0.01,
    recent_window=5,
):
    """Simple Bayesian Online Changepoint Detection score.

    Returns the posterior probability of a recent changepoint.
    """
    returns = _as_array(returns)
    scores = np.zeros(len(returns))
    mu = np.array([mu0])
    kappa = np.array([kappa0])
    alpha = np.array([alpha0])
    beta = np.array([beta0])
    run_length = np.array([1.0])
    hazard = 1.0 / hazard_lambda

    for t, x in enumerate(returns):
        if np.isnan(x):
            scores[t] = 0.0
            continue

        df = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (kappa * alpha))
        pred = _student_t_pdf(x, df=df, loc=mu, scale=scale)

        growth = run_length * pred * (1.0 - hazard)
        changepoint = np.sum(run_length * pred * hazard)
        new_run_length = np.append(changepoint, growth)
        total = new_run_length.sum()
        if total > 0:
            new_run_length /= total

        n_recent = min(recent_window, len(new_run_length))
        scores[t] = new_run_length[:n_recent].sum()
        run_length = new_run_length

        new_kappa = kappa + 1.0
        new_mu = (kappa * mu + x) / new_kappa
        new_alpha = alpha + 0.5
        new_beta = beta + kappa * (x - mu) ** 2 / (2.0 * new_kappa)

        mu = np.append(mu0, new_mu)
        kappa = np.append(kappa0, new_kappa)
        alpha = np.append(alpha0, new_alpha)
        beta = np.append(beta0, new_beta)
    return scores


# [DETECTEUR] detect_ma_cross
# Cherche des changements de tendance via croisements de moyennes mobiles
def detect_ma_cross(returns, fast=10, slow=50, cooldown=20):
    """Moving-average crossing detector on cumulative returns."""
    returns = _as_array(returns)
    clean_returns = np.nan_to_num(returns, nan=0.0)
    cumulative = pd.Series(clean_returns).cumsum()
    ma_fast = cumulative.rolling(fast, min_periods=fast).mean()
    ma_slow = cumulative.rolling(slow, min_periods=slow).mean()
    diff = ma_fast - ma_slow
    vol = diff.rolling(60, min_periods=20).std().replace(0, np.nan)
    score = sigmoid((diff.abs() / vol).to_numpy() - 1.5)

    detections = []
    last_detection = -cooldown - 1
    for t in range(slow + 1, len(returns)):
        if pd.isna(diff.iloc[t]) or pd.isna(diff.iloc[t - 1]):
            continue
        crossed = diff.iloc[t] * diff.iloc[t - 1] < 0
        if crossed and (t - last_detection) > cooldown:
            detections.append(t)
            last_detection = t
    return detections, score


# [DETECTEUR] detect_cusum_adaptive
# Variante CUSUM avec seuil ajuste a la volatilite locale.
def detect_cusum_adaptive(returns, ref_window=60, base_threshold=4.0, vol_window=252, cooldown=20):
    """CUSUM detector with a volatility-adjusted threshold."""
    returns = _as_array(returns)
    vol = pd.Series(returns).rolling(vol_window, min_periods=60).std().to_numpy()
    positive_vol = vol[~np.isnan(vol) & (vol > 0)]
    median_vol = np.median(positive_vol) if len(positive_vol) else 1.0

    detections = []
    last_detection = -cooldown - 1
    s_pos = 0.0
    s_neg = 0.0
    k = 0.5

    for t in range(ref_window, len(returns)):
        ref = returns[t - ref_window:t]
        ref = ref[~np.isnan(ref)]
        if len(ref) < 5 or np.isnan(returns[t]):
            continue
        sigma = ref.std(ddof=1)
        if sigma < 1e-12:
            continue

        z = (returns[t] - ref.mean()) / sigma
        s_pos = max(0.0, s_pos + z - k)
        s_neg = max(0.0, s_neg - z - k)

        local_vol = vol[t] if not np.isnan(vol[t]) and vol[t] > 0 else median_vol
        threshold = base_threshold * local_vol / median_vol
        if max(s_pos, s_neg) > threshold and (t - last_detection) > cooldown:
            detections.append(t)
            last_detection = t
            s_pos = 0.0
            s_neg = 0.0
    return detections


# [ENSEMBLE] ensemble_cpd
# Moyenne plusieurs scores CPD pour stabiliser la detection finale.
def ensemble_cpd(scores_list, threshold=0.5, cooldown=30):
    """Average several scores and return detections plus the average score."""
    stacked = np.column_stack(scores_list)
    valid_counts = np.sum(~np.isnan(stacked), axis=1)
    summed = np.nansum(stacked, axis=1)
    average = np.divide(
        summed,
        valid_counts,
        out=np.full(len(stacked), np.nan),
        where=valid_counts > 0,
    )
    detections = threshold_score(average, threshold=threshold, cooldown=cooldown)
    return detections, average


# [FEATURES] compute_change_features
# Resume ce qui change autour d'une rupture : moyenne, volatilite, persistance.
def compute_change_features(returns, detections, scores=None, half_window=20):
    """Describe what changed around each detected changepoint."""
    returns = _as_array(returns)
    rows = []
    for detection in detections:
        pre = returns[max(0, detection - half_window):detection]
        post = returns[detection:min(len(returns), detection + half_window)]
        pre = pre[~np.isnan(pre)]
        post = post[~np.isnan(post)]
        if len(pre) < 5 or len(post) < 5:
            continue

        persistence = 0
        if scores is not None:
            t = detection
            while t < len(scores) and not np.isnan(scores[t]) and scores[t] > 0.5:
                persistence += 1
                t += 1

        rows.append({
            "detection": detection,
            "delta_mean": post.mean() - pre.mean(),
            "delta_vol": post.std() / pre.std() if pre.std() > 1e-12 else np.nan,
            "persistence": persistence,
        })
    return pd.DataFrame(rows)


# [PANEL] compute_cpd_panel
# Construit une table CPD complete pour une seule serie de rendements.
def compute_cpd_panel(returns, dates=None, threshold=0.5, cooldown=30):
    """Compute a clean CPD table for one return series.

    The output has one row per date and can be merged with feature data.
    """
    returns = _as_array(returns)
    if dates is None:
        dates = np.arange(len(returns))

    cusum_score = cusum_continuous(returns)
    jump_score = jump_continuous(returns)
    ttest_score = ttest_continuous(returns)
    bocpd_score = bocpd(returns)
    ma_detections, ma_score = detect_ma_cross(returns)
    detections, ensemble_score = ensemble_cpd(
        [cusum_score, jump_score, ttest_score, bocpd_score, ma_score],
        threshold=threshold,
        cooldown=cooldown,
    )

    output = pd.DataFrame({
        "date": dates,
        "return": returns,
        "cusum_score": cusum_score,
        "jump_score": jump_score,
        "ttest_score": ttest_score,
        "bocpd_score": bocpd_score,
        "ma_score": ma_score,
        "ensemble_score": ensemble_score,
        "is_changepoint": False,
    })
    output.loc[detections, "is_changepoint"] = True
    output["ma_cross"] = False
    output.loc[ma_detections, "ma_cross"] = True
    return output


# [GP] detect_gp_cpd
# Applique la methode GP sur toute la serie ; a utiliser sur petits echantillons.
def detect_gp_cpd(returns, lbw=21, nu_threshold=0.85, gamma_min=0.5, cooldown=20, stride=1):
    """Run the paper-style GP CPD through a full time series."""
    returns = _as_array(returns)
    severity = np.full(len(returns), np.nan)
    location = np.full(len(returns), np.nan)
    detections = []
    last_detection = -cooldown - 1

    for t in range(lbw, len(returns), stride):
        nu, gamma = cpd_scores(returns[t - lbw:t], lbw)
        severity[t] = nu
        location[t] = gamma
        if nu > nu_threshold and gamma > gamma_min and (t - last_detection) > cooldown:
            detections.append(t)
            last_detection = t
    return detections, severity, location
