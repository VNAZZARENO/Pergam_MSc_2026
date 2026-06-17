# Pipeline LightGBM V2 — DMN momentum STOXX 600
# Architecture : 22 features → LightGBM (L2/MSE) → sigmoid(alpha* × score) → EMA → positions

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap


# --- constantes ---

MOMENTUM_FEATURES = [
    "norm_ret_1d",      "norm_ret_21d",      "norm_ret_63d",
    "norm_ret_126d",    "norm_ret_252d",
    "norm_ret_1d_lag1", "norm_ret_21d_lag1", "norm_ret_63d_lag1",
    "norm_ret_126d_lag1", "norm_ret_252d_lag1",
    "macd_8_24",  "macd_16_48",  "macd_32_96",
    "macd_8_24_lag1", "macd_16_48_lag1", "macd_32_96_lag1",
    "ewma_vol",   "ewma_vol_lag1",
]

SECTOR_FEATURES = ["region_rel_1d", "region_rel_1d_lag1"]

CPD_FEATURES = [
    "nu_cusum_lag1",  # CUSUM stock-level, AUC=0.605, recall 78.7%
    "nu_bocpd_lag1",  # BOCPD stock-level, corr CUSUM=0.11 (signal complémentaire)
]

FEATURE_COLS      = MOMENTUM_FEATURES + SECTOR_FEATURES + CPD_FEATURES
TARGET_COL        = "next_return"
TC_BPS            = 25    # coûts Pergam — doit correspondre à dmn.transaction_cost dans default.yaml
RANDOM_SEED       = 42
TEST_START        = 2011
VAL_FRAC          = 0.10  # fraction validation — doit correspondre à lgbm.val_frac dans default.yaml
POSITION_HALFLIFE = 10    # lissage EMA — doit correspondre à lgbm.position_halflife dans default.yaml
WINDOW_YEARS      = None  # None = expanding (tout l'historique disponible)

LGB_PARAMS = {
    "learning_rate":      0.05,
    "max_depth":          4,
    "num_leaves":         15,
    "min_child_samples":  20,
    "subsample":          0.8,
    "subsample_freq":     1,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         0.1,
    "random_state":       RANDOM_SEED,
    "n_jobs":             -1,
    "verbose":            -1,
}

LGB_N_ESTIMATORS = 300


# --- chemins ---

def resolve_project_root(start=None):
    root = (Path.cwd() if start is None else Path(start)).resolve()
    for p in [root, root.parent]:
        if (p / "configs" / "default.yaml").exists():
            return p
    raise FileNotFoundError("configs/default.yaml not found")


def nb03_paths(root=None):
    r = resolve_project_root(root)
    d = r / "data" / "processed" / "stoxx600"
    return {
        "project_root":  r,
        "processed_dir": d,
        "panel":         d / "panel.parquet",
        "stocks_csv":    d / "stoxx600_processed.csv",   # sortie NB01 — fallback si panel.parquet absent
        "cpd_features":  d / "cpd_features_nb03.parquet",
        "positions":     d / "positions_v2.parquet",
        "fold_metrics":  d / "fold_metrics_v2.parquet",
    }


# --- construction du panel depuis le CSV NB01 (fallback) ---

def _build_panel_from_csv(csv_path):
    """Reconstruit le panel NB03-V2 depuis stoxx600_processed.csv (NB01).

    Utilisé uniquement si panel.parquet est absent (clone fraîche).
    Calcule : norm_ret_*, region_rel_1d, next_return, et tous les lag-1.
    """
    EPS = 1e-8
    horizons = [(1, "1d"), (21, "21d"), (63, "63d"), (126, "126d"), (252, "252d")]

    usecols = (
        ["date", "ticker", "price", "1d_arith_ret", "60d_ewm_vol", "region"]
        + [f"{label}_arith_ret" for h, label in horizons if h > 1]
        + ["macd_8_24", "macd_16_48", "macd_32_96"]
    )
    panel = pd.read_csv(csv_path, usecols=usecols, parse_dates=["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # norm_ret_Xd = Xd_ret / (daily_vol × √X)
    daily_vol = (panel["60d_ewm_vol"] / np.sqrt(252)).clip(lower=EPS)
    for h, label in horizons:
        src = "1d_arith_ret" if h == 1 else f"{label}_arith_ret"
        panel[f"norm_ret_{label}"] = panel[src] / (daily_vol * np.sqrt(h))

    panel["ewma_vol"] = panel["60d_ewm_vol"]

    # rendement relatif à la région (même logique que sector_rel dans NB01)
    region_mean = (panel.groupby(["date", "region"])["1d_arith_ret"]
                        .transform("mean")
                        .fillna(0.0))
    panel["region_rel_1d"] = panel["1d_arith_ret"] - region_mean

    # cible J+1 par ticker
    panel["next_return"] = panel.groupby("ticker")["1d_arith_ret"].shift(-1)

    # lag-1 de toutes les features momentum + CPD region
    lag_src = (
        [f"norm_ret_{label}" for _, label in horizons]
        + ["macd_8_24", "macd_16_48", "macd_32_96", "ewma_vol", "region_rel_1d"]
    )
    for col in lag_src:
        panel[f"{col}_lag1"] = panel.groupby("ticker")[col].shift(1)

    return panel


# --- chargement des données ---

def load_nb03_inputs(root=None):
    paths = nb03_paths(root)
    if paths["panel"].exists():
        panel = pd.read_parquet(paths["panel"])
    elif paths["stocks_csv"].exists():
        # fallback : reconstruit le panel depuis la sortie directe de NB01
        panel = _build_panel_from_csv(paths["stocks_csv"])
    else:
        raise FileNotFoundError(
            f"Missing: {paths['panel']} et fallback {paths['stocks_csv']} "
            "— relancer NB01 d'abord"
        )
    if not paths["cpd_features"].exists():
        raise FileNotFoundError(
            f"Missing input: {paths['cpd_features']} — relancer NB02 d'abord"
        )
    cpd = pd.read_parquet(paths["cpd_features"])
    for df in (panel, cpd):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
    return {"paths": paths, "panel": panel, "cpd_features": cpd}


def build_feature_matrix(panel, cpd_features):
    base_cols = (["date", "ticker", "price"]
                 + [c for c in FEATURE_COLS
                    if c in panel.columns and c not in CPD_FEATURES]
                 + [TARGET_COL])
    base = panel[[c for c in base_cols if c in panel.columns]].copy()

    cpd_cols = (["date", "ticker"]
                + [c for c in CPD_FEATURES if c in cpd_features.columns])
    feat = base.merge(cpd_features[cpd_cols], on=["date", "ticker"], how="left")

    for col in CPD_FEATURES:
        if col in feat.columns:
            feat[col] = feat[col].fillna(0.0)

    must_have = [c for c in MOMENTUM_FEATURES[:3] if c in feat.columns]
    feat = feat.dropna(subset=must_have + [TARGET_COL])
    return feat.sort_values(["ticker", "date"]).reset_index(drop=True)


def input_summary(feat):
    fcols = [c for c in FEATURE_COLS if c in feat.columns]
    return pd.DataFrame([
        {"item": "rows",        "value": f"{len(feat):,}"},
        {"item": "tickers",     "value": f"{feat['ticker'].nunique():,}"},
        {"item": "date range",  "value": f"{feat['date'].min().date()} -> {feat['date'].max().date()}"},
        {"item": "features",    "value": str(len(fcols))},
        {"item": "model",       "value": "LightGBM L2 + calibration alpha* Sharpe-net (V2)"},
        {"item": "target",      "value": TARGET_COL},
    ])


# --- calibration TC-aware ---
#
# Entraînement : L2/MSE (use_sharpe_loss=False dans run_walk_forward)
# Objectif Sharpe disponible ci-dessous mais non activé — L2 plus stable sur ce signal faible (IC ≈ 0.005)
#
# Calibration alpha* sur validation :
#   position = sigmoid(alpha × score)  →  alpha* = argmax Sharpe_net(25bps) sur validation

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _sharpe_objective_factory(dates_arr, eps=1e-8):
    # objectif custom LightGBM : maximise le Sharpe cross-sectionnel quotidien
    _, inverse, counts = np.unique(dates_arr, return_inverse=True, return_counts=True)
    T   = len(counts)
    N_d = counts[inverse].astype(np.float64)

    def _obj(y_pred, dataset):
        y_true = dataset.get_label()
        R      = np.bincount(inverse, weights=y_pred * y_true) / counts
        mu     = R.mean()
        sigma  = R.std() + eps
        coef   = np.sqrt(252) * (
            1.0 / (T * sigma)
            - mu * (R - mu) / (max(T - 1, 1) * sigma ** 3)
        )
        grad = -coef[inverse] * y_true / N_d
        hess = np.ones_like(y_pred)
        return grad, hess

    return _obj


def _raw_sharpe(positions, returns):
    p_net = positions * returns
    return float(p_net.mean() / (p_net.std() + 1e-8) * np.sqrt(252))


def _net_sharpe_calibration(pos_flat, y_flat, dates, tickers, tc=TC_BPS / 10_000):
    # Sharpe net après EMA + TC — utilisé pour calibrer alpha
    ema_a = float(1.0 - np.exp(-np.log(2.0) / POSITION_HALFLIFE))
    df = pd.DataFrame({"date": dates, "ticker": tickers,
                       "pos": pos_flat, "ret": y_flat})
    df = df.sort_values(["ticker", "date"])
    df["pos_s"] = (df.groupby("ticker")["pos"]
                     .transform(lambda s: s.ewm(alpha=ema_a, adjust=False).mean()))
    df["prev"] = df.groupby("ticker")["pos_s"].shift(1)
    df["to"]   = (df["pos_s"] - df["prev"]).abs().fillna(0.0)
    df["pnl"]  = df["pos_s"] * df["ret"] - tc * df["to"]
    return float(df["pnl"].mean() / (df["pnl"].std() + 1e-8) * np.sqrt(252))


def calibrate_alpha(pred_returns, y_val, alphas=None, alpha_max=200.0,
                    dates=None, tickers=None, tc=TC_BPS / 10_000):
    # alpha* = argmax Sharpe_net(sigmoid(alpha × score)) sur validation
    if alphas is None:
        alphas = np.logspace(0, np.log10(alpha_max), 30)
    use_net = (dates is not None and tickers is not None)
    best_sr, best_a = -np.inf, 1.0
    for a in alphas:
        pos = _sigmoid(a * pred_returns)
        sr  = (_net_sharpe_calibration(pos, y_val, dates, tickers, tc)
               if use_net else _raw_sharpe(pos, y_val))
        if sr > best_sr:
            best_sr = sr
            best_a  = float(a)
    return best_a


# --- entraînement ---

def _ic_eval(y_pred, dataset):
    # IC = corrélation prédictions/réels — plus stable que RMSE sur rendements journaliers
    y_true = dataset.get_label()
    corr   = float(np.corrcoef(y_pred, y_true)[0, 1])
    return "IC", (0.0 if np.isnan(corr) else corr), True


def train_fold_lgb(X_train, y_train, X_val, y_val, seed=RANDOM_SEED,
                   vl_dates=None, vl_tickers=None, tr_dates=None,
                   use_sharpe_loss=True):
    use_sharpe = use_sharpe_loss and (tr_dates is not None)

    if use_sharpe:
        params = {**LGB_PARAMS, "random_state": seed,
                  "objective": _sharpe_objective_factory(tr_dates)}
        feval  = _ic_eval
    else:
        params = {**LGB_PARAMS, "random_state": seed,
                  "objective": "regression", "metric": "l2"}
        feval  = None

    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set   = lgb.Dataset(X_val,   label=y_val,   reference=train_set,
                             free_raw_data=False)
    evals_result = {}
    callbacks = [lgb.log_evaluation(period=0), lgb.record_evaluation(evals_result)]

    model = lgb.train(
        params, train_set,
        num_boost_round=LGB_N_ESTIMATORS,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        feval=feval,
        callbacks=callbacks,
    )

    pred_val = model.predict(X_val)
    alpha    = calibrate_alpha(pred_val, y_val, dates=vl_dates, tickers=vl_tickers)
    return model, alpha, evals_result


# --- walk-forward ---

def walk_forward_splits(feat, test_start=TEST_START, window_years=WINDOW_YEARS):
    years      = sorted(feat["date"].dt.year.unique())
    test_years = [y for y in years if y >= test_start]
    splits     = []
    for ty in test_years:
        tr_end   = pd.Timestamp(f"{ty - 1}-12-31")
        te_start = pd.Timestamp(f"{ty}-01-01")
        te_end   = pd.Timestamp(f"{ty}-12-31")
        tr_start = (pd.Timestamp(f"{ty - 1 - window_years}-01-01")
                    if window_years is not None else None)
        if feat.loc[feat["date"] <= tr_end].shape[0] < 1_000:
            continue
        splits.append({
            "test_year":   ty,
            "train_start": tr_start,
            "train_end":   tr_end,
            "test_start":  te_start,
            "test_end":    te_end,
        })
    return splits


def run_walk_forward(feat, feature_cols=None, verbose=True, window_years=WINDOW_YEARS):
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in feat.columns]

    splits    = walk_forward_splits(feat, window_years=window_years)
    all_pos   = []
    fold_rows = []

    for fold_idx, sp in enumerate(splits):
        ty = sp["test_year"]
        t0 = time.perf_counter()

        mask = feat["date"] <= sp["train_end"]
        if sp.get("train_start") is not None:
            mask &= feat["date"] >= sp["train_start"]
        tr_feat = feat.loc[mask].sort_values(["date", "ticker"])

        X_all = tr_feat[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
        n_val = max(1_000, int(len(tr_feat) * VAL_FRAC))
        X_tr, X_vl = X_all[:-n_val], X_all[-n_val:]
        y_all       = tr_feat[TARGET_COL].to_numpy(dtype=np.float64)
        y_tr, y_vl  = y_all[:-n_val], y_all[-n_val:]

        if len(X_tr) < 500:
            if verbose:
                print(f"  fold {ty} — trop peu de données, ignoré")
            continue

        tr_slice = tr_feat.iloc[:-n_val]
        vl_slice = tr_feat.iloc[-n_val:]
        model, alpha, _ = train_fold_lgb(
            X_tr, y_tr, X_vl, y_vl,
            seed=RANDOM_SEED + fold_idx,
            vl_dates=vl_slice["date"].to_numpy(),
            vl_tickers=vl_slice["ticker"].to_numpy(),
            tr_dates=tr_slice["date"].to_numpy(),
            use_sharpe_loss=False,
        )

        pred_vl    = model.predict(X_vl)
        p_vl       = _sigmoid(alpha * pred_vl)
        val_sharpe = _raw_sharpe(p_vl, y_vl)
        val_ic     = float(np.corrcoef(pred_vl, y_vl)[0, 1])

        te_feat = feat.loc[
            (feat["date"] >= sp["test_start"]) &
            (feat["date"] <= sp["test_end"])
        ].sort_values(["date", "ticker"]).copy()

        if te_feat.empty:
            continue

        X_te    = te_feat[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
        pred_te = model.predict(X_te)
        raw_pos = _sigmoid(alpha * pred_te)

        pos_df             = te_feat[["date", "ticker"]].copy()
        pos_df["position"] = raw_pos
        all_pos.append(pos_df)

        elapsed = time.perf_counter() - t0
        fold_rows.append({
            "test_year":        ty,
            "val_sharpe":       round(val_sharpe, 3),
            "val_ic":           round(val_ic, 4),
            "alpha_calibrated": round(alpha, 1),
            "n_estimators":     model.num_trees(),
            "n_train_rows":     len(X_tr),
            "seconds":          round(elapsed, 1),
        })

        if verbose:
            print(f"  fold {ty} | val_sharpe={val_sharpe:.3f} | val_ic={val_ic:.4f} | "
                  f"alpha={alpha:.0f} | trees={model.num_trees()} | {elapsed:.0f}s")

    positions_df = (pd.concat(all_pos, ignore_index=True) if all_pos
                    else pd.DataFrame(columns=["date", "ticker", "position"]))
    fold_metrics = pd.DataFrame(fold_rows)
    return positions_df, fold_metrics


# --- SHAP ---

def compute_shap(model, feat, feature_cols, n_sample=5_000):
    rng        = np.random.default_rng(RANDOM_SEED)
    idx        = rng.choice(len(feat), size=min(n_sample, len(feat)), replace=False)
    X_samp     = feat.iloc[idx][feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_samp)
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    return (pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


# --- filtre CPD ---
#
# Quand le score CPD (nu) est élevé → rupture de tendance détectée.
# On ramène la position vers 0.5 (neutre) proportionnellement à nu :
#   pos_filtrée = 0.5 + (1 − strength × nu) × (pos − 0.5)
# ~4% des observations sont affectées.

CPD_FILTER_NU_COLS  = ["nu_cusum_lag1"]
CPD_FILTER_STRENGTH = 1.0


def apply_cpd_filter(positions, feat, strength=CPD_FILTER_STRENGTH):
    nu_cols = [c for c in CPD_FILTER_NU_COLS if c in feat.columns]
    if not nu_cols:
        return positions
    cpd = feat[["date", "ticker"] + nu_cols].copy()
    cpd["nu_composite"] = cpd[nu_cols].mean(axis=1).clip(0.0, 1.0)
    merged = positions.merge(cpd[["date", "ticker", "nu_composite"]],
                             on=["date", "ticker"], how="left")
    merged["nu_composite"] = merged["nu_composite"].fillna(0.0)
    confidence        = (1.0 - strength * merged["nu_composite"]).clip(0.0, 1.0)
    merged["position"] = 0.5 + confidence * (merged["position"] - 0.5)
    return merged[["date", "ticker", "position"]]


# --- lissage EMA des positions ---
#
# LightGBM prédit chaque jour indépendamment → turnover ~45%/jour → ~2300 bps/an.
# EMA halflife=10j : on garde 93% de la position d'hier + 7% de la nouvelle.
# Effet : turnover ÷ 10 → ~230 bps/an.

def smooth_positions(positions, halflife=POSITION_HALFLIFE):
    alpha = float(1 - np.exp(-np.log(2) / halflife))
    out   = positions.sort_values(["ticker", "date"]).copy()
    out["position"] = (
        out.groupby("ticker")["position"]
           .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    )
    return out


# --- sauvegarde ---

def save_nb03_outputs(root, positions, fold_metrics):
    paths = nb03_paths(root)
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    positions.to_parquet(paths["positions"],      index=False)
    fold_metrics.to_parquet(paths["fold_metrics"], index=False)
    return pd.DataFrame([
        {"output": "positions",    "rows": f"{len(positions):,}",
         "path": paths["positions"].relative_to(paths["project_root"]).as_posix()},
        {"output": "fold_metrics", "rows": f"{len(fold_metrics):,}",
         "path": paths["fold_metrics"].relative_to(paths["project_root"]).as_posix()},
    ])
