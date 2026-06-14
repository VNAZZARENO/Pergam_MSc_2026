"""NB03 V2: Deep Momentum Network — LightGBM + alpha calibration + EMA smoothing.

V1 (NB03)  : LSTM(20) -> tanh/sigmoid -> position in (-1, 1) or (0, 1)
V2 (NB03v2): LightGBM -> sigmoid(alpha* x score) -> EMA -> position in (0, 1)

Why LightGBM instead of LSTM?
  - Native interpretability via SHAP: shows exactly which CPD and momentum
    signals the model uses.
  - No sequences: training ~9 min for 8 folds vs several hours for the LSTM.
  - Robust regularisation (max_depth, L1+L2) reduces overfitting on 600 stocks.

Transaction cost problem and solutions:
  - LightGBM predicts daily -> unstable positions -> ~45% daily turnover
    -> ~2308 bps/year gross TC.
  - Fix 1 — EMA(halflife=10d): exponential smoothing per stock, TC -> ~244 bps/yr.
  - Fix 2 — alpha calibration: alpha* = argmax Sharpe_net(sigmoid(alpha * score))
    on validation. TC managed by EMA.

Architecture:
  22 features -> LightGBM (MSE on next_return 1d)
              -> raw score -> sigmoid(alpha* x score) -> position in (0, 1)
              -> EMA smoothing (halflife=10d) -> positions_v2.parquet

Features (22 total):
  - 18 momentum : norm_ret + macd + ewma_vol, current + lag-1  (NB01)
  - 2  region-rel: region_rel_1d current + lag-1               (NB01)
  - 1  CPD stock CUSUM lag-1  : nu_cusum_lag1 (AUC=0.605)      (NB02)
  - 1  CPD stock BOCPD lag-1  : nu_bocpd_lag1 (corr CUSUM=0.11)(NB02)
    Note: only online (causal) CPD methods are used to avoid look-ahead bias.
    Sector-level CPD was tested but removed: sector_coverage.csv covers only
    44% of panel tickers (581/1312) — the feature was confounded with missing data.

Walk-forward: expanding-window protocol, 8 annual folds (2019 -> 2026).
Output format identical to V1 -> NB04 runs without modification.
"""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Features momentum courants (t) et decales (t-1) — encodent la dynamique
# multi-horizons sans avoir besoin d'une sequence explicite comme le LSTM.
MOMENTUM_FEATURES = [
    "norm_ret_1d",      "norm_ret_21d",      "norm_ret_63d",
    "norm_ret_126d",    "norm_ret_252d",
    "norm_ret_1d_lag1", "norm_ret_21d_lag1", "norm_ret_63d_lag1",
    "norm_ret_126d_lag1", "norm_ret_252d_lag1",
    "macd_8_24",  "macd_16_48",  "macd_32_96",
    "macd_8_24_lag1", "macd_16_48_lag1", "macd_32_96_lag1",
    "ewma_vol",   "ewma_vol_lag1",
]

# Signal regional : identifie si l'action sur/sous-performe sa region europeenne
SECTOR_FEATURES = ["region_rel_1d", "region_rel_1d_lag1"]

# CPD score lage d'1 jour (calcule par NB02, exporte dans cpd_features_nb03.parquet).
# CUSUM (mean + variance) retenu : meilleur AUC (0.605) et recall (78.7%)
# parmi toutes les methodes online sur le panel STOXX 600 (NB02).
CPD_FEATURES = [
    "nu_cusum_lag1",  # CUSUM stock-level, best AUC (0.605), recall 78.7%
    "nu_bocpd_lag1",  # BOCPD stock-level, P(run_length<=5), corr CUSUM=0.11
]

FEATURE_COLS = MOMENTUM_FEATURES + SECTOR_FEATURES + CPD_FEATURES  # 22 features
TARGET_COL        = "next_return"
TC_BPS            = 25    # couts de transaction Pergam (aller-retour)
RANDOM_SEED       = 42
TEST_START        = 2019
VAL_FRAC          = 0.10  # fraction chronologique pour la validation interne
POSITION_HALFLIFE = 10    # lissage EMA (jours ouvres)

# Fenetre d'entrainement walk-forward
# None = expanding (meme protocole que V1/papier)
WINDOW_YEARS: int | None = None

LGB_PARAMS: dict = {
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


# ---------------------------------------------------------------------------
# Paths (identiques a V1 — meme interface)
# ---------------------------------------------------------------------------

def resolve_project_root(start=None) -> Path:
    """Remonte l'arborescence jusqu'au dossier contenant configs/default.yaml."""
    root = (Path.cwd() if start is None else Path(start)).resolve()
    for p in [root, root.parent]:
        if (p / "configs" / "default.yaml").exists():
            return p
    raise FileNotFoundError("configs/default.yaml not found")


def nb03_paths(root=None) -> dict[str, Path]:
    """Retourne les chemins d'entree/sortie du pipeline V2."""
    r = resolve_project_root(root)
    d = r / "data" / "processed" / "stoxx600"
    return {
        "project_root":  r,
        "processed_dir": d,
        "panel":         d / "panel.parquet",
        "cpd_features":  d / "cpd_features_nb03.parquet",
        "positions":     d / "positions_v2.parquet",
        "fold_metrics":  d / "fold_metrics_v2.parquet",
    }


# ---------------------------------------------------------------------------
# Data loading & feature matrix
# ---------------------------------------------------------------------------

def load_nb03_inputs(root=None) -> dict:
    """Charge le panel NB01 et les scores CPD NB02 (CUSUM + BOCPD)."""
    paths = nb03_paths(root)
    for key in ("panel", "cpd_features"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing input: {paths[key]}")

    panel = pd.read_parquet(paths["panel"])
    cpd   = pd.read_parquet(paths["cpd_features"])
    for df in (panel, cpd):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    return {"paths": paths, "panel": panel, "cpd_features": cpd}


def build_feature_matrix(panel: pd.DataFrame,
                         cpd_features: pd.DataFrame) -> pd.DataFrame:
    """Merge panel + CPD features en une matrice ligne par ligne.

    V2 vs V1 : feature set enrichi (21 vs 10) — lag-1 momentum + region-rel + CPD.
    Pas de construction de sequences : LightGBM travaille ligne par ligne.
    """
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


def input_summary(feat: pd.DataFrame) -> pd.DataFrame:
    """Tableau recapitulatif du dataset charge."""
    fcols = [c for c in FEATURE_COLS if c in feat.columns]
    return pd.DataFrame([
        {"item": "rows",        "value": f"{len(feat):,}"},
        {"item": "tickers",     "value": f"{feat['ticker'].nunique():,}"},
        {"item": "date range",  "value": f"{feat['date'].min().date()} -> {feat['date'].max().date()}"},
        {"item": "features",    "value": str(len(fcols))},
        {"item": "model",       "value": "LightGBM MSE + calibration alpha (V2)"},
        {"item": "target",      "value": TARGET_COL},
    ])


# ---------------------------------------------------------------------------
# Calibration TC-aware
# ---------------------------------------------------------------------------
#
# Architecture de la loss en deux etapes :
#
#   Etape 1 — Prediction MSE (LightGBM standard)
#     On predit directement next_return avec une loss quadratique.
#     Gradients bien conditionnes (O(1)) : LightGBM apprend normalement.
#     Pourquoi pas Sharpe directement ? Le Sharpe exact sur n=1.5M donne
#     des gradients O(1/n) ~ 0 -> positions toutes bloquees a 0.5.
#
#   Etape 2 — Calibration TC-aware (sur validation)
#     On cherche le facteur alpha* tel que :
#       position_i = sigmoid(alpha * predicted_return_i)
#     maximise le Sharpe net de 25 bps sur l'ensemble de validation.
#     alpha* est appris sur les donnees de validation (in-sample a cette etape)
#     et applique sur le test.

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _raw_sharpe(positions: np.ndarray, returns: np.ndarray) -> float:
    """Sharpe annualise brut — utilise pour afficher val_sharpe dans fold_metrics."""
    p_net = positions * returns
    mu    = p_net.mean()
    sigma = p_net.std() + 1e-8
    return float(mu / sigma * np.sqrt(252))


def _net_sharpe_calibration(pos_flat: np.ndarray, y_flat: np.ndarray,
                             dates: np.ndarray, tickers: np.ndarray,
                             tc: float = TC_BPS / 10_000) -> float:
    """Sharpe net pour calibration alpha — applique EMA halflife=POSITION_HALFLIFE + TC.

    Reproduit le lissage de smooth_positions() afin que la calibration reflete
    les conditions post-lissage reelles (turnover ~x10 inferieur aux positions brutes).
    Appele ~30x par fold dans calibrate_alpha (une fois par alpha candidat).
    """
    ema_a = float(1.0 - np.exp(-np.log(2.0) / POSITION_HALFLIFE))
    df = pd.DataFrame({"date": dates, "ticker": tickers,
                       "pos":  pos_flat, "ret":   y_flat})
    df = df.sort_values(["ticker", "date"])
    df["pos_s"] = (df.groupby("ticker")["pos"]
                     .transform(lambda s: s.ewm(alpha=ema_a, adjust=False).mean()))
    df["prev"]  = df.groupby("ticker")["pos_s"].shift(1)
    df["to"]    = (df["pos_s"] - df["prev"]).abs().fillna(0.0)
    df["pnl"]   = df["pos_s"] * df["ret"] - tc * df["to"]
    mu    = df["pnl"].mean()
    sigma = df["pnl"].std() + 1e-8
    return float(mu / sigma * np.sqrt(252))


def calibrate_alpha(pred_returns: np.ndarray, y_val: np.ndarray,
                    alphas: np.ndarray | None = None,
                    alpha_max: float = 200.0,
                    dates: np.ndarray | None = None,
                    tickers: np.ndarray | None = None,
                    tc: float = TC_BPS / 10_000) -> float:
    """Trouve alpha* = argmax Sharpe_net(sigmoid(alpha * pred_returns)).

    Si dates/tickers fournis -> calibration sur Sharpe NET apres EMA + TC.
    Sinon -> Sharpe brut (mode compatibilite backward).

    alpha controle l'agressivite des positions :
      - alpha petit  -> positions proches de 0.5 (signal dilue, faibles paris)
      - alpha grand  -> positions proches de 0/1  (signal amplifie, paris forts)

    La plage 1 -> alpha_max est adaptee aux predictions journalieres (~0.001 a 0.01).
    Avec calibration nette, le TC penalise les alpha eleves qui generent trop de turnover.
    alpha_max=200 : cap original, laisse la calibration amplifier les signaux
    faibles quand le Sharpe net validation le justifie.
    """
    if alphas is None:
        alphas = np.logspace(0, np.log10(alpha_max), 30)

    use_net = (dates is not None and tickers is not None)
    best_sr = -np.inf
    best_a  = 1.0

    for a in alphas:
        pos = _sigmoid(a * pred_returns)
        if use_net:
            sr = _net_sharpe_calibration(pos, y_val, dates, tickers, tc)
        else:
            sr = _raw_sharpe(pos, y_val)
        if sr > best_sr:
            best_sr = sr
            best_a  = float(a)

    return best_a


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _ic_eval(y_pred: np.ndarray, dataset: lgb.Dataset):
    """Metrique custom LightGBM : IC = correlation entre predictions et reels.

    Pourquoi IC et pas RMSE ?
      RMSE sur rendements journaliers sature immediatement (bruit >> signal).
      IC continue d'augmenter jusqu'a ~200 arbres -> exploite mieux le signal.
    """
    y_true = dataset.get_label()
    corr = float(np.corrcoef(y_pred, y_true)[0, 1])
    if np.isnan(corr):
        corr = 0.0
    return "IC", corr, True


def train_fold_lgb(X_train: np.ndarray, y_train: np.ndarray,
                   X_val:   np.ndarray, y_val:   np.ndarray,
                   seed:    int = RANDOM_SEED,
                   vl_dates:   np.ndarray | None = None,
                   vl_tickers: np.ndarray | None = None,
                   ) -> tuple[lgb.Booster, float, dict]:
    """Entraine un LightGBM (MSE) et calibre alpha sur validation.

    Args:
        X_train, y_train : donnees d'entrainement (features, next_return 1j)
        X_val, y_val     : donnees de validation (next_return 1j — calibration TC correcte)
        seed             : graine aleatoire pour la reproductibilite
        vl_dates, vl_tickers : si fournis, calibration alpha sur Sharpe NET
                               apres EMA + TC (sinon Sharpe brut)

    Returns:
        (model, alpha_calibre, evals_result)
        - alpha : facteur sigmoid TC-aware pour sigmoid(alpha * predicted_return)
        - evals_result : dict {"train": {"l2": [...]}, "val": {"l2": [...]}}

    Note: pas d'early stopping — IC trop bruite sur rendements journaliers.
    On utilise 300 arbres fixes avec forte regularisation (max_depth=4, L1+L2).
    """
    params = {
        **LGB_PARAMS,
        "random_state": seed,
        "objective":    "regression",
        "metric":       "l2",
    }

    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set   = lgb.Dataset(X_val,   label=y_val,   reference=train_set, free_raw_data=False)

    evals_result: dict = {}
    callbacks = [
        lgb.log_evaluation(period=0),
        lgb.record_evaluation(evals_result),
    ]

    model = lgb.train(
        params,
        train_set,
        num_boost_round=LGB_N_ESTIMATORS,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    pred_val = model.predict(X_val)
    alpha    = calibrate_alpha(pred_val, y_val,
                               dates=vl_dates, tickers=vl_tickers)

    return model, alpha, evals_result


# ---------------------------------------------------------------------------
# Walk-forward (meme protocole que V1)
# ---------------------------------------------------------------------------

def walk_forward_splits(feat: pd.DataFrame,
                        test_start:   int = TEST_START,
                        window_years: int | None = WINDOW_YEARS) -> list[dict]:
    """Fenetre expansive (window_years=None) ou glissante (window_years=N ans).

    window_years=None -> expanding : entraine sur tout l'historique disponible.
    window_years=N    -> rolling   : entraine sur les N dernieres annees.
    """
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


def run_walk_forward(feat: pd.DataFrame,
                     feature_cols: list[str] | None = None,
                     verbose:      bool = True,
                     window_years: int | None = WINDOW_YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward LightGBM — retourne (positions, fold_metrics).

    Interface identique a V1 : NB04 peut tourner sans modification.
    window_years : None = expanding (papier), N = rolling N ans.
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in feat.columns]

    splits                   = walk_forward_splits(feat, window_years=window_years)
    all_pos: list[pd.DataFrame]   = []
    fold_rows: list[dict]         = []

    for fold_idx, sp in enumerate(splits):
        ty = sp["test_year"]
        t0 = time.perf_counter()

        # --- donnees d'entrainement (chronologique) ---------------------------
        mask = feat["date"] <= sp["train_end"]
        if sp.get("train_start") is not None:
            mask &= feat["date"] >= sp["train_start"]
        tr_feat = feat.loc[mask].sort_values(["date", "ticker"])

        X_all = tr_feat[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)

        n_val       = max(1_000, int(len(tr_feat) * VAL_FRAC))
        X_tr, X_vl  = X_all[:-n_val], X_all[-n_val:]
        y_all = tr_feat[TARGET_COL].to_numpy(dtype=np.float64)
        y_tr, y_vl  = y_all[:-n_val], y_all[-n_val:]

        if len(X_tr) < 500:
            if verbose:
                print(f"  fold {ty} — trop peu de donnees, ignore")
            continue

        # --- entrainement + calibration alpha ---------------------------------
        vl_slice = tr_feat.iloc[-n_val:]
        model, alpha, _ = train_fold_lgb(X_tr, y_tr, X_vl, y_vl,
                                         seed=RANDOM_SEED + fold_idx,
                                         vl_dates=vl_slice["date"].to_numpy(),
                                         vl_tickers=vl_slice["ticker"].to_numpy())

        # --- metriques de validation avec alpha calibre -----------------------
        pred_vl    = model.predict(X_vl)
        p_vl       = _sigmoid(alpha * pred_vl)
        val_sharpe = _raw_sharpe(p_vl, y_vl)
        val_ic     = float(np.corrcoef(pred_vl, y_vl)[0, 1])

        # --- prediction sur l'annee de test -----------------------------------
        te_feat = feat.loc[
            (feat["date"] >= sp["test_start"]) &
            (feat["date"] <= sp["test_end"])
        ].sort_values(["date", "ticker"]).copy()

        if te_feat.empty:
            continue

        X_te    = te_feat[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)
        pred_te = model.predict(X_te)
        raw_pos = _sigmoid(alpha * pred_te)

        pos_df = te_feat[["date", "ticker"]].copy()
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


# ---------------------------------------------------------------------------
# SHAP — interpretabilite (nouveau en V2)
# ---------------------------------------------------------------------------

def compute_shap(model: lgb.Booster,
                 feat:  pd.DataFrame,
                 feature_cols: list[str],
                 n_sample: int = 5_000) -> pd.DataFrame:
    """Calcule les valeurs SHAP sur un echantillon aleatoire.

    Args:
        model        : modele LightGBM entraine sur le panel complet
        feat         : feature matrix (utilise pour l'echantillonnage)
        feature_cols : liste des features utilisees par le modele
        n_sample     : taille de l'echantillon (defaut 5000 pour vitesse)

    Returns:
        DataFrame (feature, mean_abs_shap) trie par importance decroissante.
    """
    rng     = np.random.default_rng(RANDOM_SEED)
    idx     = rng.choice(len(feat), size=min(n_sample, len(feat)), replace=False)
    X_samp  = feat.iloc[idx][feature_cols].fillna(0.0).to_numpy(dtype=np.float64)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_samp)

    mean_abs = np.abs(shap_values).mean(axis=0)
    return (pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Filtre CPD — "Fast Reversion"
# ---------------------------------------------------------------------------
#
# Logique : le modele LightGBM predit du momentum. Mais quand NB02 detecte
# une rupture de tendance (score nu eleve), le signal momentum n'est plus
# fiable. On ramene alors la position vers 0.5 (neutre) proportionnellement
# a l'intensite de la rupture :
#
#   pos_filtered = 0.5 + confidence x (pos_raw - 0.5)
#   confidence   = clip(1 - strength x nu, 0, 1)
#
# Exemples avec strength=1 :
#   nu=0.0  (pas de rupture)  -> confidence=1.0 -> position inchangee
#   nu=0.5  (rupture moderee) -> confidence=0.5 -> position reduite 50% vers neutre
#   nu=1.0+ (rupture forte)   -> confidence=0.0 -> position=0.5 (neutre force)
#
# Ce filtre s'applique APRES le lissage EMA (smooth_positions).
# Selectif : ~4% des observations sont affectees (nu > seuil).

CPD_FILTER_NU_COLS = [
    "nu_cusum_lag1",
]
CPD_FILTER_STRENGTH = 1.0


def apply_cpd_filter(positions: pd.DataFrame,
                     feat:      pd.DataFrame,
                     strength:  float = CPD_FILTER_STRENGTH) -> pd.DataFrame:
    """Reduit les positions vers 0.5 lors de ruptures de regime (nu eleve).

    Args:
        positions : DataFrame (date, ticker, position) — apres EMA smoothing
        feat      : feature matrix complete (contient les scores nu CPD)
        strength  : intensite du filtre (defaut 1.0 ; > 1 = plus agressif)

    Returns:
        DataFrame (date, ticker, position) avec positions filtrees.
        Les positions non affectees (nu faible) sont inchangees.
    """
    nu_cols = [c for c in CPD_FILTER_NU_COLS if c in feat.columns]
    if not nu_cols:
        return positions

    cpd = feat[["date", "ticker"] + nu_cols].copy()
    cpd["nu_composite"] = cpd[nu_cols].mean(axis=1).clip(0.0, 1.0)

    merged = positions.merge(
        cpd[["date", "ticker", "nu_composite"]],
        on=["date", "ticker"], how="left"
    )
    merged["nu_composite"] = merged["nu_composite"].fillna(0.0)

    confidence = (1.0 - strength * merged["nu_composite"]).clip(0.0, 1.0)
    merged["position"] = 0.5 + confidence * (merged["position"] - 0.5)
    return merged[["date", "ticker", "position"]]


# ---------------------------------------------------------------------------
# Lissage temporel des positions (reduction du turnover)
# ---------------------------------------------------------------------------
#
# Probleme : LightGBM recalcule les predictions chaque jour -> positions
# varient beaucoup -> turnover journalier ~45% -> TC ~2308 bps/an.
#
# Solution : lissage EMA par action individuellement.
# Avec halflife=10j : alpha_ema ~ 0.067 par jour.
# Chaque jour on conserve 93.3% de la position d'hier + 6.7% de la nouvelle.
# Effet : turnover divise par ~10 -> TC ~244 bps/an.

def smooth_positions(positions: pd.DataFrame,
                     halflife:  int = POSITION_HALFLIFE) -> pd.DataFrame:
    """Lisse les positions par EMA pour reduire le turnover.

    Args:
        positions : DataFrame (date, ticker, position) — sorties de run_walk_forward
        halflife  : demi-vie de l'EMA en jours ouvrables (defaut 10)

    Returns:
        DataFrame (date, ticker, position) avec positions lissees.
        Les positions restent dans [0, 1] (propriete preservee par l'EMA).
    """
    alpha = float(1 - np.exp(-np.log(2) / halflife))
    out   = (positions
             .sort_values(["ticker", "date"])
             .copy())
    out["position"] = (
        out.groupby("ticker")["position"]
           .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    )
    return out


# ---------------------------------------------------------------------------
# Save (interface identique a V1 -> NB04 tourne sans modification)
# ---------------------------------------------------------------------------

def save_nb03_outputs(root,
                      positions:    pd.DataFrame,
                      fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Sauvegarde positions et fold_metrics dans data/processed/stoxx600/.

    Args:
        root         : racine du projet (ou None pour auto-detection)
        positions    : DataFrame (date, ticker, position) — sorties filtrees
        fold_metrics : DataFrame avec metriques par fold

    Returns:
        DataFrame recapitulatif des fichiers sauvegardes.
    """
    paths = nb03_paths(root)
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    positions.to_parquet(paths["positions"],    index=False)
    fold_metrics.to_parquet(paths["fold_metrics"], index=False)
    return pd.DataFrame([
        {"output": "positions",    "rows": f"{len(positions):,}",
         "path": paths["positions"].relative_to(paths["project_root"]).as_posix()},
        {"output": "fold_metrics", "rows": f"{len(fold_metrics):,}",
         "path": paths["fold_metrics"].relative_to(paths["project_root"]).as_posix()},
    ])
