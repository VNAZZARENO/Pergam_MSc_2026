# Presentation 1 - Progress And Current Results Snapshot

This note freezes the current state of the project before adding a heavier
LSTM/Sharpe-loss model. It is meant to support the first monthly presentation:
what is already implemented, what the current results say, and what remains to
be improved. It also keeps a trace of the main corrections and decisions made
along the way.

## 0. Progress Log

| Step | What changed | Why it mattered | Current status |
|---|---|---|---|
| Git branch | Work moved and pushed on `justine/submission`. | Vincent explicitly asked everyone to work on a Git branch. | Done. |
| Data source | `01_build_dataset.py` now starts in 2006 and uses yearly CSV files by default, then appends a rescaled PRICE ATLAS Excel tail for 2025-2026. | The Excel workbook alone starts in 2013, while Vincent asked for a 2006-to-today backtest; the rescaling avoids artificial source-switch jumps. | Done and rebuilt. |
| Long-format panel | Built `stoxx600_processed.csv` with price, returns, volatility, metadata and relative returns. | This matches the requested format: `date`, `ticker`, `price`, features. | Done. |
| Idiosyncratic returns | Added market-relative and sector-relative returns. | Vincent highlighted idiosyncratic shocks as the useful detection target, not only macro shocks. | Done, can be improved with earnings dates later. |
| CPD layer | Implemented fast CPD scores and GP-style reference logic. | The paper's key contribution is the CPD signal; the fast layer makes experiments scalable. | Full 2006-2026 stock-vs-sector refresh done. |
| Backtest | Added a first rule-based backtest in `04_run_backtest.py`. | This created the evaluation layer needed to compare all future models. | Done. |
| DMN-lite | Added a ridge-based supervised allocation model in `03_train_dmn.py`. | It closes the full pipeline before moving to the heavier LSTM. | Done, preliminary baseline. |
| Walk-forward | DMN-lite uses expanding annual walk-forward folds. | Vincent emphasized walk-forward validation; this avoids random temporal leakage. | Done, validation split still to improve. |
| LSTM DMN | Added `03_train_lstm_dmn.py` with a PyTorch LSTM and differentiable Sharpe loss. | This is the first implementation step that moves the model closer to the paper. | First full CPU run done. |
| Notebooks | Added notebooks `03` and `04`; converted main visuals to Plotly. | The presentation needs clear, reproducible outputs and graphs. | Done. |
| Current snapshot | This file freezes results, limitations and next steps. | It keeps a clean trace for the report and PowerPoint. | Done. |

## 1. Scope Implemented So Far

The current pipeline covers the full research chain:

1. `01_data_loading.ipynb` / `scripts/01_build_dataset.py`
   - STOXX 600 stock prices.
   - Period: 2006-01-02 to 2026-04-10.
   - Long-format panel: `date`, `ticker`, `price`, returns, volatility and metadata.
   - Market-relative and sector-relative returns for idiosyncratic shock analysis.

2. `02_changepoint_detection.ipynb` / `scripts/02_compute_cpd.py`
   - CPD methods: CUSUM, jump score, rolling t-test, BOCPD option, GP-style CPD reference.
   - CPD scores on raw stock returns, market-relative returns, sector-relative returns and sector-level series.
   - Latest full refresh: stock-vs-sector CPD scores for all 828 tickers, plus sector-level series.
   - Output: 1,763,349 CPD rows and 10,198 detected changepoints.

3. `03_train_dmn.ipynb` / `scripts/03_train_dmn.py`
   - First supervised model layer: `DMN-lite`.
   - Expanding walk-forward training by test year.
   - Features: momentum, volatility, market/sector relative returns, CPD score.
   - Output: stock-level positions in `dmn_lite_positions.csv`.

4. `scripts/03_train_lstm_dmn.py`
   - First PyTorch LSTM implementation closer to the paper.
   - Rolling stock-level feature sequences.
   - Differentiable negative Sharpe-ratio loss.
   - Same annual expanding walk-forward structure.
   - Output: stock-level positions in `dmn_lstm_positions.csv`.
   - Ablation output without CPD: `dmn_lstm_no_cpd_positions.csv`.

5. `04_run_backtest.ipynb` / `scripts/04_run_backtest.py`
   - Backtest of rule-based baselines and DMN-lite positions.
   - Metrics: annual return, annual volatility, Sharpe, Sortino, Calmar, max drawdown, hit ratio, average assets, turnover.

## 2. Current Backtest Results

Source file: `data/processed/stoxx600/backtest_summary_final_comparison.csv`

| Strategy | Period | Ann. Return | Ann. Vol | Sharpe | Sortino | Calmar | Max Drawdown | Hit Ratio | Avg Assets | Avg Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DMN-lite | 2010-01-04 to 2026-04-10 | 2.85% | 7.43% | 0.42 | 0.53 | 0.15 | -19.31% | 52.47% | 317.96 | 3.59% |
| Slow momentum | 2006-12-21 to 2026-04-10 | 3.97% | 14.74% | 0.34 | 0.44 | 0.12 | -34.29% | 53.05% | 251.41 | 4.19% |
| LSTM DMN without CPD | 2010-01-04 to 2026-04-10 | 3.35% | 13.39% | 0.31 | 0.38 | 0.11 | -30.85% | 53.50% | 312.10 | 9.60% |
| CPD-adjusted | 2006-12-21 to 2026-04-10 | 2.78% | 13.25% | 0.27 | 0.34 | 0.09 | -31.35% | 53.41% | 251.36 | 5.97% |
| LSTM DMN with CPD | 2010-01-04 to 2026-04-10 | 2.43% | 13.56% | 0.25 | 0.31 | 0.08 | -29.26% | 52.97% | 312.10 | 8.40% |
| Slow + fast | 2006-12-21 to 2026-04-10 | 1.81% | 12.93% | 0.20 | 0.25 | 0.05 | -33.68% | 53.45% | 251.36 | 4.53% |

## 3. Main Interpretation

The current results are encouraging but preliminary.

- DMN-lite has the highest Sharpe ratio in the current comparison.
- DMN-lite also has the smallest max drawdown.
- The rule-based CPD-adjusted strategy reduces drawdown versus slow momentum,
  but also lowers annual return and Sharpe.
- The first LSTM/Sharpe-loss implementation is working end-to-end, but it is
  not yet better than the simpler DMN-lite baseline.
- In this first run, the LSTM without CPD performs better than the LSTM with
  CPD. This does not prove CPD is useless; it suggests the CPD feature and LSTM
  regularization need more tuning before claiming an improvement.
- The most defensible presentation message is therefore: the full pipeline is
  reproducible, walk-forward, and now includes a paper-style LSTM baseline, but
  the strongest current empirical result is still the simpler DMN-lite model.

## 3.1 How The Work Evolved

The project started from the paper structure: slow momentum, fast reversion,
CPD, then a DMN-style allocation model. The first implementation effort focused
on making the data reliable and reproducible before optimizing the model.

The main correction was the separation between the universe source and the
historical price source. Vincent asked us to use the static 2025-2026 Excel file
for the universe, and also said that the backtest can run from 2006 to today.
The Excel workbook alone starts in 2013, so it cannot be the only source for a
2006-to-today empirical backtest.

The production pipeline therefore uses the yearly CSV files as the continuous
historical price source for 2006-2024, then appends the 2025-2026 PRICE ATLAS
tail. The tail is rescaled ticker by ticker on the latest overlapping date
before being appended, because the CSV and Excel sources are close in returns
but not identical in raw price levels. Without this rescaling, the switch of
source can create artificial one-day returns around the 2024-2025 boundary.

The interpretation is now:

- `prices_2006.csv` to `prices_2024.csv` are raw historical price files. They
  are read-only inputs and must not be modified.
- `2025_2026_PRICE_ATLAS_data_sxxr_static.xlsx` is used for the static universe
  / metadata reference and for the recent 2025-2026 price tail.
- The 2025-2026 Excel tail is rebased on the last overlapping CSV date before
  being joined to the historical CSV archive.
- All transformations, feature engineering and cleaning outputs are written
  downstream in `data/processed/stoxx600/`.

This is consistent with Vincent's instruction: the static 2025-2026 workbook is
the universe reference, while the annual CSV files provide the historical price
depth needed for a 2006-to-today empirical backtest. It also avoids using one
price source for CPD/model features and another one for the backtest.

Latest rebuild check:

- `stoxx600_processed.csv` date range: 2006-01-02 to 2026-04-10.
- Feature rows: 1,705,321.
- Unique tickers observed across the full raw history: 828.
- Duplicate `(date, ticker)` rows: 0.
- The raw annual CSV files remain unchanged; only processed outputs are
  regenerated.

## 3.2 Basic Feature Definitions

The main processed file is `stoxx600_processed.csv`. It is a long-format panel:
one row corresponds to one stock on one date. The important columns are not raw
Bloomberg fields; most of them are features computed from the raw prices.

The return columns measure past price performance over different horizons:

- `1d_arith_ret`: one-day return, computed from yesterday's price to today's
  price.
- `21d_arith_ret`: approximately one trading month.
- `63d_arith_ret`: approximately one trading quarter.
- `126d_arith_ret`: approximately six trading months.
- `252d_arith_ret`: approximately one trading year.

Using several horizons matters because the paper's economic idea combines slow
momentum and fast reversion. Long horizons, such as 126d or 252d, capture slow
trend information. Short horizons, such as 1d or 21d, capture recent shocks or
short-term reversal information.

The volatility columns, such as `20d_vol`, `60d_vol` and `252d_vol`, are also
computed from historical returns. For example, `60d_vol` is the annualized
standard deviation of the stock's recent daily returns over roughly 60 trading
days. These variables help the model distinguish a normal move from a large
move relative to the stock's own risk.

The relative-return columns compare each stock to a benchmark or group on the
same day:

- `1d_ret_vs_ew`: stock return minus the equal-weight STOXX 600 return.
- `1d_ret_vs_sxxr`: stock return minus the SXXR benchmark return.
- `1d_ret_vs_sector`: stock return minus its sector return.
- `1d_ret_vs_country` and `1d_ret_vs_region`: stock return minus its local
  group return.

These columns are useful for idiosyncratic shocks. A stock may fall because the
whole market falls, or because something specific happened to that stock. The
relative-return variables help separate stock-specific moves from macro or
sector-wide moves.

Columns ending in `_lag1` are shifted by one trading day. This means the model
uses yesterday's observed information to make today's decision. This is
important to avoid lookahead bias: the strategy must not use information from a
date before that information would have been observable in real time.

For supervised training, the scripts also create next-day targets internally.
The model observes features available at date `t`, predicts a position for date
`t+1`, and the backtest evaluates that position on the realized return at
`t+1`. This keeps the timing of information, prediction and PnL consistent.

## 3.3 Notes De Compréhension - Partie 01

Cette section résume les points importants du notebook `01_data_loading.ipynb`.
Elle sert de mémo de compréhension pour préparer la présentation.

### Notebook 01 vs script de production

Le notebook 01 sert surtout à expliquer et vérifier les définitions de données.
Il lit le fichier Excel `2025_2026_PRICE_ATLAS_data_sxxr_static.xlsx`, qui est
facile à inspecter mais commence en 2013. C'est pour cela que le notebook peut
afficher une période 2013-2026.

Le dataset final utilisé par les notebooks 02-04 est généré par
`scripts/01_build_dataset.py`. Ce script lit les CSV annuels
`prices_2006.csv` à `prices_2024.csv`, puis ajoute la fin 2025-2026 depuis
l'Excel. Avant de raccorder l'Excel, le script recale les niveaux de prix de
l'Excel sur la dernière date commune avec les CSV. Cela évite qu'un changement
de source crée un faux rendement extrême au passage 2024-2025.

On ne mélange donc pas les sources de manière arbitraire :

- l'Excel static 2025-2026 sert à définir l'univers demandé par Vincent et à
  compléter la période récente ;
- les CSV annuels servent de source historique continue pour les prix ;
- les features, la CPD, le modèle et le backtest utilisent ensuite le même
  panel traité.

Le fichier de production `stoxx600_processed.csv` couvre donc bien 2006-01-02 à
2026-04-10.

Le notebook 01 ne doit pas écraser les fichiers de production. Les fichiers
principaux dans `data/processed/stoxx600/` sont générés par le script.

### Fichiers `src` liés à la partie 01

Les fonctions réutilisables sont dans `src/`, tandis que le script assemble ces
fonctions pour construire le pipeline complet.

- `src/data_loader.py`
  - `load_stoxx600_prices()` : lit les CSV annuels de prix.
  - `load_price_atlas_prices()` : lit les prix depuis l'Excel PRICE ATLAS.
  - `append_price_atlas_tail()` : ajoute les dates 2025-2026 de l'Excel après
    l'historique CSV, en recalant d'abord les niveaux de prix sur la dernière
    date commune entre CSV et Excel.
  - `clean_prices()` : retire les prix aberrants et bouche seulement les petits
    trous.
  - `prices_to_panel()` : transforme les prix du format large vers le format
    long `date`, `ticker`, `price`.
  - `add_geography()` : ajoute `exchange`, `country` et `region` à partir du
    suffixe Bloomberg du ticker.

- `src/preprocessing.py`
  - `arithmetic_returns()` : calcule les rendements simples utilisés pour le
    PnL et le backtest.
  - `log_returns()` : calcule les log-rendements, pratiques pour les calculs
    statistiques et la volatilité.
  - `rolling_vol()` : calcule la volatilité réalisée sur une fenêtre glissante.

- `src/features.py`
  - `equal_weight_return()` : calcule le rendement moyen équipondéré de
    l'univers disponible.
  - `add_relative_returns()` : ajoute les rendements relatifs au marché, à la
    géographie et au secteur.
  - `normalized_returns()` et `macd()` : construisent des features de momentum
    supplémentaires pour les expériences de modèle.

- `src/sector_mapping.py`
  - `SECTOR_MAP` : mapping statique `ticker -> secteur`.
  - Ce mapping ne vient pas des prix. Les prix ne permettent pas de savoir
    qu'ASML est en Information Technology ou que BNP est en Financials. Il
    s'agit d'une classification sectorielle externe encodée dans le projet pour
    une première implémentation.

### SXXR et Equal-Weight

`SXXR` est le benchmark officiel STOXX Europe 600, plutôt pondéré par les
capitalisations. Les grandes valeurs ont donc plus de poids dans son mouvement.

`EW` signifie equal-weight. C'est un indice construit dans le projet en donnant
le même poids à chaque action disponible. Il représente davantage le mouvement
du stock européen moyen.

Les deux références sont utiles :

- `SXXR` sert de benchmark officiel de marché.
- `EW` sert de référence interne moins dominée par les grandes capitalisations.
- Pour isoler les chocs spécifiques aux stocks, `EW` et surtout les rendements
  relatifs au secteur sont souvent plus informatifs que le benchmark officiel
  seul.

Dans le tableau benchmark du notebook, `EW` commence à 100 parce que nous le
construisons comme un indice base 100. `SXXR` garde son niveau de prix réel dans
la source. Dans les graphes, les séries sont rebased à 100 pour comparer leurs
trajectoires.

### Tickers, exchanges, countries et regions

Un ticker Bloomberg contient souvent deux parties, par exemple `ASML NA` :

- `ASML` : code de la société.
- `NA` : suffixe de place de cotation.

La place de cotation est le marché sur lequel l'action est cotée. Exemples :

- `FP` : France / Paris.
- `GY` : Allemagne / Xetra.
- `LN` : Londres.
- `IM` : Italie.
- `NA` : Pays-Bas / Amsterdam.
- `SS` : Suède / Stockholm.
- `BB` : Belgique.

Le suffixe est utile car un même symbole peut être ambigu ou exister sur
plusieurs marchés. Le ticker complet `ticker + exchange` identifie donc plus
clairement l'actif.

`exchange`, `country` et `region` sont trois niveaux de granularité :

- `exchange` : place de cotation, niveau le plus précis.
- `country` : pays associé à la cotation.
- `region` : regroupement géographique plus large, par exemple Western Europe,
  Nordic, Southern Europe ou UK & Ireland.

Ces variables ne sont pas le coeur du projet, mais elles permettent de contrôler
des mouvements géographiques. Elles aident à distinguer un choc propre au stock
d'un choc de marché local, national ou régional.

### Rendements arithmétiques, log-rendements et horizons

Les rendements ne sont pas donnés directement dans les fichiers raw. Ils sont
calculés à partir des prix.

Le rendement arithmétique est :

```text
price_today / price_yesterday - 1
```

Il est intuitif et sert au PnL et au backtest.

Le log-rendement est :

```text
log(price_today) - log(price_yesterday)
```

Il est utile pour les calculs statistiques car il s'additionne mieux dans le
temps.

Les horizons mesurent des mouvements passés à différentes fréquences :

- `1d` : rendement journalier.
- `21d` : environ un mois de bourse.
- `63d` : environ trois mois.
- `126d` : environ six mois.
- `252d` : environ un an.

Plusieurs horizons sont utiles parce que le papier combine slow momentum et
fast reversion. Les horizons longs capturent la tendance lente. Les horizons
courts capturent les chocs récents et les effets de retour rapide.

### Volatilité

La volatilité est aussi calculée à partir des rendements passés. Par exemple,
`60d_vol` est l'écart-type annualisé des rendements quotidiens sur environ 60
jours de bourse.

On utilise `20d_vol`, `60d_vol` et `252d_vol` :

- 20 jours : volatilité courte, environ un mois.
- 60 jours : volatilité intermédiaire, environ trois mois.
- 252 jours : volatilité longue, environ un an.

On ne calcule pas vraiment une volatilité `1d`, car une volatilité estimée sur
un seul jour n'a pas de sens statistique. Il faut une fenêtre d'observations.

### Rendements relatifs et chocs idiosyncratiques

Un rendement relatif est défini comme :

```text
rendement du stock - rendement du groupe de référence
```

Exemple :

```text
ASML = +1%
secteur Tech = -2%
1d_ret_vs_sector = +1% - (-2%) = +3%
```

ASML a donc fait 3 points de pourcentage mieux que son secteur.

Les rendements relatifs calculés sont :

- `1d_ret_vs_sxxr` : stock moins benchmark SXXR.
- `1d_ret_vs_ew` : stock moins marché equal-weight.
- `1d_ret_vs_exchange` : stock moins les actions de la même place de cotation.
- `1d_ret_vs_country` : stock moins les actions du même pays.
- `1d_ret_vs_region` : stock moins les actions de la même région.
- `1d_ret_vs_sector` : stock moins les actions du même secteur.

Ces variables servent à isoler les chocs idiosyncratiques, c'est-à-dire les
mouvements propres au stock qui ne s'expliquent pas seulement par le marché, le
pays, la région ou le secteur. Pour Vincent, la variable la plus importante à
défendre est probablement `1d_ret_vs_sector`, car elle retire le mouvement
commun du secteur.

### Sector mapping et sector coverage

Les prix ne contiennent pas le secteur des entreprises. Le projet ajoute donc
un mapping statique dans `src/sector_mapping.py` :

```text
ticker -> secteur GICS
```

Cela permet de construire :

- `sector_1d_ret` : rendement moyen du secteur à une date donnée.
- `1d_ret_vs_sector` : rendement du stock moins rendement moyen de son secteur.
- `sector_returns.csv` : séries sectorielles utilisées ensuite par le notebook
  02 pour la détection de changepoints au niveau secteur.

`sector coverage` mesure la proportion de lignes stock auxquelles on a pu
attribuer un secteur. Par exemple, 97.51% signifie qu'environ 97.51% des lignes
ont un secteur disponible. C'est suffisant pour utiliser les variables
sectorielles dans une première version, mais le mapping devra idéalement être
validé contre Bloomberg ou une source officielle.

### Sector-relative returns vs sector-level indices

`1d_ret_vs_sector` est une variable stock-level : elle indique si un stock a
surperformé ou sous-performé son secteur.

`sector_returns.csv` est une table sector-level : elle donne le rendement moyen
de chaque secteur dans le temps.

Les deux sont utiles :

- le stock-vs-sector sert à détecter les mouvements spécifiques à une action ;
- les séries sectorielles servent à identifier les mouvements communs au
  secteur entier.

Cela permet ensuite de distinguer un choc sectoriel d'un choc vraiment propre
au stock.

### Lag et lookahead bias

Les colonnes qui se terminent par `_lag1` sont décalées d'un jour par ticker :

```python
df.groupby("ticker")[col].shift(1)
```

La valeur observée lundi devient donc une information utilisable mardi. C'est
important pour éviter le lookahead bias. Le modèle ne doit jamais utiliser une
information du jour `t` si cette information n'aurait été connue qu'à la fin du
jour `t`.

La logique correcte est :

```text
features disponibles à t -> position pour t+1 -> PnL réalisé à t+1
```

### Sanity checks

Un sanity check est une vérification de cohérence simple. Ce n'est pas une
preuve statistique, mais cela permet de détecter rapidement une erreur de
pipeline.

Le sanity check COVID sur ASML compare par exemple :

- le rendement brut du stock ;
- le rendement du marché SXXR ;
- le rendement equal-weight ;
- le rendement du secteur ;
- les rendements relatifs.

Si ASML monte alors que le marché et son secteur baissent fortement, le
rendement relatif devient fortement positif. Cela illustre l'objectif du
pipeline : séparer ce qui est dû au marché ou au secteur de ce qui est propre au
stock.

The second correction was methodological. A CPD detector alone is not enough:
we need to test whether CPD improves actual portfolio PnL. This is why the
backtest layer was implemented before the final LSTM. It gives us a stable
evaluation framework.

The third correction was validation. Instead of random train/test splits, the
current supervised model uses yearly walk-forward folds. This is closer to how
the strategy would be evaluated in practice and matches Vincent's comments.

The current DMN-lite model should therefore be understood as a bridge: it is
not the final paper-style LSTM, but it proves that the pipeline can transform
daily data and CPD features into out-of-sample positions and measurable
backtest results.

The first LSTM/Sharpe-loss implementation has now been added as a separate
script, not as a replacement for DMN-lite. This keeps the baseline comparable
while allowing the project to move toward the paper's architecture. The next
step is to run the LSTM on the full intended universe/date range, then backtest
`dmn_lstm_positions.csv` next to the existing strategies.

The first full CPU LSTM experiment has now been run with annual expanding
walk-forward folds from 2010 to 2026. It uses a 63-day input sequence, two
training epochs per fold, and a capped random sample of 60,000 training
sequences per fold to keep the computation feasible on a laptop. This should be
presented as a first working LSTM baseline, not as a fully tuned final network.

## 4. Walk-Forward Status

The current DMN-lite model already uses an expanding walk-forward protocol:

- first test year: 2010;
- train set always ends before the test year starts;
- test folds are yearly;
- latest fold: train 2006-03-27 to 2025-12-31, test 2026-01-01 to 2026-04-10.

This is aligned with Vincent's emphasis on walk-forward validation. The next
improvement is to add a separate validation block inside each fold for
hyperparameter selection and early stopping.

## 5. Important Limitations To State Clearly

1. The current `DMN-lite` is not the full LSTM from the paper.
   - It is a ridge-based supervised allocation layer.
   - It closes the full pipeline, but does not fully reproduce the model architecture.

2. The paper's original universe is futures, while this project uses STOXX 600 equities.
   - Results are not directly comparable to the paper's reported Sharpe ratios.

3. The CPD rule-based strategy is still simple.
   - Better use of CPD may require a learned model or a more persistent regime layer.

4. The current portfolio construction is still simplified.
   - Long-only, benchmark-aware and tracking-error-aware construction remains a future extension.

## 6. Next Technical Step

The next step is to turn the new LSTM script into a full result:

1. Add a validation split inside each walk-forward fold for hyperparameter
   selection and early stopping.
2. Tune LSTM regularization and turnover control, because the first LSTM runs
   have materially higher turnover than DMN-lite.
3. Test alternative CPD feature transformations, for example persistence,
   recent maximum score, or shock sign, instead of using only the raw ensemble
   score.
4. Improve the portfolio layer with benchmark-aware or turnover-aware
   constraints.
5. Convert the current results into the first PowerPoint narrative.

The key presentation message is:

> We first built a complete and reproducible pipeline. The next step is to
> replace the DMN-lite model by the paper-style LSTM trained with a Sharpe loss,
> then test whether CPD improves performance out of sample.
