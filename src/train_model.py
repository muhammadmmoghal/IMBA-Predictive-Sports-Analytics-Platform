"""
train_model.py  (V2 — tier-aware + STL/BLK)
Trains models for pts / reb / ast / stl / blk and saves the best model
per target to models/.

What changed from V1:
  - Feature set expanded from 24 (V1) to 66 (V2) by adding tier-aware
    rolling/expanding averages and tier game counts.
  - New targets: target_stl, target_blk (5 targets total).
  - V1 vs V2 comparison reported for pts / reb / ast.
  - NaN in tier-aware features is filled with 0 before training (these
    NaNs arise when a player has no prior games in that tier; 0 is the
    natural "no tier history" sentinel).  V1-feature NaN rows (players'
    first game ever) are still dropped.

Time split:
  Train <= 2025-10-31  (D1/D2 2024 and 2025 Summer seasons)
  Test  >  2025-10-31  (D1/D2 2025-26 Winter — the current season)

Models per target:
  1. CareerAvgBaseline  — always predict career_{stat}_avg
  2. LinearRegression   — Pipeline(StandardScaler → LinearRegression)
  3. RandomForest       — 200 trees, min_samples_leaf=5

Best model saved to models/{stat}_model.pkl.
Results saved to reports/model_results.csv.

Usage:  python src/train_model.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURES_PATH = Path("data/processed/features.csv")
MODELS_DIR    = Path("models")
REPORTS_DIR   = Path("reports")
RESULTS_PATH  = REPORTS_DIR / "model_results.csv"

TRAIN_CUTOFF = pd.Timestamp("2025-10-31")

STATS = ["pts", "reb", "ast", "stl", "blk"]

# V1: 24 features (original, all-tier-blended) — kept for comparison only
V1_FEATURE_COLS = [
    "tier_d1",
    "days_since_last_game",
    "games_played_before",
    "season_games_before",
    "last_3_pts_avg", "last_3_reb_avg", "last_3_ast_avg",
    "last_3_stl_avg", "last_3_blk_avg",
    "last_5_pts_avg", "last_5_reb_avg", "last_5_ast_avg",
    "last_5_stl_avg", "last_5_blk_avg",
    "season_pts_avg", "season_reb_avg", "season_ast_avg",
    "season_stl_avg", "season_blk_avg",
    "career_pts_avg", "career_reb_avg", "career_ast_avg",
    "career_stl_avg", "career_blk_avg",
]

# V2: tier-aware with context-routed features (66 total)
#
# ctx_* features are the key fix:
#   At training time, ctx_last_5_pts_avg = last_5_tier1_pts_avg for a D1 row
#                                        = last_5_tier2_pts_avg for a D2 row
#   At predict time, ctx_* = tier1-specific when predicting for tier1,
#                           = tier2-specific when predicting for tier2.
#
# This means the top-importance features (last_5, season, career) will carry
# genuinely different values for tier1 vs tier2 predictions, so the models
# produce tier-differentiated output.
V2_FEATURE_COLS = (
    [
        "tier_d1",
        "days_since_last_game",
        "games_played_before",
        "season_games_before",
        "tier1_games_before",
        "tier2_games_before",
    ]
    # Primary signal: context-routed to the game's (or requested) tier
    + [f"ctx_last_{n}_{s}_avg" for n in [3, 5] for s in STATS]
    + [f"ctx_season_{s}_avg" for s in STATS]
    + [f"ctx_career_{s}_avg" for s in STATS]
    # Cross-tier context: full tier1 and tier2 histories
    + [f"last_{n}_tier1_{s}_avg" for n in [3, 5] for s in STATS]
    + [f"season_tier1_{s}_avg" for s in STATS]
    + [f"career_tier1_{s}_avg" for s in STATS]
    + [f"last_{n}_tier2_{s}_avg" for n in [3, 5] for s in STATS]
    + [f"season_tier2_{s}_avg" for s in STATS]
    + [f"career_tier2_{s}_avg" for s in STATS]
)

# All five prediction targets
TARGETS = {
    "pts": "target_pts",
    "reb": "target_reb",
    "ast": "target_ast",
    "stl": "target_stl",
    "blk": "target_blk",
}

# Tier-context career avg as the no-model baseline.
# ctx_career_*_avg = D1 career avg for D1 rows, D2 career avg for D2 rows.
CAREER_AVG_COL = {
    "pts": "ctx_career_pts_avg",
    "reb": "ctx_career_reb_avg",
    "ast": "ctx_career_ast_avg",
    "stl": "ctx_career_stl_avg",
    "blk": "ctx_career_blk_avg",
}

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


# ---------------------------------------------------------------------------
# Train + evaluate one target with a given feature set
# ---------------------------------------------------------------------------

def train_target(
    stat: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    version: str = "V2",
) -> tuple[dict, object, str]:
    """
    Trains all models for one target using the supplied feature set.
    Returns (results_dict, best_model, best_model_name).
    """
    results: dict = {}

    # ── 1. Career-average baseline (no training) ─────────────────────────
    baseline_col = CAREER_AVG_COL[stat]
    y_pred_base  = X_test[baseline_col].fillna(0).values
    results["CareerAvgBaseline"] = metrics(y_test, y_pred_base)
    log.info("    CareerAvgBaseline  MAE=%.3f  RMSE=%.3f  R2=%.3f",
             *results["CareerAvgBaseline"].values())

    # ── 2. Linear Regression ─────────────────────────────────────────────
    lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    lr.fit(X_train[feature_cols], y_train)
    y_pred_lr = lr.predict(X_test[feature_cols])
    results["LinearRegression"] = metrics(y_test, y_pred_lr)
    log.info("    LinearRegression   MAE=%.3f  RMSE=%.3f  R2=%.3f",
             *results["LinearRegression"].values())

    # ── 3. Random Forest ─────────────────────────────────────────────────
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train[feature_cols], y_train)
    y_pred_rf = rf.predict(X_test[feature_cols])
    results["RandomForest"] = metrics(y_test, y_pred_rf)
    log.info("    RandomForest       MAE=%.3f  RMSE=%.3f  R2=%.3f",
             *results["RandomForest"].values())

    # ── Pick best sklearn model ───────────────────────────────────────────
    sklearn_models = {"LinearRegression": lr, "RandomForest": rf}
    best_name  = min(sklearn_models, key=lambda m: results[m]["mae"])
    best_model = sklearn_models[best_name]
    log.info("    Best (%s): %s", version, best_name)

    best_model.target       = stat
    best_model.model_name   = best_name
    best_model.feature_cols = feature_cols
    best_model.test_metrics = results[best_name]
    best_model.version      = version

    return results, best_model, best_name


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------

def print_audit(
    df_model: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    v1_results: dict,
    v2_results: dict,
    v2_best_names: dict,
) -> None:
    sep = "=" * 72

    print()
    print(sep)
    print("MODEL TRAINING AUDIT REPORT  (V2 — tier-aware + STL/BLK)")
    print(sep)

    # ── Data split ────────────────────────────────────────────────────────
    print(f"\n  Total usable rows  : {len(df_model)}")
    print(f"  Train rows         : {train_mask.sum()}  (up to {TRAIN_CUTOFF.date()})")
    print(f"  Test rows          : {test_mask.sum()}   (from {TRAIN_CUTOFF.date() + pd.Timedelta(days=1)})")

    print("\n  Train seasons:")
    for s, c in df_model[train_mask]["season"].value_counts().sort_values(ascending=False).items():
        print(f"    {s:<25}  {c:>4} rows")

    print("\n  Test seasons:")
    for s, c in df_model[test_mask]["season"].value_counts().sort_values(ascending=False).items():
        print(f"    {s:<25}  {c:>4} rows")

    # ── Feature count ────────────────────────────────────────────────────
    print(f"\n  V1 feature count : {len(V1_FEATURE_COLS)}")
    print(f"  V2 feature count : {len(V2_FEATURE_COLS)}  "
          f"(+{len(V2_FEATURE_COLS)-len(V1_FEATURE_COLS)} tier-aware features)")

    # ── V1 vs V2 comparison for pts / reb / ast ───────────────────────────
    comparison_targets = ["pts", "reb", "ast"]
    print(f"\n{'-'*72}")
    print("  V1 vs V2 COMPARISON  (pts / reb / ast)")
    print(f"{'-'*72}")
    print(f"  {'Target':<12}  {'Model':<10}  "
          f"{'V1 MAE':>8}  {'V2 MAE':>8}  {'Delta':>8}  {'V2 Best Model':<22}")
    print("  " + "-" * 68)
    for stat in comparison_targets:
        if stat not in v1_results or stat not in v2_results:
            continue
        v1_best_name = min(
            [k for k in v1_results[stat] if k != "CareerAvgBaseline"],
            key=lambda m: v1_results[stat][m]["mae"],
        )
        v2_best_name = v2_best_names[stat]
        v1_mae = v1_results[stat][v1_best_name]["mae"]
        v2_mae = v2_results[stat][v2_best_name]["mae"]
        delta  = v2_mae - v1_mae
        sign   = "+" if delta >= 0 else ""
        print(f"  {'target_'+stat:<12}  {'ML':>10}  "
              f"{v1_mae:>8.3f}  {v2_mae:>8.3f}  {sign}{delta:>7.3f}  {v2_best_name:<22}")

    # ── Results per target ────────────────────────────────────────────────
    for stat, target_col in TARGETS.items():
        print(f"\n{'-'*72}")
        print(f"  TARGET: {target_col.upper()}")
        print(f"{'-'*72}")

        y_test = df_model[test_mask][target_col]
        print(f"  Test distribution:  mean={y_test.mean():.2f}  std={y_test.std():.2f}  "
              f"median={y_test.median():.0f}  max={y_test.max():.0f}")
        print()

        model_results = v2_results[stat]
        best = v2_best_names[stat]

        header = f"  {'Model':<22}  {'MAE':>7}  {'RMSE':>7}  {'R2':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for model_name, m in model_results.items():
            marker = " <-- best" if model_name == best else ""
            print(f"  {model_name:<22}  {m['mae']:>7.3f}  {m['rmse']:>7.3f}  "
                  f"{m['r2']:>7.3f}{marker}")

        print(f"\n  Saved: models/{stat}_model.pkl  ({best})")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print("  SUMMARY: Best V2 model per target")
    print(f"{'-'*72}")
    print(f"  {'Target':<12}  {'Best Model':<22}  {'MAE':>7}  {'RMSE':>7}  {'R2':>7}")
    print("  " + "-" * 65)
    for stat, target_col in TARGETS.items():
        best = v2_best_names[stat]
        m    = v2_results[stat][best]
        print(f"  {target_col:<12}  {best:<22}  {m['mae']:>7.3f}  {m['rmse']:>7.3f}  {m['r2']:>7.3f}")

    print(f"\n  Feature count  : V1={len(V1_FEATURE_COLS)}  V2={len(V2_FEATURE_COLS)}")
    print(f"  Results saved to: {RESULTS_PATH}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not FEATURES_PATH.exists():
        log.error("Features file not found: %s — run feature_engineering.py first", FEATURES_PATH)
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────
    log.info("Loading %s", FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    log.info("Loaded %d rows x %d columns", *df.shape)

    # Check for V2 features
    if "career_tier1_pts_avg" not in df.columns:
        log.error("V2 tier features not found — run feature_engineering.py first")
        sys.exit(1)

    # ── Drop rows where days_since_last_game is NaN (first-game rows) ────
    # These rows have no prior history at all — no rolling or career features.
    before = len(df)
    df_model = df.dropna(subset=["days_since_last_game"]).copy()
    log.info("Dropped %d first-game rows -> %d usable rows", before - len(df_model), len(df_model))

    # ── Fill tier / ctx NaN with 0 ("no prior tier experience") ──────────
    # ctx_* is NaN when a player has no prior games in their own tier yet.
    # tier1/2_* is NaN when a player has no prior games in that tier yet.
    # 0 is the correct "no experience" sentinel; tier_games_before = 0 tells
    # the model this player has no history in that tier.
    v2_only_cols = [c for c in V2_FEATURE_COLS if c not in V1_FEATURE_COLS]
    df_model[v2_only_cols] = df_model[v2_only_cols].fillna(0)
    log.info("Filled NaN in %d V2 columns with 0", len(v2_only_cols))

    # ── Time split ───────────────────────────────────────────────────────
    train_mask = df_model["date"] <= TRAIN_CUTOFF
    test_mask  = df_model["date"] >  TRAIN_CUTOFF
    log.info("Train: %d rows  |  Test: %d rows", train_mask.sum(), test_mask.sum())

    X_train = df_model[train_mask]
    X_test  = df_model[test_mask]

    # ── Train V2 models for all five targets ──────────────────────────────
    log.info("Training V2 models (pts / reb / ast / stl / blk)...")
    v2_results:    dict = {}
    v2_best_names: dict = {}
    results_rows:  list = []

    for stat, target_col in TARGETS.items():
        log.info("  V2  target: %s", target_col)
        y_train = X_train[target_col]
        y_test  = X_test[target_col]

        results, best_model, best_name = train_target(
            stat, X_train, y_train, X_test, y_test,
            V2_FEATURE_COLS, version="V2",
        )
        v2_results[stat]    = results
        v2_best_names[stat] = best_name

        # Save best V2 model
        model_path = MODELS_DIR / f"{stat}_model.pkl"
        joblib.dump(best_model, model_path)
        log.info("  Saved %s -> %s", best_name, model_path)

        for model_name, m in results.items():
            results_rows.append({
                "version":      "V2",
                "target":       target_col,
                "model":        model_name,
                "mae":          m["mae"],
                "rmse":         m["rmse"],
                "r2":           m["r2"],
                "is_best":      model_name == best_name,
                "train_rows":   int(train_mask.sum()),
                "test_rows":    int(test_mask.sum()),
                "n_features":   len(V2_FEATURE_COLS),
                "train_cutoff": str(TRAIN_CUTOFF.date()),
            })

    # ── Save results CSV ─────────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(RESULTS_PATH, index=False)
    log.info("Results saved to %s", RESULTS_PATH)

    # ── Audit report ─────────────────────────────────────────────────────
    print_audit(df_model, train_mask, test_mask, {}, v2_results, v2_best_names)


if __name__ == "__main__":
    main()
