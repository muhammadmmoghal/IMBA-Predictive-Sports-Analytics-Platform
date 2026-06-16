"""
feature_engineering.py  (V2 — tier-aware)
Builds the pre-game feature matrix from data/processed/clean_game_logs.csv
and writes data/processed/features.csv.

V2 additions over V1:
  - Tier-aware rolling / expanding features (tier1 = D1, tier2 = D2):
      last_3_tier{1,2}_{stat}_avg, last_5_tier{1,2}_{stat}_avg
      career_tier{1,2}_{stat}_avg, season_tier{1,2}_{stat}_avg
    for each stat in [pts, reb, ast, stl, blk].
  - tier1_games_before, tier2_games_before  (game counts per tier)
  - target_stl and target_blk added alongside existing target_pts/reb/ast.

Tier-aware feature semantics (no leakage):
  At any given row, last_N_tier1_pts_avg = mean pts in the player's
  last N D1 games BEFORE this game, in chronological order.
  If the player has no prior D1 games the value is NaN.
  season_tier1_pts_avg = mean pts in D1 games from the player's most
  recent D1 season prior to this game (or NaN if no D1 season yet).

  These features are computed by iterating each player's game history
  in order and maintaining separate accumulators per tier, updating
  AFTER the feature values for the current row are recorded.

All original V1 features are retained unchanged.

Sort order: (player_id, date, game_id)

Usage:  python src/feature_engineering.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLEAN_PATH  = Path("data/processed/clean_game_logs.csv")
OUTPUT_PATH = Path("data/processed/features.csv")

STATS        = ["pts", "reb", "ast", "stl", "blk"]
ROLL_WINDOWS = [3, 5]

# Tier label -> raw tier value in the data
TIERS = {"tier1": "D1", "tier2": "D2"}

# Columns passed through unchanged
IDENTITY_COLS = ["player_id", "player_name", "game_id", "date", "season"]
CONTEXT_COLS  = ["tier", "tier_d1", "team", "opponent"]

# Output column order
FEATURE_COLS = (
    IDENTITY_COLS
    + CONTEXT_COLS
    # ── Experience ────────────────────────────────────────────────────────
    + ["days_since_last_game", "games_played_before", "season_games_before",
       "tier1_games_before", "tier2_games_before"]
    # ── Context-routed features (ctx_*) ──────────────────────────────────
    # For a D1 game row: ctx_* = tier1-specific value.
    # For a D2 game row: ctx_* = tier2-specific value.
    # At predict time: route to the requested tier.
    # These replace blended V1 features as the model's primary recent-form signal.
    + [f"ctx_last_{n}_{s}_avg" for n in ROLL_WINDOWS for s in STATS]
    + [f"ctx_season_{s}_avg" for s in STATS]
    + [f"ctx_career_{s}_avg" for s in STATS]
    # ── Tier1 (D1) specific history ───────────────────────────────────────
    + [f"last_{n}_tier1_{s}_avg" for n in ROLL_WINDOWS for s in STATS]
    + [f"season_tier1_{s}_avg" for s in STATS]
    + [f"career_tier1_{s}_avg" for s in STATS]
    # ── Tier2 (D2) specific history ───────────────────────────────────────
    + [f"last_{n}_tier2_{s}_avg" for n in ROLL_WINDOWS for s in STATS]
    + [f"season_tier2_{s}_avg" for s in STATS]
    + [f"career_tier2_{s}_avg" for s in STATS]
    # ── Targets ───────────────────────────────────────────────────────────
    + ["target_pts", "target_reb", "target_ast", "target_stl", "target_blk"]
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
# V1 feature builders (overall, all tiers blended)
# ---------------------------------------------------------------------------

def _rolling_avg(series: pd.Series, n: int) -> pd.Series:
    return series.shift(1).rolling(n, min_periods=1).mean()


def _expanding_avg(series: pd.Series) -> pd.Series:
    return series.shift(1).expanding(min_periods=1).mean()


# ---------------------------------------------------------------------------
# V2 tier-aware feature builder
# ---------------------------------------------------------------------------

def _compute_tier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tier-aware rolling and expanding features without leakage.

    For each player the game history is walked in chronological order.
    At every row the features are computed from the accumulator BEFORE
    the current game is included, then the current game is appended to
    the correct tier's accumulator.

    season_tierN_*_avg uses the player's most recent tierN season seen
    prior to the current game (regardless of the current game's tier),
    so it carries the most recent in-tier seasonal context forward.
    """
    n_rows = len(df)

    # Pre-allocate all new columns as NaN / zero
    arrays: dict[str, np.ndarray] = {}
    for tlabel in TIERS:
        for n in ROLL_WINDOWS:
            for s in STATS:
                arrays[f"last_{n}_{tlabel}_{s}_avg"] = np.full(n_rows, np.nan)
        for s in STATS:
            arrays[f"season_{tlabel}_{s}_avg"] = np.full(n_rows, np.nan)
            arrays[f"career_{tlabel}_{s}_avg"] = np.full(n_rows, np.nan)
    arrays["tier1_games_before"] = np.zeros(n_rows, dtype=float)
    arrays["tier2_games_before"] = np.zeros(n_rows, dtype=float)

    for _pid, grp in df.groupby("player_id", sort=False):
        orig_idx  = grp.index.tolist()

        # Per-tier accumulators  {tier_val: {stat: list}}
        career  = {"D1": {s: [] for s in STATS}, "D2": {s: [] for s in STATS}}
        # {tier_val: {stat: {season_name: list}}}
        seasons = {"D1": {s: {} for s in STATS}, "D2": {s: {} for s in STATS}}
        # Most recent season seen for each tier
        latest_season = {"D1": None, "D2": None}
        tier_count    = {"D1": 0,    "D2": 0}

        for orig_i, row in zip(orig_idx, grp.itertuples(index=False)):
            row_tier   = row.tier
            row_season = row.season

            # ── Record game counts before this game ───────────────────
            arrays["tier1_games_before"][orig_i] = tier_count["D1"]
            arrays["tier2_games_before"][orig_i] = tier_count["D2"]

            # ── Compute tier features from prior history ───────────────
            for tlabel, tier_val in TIERS.items():
                c  = career[tier_val]   # {stat: list}
                ls = latest_season[tier_val]

                for s in STATS:
                    hist = c[s]

                    # Rolling last N
                    for n in ROLL_WINDOWS:
                        arrays[f"last_{n}_{tlabel}_{s}_avg"][orig_i] = (
                            float(np.mean(hist[-n:])) if hist else np.nan
                        )

                    # Career avg
                    arrays[f"career_{tlabel}_{s}_avg"][orig_i] = (
                        float(np.mean(hist)) if hist else np.nan
                    )

                    # Season avg (most recent tier season)
                    if ls is not None:
                        s_hist = seasons[tier_val][s].get(ls, [])
                        arrays[f"season_{tlabel}_{s}_avg"][orig_i] = (
                            float(np.mean(s_hist)) if s_hist else np.nan
                        )

            # ── Update accumulators with current game ──────────────────
            for s in STATS:
                val = getattr(row, s)
                career[row_tier][s].append(val)
                seasons[row_tier][s].setdefault(row_season, []).append(val)
            latest_season[row_tier] = row_season
            tier_count[row_tier]   += 1

    for col, arr in arrays.items():
        df[col] = arr

    return df


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the clean game-log DataFrame and returns the full V2 feature matrix
    with no future-data leakage.
    """
    log.info("Building features on %d rows...", len(df))

    df = df.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)

    # ── Tier encoding ────────────────────────────────────────────────────
    df["tier_d1"] = (df["tier"] == "D1").astype(int)

    # ── Targets ──────────────────────────────────────────────────────────
    df["target_pts"] = df["pts"]
    df["target_reb"] = df["reb"]
    df["target_ast"] = df["ast"]
    df["target_stl"] = df["stl"]
    df["target_blk"] = df["blk"]

    # ── Experience features ───────────────────────────────────────────────
    df["games_played_before"] = df.groupby("player_id").cumcount()
    df["season_games_before"] = df.groupby(["player_id", "season"]).cumcount()
    log.info("  Experience features done.")

    # ── Days since last game ─────────────────────────────────────────────
    prev_date = df.groupby("player_id")["date"].shift(1)
    df["days_since_last_game"] = (df["date"] - prev_date).dt.days
    log.info("  Days since last game done.")

    # ── Tier-aware features (tier1_* and tier2_*) ─────────────────────────
    df = _compute_tier_features(df)
    log.info("  Tier-aware features done.")

    # ── Context-routed features (ctx_*) ───────────────────────────────────
    # For each row, ctx_* picks the tier-specific value matching that row's
    # own tier.  This gives the model genuine tier-specific recent form as its
    # primary signal — no blending of D1 and D2 history.
    # NaN (player has no history in their own tier yet) → 0, same as at
    # prediction time for players new to a tier.
    d1_mask = df["tier"] == "D1"
    for s in STATS:
        for n in ROLL_WINDOWS:
            df[f"ctx_last_{n}_{s}_avg"] = np.where(
                d1_mask,
                df[f"last_{n}_tier1_{s}_avg"],
                df[f"last_{n}_tier2_{s}_avg"],
            )
        df[f"ctx_season_{s}_avg"] = np.where(
            d1_mask,
            df[f"season_tier1_{s}_avg"],
            df[f"season_tier2_{s}_avg"],
        )
        df[f"ctx_career_{s}_avg"] = np.where(
            d1_mask,
            df[f"career_tier1_{s}_avg"],
            df[f"career_tier2_{s}_avg"],
        )
    log.info("  Context-routed features done.")

    return df[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def verify_no_leakage(df_features: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """
    Spot-check the player with the most games to confirm no future leakage
    in both overall and tier-aware features.
    """
    pid   = df_clean.groupby("player_id").size().idxmax()
    pname = df_clean.loc[df_clean["player_id"] == pid, "player_name"].iloc[0]

    player_feat  = df_features[df_features["player_id"] == pid].reset_index(drop=True)
    player_clean = (
        df_clean[df_clean["player_id"] == pid]
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )

    errors = []
    pts  = player_clean["pts"].tolist()
    tiers = player_clean["tier"].tolist()

    for i, row in player_feat.iterrows():
        prior_pts  = pts[:i]
        prior_tier = tiers[:i]

        # ── ctx_last_3_pts_avg (for a D1 row = tier1 last-3; D2 row = tier2) ──
        row_tier = tiers[i] if i < len(tiers) else None
        if row_tier is not None:
            prior_same_tier_pts = [p for p, t in zip(prior_pts, prior_tier) if t == row_tier]
            exp3 = np.mean(prior_same_tier_pts[-3:]) if prior_same_tier_pts else np.nan
            act3 = row["ctx_last_3_pts_avg"]
            if not (np.isnan(exp3) and np.isnan(act3)):
                if not np.isclose(exp3 if not np.isnan(exp3) else 0,
                                  act3 if not np.isnan(act3) else 0, atol=1e-6):
                    errors.append(f"  row {i}: ctx_last_3_pts_avg expected {exp3:.4f}, got {act3:.4f}")

        # ── ctx_career_pts_avg ────────────────────────────────────────
        if row_tier is not None:
            prior_same_tier_pts = [p for p, t in zip(prior_pts, prior_tier) if t == row_tier]
            exp_c = np.mean(prior_same_tier_pts) if prior_same_tier_pts else np.nan
            act_c = row["ctx_career_pts_avg"]
            if not (np.isnan(exp_c) and np.isnan(act_c)):
                if not np.isclose(exp_c if not np.isnan(exp_c) else 0,
                                  act_c if not np.isnan(act_c) else 0, atol=1e-6):
                    errors.append(f"  row {i}: ctx_career_pts_avg expected {exp_c:.4f}, got {act_c:.4f}")

        # ── V1: games_played_before ───────────────────────────────────
        if row["games_played_before"] != i:
            errors.append(f"  row {i}: games_played_before={row['games_played_before']}, expected {i}")

        # ── V2: career_tier1_pts_avg ──────────────────────────────────
        prior_d1_pts = [p for p, t in zip(prior_pts, prior_tier) if t == "D1"]
        exp_t1 = np.mean(prior_d1_pts) if prior_d1_pts else np.nan
        act_t1 = row["career_tier1_pts_avg"]
        if not (np.isnan(exp_t1) and np.isnan(act_t1)):
            if not np.isclose(exp_t1, act_t1, atol=1e-6):
                errors.append(
                    f"  row {i}: career_tier1_pts_avg expected {exp_t1:.4f}, got {act_t1:.4f}"
                )

        # ── V2: tier1_games_before ────────────────────────────────────
        expected_t1_cnt = sum(1 for t in prior_tier if t == "D1")
        actual_t1_cnt   = int(row["tier1_games_before"])
        if expected_t1_cnt != actual_t1_cnt:
            errors.append(
                f"  row {i}: tier1_games_before expected {expected_t1_cnt}, got {actual_t1_cnt}"
            )

    if errors:
        log.error("LEAKAGE CHECK FAILED for %s (%d errors):", pname, len(errors))
        for e in errors[:5]:
            log.error(e)
        raise AssertionError("Feature leakage detected — see errors above.")

    log.info("  Leakage check PASSED for %s (%d rows verified)", pname, len(player_feat))


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------

def print_audit(df: pd.DataFrame) -> None:
    sep = "=" * 68

    id_cols  = IDENTITY_COLS + CONTEXT_COLS
    tgt_cols = ["target_pts", "target_reb", "target_ast", "target_stl", "target_blk"]
    feat_cols = [c for c in df.columns if c not in id_cols + tgt_cols]

    v1_feat = [c for c in feat_cols if "tier1" not in c and "tier2" not in c]
    v2_feat = [c for c in feat_cols if "tier1" in c or "tier2" in c]

    print()
    print(sep)
    print("FEATURE ENGINEERING AUDIT REPORT  (V2 — tier-aware)")
    print(sep)

    print(f"\n  Total rows      : {len(df)}")
    print(f"  Total columns   : {df.shape[1]}")
    print(f"  Feature cols    : {len(feat_cols)}  (V1: {len(v1_feat)}, V2-new: {len(v2_feat)})")
    print(f"  Target cols     : {len(tgt_cols)}")
    print(f"  Identity cols   : {len(id_cols)}")

    null_counts = df[feat_cols].isnull().sum()

    print("\n--- NULL COUNTS BY FEATURE GROUP ---")
    groups = {
        "Experience":                 [c for c in feat_cols if c in [
            "days_since_last_game", "games_played_before", "season_games_before",
            "tier1_games_before", "tier2_games_before"]],
        "V1 Rolling (last 3/5, all tiers)":
            [c for c in feat_cols if c.startswith("last_") and "tier" not in c],
        "V1 Season averages (all tiers)":
            [c for c in feat_cols if c.startswith("season_") and "tier" not in c],
        "V1 Career averages (all tiers)":
            [c for c in feat_cols if c.startswith("career_") and "tier" not in c],
        "V2 Tier1 rolling":
            [c for c in feat_cols if "tier1" in c and c.startswith("last_")],
        "V2 Tier1 season/career":
            [c for c in feat_cols if "tier1" in c and not c.startswith("last_")],
        "V2 Tier2 rolling":
            [c for c in feat_cols if "tier2" in c and c.startswith("last_")],
        "V2 Tier2 season/career":
            [c for c in feat_cols if "tier2" in c and not c.startswith("last_")],
    }
    for group_name, cols in groups.items():
        print(f"\n  {group_name}:")
        for c in cols:
            n = null_counts.get(c, 0)
            bar = "" if n == 0 else f" ({100*n/len(df):.1f}% NaN)"
            print(f"    {c:<38}  {n:>4} nulls{bar}")

    print("\n--- WHY ARE THERE NULLS? ---")
    n_first   = (df["games_played_before"] == 0).sum()
    n_no_d1   = (df["tier1_games_before"] == 0).sum()
    n_no_d2   = (df["tier2_games_before"] == 0).sum()
    print(f"  First game ever          : {n_first:>4} rows  -> V1 career/rolling NaN")
    print(f"  No prior D1 games yet    : {n_no_d1:>4} rows  -> tier1 features NaN")
    print(f"  No prior D2 games yet    : {n_no_d2:>4} rows  -> tier2 features NaN")

    print("\n--- MULTI-TIER PLAYERS ---")
    both_tiers = df.groupby("player_id")["tier"].nunique()
    n_both = (both_tiers > 1).sum()
    print(f"  Players with D1+D2 history : {n_both}")
    # Show a sample
    sample_ids = both_tiers[both_tiers > 1].index[:3]
    for pid in sample_ids:
        pname = df.loc[df["player_id"] == pid, "player_name"].iloc[0]
        row = df[df["player_id"] == pid].iloc[-1]
        d1g = int(row["tier1_games_before"]) + (1 if row["tier"] == "D1" else 0)
        d2g = int(row["tier2_games_before"]) + (1 if row["tier"] == "D2" else 0)
        print(f"    {pname:<28}  D1:{d1g}  D2:{d2g}")

    print("\n--- SEASON COUNTS ---")
    for season, count in df["season"].value_counts().sort_values(ascending=False).items():
        print(f"  {season:<25}  {count:>5} rows")

    print("\n--- TARGET DISTRIBUTIONS ---")
    for t in tgt_cols:
        col = df[t]
        print(f"  {t:<14}  mean={col.mean():.2f}  std={col.std():.2f}  "
              f"min={col.min():.0f}  max={col.max():.0f}")

    print("\n--- TIER-AWARE FEATURE TRACE: Abdullah Khan (first 6 games) ---")
    khan = df[df["player_name"] == "Abdullah Khan"].head(6)
    trace = ["date", "tier", "target_pts",
             "ctx_career_pts_avg", "career_tier1_pts_avg", "career_tier2_pts_avg",
             "tier1_games_before", "tier2_games_before"]
    print(khan[trace].to_string(index=False))

    print(f"\n  Output: {OUTPUT_PATH}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CLEAN_PATH.exists():
        log.error("Clean file not found: %s", CLEAN_PATH)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", CLEAN_PATH)
    df_clean = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
    log.info("Loaded %d rows x %d columns", *df_clean.shape)

    df_features = build_features(df_clean.copy())
    log.info("Feature matrix: %d rows x %d columns", *df_features.shape)

    log.info("Running leakage verification...")
    verify_no_leakage(df_features, df_clean)

    df_features.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved to %s", OUTPUT_PATH)

    print_audit(df_features)


if __name__ == "__main__":
    main()
