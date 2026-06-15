"""
feature_engineering.py
Builds the pre-game feature matrix from data/processed/clean_game_logs.csv
and writes data/processed/features.csv.

All features are computed using only games that occurred BEFORE the current
game for each player.  The primary mechanism is shift(1) applied inside each
per-player group before any rolling or expanding operation.  This is verified
explicitly in the leakage check at the end of the audit.

Sort order: (player_id, date, game_id)
  - game_id is the tiebreaker for same-day double/triple-headers (63 rows
    across 3 746 total).  For the second/third game on a given day the first
    same-day game IS included in that row's rolling features, which is
    correct — the player had already played it.

NaN handling:
  - A player's very first game has NaN for every rolling and expanding feature
    (no prior data).  These rows are retained in features.csv as valid
    prediction targets; the modeling step decides whether to filter them.
  - Rolling windows use min_periods=1, so last_3_* shows a 1- or 2-game
    partial average before 3 prior games exist.
  - season_* features are NaN only for a player's first game of each season.
  - career_* features are NaN only for a player's very first game ever.
  - days_since_last_game is NaN for each player's first game.

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

STATS = ["pts", "reb", "ast", "stl", "blk"]   # stats to roll over
ROLL_WINDOWS = [3, 5]

# Columns passed through to the output unchanged
IDENTITY_COLS = ["player_id", "player_name", "game_id", "date", "season"]
CONTEXT_COLS  = ["tier", "tier_d1", "team", "opponent"]

# Output column order (features.csv)
FEATURE_COLS = (
    IDENTITY_COLS
    + CONTEXT_COLS
    + ["days_since_last_game", "games_played_before", "season_games_before"]
    + [f"last_{n}_{s}_avg" for n in ROLL_WINDOWS for s in STATS]
    + [f"season_{s}_avg" for s in STATS]
    + [f"career_{s}_avg" for s in STATS]
    + ["target_pts", "target_reb", "target_ast"]
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
# Feature builders
# ---------------------------------------------------------------------------

def _rolling_avg(series: pd.Series, n: int) -> pd.Series:
    """
    Per-player rolling mean of the N games immediately before each row.
    shift(1) ensures the current game is excluded.
    min_periods=1 allows partial windows for the first few games.
    """
    return series.shift(1).rolling(n, min_periods=1).mean()


def _expanding_avg(series: pd.Series) -> pd.Series:
    """
    Per-player expanding (cumulative) mean of all games before each row.
    shift(1) excludes the current game.
    """
    return series.shift(1).expanding(min_periods=1).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the clean game-log DataFrame (sorted chronologically per player)
    and returns a feature matrix with no future-data leakage.
    """
    log.info("Building features on %d rows...", len(df))

    # ── Sort: primary=player, secondary=date, tertiary=game_id (stable) ──
    df = df.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)

    # ── Tier encoding (known before the game) ────────────────────────────
    df["tier_d1"] = (df["tier"] == "D1").astype(int)

    # ── Targets (current game outcomes — what we want to predict) ─────────
    df["target_pts"] = df["pts"]
    df["target_reb"] = df["reb"]
    df["target_ast"] = df["ast"]

    # ── Rolling averages (last 3 and 5 games before current) ─────────────
    for stat in STATS:
        grp = df.groupby("player_id", sort=False)[stat]
        for n in ROLL_WINDOWS:
            df[f"last_{n}_{stat}_avg"] = grp.transform(
                lambda x, _n=n: _rolling_avg(x, _n)
            )
    log.info("  Rolling averages done.")

    # ── Season averages before current game ──────────────────────────────
    for stat in STATS:
        df[f"season_{stat}_avg"] = (
            df.groupby(["player_id", "season"], sort=False)[stat]
            .transform(_expanding_avg)
        )
    log.info("  Season averages done.")

    # ── Career averages before current game ──────────────────────────────
    for stat in STATS:
        df[f"career_{stat}_avg"] = (
            df.groupby("player_id", sort=False)[stat]
            .transform(_expanding_avg)
        )
    log.info("  Career averages done.")

    # ── Experience features ───────────────────────────────────────────────
    # cumcount() gives 0 for a player's first row, 1 for second, etc.
    # = number of games before the current game (exactly what we want)
    df["games_played_before"]  = df.groupby("player_id").cumcount()
    df["season_games_before"]  = df.groupby(["player_id", "season"]).cumcount()
    log.info("  Experience features done.")

    # ── Days since last game (NaN for each player's first appearance) ─────
    prev_date = df.groupby("player_id")["date"].shift(1)
    df["days_since_last_game"] = (df["date"] - prev_date).dt.days
    log.info("  Days since last game done.")

    return df[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def verify_no_leakage(df_features: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """
    Spot-check the player with the most games to confirm that each row's
    rolling and expanding features equal the expected hand-computed values
    derived only from prior rows.
    """
    pid = df_clean.groupby("player_id").size().idxmax()
    pname = df_clean.loc[df_clean["player_id"] == pid, "player_name"].iloc[0]

    player_feat  = df_features[df_features["player_id"] == pid].reset_index(drop=True)
    player_clean = (
        df_clean[df_clean["player_id"] == pid]
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )

    errors = []
    pts = player_clean["pts"].tolist()

    for i, row in player_feat.iterrows():
        prior_pts = pts[:i]   # games before row i

        # last_3_pts_avg check
        expected_3 = np.mean(prior_pts[-3:]) if prior_pts else np.nan
        actual_3   = row["last_3_pts_avg"]
        if not (np.isnan(expected_3) and np.isnan(actual_3)):
            if not np.isclose(expected_3, actual_3, atol=1e-6):
                errors.append(f"  row {i}: last_3_pts_avg expected {expected_3:.4f}, got {actual_3:.4f}")

        # career_pts_avg check
        expected_c = np.mean(prior_pts) if prior_pts else np.nan
        actual_c   = row["career_pts_avg"]
        if not (np.isnan(expected_c) and np.isnan(actual_c)):
            if not np.isclose(expected_c, actual_c, atol=1e-6):
                errors.append(f"  row {i}: career_pts_avg expected {expected_c:.4f}, got {actual_c:.4f}")

        # games_played_before check
        if row["games_played_before"] != i:
            errors.append(f"  row {i}: games_played_before={row['games_played_before']}, expected {i}")

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
    sep = "=" * 66

    # Separate feature cols from identity/target
    id_cols  = IDENTITY_COLS + CONTEXT_COLS
    tgt_cols = ["target_pts", "target_reb", "target_ast"]
    feat_cols = [c for c in df.columns if c not in id_cols + tgt_cols]

    print()
    print(sep)
    print("FEATURE ENGINEERING AUDIT REPORT")
    print(sep)

    print(f"\n  Total rows    : {len(df)}")
    print(f"  Total columns : {df.shape[1]}")
    print(f"  Feature cols  : {len(feat_cols)}")
    print(f"  Target cols   : {len(tgt_cols)}")
    print(f"  Identity cols : {len(id_cols)}")

    print("\n--- NULL COUNTS BY FEATURE ---")
    null_counts = df[feat_cols].isnull().sum()
    max_null = null_counts.max()

    groups = {
        "Rolling (last 3 / 5 games)": [c for c in feat_cols if c.startswith("last_")],
        "Season averages":            [c for c in feat_cols if c.startswith("season_")],
        "Career averages":            [c for c in feat_cols if c.startswith("career_")],
        "Experience":                 ["days_since_last_game",
                                       "games_played_before",
                                       "season_games_before"],
        "Context":                    [c for c in CONTEXT_COLS if c in feat_cols],
    }
    for group_name, cols in groups.items():
        print(f"\n  {group_name}:")
        for c in cols:
            n = null_counts.get(c, 0)
            bar = "" if n == 0 else f" ({100*n/len(df):.1f}% NaN)"
            print(f"    {c:<30}  {n:>4} nulls{bar}")

    print("\n--- WHY ARE THERE NULLS? ---")
    n_first_game    = (df["games_played_before"] == 0).sum()
    n_first_season  = (df["season_games_before"] == 0).sum()
    print(f"  Player first game ever          : {n_first_game} rows  "
          f"-> career_* and rolling features are NaN")
    print(f"  Player first game of any season : {n_first_season} rows  "
          f"-> season_* is NaN for these rows")
    print(f"  (These rows are retained for use as prediction targets.)")

    print("\n--- SEASON COUNTS ---")
    for season, count in df["season"].value_counts().sort_values(ascending=False).items():
        print(f"  {season:<25}  {count:>5} rows")

    print("\n--- PLAYER COUNTS ---")
    gpd = df.groupby("player_id").size()
    print(f"  Unique players     : {df['player_id'].nunique()}")
    print(f"  Unique games       : {df['game_id'].nunique()}")
    print(f"  Players >= 20 rows : {(gpd >= 20).sum()}")
    print(f"  Max rows/player    : {gpd.max()}")

    print("\n--- TIER DISTRIBUTION ---")
    for tier, cnt in df["tier"].value_counts().items():
        print(f"  {tier}: {cnt} rows  (tier_d1 = {int(df.loc[df['tier']==tier,'tier_d1'].iloc[0])})")

    print("\n--- TARGET DISTRIBUTIONS ---")
    for t in tgt_cols:
        col = df[t]
        print(f"  {t:<14}  mean={col.mean():.2f}  "
              f"std={col.std():.2f}  min={col.min()}  max={col.max()}")

    print("\n--- TOP 10 PLAYERS BY GAME COUNT ---")
    top10 = df.groupby(["player_id","player_name"]).size().sort_values(ascending=False).head(10)
    for (_, pname), cnt in top10.items():
        print(f"  {pname:<30}  {cnt:>3} rows")

    print("\n--- EXAMPLE FEATURE TRACE: Abdullah Khan (first 8 games) ---")
    khan = df[df["player_name"] == "Abdullah Khan"].head(8)
    trace_cols = ["date", "season", "target_pts", "target_reb", "target_ast",
                  "last_3_pts_avg", "last_5_pts_avg", "season_pts_avg",
                  "career_pts_avg", "games_played_before", "days_since_last_game"]
    print(khan[trace_cols].to_string(index=False))

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

    # Build features
    df_features = build_features(df_clean.copy())
    log.info("Feature matrix: %d rows x %d columns", *df_features.shape)

    # Leakage verification
    log.info("Running leakage verification...")
    verify_no_leakage(df_features, df_clean)

    # Save
    df_features.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved to %s", OUTPUT_PATH)

    print_audit(df_features)


if __name__ == "__main__":
    main()
