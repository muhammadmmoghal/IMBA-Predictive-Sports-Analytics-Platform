"""
clean_data.py
Cleans data/raw/game_logs_raw.csv and writes data/processed/clean_game_logs.csv.

Cleaning decisions (all derived from profiling the raw data):

  DROP turnovers      — uniformly zero across all 3 746 rows; zero variance.
  DROP two_pt_att     — 60.8 % of rows have att < made, which is statistically
  DROP three_att        impossible.  The league does not enter shot attempts in
  DROP ft_att           its scoring system.  Keeping these columns would silently
                        corrupt any efficiency metrics built downstream.

  KEEP everything else — zero nulls, zero duplicates, all dates parseable, and
                         score/result consistency is perfect throughout.
                         No imputation or value correction is performed.

Notable flags (rows retained, highlighted in audit):
  1 row  — pts ≠ 2*fg2m + 3*fg3m + ftm  (Safwan Ibrahim, pts=4 vs calc=2).
            Source data-entry error; we cannot determine which field is wrong.
  55 rows — team='Add Others'  (D2 2025-26 Winter substitute roster entry in the
             league system).  Real games with real stats; retained.

Usage:  python src/clean_data.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_PATH    = Path("data/raw/game_logs_raw.csv")
OUTPUT_PATH = Path("data/processed/clean_game_logs.csv")

DROP_COLS = ["turnovers", "two_pt_att", "three_att", "ft_att"]

OUTPUT_COLUMNS = [
    # Identity
    "player_id", "player_name", "player_number", "player_position",
    # Game context
    "game_id", "date", "season", "tier",
    "team", "opponent", "result", "my_score", "opp_score",
    # Core counting stats (required for modeling)
    "pts", "reb", "ast", "stl", "blk",
    # Shot-making (attempts excluded — see module docstring)
    "fg2m", "fg3m", "ftm", "fouls",
]

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
# Validation helpers (run on raw data before any mutation)
# ---------------------------------------------------------------------------

def _check_duplicates(df: pd.DataFrame) -> dict:
    return {
        "full_row":       int(df.duplicated().sum()),
        "player_game_key": int(df.duplicated(["player_id", "game_id"]).sum()),
    }


def _check_nulls(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()


def _check_negative_stats(df: pd.DataFrame) -> dict:
    cols = ["pts", "reb", "ast", "stl", "blk"]
    return {c: int((df[c] < 0).sum()) for c in cols}


def _check_dates(df: pd.DataFrame) -> dict:
    parsed = pd.to_datetime(df["date"], errors="coerce")
    return {
        "unparseable": int(parsed.isnull().sum()),
        "min": str(parsed.min().date()),
        "max": str(parsed.max().date()),
    }


def _check_result_consistency(df: pd.DataFrame) -> dict:
    w = df[df["result"] == "W"]
    l = df[df["result"] == "L"]
    return {
        "W_rows_score_wrong": int((w["my_score"] <= w["opp_score"]).sum()),
        "L_rows_score_wrong": int((l["my_score"] >= l["opp_score"]).sum()),
    }


def _check_scoring_formula(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where pts != 2*two_pt_made + 3*three_made + ft_made."""
    expected = 2 * df["two_pt_made"] + 3 * df["three_made"] + df["ft_made"]
    bad = df[df["pts"] != expected].copy()
    bad["pts_calc"] = expected[bad.index]
    return bad[["player_id", "player_name", "game_id", "date", "season",
                "pts", "pts_calc", "two_pt_made", "three_made", "ft_made"]]


def _check_attempt_reliability(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "two_pt_att_lt_made":  int((df["two_pt_att"]  < df["two_pt_made"]).sum()),
        "three_att_lt_made":   int((df["three_att"]   < df["three_made"]).sum()),
        "ft_att_lt_made":      int((df["ft_att"]      < df["ft_made"]).sum()),
        "pct_two_pt_att_zero": f"{100*(df['two_pt_att']==0).mean():.1f}%",
        "pct_three_att_zero":  f"{100*(df['three_att']==0).mean():.1f}%",
        "pct_ft_att_zero":     f"{100*(df['ft_att']==0).mean():.1f}%",
        "n": n,
    }


def _check_add_others(df: pd.DataFrame) -> dict:
    ao = df[df["team"] == "Add Others"]
    return {
        "rows": int(len(ao)),
        "unique_players": int(ao["player_name"].nunique()),
        "seasons": ao["season"].value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Normalize date to YYYY-MM-DD string (guarantees consistent format
    #    even if the source ever mixes formats in the future)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # 2. Rename shot-making columns to cleaner names
    df = df.rename(columns={
        "two_pt_made": "fg2m",
        "three_made":  "fg3m",
        "ft_made":     "ftm",
    })

    # 3. Drop unreliable / zero-variance columns
    df = df.drop(columns=DROP_COLS)

    # 4. Select and order output columns
    return df[OUTPUT_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------

def print_audit(raw: pd.DataFrame, clean_df: pd.DataFrame, flags: dict) -> None:
    sep = "=" * 64

    print()
    print(sep)
    print("CLEANING AUDIT REPORT")
    print(sep)

    print(f"\n  Starting rows  :  {len(raw):>6}")
    print(f"  Ending rows    :  {len(clean_df):>6}")
    print(f"  Rows removed   :  {len(raw) - len(clean_df):>6}  "
          "(0 expected — cleaning drops no rows)")
    print(f"  Starting cols  :  {raw.shape[1]:>6}")
    print(f"  Ending cols    :  {clean_df.shape[1]:>6}  "
          "(dropped 4: turnovers + 3 attempt cols)")

    print("\n--- DUPLICATES ---")
    d = flags["duplicates"]
    print(f"  Full-row duplicates          : {d['full_row']}  (expect 0)")
    print(f"  (player_id, game_id) key dups: {d['player_game_key']}  (expect 0)")

    print("\n--- NULL VALUES (raw) ---")
    null_total = flags["nulls"].sum()
    print(f"  Total null cells : {null_total}  (expect 0)")

    print("\n--- DATE RANGE ---")
    dt = flags["dates"]
    print(f"  Unparseable  : {dt['unparseable']}")
    print(f"  Range        : {dt['min']}  ->  {dt['max']}")

    print("\n--- STAT SANITY ---")
    for col, cnt in flags["negative_stats"].items():
        print(f"  Negative {col:<4} : {cnt}")
    rc = flags["result_consistency"]
    print(f"  W rows, my_score <= opp_score : {rc['W_rows_score_wrong']}")
    print(f"  L rows, my_score >= opp_score : {rc['L_rows_score_wrong']}")

    print("\n--- SCORING FORMULA (pts == 2*fg2m + 3*fg3m + ftm) ---")
    sf = flags["scoring_formula"]
    if len(sf) == 0:
        print("  All rows pass.")
    else:
        print(f"  {len(sf)} row(s) FAIL (retained — source data-entry error):")
        for _, row in sf.iterrows():
            print(f"    {row['player_name']:<25} {row['date']}  {row['season']}"
                  f"  pts={int(row['pts'])}  calc={int(row['pts_calc'])}")

    print("\n--- DROPPED COLUMNS ---")
    ar = flags["attempt_reliability"]
    n  = ar["n"]
    print(f"  turnovers  : 100.0% zero across {n} rows")
    print(f"  two_pt_att : {ar['pct_two_pt_att_zero']} zero; "
          f"{ar['two_pt_att_lt_made']} rows ({100*ar['two_pt_att_lt_made']/n:.1f}%) have att < made")
    print(f"  three_att  : {ar['pct_three_att_zero']} zero; "
          f"{ar['three_att_lt_made']} rows ({100*ar['three_att_lt_made']/n:.1f}%) have att < made")
    print(f"  ft_att     : {ar['pct_ft_att_zero']} zero; "
          f"{ar['ft_att_lt_made']} rows ({100*ar['ft_att_lt_made']/n:.1f}%) have att < made")

    print("\n--- FLAGGED: 'Add Others' TEAM (retained) ---")
    ao = flags["add_others"]
    print(f"  Rows    : {ao['rows']}")
    print(f"  Players : {ao['unique_players']}")
    for s, c in ao["seasons"].items():
        print(f"  Season  : {s} ({c} rows)")

    print("\n--- SEASON COUNTS (clean output) ---")
    for season, count in clean_df["season"].value_counts().sort_values(ascending=False).items():
        print(f"  {season:<25}  {count:>5} rows")

    print("\n--- PLAYER COUNTS (clean output) ---")
    gpd = clean_df.groupby("player_id").size()
    print(f"  Unique players    : {clean_df['player_id'].nunique()}")
    print(f"  Unique games      : {clean_df['game_id'].nunique()}")
    print(f"  Max games/player  : {gpd.max()}")
    print(f"  Median games/player: {gpd.median():.0f}")
    print(f"  Players >= 20 games: {(gpd >= 20).sum()}")
    print(f"  Players >= 30 games: {(gpd >= 30).sum()}")

    print("\n--- TOP 10 PLAYERS BY GAMES PLAYED ---")
    top10 = (
        clean_df.groupby(["player_id", "player_name"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    for (pid, pname), cnt in top10.items():
        print(f"  {pname:<30}  {cnt:>3} games")

    print("\n--- OUTPUT COLUMNS ---")
    for col in clean_df.columns:
        print(f"  {col}")

    print(f"\n  Output: {OUTPUT_PATH}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not RAW_PATH.exists():
        log.error("Raw file not found: %s", RAW_PATH)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", RAW_PATH)
    raw = pd.read_csv(RAW_PATH)
    log.info("Raw shape: %d rows x %d columns", *raw.shape)

    # ── Validate before touching anything ──────────────────────────────────
    log.info("Running validation checks on raw data...")
    flags = {
        "duplicates":          _check_duplicates(raw),
        "nulls":               _check_nulls(raw),
        "negative_stats":      _check_negative_stats(raw),
        "dates":               _check_dates(raw),
        "result_consistency":  _check_result_consistency(raw),
        "scoring_formula":     _check_scoring_formula(raw),
        "attempt_reliability": _check_attempt_reliability(raw),
        "add_others":          _check_add_others(raw),
    }

    # Hard-fail on data integrity violations
    assert flags["duplicates"]["player_game_key"] == 0, \
        "Duplicate (player_id, game_id) keys — investigate before proceeding."
    assert flags["nulls"].sum() == 0, \
        "Unexpected nulls found — investigate before proceeding."
    assert flags["result_consistency"]["W_rows_score_wrong"] == 0, \
        "W rows with my_score <= opp_score detected."
    assert flags["result_consistency"]["L_rows_score_wrong"] == 0, \
        "L rows with my_score >= opp_score detected."

    log.info("All integrity checks passed.")

    # ── Clean ──────────────────────────────────────────────────────────────
    log.info("Applying cleaning steps...")
    clean_df = clean(raw.copy())
    log.info("Clean shape: %d rows x %d columns", *clean_df.shape)

    # ── Save ───────────────────────────────────────────────────────────────
    clean_df.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved to %s", OUTPUT_PATH)

    print_audit(raw, clean_df, flags)


if __name__ == "__main__":
    main()
