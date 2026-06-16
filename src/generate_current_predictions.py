"""
generate_current_predictions.py  (V5 — confidence scores + prediction ranges)
Generates next-game predictions for every active player in the 2026 Summer
divisions and writes leaderboard CSVs.

Divisions covered:
  D2 Comp 2026 Summer  ->  Tier 1 (uses tier1/D1 model context)
  D2 Rec  2026 Summer  ->  Tier 2 (uses tier2/D2 model context)

Workflow:
  1. Scrape played 2026 Summer box scores from the API (inline).
  2. Map those games onto the historical tier schema (D2 Comp -> D1, D2 Rec -> D2).
  3. Combine with existing clean_game_logs.csv as full player history.
  4. For each active player × division, build V2 features and predict.
  5. Write output CSVs and leaderboards; print audit report.

Players who have appeared in both divisions get one prediction row per division.
Players with zero combined history are skipped (logged).

Outputs:
  data/processed/current_predictions_all.csv
  data/processed/current_predictions_d2_comp.csv
  data/processed/current_predictions_d2_rec.csv
  reports/top_projected_scorers.csv
  reports/top_projected_rebounders.csv
  reports/top_projected_assists.csv
  reports/top_double_double_candidates.csv

Usage:  python src/generate_current_predictions.py
"""

import sys
import time
import logging
import warnings
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import joblib

# Suppress sklearn joblib parallel config warnings (benign, fired by RF n_jobs=-1)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Reuse feature-building and confidence helpers from predict.py
sys.path.insert(0, str(Path(__file__).parent))
from predict import (
    build_next_game_features, V2_FEATURE_COLS, STATS,
    compute_confidence_score, confidence_score_to_label, compute_stat_ranges,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL      = "https://www.imbaonline.com"
REQUEST_DELAY = 0.35
MAX_RETRIES   = 3
RETRY_DELAY   = 2.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# 2026 Summer seasons and their tier mapping in the historical schema
# (D2 Comp = the competitive division, maps to D1/tier1 context)
# (D2 Rec  = the recreational division, maps to D2/tier2 context)
SUMMER_SEASONS: dict[str, str] = {
    "D2 Comp 2026 Summer": "D1",
    "D2 Rec 2026 Summer":  "D2",
}

TIER_TO_TIER_LABEL = {"D1": "tier1", "D2": "tier2"}
TIER_TO_DISPLAY    = {"D1": "Tier 1 (D2 Comp)", "D2": "Tier 2 (D2 Rec)"}

CLEAN_PATH  = Path("data/processed/clean_game_logs.csv")
MODELS_DIR  = Path("models")
DATA_DIR    = Path("data/processed")
REPORTS_DIR = Path("reports")

OUTPUT_ALL       = DATA_DIR    / "current_predictions_all.csv"
OUTPUT_D2_COMP   = DATA_DIR    / "current_predictions_d2_comp.csv"
OUTPUT_D2_REC    = DATA_DIR    / "current_predictions_d2_rec.csv"
TOP_SCORERS      = REPORTS_DIR / "top_projected_scorers.csv"
TOP_REBOUNDERS   = REPORTS_DIR / "top_projected_rebounders.csv"
TOP_ASSISTS      = REPORTS_DIR / "top_projected_assists.csv"
TOP_DOUBLE_DOUBLE = REPORTS_DIR / "top_double_double_candidates.csv"

LEADERBOARD_SIZE = 20

MODEL_FILES = {
    "pts": MODELS_DIR / "pts_model.pkl",
    "reb": MODELS_DIR / "reb_model.pkl",
    "ast": MODELS_DIR / "ast_model.pkl",
    "stl": MODELS_DIR / "stl_model.pkl",
    "blk": MODELS_DIR / "blk_model.pkl",
}

PROB_MODEL_FILES = {
    "pts_10_plus":   MODELS_DIR / "pts_10_plus_model.pkl",
    "pts_15_plus":   MODELS_DIR / "pts_15_plus_model.pkl",
    "pts_20_plus":   MODELS_DIR / "pts_20_plus_model.pkl",
    "reb_5_plus":    MODELS_DIR / "reb_5_plus_model.pkl",
    "reb_10_plus":   MODELS_DIR / "reb_10_plus_model.pkl",
    "ast_5_plus":    MODELS_DIR / "ast_5_plus_model.pkl",
    "double_double": MODELS_DIR / "double_double_model.pkl",
}

# Final output column order
OUTPUT_COLUMNS = [
    "player_id", "player_name", "division", "tier", "team",
    "predicted_pts", "predicted_reb", "predicted_ast", "predicted_stl", "predicted_blk",
    "pts_low", "pts_high", "reb_low", "reb_high",
    "ast_low", "ast_high", "stl_low", "stl_high", "blk_low", "blk_high",
    "prob_10_plus_pts", "prob_15_plus_pts", "prob_20_plus_pts",
    "prob_5_plus_reb", "prob_10_plus_reb",
    "prob_5_plus_ast",
    "prob_double_double",
    "confidence_level",
    "confidence_score",
    "games_played_history",
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
# HTTP helper
# ---------------------------------------------------------------------------

def _get(session: requests.Session, url: str, params: dict | None = None) -> dict | list | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                log.error("FAILED %s %s — %s", url, params or "", exc)
                return None
            log.warning("Retry %d/%d for %s: %s", attempt, MAX_RETRIES, url, exc)
            time.sleep(RETRY_DELAY * attempt)
    return None


# ---------------------------------------------------------------------------
# Scrape 2026 Summer box scores
# ---------------------------------------------------------------------------

def _parse_summer_stat(stat: dict, game: dict, season: str, tier: str) -> dict | None:
    """
    Parse one playerStats entry from a 2026 Summer box score into a row that
    matches the clean_game_logs.csv schema exactly.

    tier: "D1" for D2 Comp, "D2" for D2 Rec (historical tier mapping).
    """
    player    = stat.get("player") or {}
    team_meta = stat.get("team") or {}
    team_id   = stat.get("teamId", "")

    # Skip rows with no player ID
    player_id = stat.get("playerId", "")
    if not player_id:
        return None

    is_home   = (team_id == game.get("homeTeamId"))
    my_score  = game.get("homeScore") if is_home else game.get("awayScore")
    opp_score = game.get("awayScore") if is_home else game.get("homeScore")
    opp_team  = (game.get("awayTeam") or {}) if is_home else (game.get("homeTeam") or {})
    opponent  = opp_team.get("name", "")

    if my_score is not None and opp_score is not None:
        result = "W" if my_score > opp_score else ("L" if my_score < opp_score else "T")
    else:
        result = ""

    date_raw = game.get("date", "")
    date_str = date_raw[:10] if date_raw else ""

    # Core stats — guard against None
    def _i(v): return int(v) if v is not None else 0

    return {
        "player_id":       player_id,
        "player_name":     player.get("name", ""),
        "player_number":   player.get("number"),
        "player_position": player.get("position", ""),
        "game_id":         stat.get("gameId", ""),
        "date":            date_str,
        "season":          season,
        "tier":            tier,
        "team":            team_meta.get("name", ""),
        "opponent":        opponent,
        "result":          result,
        "my_score":        my_score,
        "opp_score":       opp_score,
        "pts":             _i(stat.get("points")),
        "reb":             _i(stat.get("rebounds")),
        "ast":             _i(stat.get("assists")),
        "stl":             _i(stat.get("steals")),
        "blk":             _i(stat.get("blocks")),
        "fg2m":            _i(stat.get("twoPtMade")),
        "fg3m":            _i(stat.get("threeMade")),
        "ftm":             _i(stat.get("ftMade")),
        "fouls":           _i(stat.get("fouls")),
    }


def scrape_summer_2026(session: requests.Session) -> tuple[pd.DataFrame, dict]:
    """
    Fetches all played box scores for the 2026 Summer divisions.
    Returns (rows_df, meta) where meta has per-season game counts.
    """
    all_rows: list[dict] = []
    meta: dict = {}
    seen_keys: set[tuple] = set()   # (player_id, game_id)

    for season, tier in SUMMER_SEASONS.items():
        log.info("Fetching game list for: %s", season)
        data = _get(session, f"{BASE_URL}/api/games", params={"season": season})
        time.sleep(REQUEST_DELAY)

        if not data:
            log.warning("  No data returned for %s", season)
            meta[season] = {"total": 0, "played": 0, "rows": 0}
            continue

        played = [g for g in data if g.get("played") is True]
        log.info("  %d played / %d total games", len(played), len(data))
        meta[season] = {"total": len(data), "played": len(played), "rows": 0}

        for idx, game_meta in enumerate(played, 1):
            game_id = game_meta["id"]
            box = _get(session, f"{BASE_URL}/api/games/{game_id}")
            time.sleep(REQUEST_DELAY)

            if box is None:
                log.warning("  Failed to fetch box score: %s", game_id)
                continue

            for stat in box.get("playerStats", []):
                pid  = stat.get("playerId", "")
                gid  = stat.get("gameId", "")
                key  = (pid, gid)
                if key in seen_keys or not pid:
                    continue
                seen_keys.add(key)
                row = _parse_summer_stat(stat, box, season, tier)
                if row:
                    all_rows.append(row)

            if idx % 10 == 0 or idx == len(played):
                log.info("  [%d/%d] %d rows so far", idx, len(played), len(all_rows))

        meta[season]["rows"] = sum(1 for r in all_rows if r["season"] == season)

    if not all_rows:
        return pd.DataFrame(), meta

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df, meta


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_models() -> tuple[dict, dict]:
    models = {}
    for stat, path in MODEL_FILES.items():
        if not path.exists():
            log.error("Regression model not found: %s — run train_model.py first", path)
            sys.exit(1)
        models[stat] = joblib.load(path)

    prob_models = {}
    for name, path in PROB_MODEL_FILES.items():
        if not path.exists():
            log.error("Probability model not found: %s — run train_model.py first", path)
            sys.exit(1)
        prob_models[name] = joblib.load(path)

    log.info("Loaded %d regression + %d probability models", len(models), len(prob_models))
    return models, prob_models


# ---------------------------------------------------------------------------
# Predict one player × division
# ---------------------------------------------------------------------------

def predict_one(
    player_df: pd.DataFrame,
    player_id: str,
    player_name: str,
    division: str,
    tier: str,       # "D1" or "D2"
    team: str,
    models: dict,
    prob_models: dict,
) -> dict | None:
    """
    Build V2 features and run all models for one player in one division context.
    Returns None if insufficient history to build features.
    Includes confidence score and prediction ranges (V5).
    """
    if len(player_df) == 0:
        return None

    tier_label = TIER_TO_TIER_LABEL[tier]  # "tier1" or "tier2"
    tier_d1    = 1 if tier == "D1" else 0

    try:
        feat = build_next_game_features(player_df, tier_d1_val=tier_d1)
    except Exception as exc:
        log.warning("  Feature build failed for %s (%s): %s", player_name, division, exc)
        return None

    X = pd.DataFrame([feat])[V2_FEATURE_COLS]

    preds = {stat: max(0.0, float(models[stat].predict(X)[0]))
             for stat in STATS}

    probs: dict = {}
    for name, model in prob_models.items():
        try:
            probs[name] = round(float(model.predict_proba(X)[0, 1]) * 100, 1)
        except Exception:
            probs[name] = None

    # V5: confidence score and prediction ranges
    tier_games_df = player_df[player_df["tier"] == tier]
    conf_score    = compute_confidence_score(feat, tier_games_df, tier_label)
    conf_label    = confidence_score_to_label(conf_score)
    ranges        = compute_stat_ranges(preds, tier_games_df, conf_score)

    return {
        "player_id":       player_id,
        "player_name":     player_name,
        "division":        division,
        "tier":            TIER_TO_DISPLAY[tier],
        "team":            team,
        "predicted_pts":   round(preds["pts"],  1),
        "predicted_reb":   round(preds["reb"],  1),
        "predicted_ast":   round(preds["ast"],  1),
        "predicted_stl":   round(preds["stl"],  1),
        "predicted_blk":   round(preds["blk"],  1),
        "pts_low":  ranges["pts"][0],  "pts_high":  ranges["pts"][1],
        "reb_low":  ranges["reb"][0],  "reb_high":  ranges["reb"][1],
        "ast_low":  ranges["ast"][0],  "ast_high":  ranges["ast"][1],
        "stl_low":  ranges["stl"][0],  "stl_high":  ranges["stl"][1],
        "blk_low":  ranges["blk"][0],  "blk_high":  ranges["blk"][1],
        "prob_10_plus_pts":   probs.get("pts_10_plus"),
        "prob_15_plus_pts":   probs.get("pts_15_plus"),
        "prob_20_plus_pts":   probs.get("pts_20_plus"),
        "prob_5_plus_reb":    probs.get("reb_5_plus"),
        "prob_10_plus_reb":   probs.get("reb_10_plus"),
        "prob_5_plus_ast":    probs.get("ast_5_plus"),
        "prob_double_double": probs.get("double_double"),
        "confidence_level":   conf_label,
        "confidence_score":   conf_score,
        "games_played_history": int(feat["games_played_before"]),
    }


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------

def print_audit(
    df_all: pd.DataFrame,
    summer_meta: dict,
    skipped: list[tuple],
    comp_players: set,
    rec_players: set,
) -> None:
    sep = "=" * 72

    both = comp_players & rec_players
    only_comp = comp_players - rec_players
    only_rec  = rec_players - comp_players

    print()
    print(sep)
    print("CURRENT SEASON BATCH PREDICTIONS AUDIT  (V5 — confidence scores + ranges)")
    print(sep)

    # ── Scrape summary ─────────────────────────────────────────────────────
    print("\n  2026 SUMMER SEASON GAME DATA")
    print(f"  {'-'*68}")
    for season, info in summer_meta.items():
        print(f"  {season:<28}  played={info['played']:>3}  total={info['total']:>3}  "
              f"player-game rows={info['rows']:>4}")

    # ── Player counts ──────────────────────────────────────────────────────
    print(f"\n  ACTIVE PLAYER COUNTS")
    print(f"  {'-'*68}")
    print(f"  D2 Comp 2026 Summer players  : {len(comp_players)}")
    print(f"  D2 Rec  2026 Summer players  : {len(rec_players)}")
    print(f"  Players in BOTH divisions    : {len(both)}")
    print(f"  Only D2 Comp                 : {len(only_comp)}")
    print(f"  Only D2 Rec                  : {len(only_rec)}")
    print(f"  Total prediction rows        : {len(df_all)}")
    print(f"  Skipped players              : {len(skipped)}")

    if skipped:
        print(f"\n  SKIPPED PLAYERS:")
        for pname, division, reason in skipped[:20]:
            print(f"    {pname:<28}  {division:<28}  {reason}")

    # ── Confidence distribution ────────────────────────────────────────────
    print(f"\n  CONFIDENCE DISTRIBUTION")
    print(f"  {'-'*68}")
    for div in ["D2 Comp 2026 Summer", "D2 Rec 2026 Summer"]:
        sub = df_all[df_all["division"] == div]
        if sub.empty:
            continue
        counts = sub["confidence_level"].value_counts()
        scores = sub["confidence_score"]
        print(f"  {div}:")
        for lvl in ["High", "Medium", "Low"]:
            c = counts.get(lvl, 0)
            print(f"    {lvl:<8}  {c:>3} players")
        print(f"    Score   — mean={scores.mean():.1f}  median={scores.median():.0f}  "
              f"min={scores.min()}  max={scores.max()}")

    # ── Games-played-history distribution ─────────────────────────────────
    print(f"\n  HISTORY DEPTH (games_played_history)")
    print(f"  {'-'*68}")
    for div in ["D2 Comp 2026 Summer", "D2 Rec 2026 Summer"]:
        sub = df_all[df_all["division"] == div]["games_played_history"]
        if sub.empty:
            continue
        print(f"  {div}: mean={sub.mean():.1f}  median={sub.median():.0f}  "
              f"min={sub.min()}  max={sub.max()}")

    # ── Top 10 projected scorers by division ───────────────────────────────
    print(f"\n  TOP 10 PROJECTED SCORERS — D2 COMP 2026 SUMMER")
    print(f"  {'-'*68}")
    comp_df = df_all[df_all["division"] == "D2 Comp 2026 Summer"] \
                .sort_values("predicted_pts", ascending=False)
    print(f"  {'Player':<24}  {'Team':<16}  {'PTS':>5}  {'REB':>5}  "
          f"{'AST':>5}  {'Conf':<6}  {'10+PTS':>7}")
    print("  " + "-" * 68)
    for _, row in comp_df.head(10).iterrows():
        p10 = f"{row['prob_10_plus_pts']:.0f}%" if row['prob_10_plus_pts'] is not None else "N/A"
        print(f"  {row['player_name']:<24}  {row['team']:<16}  "
              f"{row['predicted_pts']:>5.1f}  {row['predicted_reb']:>5.1f}  "
              f"{row['predicted_ast']:>5.1f}  {row['confidence_level']:<6}  {p10:>7}")

    print(f"\n  TOP 10 PROJECTED SCORERS — D2 REC 2026 SUMMER")
    print(f"  {'-'*68}")
    rec_df = df_all[df_all["division"] == "D2 Rec 2026 Summer"] \
               .sort_values("predicted_pts", ascending=False)
    print(f"  {'Player':<24}  {'Team':<16}  {'PTS':>5}  {'REB':>5}  "
          f"{'AST':>5}  {'Conf':<6}  {'10+PTS':>7}")
    print("  " + "-" * 68)
    for _, row in rec_df.head(10).iterrows():
        p10 = f"{row['prob_10_plus_pts']:.0f}%" if row['prob_10_plus_pts'] is not None else "N/A"
        print(f"  {row['player_name']:<24}  {row['team']:<16}  "
              f"{row['predicted_pts']:>5.1f}  {row['predicted_reb']:>5.1f}  "
              f"{row['predicted_ast']:>5.1f}  {row['confidence_level']:<6}  {p10:>7}")

    # ── Top double-double candidates ───────────────────────────────────────
    print(f"\n  TOP 10 DOUBLE-DOUBLE CANDIDATES (all divisions)")
    print(f"  {'-'*68}")
    dd_df = df_all.sort_values("prob_double_double", ascending=False)
    print(f"  {'Player':<24}  {'Division':<24}  {'Dbl-Dbl%':>8}  "
          f"{'PTS':>5}  {'REB':>5}")
    print("  " + "-" * 66)
    for _, row in dd_df.head(10).iterrows():
        dd = f"{row['prob_double_double']:.0f}%" if row['prob_double_double'] is not None else "N/A"
        print(f"  {row['player_name']:<24}  {row['division']:<24}  {dd:>8}  "
              f"{row['predicted_pts']:>5.1f}  {row['predicted_reb']:>5.1f}")

    # ── Output files ───────────────────────────────────────────────────────
    print(f"\n  OUTPUT FILES")
    print(f"  {'-'*68}")
    print(f"  {str(OUTPUT_ALL):<55} {len(df_all):>4} rows")
    comp_rows = len(df_all[df_all['division'] == 'D2 Comp 2026 Summer'])
    rec_rows  = len(df_all[df_all['division'] == 'D2 Rec 2026 Summer'])
    print(f"  {str(OUTPUT_D2_COMP):<55} {comp_rows:>4} rows")
    print(f"  {str(OUTPUT_D2_REC):<55} {rec_rows:>4} rows")
    print(f"  {str(TOP_SCORERS):<55} top {LEADERBOARD_SIZE}")
    print(f"  {str(TOP_REBOUNDERS):<55} top {LEADERBOARD_SIZE}")
    print(f"  {str(TOP_ASSISTS):<55} top {LEADERBOARD_SIZE}")
    print(f"  {str(TOP_DOUBLE_DOUBLE):<55} top {LEADERBOARD_SIZE}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEAN_PATH.exists():
        log.error("clean_game_logs.csv not found: %s — run clean_data.py first", CLEAN_PATH)
        sys.exit(1)

    # ── Load historical data ──────────────────────────────────────────────
    log.info("Loading historical game logs from %s", CLEAN_PATH)
    df_hist = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
    log.info("  Historical rows: %d", len(df_hist))

    # ── Load models ───────────────────────────────────────────────────────
    models, prob_models = load_models()

    # ── Scrape 2026 Summer box scores ─────────────────────────────────────
    session = requests.Session()
    session.headers.update(HEADERS)

    log.info("Scraping 2026 Summer box scores...")
    df_summer, summer_meta = scrape_summer_2026(session)

    if df_summer.empty:
        log.error("No 2026 Summer data scraped — cannot continue.")
        sys.exit(1)

    log.info("  Summer rows scraped: %d (across %d seasons)",
             len(df_summer), len(SUMMER_SEASONS))

    # ── Identify active players per division ──────────────────────────────
    # Active = played at least 1 game in the 2026 Summer season
    comp_season = "D2 Comp 2026 Summer"
    rec_season  = "D2 Rec 2026 Summer"

    comp_df = df_summer[df_summer["season"] == comp_season]
    rec_df  = df_summer[df_summer["season"] == rec_season]

    # Per-player, per-division: last team played for in that division
    def _player_team_map(div_df: pd.DataFrame) -> dict[str, tuple[str, str]]:
        """Returns {player_id: (player_name, last_team)} for a division df."""
        result = {}
        for pid, grp in div_df.groupby("player_id"):
            last = grp.sort_values("date").iloc[-1]
            result[pid] = (last["player_name"], last["team"])
        return result

    comp_players_info = _player_team_map(comp_df)
    rec_players_info  = _player_team_map(rec_df)

    comp_player_ids = set(comp_players_info.keys())
    rec_player_ids  = set(rec_players_info.keys())

    log.info("Active players — D2 Comp: %d  |  D2 Rec: %d  |  Both: %d",
             len(comp_player_ids), len(rec_player_ids),
             len(comp_player_ids & rec_player_ids))

    # ── Build combined history (historical + 2026 Summer) ─────────────────
    # The 2026 Summer games are already played games — include them as prior
    # history so predictions reflect the most recent performance.
    # We select only the columns that exist in clean_game_logs.csv for safety.
    hist_cols = list(df_hist.columns)
    summer_for_hist = df_summer[[c for c in hist_cols if c in df_summer.columns]].copy()
    df_combined = pd.concat([df_hist, summer_for_hist], ignore_index=True)
    df_combined["date"] = pd.to_datetime(df_combined["date"])
    df_combined = df_combined.sort_values(["player_id", "date", "game_id"]).reset_index(drop=True)
    log.info("Combined history: %d rows (hist=%d + summer=%d)",
             len(df_combined), len(df_hist), len(df_summer))

    # ── Generate predictions ───────────────────────────────────────────────
    prediction_rows: list[dict] = []
    skipped: list[tuple[str, str, str]] = []

    # D2 Comp predictions (tier1 context)
    log.info("Generating predictions for D2 Comp 2026 Summer (%d players)...",
             len(comp_player_ids))
    for pid, (pname, team) in sorted(comp_players_info.items(), key=lambda x: x[1][0]):
        player_hist = df_combined[df_combined["player_id"] == pid].copy()
        if len(player_hist) == 0:
            skipped.append((pname, comp_season, "No history rows found"))
            continue
        row = predict_one(
            player_hist, pid, pname, comp_season, "D1", team, models, prob_models
        )
        if row is None:
            skipped.append((pname, comp_season, "Feature build failed"))
        else:
            prediction_rows.append(row)

    # D2 Rec predictions (tier2 context)
    log.info("Generating predictions for D2 Rec 2026 Summer (%d players)...",
             len(rec_player_ids))
    for pid, (pname, team) in sorted(rec_players_info.items(), key=lambda x: x[1][0]):
        player_hist = df_combined[df_combined["player_id"] == pid].copy()
        if len(player_hist) == 0:
            skipped.append((pname, rec_season, "No history rows found"))
            continue
        row = predict_one(
            player_hist, pid, pname, rec_season, "D2", team, models, prob_models
        )
        if row is None:
            skipped.append((pname, rec_season, "Feature build failed"))
        else:
            prediction_rows.append(row)

    if not prediction_rows:
        log.error("No predictions generated — check models and data.")
        sys.exit(1)

    log.info("Generated %d prediction rows (%d skipped)", len(prediction_rows), len(skipped))

    # ── Build output DataFrame ────────────────────────────────────────────
    df_all = pd.DataFrame(prediction_rows)[OUTPUT_COLUMNS]

    df_d2_comp = df_all[df_all["division"] == comp_season].sort_values(
        "predicted_pts", ascending=False
    ).reset_index(drop=True)
    df_d2_rec  = df_all[df_all["division"] == rec_season].sort_values(
        "predicted_pts", ascending=False
    ).reset_index(drop=True)

    # ── Save main CSVs ────────────────────────────────────────────────────
    df_all.to_csv(OUTPUT_ALL,     index=False)
    df_d2_comp.to_csv(OUTPUT_D2_COMP, index=False)
    df_d2_rec.to_csv(OUTPUT_D2_REC,  index=False)
    log.info("Saved prediction CSVs to %s", DATA_DIR)

    # ── Save leaderboards ─────────────────────────────────────────────────
    leaderboard_cols = [
        "player_name", "division", "team", "confidence_level", "confidence_score",
        "games_played_history",
        "predicted_pts", "predicted_reb", "predicted_ast", "predicted_stl", "predicted_blk",
        "pts_low", "pts_high", "reb_low", "reb_high",
        "ast_low", "ast_high", "stl_low", "stl_high", "blk_low", "blk_high",
        "prob_10_plus_pts", "prob_15_plus_pts", "prob_20_plus_pts",
        "prob_5_plus_reb", "prob_10_plus_reb", "prob_5_plus_ast", "prob_double_double",
    ]

    df_all.sort_values("predicted_pts", ascending=False) \
          .head(LEADERBOARD_SIZE)[leaderboard_cols] \
          .reset_index(drop=True) \
          .to_csv(TOP_SCORERS, index=False)

    df_all.sort_values("predicted_reb", ascending=False) \
          .head(LEADERBOARD_SIZE)[leaderboard_cols] \
          .reset_index(drop=True) \
          .to_csv(TOP_REBOUNDERS, index=False)

    df_all.sort_values("predicted_ast", ascending=False) \
          .head(LEADERBOARD_SIZE)[leaderboard_cols] \
          .reset_index(drop=True) \
          .to_csv(TOP_ASSISTS, index=False)

    df_all.sort_values("prob_double_double", ascending=False) \
          .head(LEADERBOARD_SIZE)[leaderboard_cols] \
          .reset_index(drop=True) \
          .to_csv(TOP_DOUBLE_DOUBLE, index=False)

    log.info("Saved leaderboard CSVs to %s", REPORTS_DIR)

    # ── Audit report ──────────────────────────────────────────────────────
    print_audit(df_all, summer_meta, skipped, comp_player_ids, rec_player_ids)


if __name__ == "__main__":
    main()
