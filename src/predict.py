"""
predict.py  (V5 — confidence scores + prediction ranges)
Generates next-game predictions using the V2 regression models,
V3 probability classification models, and V5 confidence scoring.

Changes from V4 / V3:
  - Confidence score (0-100) computed from four components:
      * Total games played           (max 25 pts)
      * Games in the requested tier  (max 35 pts)
      * Days since last game         (max 20 pts)
      * PTS consistency in tier (CV) (max 20 pts)
  - Confidence label derived from score:
      * 65-100 -> High   35-64 -> Medium   0-34 -> Low
  - Prediction ranges for all 5 stats:
      * Base half-width = MODEL_MAE x confidence multiplier
        (High x1.00, Medium x1.25, Low x1.60)
      * Blended with player's own tier std-dev when >= 5 tier games
  - All output modes (single, --tier, --both, sample) show score and ranges.

Changes from V2 (V3):
  - Probability outputs for seven thresholds added.

Changes from V1 (V2):
  - Tier-aware predictions with --tier / --both flags.

Usage:
  python src/predict.py
  python src/predict.py --player "Abdullah Khan"
  python src/predict.py --player "Abdullah Khan" --tier tier1
  python src/predict.py --player "Abdullah Khan" --tier tier2
  python src/predict.py --player "Abdullah Khan" --both
  python src/predict.py --id <player_id> --both
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CLEAN_PATH = Path("data/processed/clean_game_logs.csv")
MODELS_DIR = Path("models")

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

PROB_LABELS = {
    "pts_10_plus":   "10+ PTS",
    "pts_15_plus":   "15+ PTS",
    "pts_20_plus":   "20+ PTS",
    "reb_5_plus":    " 5+ REB",
    "reb_10_plus":   "10+ REB",
    "ast_5_plus":    " 5+ AST",
    "double_double": "Double-Double",
}

STATS = ["pts", "reb", "ast", "stl", "blk"]

# Must match V2_FEATURE_COLS from train_model.py exactly
V2_FEATURE_COLS = (
    [
        "tier_d1",
        "days_since_last_game",
        "games_played_before",
        "season_games_before",
        "tier1_games_before",
        "tier2_games_before",
    ]
    # Context-routed: set to tier1-specific when predicting tier1, tier2 for tier2
    + [f"ctx_last_{n}_{s}_avg" for n in [3, 5] for s in STATS]
    + [f"ctx_season_{s}_avg" for s in STATS]
    + [f"ctx_career_{s}_avg" for s in STATS]
    # Full tier1 and tier2 histories as cross-tier context
    + [f"last_{n}_tier1_{s}_avg" for n in [3, 5] for s in STATS]
    + [f"season_tier1_{s}_avg" for s in STATS]
    + [f"career_tier1_{s}_avg" for s in STATS]
    + [f"last_{n}_tier2_{s}_avg" for n in [3, 5] for s in STATS]
    + [f"season_tier2_{s}_avg" for s in STATS]
    + [f"career_tier2_{s}_avg" for s in STATS]
)

TIER_MAP = {"tier1": "D1", "tier2": "D2"}
TIER_LABEL_DISPLAY = {"tier1": "Tier 1 (D1 / D2 Comp)", "tier2": "Tier 2 (D2 / D2 Rec)"}

# MAE from V2 training — used as base uncertainty for prediction ranges
MODEL_MAE = {"pts": 4.74, "reb": 2.35, "ast": 1.20, "stl": 0.90, "blk": 0.43}

# Confidence score thresholds
CONF_HIGH_THRESHOLD = 65   # score >= 65 -> High
CONF_LOW_THRESHOLD  = 35   # score <  35 -> Low


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _rolling_mean(values: list, n: int) -> float:
    if not values:
        return np.nan
    return float(np.mean(values[-n:]))


def _mean(values: list) -> float:
    if not values:
        return np.nan
    return float(np.mean(values))


def confidence_label(games_in_tier: int) -> str:
    """Legacy helper — kept for backward compatibility."""
    if games_in_tier <= 2:
        return "Low"
    if games_in_tier <= 7:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# V5: Confidence scoring and prediction ranges
# ---------------------------------------------------------------------------

def compute_confidence_score(
    feat: dict,
    tier_games_df: pd.DataFrame,
    tier_label: str,
) -> int:
    """
    Returns a 0-100 integer confidence score combining four components:
      1. Total games played           (max 25 pts)
      2. Games in the requested tier  (max 35 pts)
      3. Days since last game         (max 20 pts)
      4. PTS consistency in tier (CV) (max 20 pts)
    """
    score = 0

    # 1. Total games
    tg = int(feat["games_played_before"])
    if   tg >= 30: score += 25
    elif tg >= 20: score += 20
    elif tg >= 10: score += 15
    elif tg >= 5:  score += 10
    elif tg >= 3:  score += 6
    else:          score += 2

    # 2. Tier-specific games
    tier_g = int(feat[f"{tier_label}_games_before"])
    if   tier_g >= 20: score += 35
    elif tier_g >= 10: score += 28
    elif tier_g >= 6:  score += 21
    elif tier_g >= 3:  score += 13
    elif tier_g >= 1:  score += 6
    # else 0

    # 3. Recency
    days = float(feat["days_since_last_game"])
    if   days <=  7: score += 20
    elif days <= 14: score += 16
    elif days <= 30: score += 11
    elif days <= 60: score += 6
    elif days <= 90: score += 3
    # else 0

    # 4. PTS consistency (CV = std / mean); requires >= 3 tier games
    if len(tier_games_df) >= 3:
        m  = float(tier_games_df["pts"].mean())
        s  = float(tier_games_df["pts"].std(ddof=1))
        cv = s / (m + 1e-6)
        if   cv < 0.30: score += 20
        elif cv < 0.50: score += 15
        elif cv < 0.70: score += 10
        elif cv < 1.00: score += 5
        else:           score += 2
    else:
        score += 2   # insufficient data — minimal consistency bonus

    return min(100, score)


def confidence_score_to_label(score: int) -> str:
    if score >= CONF_HIGH_THRESHOLD:
        return "High"
    if score >= CONF_LOW_THRESHOLD:
        return "Medium"
    return "Low"


def compute_stat_ranges(
    preds: dict,
    tier_games_df: pd.DataFrame,
    confidence_score: int,
) -> dict:
    """
    Returns {stat: (low, high)} for each of the 5 stats.

    Base half-width = MODEL_MAE[stat] x confidence multiplier:
      High (>=65)   -> x1.00
      Medium (>=35) -> x1.25
      Low (<35)     -> x1.60

    If player has >=5 tier games, their own stat std-dev is blended:
      half = max(base_half, 0.80 x tier_std)

    Floor: low >= 0.
    """
    if   confidence_score >= CONF_HIGH_THRESHOLD: mult = 1.00
    elif confidence_score >= CONF_LOW_THRESHOLD:  mult = 1.25
    else:                                          mult = 1.60

    ranges: dict = {}
    for stat in STATS:
        base_half = MODEL_MAE[stat] * mult
        if len(tier_games_df) >= 5:
            tier_std = float(tier_games_df[stat].std(ddof=1))
            half = max(base_half, 0.80 * tier_std)
        else:
            half = base_half
        low  = round(max(0.0, preds[stat] - half), 1)
        high = round(preds[stat] + half, 1)
        ranges[stat] = (low, high)

    return ranges


# ---------------------------------------------------------------------------
# Feature builder for next-game prediction
# ---------------------------------------------------------------------------

def build_next_game_features(
    player_df: pd.DataFrame,
    tier_d1_val: int,
) -> dict:
    """
    Build the full 66-feature V2 row for the player's NEXT (not-yet-played) game.

    player_df  : all game rows for one player, sorted chronologically.
    tier_d1_val: 1 if predicting for a D1 game (Tier 1), 0 for D2 (Tier 2).

    All played games are treated as prior history.  Tier-specific features that
    are NaN (no prior games in that tier) are filled with 0, matching training.
    """
    df   = player_df.sort_values(["date", "game_id"]).reset_index(drop=True)
    last = df.iloc[-1]

    # ── Overall (V1) features ─────────────────────────────────────────────
    career_all = {s: df[s].tolist() for s in STATS}
    season_df  = df[df["season"] == last["season"]]
    season_all = {s: season_df[s].tolist() for s in STATS}

    last_date = pd.to_datetime(last["date"]).date()
    days_gap  = (date.today() - last_date).days

    feat: dict = {
        "tier_d1":              tier_d1_val,
        "days_since_last_game": float(days_gap),
        "games_played_before":  len(df),
        "season_games_before":  len(season_df),
    }

    for s in STATS:
        feat[f"last_3_{s}_avg"] = _rolling_mean(career_all[s], 3)
        feat[f"last_5_{s}_avg"] = _rolling_mean(career_all[s], 5)
        feat[f"season_{s}_avg"] = _mean(season_all[s])
        feat[f"career_{s}_avg"] = _mean(career_all[s])

    # ── Tier-aware (V2) features ──────────────────────────────────────────
    tier_career        = {"D1": {s: [] for s in STATS}, "D2": {s: [] for s in STATS}}
    tier_seasons       = {"D1": {s: {} for s in STATS}, "D2": {s: {} for s in STATS}}
    tier_latest_season = {"D1": None, "D2": None}
    tier_count         = {"D1": 0, "D2": 0}

    for _, row in df.iterrows():
        g_tier   = row["tier"]
        g_season = row["season"]
        for s in STATS:
            tier_career[g_tier][s].append(row[s])
            tier_seasons[g_tier][s].setdefault(g_season, []).append(row[s])
        tier_latest_season[g_tier] = g_season
        tier_count[g_tier] += 1

    feat["tier1_games_before"] = float(tier_count["D1"])
    feat["tier2_games_before"] = float(tier_count["D2"])

    for tlabel, tier_val in TIER_MAP.items():
        c  = tier_career[tier_val]
        ls = tier_latest_season[tier_val]

        for s in STATS:
            hist = c[s]
            feat[f"last_3_{tlabel}_{s}_avg"] = _rolling_mean(hist, 3) if hist else 0.0
            feat[f"last_5_{tlabel}_{s}_avg"] = _rolling_mean(hist, 5) if hist else 0.0
            feat[f"career_{tlabel}_{s}_avg"] = _mean(hist)            if hist else 0.0

            if ls is not None:
                s_hist = tier_seasons[tier_val][s].get(ls, [])
                feat[f"season_{tlabel}_{s}_avg"] = _mean(s_hist) if s_hist else 0.0
            else:
                feat[f"season_{tlabel}_{s}_avg"] = 0.0

    # ── Context-routed features ───────────────────────────────────────────
    req_label = "tier1" if tier_d1_val == 1 else "tier2"
    for s in STATS:
        for n in [3, 5]:
            feat[f"ctx_last_{n}_{s}_avg"] = feat[f"last_{n}_{req_label}_{s}_avg"]
        feat[f"ctx_season_{s}_avg"] = feat[f"season_{req_label}_{s}_avg"]
        feat[f"ctx_career_{s}_avg"] = feat[f"career_{req_label}_{s}_avg"]

    return feat


# ---------------------------------------------------------------------------
# Player lookup
# ---------------------------------------------------------------------------

def lookup_player(query: str, df: pd.DataFrame) -> pd.DataFrame:
    by_id = df[df["player_id"] == query]
    if not by_id.empty:
        return by_id

    mask    = df["player_name"].str.lower().str.contains(query.lower(), regex=False)
    matched = df[mask]["player_name"].unique()

    if len(matched) == 0:
        raise ValueError(f"No player found matching '{query}'.")
    if len(matched) > 1:
        raise ValueError(
            f"'{query}' matches multiple players: {matched.tolist()}\n"
            "Use a more specific name or pass --id."
        )
    return df[df["player_name"] == matched[0]]


# ---------------------------------------------------------------------------
# Single-tier prediction
# ---------------------------------------------------------------------------

def predict_player_tier(
    player_df: pd.DataFrame,
    models: dict,
    tier_label: str,          # "tier1" or "tier2"
    prob_models: dict | None = None,
) -> dict:
    """
    Build V2 features and run all five regression models plus optional
    probability classifiers for one player in one tier context.
    Returns a result dict including confidence score and stat ranges (V5).
    """
    tier_val = TIER_MAP[tier_label]   # "D1" or "D2"
    tier_d1  = 1 if tier_val == "D1" else 0

    feat = build_next_game_features(player_df, tier_d1_val=tier_d1)

    X = pd.DataFrame([feat])[V2_FEATURE_COLS]

    preds = {stat: max(0.0, float(models[stat].predict(X)[0]))
             for stat in STATS}

    # Probability predictions
    probs: dict = {}
    if prob_models:
        for name, model in prob_models.items():
            try:
                probs[name] = float(model.predict_proba(X)[0, 1])
            except Exception:
                probs[name] = None

    df_sorted     = player_df.sort_values(["date", "game_id"])
    last          = df_sorted.iloc[-1]
    tier_rows     = df_sorted[df_sorted["tier"] == tier_val]
    team_for_tier = tier_rows.iloc[-1]["team"] if not tier_rows.empty else last["team"]
    games_in_tier = int(feat[f"{tier_label}_games_before"])

    # V5: confidence score, label, and prediction ranges
    conf_score = compute_confidence_score(feat, tier_rows, tier_label)
    conf_label = confidence_score_to_label(conf_score)
    ranges     = compute_stat_ranges(preds, tier_rows, conf_score)

    return {
        "player_id":        last["player_id"],
        "player_name":      last["player_name"],
        "team":             team_for_tier,
        "tier_label":       tier_label,
        "tier_display":     TIER_LABEL_DISPLAY[tier_label],
        "games_in_tier":    games_in_tier,
        "games_total":      int(feat["games_played_before"]),
        "confidence":       conf_label,
        "confidence_score": conf_score,
        "last_game":        str(last["date"])[:10],
        # Regression predictions
        "pred_pts":  round(preds["pts"],  1),
        "pred_reb":  round(preds["reb"],  1),
        "pred_ast":  round(preds["ast"],  1),
        "pred_stl":  round(preds["stl"],  1),
        "pred_blk":  round(preds["blk"],  1),
        # Prediction ranges (low, high) — floor at 0
        "pts_low":  ranges["pts"][0],  "pts_high":  ranges["pts"][1],
        "reb_low":  ranges["reb"][0],  "reb_high":  ranges["reb"][1],
        "ast_low":  ranges["ast"][0],  "ast_high":  ranges["ast"][1],
        "stl_low":  ranges["stl"][0],  "stl_high":  ranges["stl"][1],
        "blk_low":  ranges["blk"][0],  "blk_high":  ranges["blk"][1],
        # Context averages for this tier
        "tier_career_pts": round(feat[f"career_{tier_label}_pts_avg"], 1),
        "tier_career_reb": round(feat[f"career_{tier_label}_reb_avg"], 1),
        "tier_career_ast": round(feat[f"career_{tier_label}_ast_avg"], 1),
        "tier_career_stl": round(feat[f"career_{tier_label}_stl_avg"], 1),
        "tier_career_blk": round(feat[f"career_{tier_label}_blk_avg"], 1),
        "tier_last3_pts":  round(feat[f"last_3_{tier_label}_pts_avg"], 1),
        "tier_last3_reb":  round(feat[f"last_3_{tier_label}_reb_avg"], 1),
        "tier_last3_ast":  round(feat[f"last_3_{tier_label}_ast_avg"], 1),
        # Probability predictions
        "probs": probs,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

CONFIDENCE_BADGE = {"High": "[HIGH]", "Medium": "[MED] ", "Low": "[LOW] "}


def _fmt_prob(p: float | None) -> str:
    if p is None:
        return " N/A"
    return f"{round(p * 100):>3}%"


def _fmt_range(low: float, high: float) -> str:
    return f"{low:.1f}-{high:.1f}"


def _print_probabilities(probs: dict) -> None:
    if not probs:
        return
    print(f"\n  Probabilities")
    p10 = _fmt_prob(probs.get("pts_10_plus"))
    p15 = _fmt_prob(probs.get("pts_15_plus"))
    p20 = _fmt_prob(probs.get("pts_20_plus"))
    print(f"    10+ PTS: {p10}    15+ PTS: {p15}    20+ PTS: {p20}")
    r5  = _fmt_prob(probs.get("reb_5_plus"))
    r10 = _fmt_prob(probs.get("reb_10_plus"))
    a5  = _fmt_prob(probs.get("ast_5_plus"))
    print(f"     5+ REB: {r5}    10+ REB: {r10}     5+ AST: {a5}")
    dd  = _fmt_prob(probs.get("double_double"))
    print(f"    Double-Double: {dd}")


def print_prediction(r: dict, idx: int | None = None) -> None:
    prefix     = f"[{idx}] " if idx is not None else ""
    conf       = CONFIDENCE_BADGE[r["confidence"]]
    games_note = f"{r['games_in_tier']} in-tier / {r['games_total']} total"
    score_note = f"{r['confidence_score']}/100"

    print(f"\n{prefix}{r['player_name']}  |  {r['team']}  |  {r['tier_display']}")
    print(f"  Last game : {r['last_game']}   Games : {games_note}   "
          f"Confidence : {conf} {r['confidence']} ({score_note})")
    print(f"  {'':32}  {'PTS':>6}  {'REB':>6}  {'AST':>6}  {'STL':>6}  {'BLK':>6}")
    print(f"  {'Predicted next game':32}  "
          f"{r['pred_pts']:>6.1f}  {r['pred_reb']:>6.1f}  {r['pred_ast']:>6.1f}  "
          f"{r['pred_stl']:>6.1f}  {r['pred_blk']:>6.1f}")
    print(f"  {'Prediction range':32}  "
          f"PTS {_fmt_range(r['pts_low'], r['pts_high'])}  "
          f"REB {_fmt_range(r['reb_low'], r['reb_high'])}  "
          f"AST {_fmt_range(r['ast_low'], r['ast_high'])}  "
          f"STL {_fmt_range(r['stl_low'], r['stl_high'])}  "
          f"BLK {_fmt_range(r['blk_low'], r['blk_high'])}")
    print(f"  {'Tier career avg':32}  "
          f"{r['tier_career_pts']:>6.1f}  {r['tier_career_reb']:>6.1f}  {r['tier_career_ast']:>6.1f}  "
          f"{r['tier_career_stl']:>6.1f}  {r['tier_career_blk']:>6.1f}")
    print(f"  {'Tier last-3 avg (PTS/REB/AST)':32}  "
          f"{r['tier_last3_pts']:>6.1f}  {r['tier_last3_reb']:>6.1f}  {r['tier_last3_ast']:>6.1f}")
    _print_probabilities(r.get("probs", {}))


def print_both(r1: dict, r2: dict, idx: int | None = None) -> None:
    """Print side-by-side comparison for --both mode."""
    prefix  = f"[{idx}] " if idx is not None else ""
    header1 = TIER_LABEL_DISPLAY["tier1"]
    header2 = TIER_LABEL_DISPLAY["tier2"]

    def _conf_str(r: dict) -> str:
        return f"{r['confidence']} ({r['confidence_score']}/100)"

    print(f"\n{prefix}{r1['player_name']}")
    print(f"  Last game : {r1['last_game']}")

    print(f"\n  {'':28}  {'--- ' + header1 + ' ---':^28}  {'--- ' + header2 + ' ---':^28}")
    print(f"  {'':28}  {_conf_str(r1):^28}  {_conf_str(r2):^28}")
    print(f"  {'':28}  {str(r1['games_in_tier'])+' games in tier':^28}  "
          f"{str(r2['games_in_tier'])+' games in tier':^28}")

    cols  = ["PTS", "REB", "AST", "STL", "BLK"]
    keys  = ["pred_pts", "pred_reb", "pred_ast", "pred_stl", "pred_blk"]
    ck    = ["tier_career_pts", "tier_career_reb", "tier_career_ast",
             "tier_career_stl", "tier_career_blk"]

    print(f"\n  {'Stat':8}  {'Predicted':>10}  {'Range':>13}  {'Career avg':>10}  "
          f"  {'Predicted':>10}  {'Range':>13}  {'Career avg':>10}")
    print("  " + "-" * 84)
    for col, k, c, sk in zip(cols, keys, ck, STATS):
        rng1 = _fmt_range(r1[f"{sk}_low"], r1[f"{sk}_high"])
        rng2 = _fmt_range(r2[f"{sk}_low"], r2[f"{sk}_high"])
        print(f"  {col:<8}  {r1[k]:>10.1f}  {rng1:>13}  {r1[c]:>10.1f}  "
              f"  {r2[k]:>10.1f}  {rng2:>13}  {r2[c]:>10.1f}")

    # Probabilities side-by-side
    p1 = r1.get("probs", {})
    p2 = r2.get("probs", {})
    if p1 or p2:
        prob_order = [
            ("pts_10_plus",   "10+ PTS"),
            ("pts_15_plus",   "15+ PTS"),
            ("pts_20_plus",   "20+ PTS"),
            ("reb_5_plus",    " 5+ REB"),
            ("reb_10_plus",   "10+ REB"),
            ("ast_5_plus",    " 5+ AST"),
            ("double_double", "Dbl-Dbl"),
        ]
        print(f"\n  {'Probability':20}  {'Tier 1':>8}  {'Tier 2':>8}")
        print("  " + "-" * 38)
        for key, label in prob_order:
            v1 = _fmt_prob(p1.get(key))
            v2 = _fmt_prob(p2.get(key))
            print(f"  {label:<20}  {v1:>8}  {v2:>8}")


def print_banner() -> None:
    print("=" * 68)
    print("  IMBA PLAYER PERFORMANCE PREDICTOR  (V5 — confidence scores + ranges)")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Data + model loading
# ---------------------------------------------------------------------------

def load_data_and_models() -> tuple[pd.DataFrame, dict, dict]:
    if not CLEAN_PATH.exists():
        print(f"ERROR: {CLEAN_PATH} not found. Run clean_data.py first.")
        sys.exit(1)

    df = pd.read_csv(CLEAN_PATH, parse_dates=["date"])

    models = {}
    for stat, path in MODEL_FILES.items():
        if not path.exists():
            print(f"ERROR: {path} not found. Run train_model.py first.")
            sys.exit(1)
        models[stat] = joblib.load(path)

    prob_models = {}
    missing_prob = []
    for name, path in PROB_MODEL_FILES.items():
        if path.exists():
            prob_models[name] = joblib.load(path)
        else:
            missing_prob.append(str(path))

    if missing_prob:
        print(f"  NOTE: {len(missing_prob)} probability model(s) not found "
              f"— run train_model.py to enable probability outputs.")
        prob_models = {}

    return df, models, prob_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict next-game stats for an IMBA player (V5 confidence + ranges)."
    )
    id_group = parser.add_mutually_exclusive_group()
    id_group.add_argument("--player", "-p", metavar="NAME",
                          help="Player name (substring match)")
    id_group.add_argument("--id", "-i", metavar="PLAYER_ID",
                          help="Exact player_id")

    tier_group = parser.add_mutually_exclusive_group()
    tier_group.add_argument("--tier", "-t", choices=["tier1", "tier2"],
                            help="Predict for a specific tier (tier1=D1, tier2=D2)")
    tier_group.add_argument("--both", "-b", action="store_true",
                            help="Show predictions for both tiers")

    args = parser.parse_args()

    df, models, prob_models = load_data_and_models()
    print_banner()

    # ── Single-player mode ────────────────────────────────────────────────
    if args.player or args.id:
        query = args.player or args.id
        try:
            player_df = lookup_player(query, df)
        except ValueError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)

        player_tiers = player_df["tier"].unique().tolist()

        if args.both:
            results = {}
            for tlabel, tval in TIER_MAP.items():
                if tval not in player_tiers:
                    print(f"\n  NOTE: {player_df['player_name'].iloc[0]} has no {tval} history "
                          f"— {tlabel} prediction uses overall averages as fallback.")
                results[tlabel] = predict_player_tier(
                    player_df, models, tlabel, prob_models or None
                )
            print_both(results["tier1"], results["tier2"])

        elif args.tier:
            tval = TIER_MAP[args.tier]
            if tval not in player_tiers:
                print(f"\n  NOTE: {player_df['player_name'].iloc[0]} has no {tval} history "
                      f"— prediction uses overall averages as fallback.")
            r = predict_player_tier(player_df, models, args.tier, prob_models or None)
            print_prediction(r)

        else:
            last_tier_val = player_df.sort_values("date").iloc[-1]["tier"]
            default_label = {v: k for k, v in TIER_MAP.items()}[last_tier_val]
            r = predict_player_tier(player_df, models, default_label, prob_models or None)
            print_prediction(r)
            if len(player_tiers) > 1:
                pname = player_df["player_name"].iloc[0]
                print(f"\n  {pname} has played in both tiers. "
                      f"Use --both to see Tier 1 and Tier 2 predictions.")
        return

    # ── Sample mode ───────────────────────────────────────────────────────
    print("\n  No player specified — showing 5 sample players.")
    print("  Run with --player NAME, --tier tier1/tier2, or --both.\n")

    top_players = (
        df.groupby(["player_id", "player_name"])
        .size()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    results = []
    for pid, pname in top_players:
        player_df     = df[df["player_id"] == pid]
        last_tier_val = player_df.sort_values("date").iloc[-1]["tier"]
        tlabel        = {v: k for k, v in TIER_MAP.items()}[last_tier_val]
        try:
            r = predict_player_tier(player_df, models, tlabel, prob_models or None)
            results.append(r)
        except Exception as e:
            print(f"  Warning: {pname}: {e}")

    for i, r in enumerate(results, 1):
        print_prediction(r, idx=i)

    if results:
        print("\n" + "=" * 68)
        print("  SUMMARY")
        print("=" * 68)
        print(f"  {'Player':<22}  {'Tier':>5}  {'Score':>5}  {'Conf':<6}  "
              f"{'PTS':>5}  {'REB':>5}  {'AST':>5}  {'STL':>5}  {'BLK':>5}")
        print("  " + "-" * 70)
        for r in results:
            tier_short = "T1" if r["tier_label"] == "tier1" else "T2"
            print(f"  {r['player_name']:<22}  {tier_short:>5}  {r['confidence_score']:>5}  "
                  f"{r['confidence']:<6}  "
                  f"{r['pred_pts']:>5.1f}  {r['pred_reb']:>5.1f}  {r['pred_ast']:>5.1f}  "
                  f"{r['pred_stl']:>5.1f}  {r['pred_blk']:>5.1f}")
        print()


if __name__ == "__main__":
    main()
