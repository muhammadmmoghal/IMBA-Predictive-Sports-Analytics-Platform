"""
predict.py  (V3 — tier-aware predictions + STL/BLK + probability models)
Generates next-game predictions using the V2 regression models and
V3 probability classification models.

Changes from V2:
  - Probability outputs added for seven thresholds:
      10+ PTS, 15+ PTS, 20+ PTS
       5+ REB, 10+ REB
       5+ AST
      Double-Double
  - Probability models loaded from models/{target_name}_model.pkl.
  - If probability models are not yet trained, stat predictions still work.

Changes from V1:
  - New --tier {tier1, tier2} flag: predict for a specific division context.
      tier1 = D1 (D2 Comp)
      tier2 = D2 (D2 Rec)
  - New --both flag: show Tier 1 and Tier 2 projections side-by-side.
  - Predictions now include STL and BLK alongside PTS, REB, AST.
  - Confidence is based on the number of games in the requested tier
    (overall career games when no tier is specified).
  - Output shows recent team in the requested tier.

Usage:
  python src/predict.py                            # 5 sample players, default tier
  python src/predict.py --player "Abdullah Khan"   # default (most recent) tier
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

# Display labels for each probability target
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

# Tier label -> raw value in the data
TIER_MAP = {"tier1": "D1", "tier2": "D2"}
TIER_LABEL_DISPLAY = {"tier1": "Tier 1 (D1 / D2 Comp)", "tier2": "Tier 2 (D2 / D2 Rec)"}

# ---------------------------------------------------------------------------
# Helpers
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
    if games_in_tier <= 2:
        return "Low"
    if games_in_tier <= 7:
        return "Medium"
    return "High"


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
    df  = player_df.sort_values(["date", "game_id"]).reset_index(drop=True)
    last = df.iloc[-1]

    # ── Overall (V1) features ─────────────────────────────────────────────
    career_all  = {s: df[s].tolist() for s in STATS}
    season_df   = df[df["season"] == last["season"]]
    season_all  = {s: season_df[s].tolist() for s in STATS}

    last_date = pd.to_datetime(last["date"]).date()
    days_gap  = (date.today() - last_date).days

    feat: dict = {
        "tier_d1":              tier_d1_val,
        "days_since_last_game": float(days_gap),
        "games_played_before":  len(df),
        "season_games_before":  len(season_df),
    }

    for s in STATS:
        feat[f"last_3_{s}_avg"]  = _rolling_mean(career_all[s], 3)
        feat[f"last_5_{s}_avg"]  = _rolling_mean(career_all[s], 5)
        feat[f"season_{s}_avg"]  = _mean(season_all[s])
        feat[f"career_{s}_avg"]  = _mean(career_all[s])

    # ── Tier-aware (V2) features ──────────────────────────────────────────
    # Walk the game history and build per-tier accumulators
    tier_career  = {"D1": {s: [] for s in STATS}, "D2": {s: [] for s in STATS}}
    tier_seasons = {"D1": {s: {} for s in STATS}, "D2": {s: {} for s in STATS}}
    tier_latest_season = {"D1": None, "D2": None}
    tier_count   = {"D1": 0, "D2": 0}

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
            feat[f"last_3_{tlabel}_{s}_avg"]  = _rolling_mean(hist, 3) if hist else 0.0
            feat[f"last_5_{tlabel}_{s}_avg"]  = _rolling_mean(hist, 5) if hist else 0.0
            feat[f"career_{tlabel}_{s}_avg"]  = _mean(hist)            if hist else 0.0

            if ls is not None:
                s_hist = tier_seasons[tier_val][s].get(ls, [])
                feat[f"season_{tlabel}_{s}_avg"] = _mean(s_hist) if s_hist else 0.0
            else:
                feat[f"season_{tlabel}_{s}_avg"] = 0.0

    # ── Context-routed features ───────────────────────────────────────────
    # Route ctx_* to the tier being predicted — mirrors the training-time
    # routing where ctx_* = the row's own-tier values.
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
    Returns a result dict.
    """
    tier_val  = TIER_MAP[tier_label]   # "D1" or "D2"
    tier_d1   = 1 if tier_val == "D1" else 0

    feat = build_next_game_features(player_df, tier_d1_val=tier_d1)

    X = pd.DataFrame([feat])[V2_FEATURE_COLS]

    preds = {stat: max(0.0, float(models[stat].predict(X)[0]))
             for stat in ["pts", "reb", "ast", "stl", "blk"]}

    # Probability predictions
    probs: dict = {}
    if prob_models:
        for name, model in prob_models.items():
            try:
                probs[name] = float(model.predict_proba(X)[0, 1])
            except Exception:
                probs[name] = None

    df_sorted = player_df.sort_values(["date", "game_id"])
    last       = df_sorted.iloc[-1]

    # Team from most recent game in this tier (fallback: overall most recent)
    tier_rows = df_sorted[df_sorted["tier"] == tier_val]
    team_for_tier = tier_rows.iloc[-1]["team"] if not tier_rows.empty else last["team"]

    games_in_tier = int(feat[f"{tier_label}_games_before"])

    return {
        "player_id":          last["player_id"],
        "player_name":        last["player_name"],
        "team":               team_for_tier,
        "tier_label":         tier_label,
        "tier_display":       TIER_LABEL_DISPLAY[tier_label],
        "games_in_tier":      games_in_tier,
        "games_total":        int(feat["games_played_before"]),
        "confidence":         confidence_label(games_in_tier),
        "last_game":          str(last["date"])[:10],
        # Regression predictions
        "pred_pts":  round(preds["pts"],  1),
        "pred_reb":  round(preds["reb"],  1),
        "pred_ast":  round(preds["ast"],  1),
        "pred_stl":  round(preds["stl"],  1),
        "pred_blk":  round(preds["blk"],  1),
        # Context averages for this tier
        "tier_career_pts": round(feat[f"career_{tier_label}_pts_avg"], 1),
        "tier_career_reb": round(feat[f"career_{tier_label}_reb_avg"], 1),
        "tier_career_ast": round(feat[f"career_{tier_label}_ast_avg"], 1),
        "tier_career_stl": round(feat[f"career_{tier_label}_stl_avg"], 1),
        "tier_career_blk": round(feat[f"career_{tier_label}_blk_avg"], 1),
        "tier_last3_pts":  round(feat[f"last_3_{tier_label}_pts_avg"], 1),
        "tier_last3_reb":  round(feat[f"last_3_{tier_label}_reb_avg"], 1),
        "tier_last3_ast":  round(feat[f"last_3_{tier_label}_ast_avg"], 1),
        # Probability predictions (may be empty if models not trained yet)
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


def _print_probabilities(probs: dict) -> None:
    if not probs:
        return
    print(f"\n  {'Probabilities':}")
    # Row 1: PTS thresholds
    p10  = _fmt_prob(probs.get("pts_10_plus"))
    p15  = _fmt_prob(probs.get("pts_15_plus"))
    p20  = _fmt_prob(probs.get("pts_20_plus"))
    print(f"    10+ PTS: {p10}    15+ PTS: {p15}    20+ PTS: {p20}")
    # Row 2: REB + AST thresholds
    r5   = _fmt_prob(probs.get("reb_5_plus"))
    r10  = _fmt_prob(probs.get("reb_10_plus"))
    a5   = _fmt_prob(probs.get("ast_5_plus"))
    print(f"     5+ REB: {r5}    10+ REB: {r10}     5+ AST: {a5}")
    # Row 3: Double-double
    dd   = _fmt_prob(probs.get("double_double"))
    print(f"    Double-Double: {dd}")


def print_prediction(r: dict, idx: int | None = None) -> None:
    prefix = f"[{idx}] " if idx is not None else ""
    conf   = CONFIDENCE_BADGE[r["confidence"]]
    games_note = (
        f"{r['games_in_tier']} in-tier / {r['games_total']} total"
    )

    print(f"\n{prefix}{r['player_name']}  |  {r['team']}  |  {r['tier_display']}")
    print(f"  Last game : {r['last_game']}   Games : {games_note}   "
          f"Confidence : {conf} {r['confidence']}")
    print(f"  {'':32}  {'PTS':>6}  {'REB':>6}  {'AST':>6}  {'STL':>6}  {'BLK':>6}")
    print(f"  {'Predicted next game':32}  "
          f"{r['pred_pts']:>6.1f}  {r['pred_reb']:>6.1f}  {r['pred_ast']:>6.1f}  "
          f"{r['pred_stl']:>6.1f}  {r['pred_blk']:>6.1f}")
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

    print(f"\n{prefix}{r1['player_name']}")
    print(f"  Last game : {r1['last_game']}")

    print(f"\n  {'':28}  {'--- ' + header1 + ' ---':^28}  {'--- ' + header2 + ' ---':^28}")
    print(f"  {'':28}  {'Conf: ' + r1['confidence']:^28}  {'Conf: ' + r2['confidence']:^28}")
    print(f"  {'':28}  {str(r1['games_in_tier'])+' games in tier':^28}  "
          f"{str(r2['games_in_tier'])+' games in tier':^28}")

    cols = ["PTS", "REB", "AST", "STL", "BLK"]
    keys = ["pred_pts", "pred_reb", "pred_ast", "pred_stl", "pred_blk"]
    ck1  = ["tier_career_pts", "tier_career_reb", "tier_career_ast",
             "tier_career_stl", "tier_career_blk"]
    ck2  = ck1

    print(f"\n  {'Stat':8}  {'Predicted':>10}  {'Tier career':>12}  "
          f"  {'Predicted':>10}  {'Tier career':>12}")
    print("  " + "-" * 60)
    for col, k, c1, c2 in zip(cols, keys, ck1, ck2):
        print(f"  {col:<8}  {r1[k]:>10.1f}  {r1[c1]:>12.1f}  "
              f"  {r2[k]:>10.1f}  {r2[c2]:>12.1f}")

    # Probabilities for both tiers side-by-side
    p1 = r1.get("probs", {})
    p2 = r2.get("probs", {})
    if p1 or p2:
        prob_order = [
            ("pts_10_plus", "10+ PTS"),
            ("pts_15_plus", "15+ PTS"),
            ("pts_20_plus", "20+ PTS"),
            ("reb_5_plus",  " 5+ REB"),
            ("reb_10_plus", "10+ REB"),
            ("ast_5_plus",  " 5+ AST"),
            ("double_double", "Dbl-Dbl"),
        ]
        print(f"\n  {'Probability':20}  {'Tier 1':>8}  {'Tier 2':>8}")
        print("  " + "-" * 38)
        for key, label in prob_order:
            v1 = _fmt_prob(p1.get(key))
            v2 = _fmt_prob(p2.get(key))
            print(f"  {label:<20}  {v1:>8}  {v2:>8}")


def print_banner() -> None:
    print("=" * 64)
    print("  IMBA PLAYER PERFORMANCE PREDICTOR  (V3 — probability models)")
    print("=" * 64)


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
        description="Predict next-game stats for an IMBA player (V2 tier-aware)."
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
            # Show both tiers; warn if player hasn't appeared in a tier
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
            # Default: use most recent tier; hint about --both if multi-tier
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
        player_df  = df[df["player_id"] == pid]
        last_tier_val = player_df.sort_values("date").iloc[-1]["tier"]
        tlabel     = {v: k for k, v in TIER_MAP.items()}[last_tier_val]
        try:
            r = predict_player_tier(player_df, models, tlabel, prob_models or None)
            results.append(r)
        except Exception as e:
            print(f"  Warning: {pname}: {e}")

    for i, r in enumerate(results, 1):
        print_prediction(r, idx=i)

    # Summary table
    if results:
        print("\n" + "=" * 64)
        print("  SUMMARY")
        print("=" * 64)
        print(f"  {'Player':<22}  {'Tier':>5}  {'Conf':<6}  "
              f"{'PTS':>5}  {'REB':>5}  {'AST':>5}  {'STL':>5}  {'BLK':>5}")
        print("  " + "-" * 62)
        for r in results:
            tier_short = "T1" if r["tier_label"] == "tier1" else "T2"
            print(f"  {r['player_name']:<22}  {tier_short:>5}  {r['confidence']:<6}  "
                  f"{r['pred_pts']:>5.1f}  {r['pred_reb']:>5.1f}  {r['pred_ast']:>5.1f}  "
                  f"{r['pred_stl']:>5.1f}  {r['pred_blk']:>5.1f}")
        print()


if __name__ == "__main__":
    main()
