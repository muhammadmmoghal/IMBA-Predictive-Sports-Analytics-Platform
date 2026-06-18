#!/usr/bin/env python3
"""
Convert IMBA prediction CSVs into JSON files for the Next.js frontend.

Run from anywhere:
    python frontend/scripts/convert_csv_to_json.py

Output goes to:
    frontend/public/data/
"""

import csv
import json
import sys
from pathlib import Path

# ── Path resolution ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # frontend/scripts/
FRONTEND_DIR = SCRIPT_DIR.parent                       # frontend/
PROJECT_ROOT = FRONTEND_DIR.parent                     # IMBA-Predictive-Sports.../

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = FRONTEND_DIR / "public" / "data"

# ── Column type mapping ───────────────────────────────────────────────────────
INT_COLS = {"confidence_score", "games_played_history"}
FLOAT_COLS = {
    "predicted_pts", "predicted_reb", "predicted_ast", "predicted_stl", "predicted_blk",
    "pts_low", "pts_high",
    "reb_low", "reb_high",
    "ast_low", "ast_high",
    "stl_low", "stl_high",
    "blk_low", "blk_high",
    "prob_10_plus_pts", "prob_15_plus_pts", "prob_20_plus_pts",
    "prob_5_plus_reb", "prob_10_plus_reb",
    "prob_5_plus_ast",
    "prob_double_double",
}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [SKIP] {path.name} not found at {path}")
        return []

    rows = []
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted: dict = {}
            for k, v in row.items():
                key = k.strip()
                val = v.strip() if v else ""
                if key in INT_COLS:
                    try:
                        converted[key] = int(float(val))
                    except (ValueError, TypeError):
                        converted[key] = 0
                elif key in FLOAT_COLS:
                    try:
                        converted[key] = round(float(val), 2)
                    except (ValueError, TypeError):
                        converted[key] = 0.0
                else:
                    converted[key] = val
            rows.append(converted)
    return rows


def write_json(data: list[dict], filename: str) -> None:
    out_path = OUTPUT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    kb = out_path.stat().st_size / 1024
    print(f"  [OK]   {filename}  ({len(data)} records, {kb:.1f} KB)")


def main() -> None:
    print("\n=== IMBA CSV -> JSON Converter ===\n")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output -> {OUTPUT_DIR}\n")

    # ── Load source data ─────────────────────────────────────────────────────
    all_players = read_csv(DATA_DIR / "current_predictions_all.csv")
    d2_comp = read_csv(DATA_DIR / "current_predictions_d2_comp.csv")
    d2_rec = read_csv(DATA_DIR / "current_predictions_d2_rec.csv")
    top_scorers = read_csv(REPORTS_DIR / "top_projected_scorers.csv")
    top_rebounders = read_csv(REPORTS_DIR / "top_projected_rebounders.csv")
    top_assists = read_csv(REPORTS_DIR / "top_projected_assists.csv")
    top_dd = read_csv(REPORTS_DIR / "top_double_double_candidates.csv")

    if not all_players:
        print("ERROR: No data loaded. Make sure prediction CSVs exist in data/processed/")
        print(f"  Expected: {DATA_DIR / 'current_predictions_all.csv'}")
        sys.exit(1)

    # Sort comp/rec by predicted_pts desc if not already sorted
    d2_comp_sorted = sorted(d2_comp, key=lambda p: p.get("predicted_pts", 0), reverse=True)
    d2_rec_sorted = sorted(d2_rec, key=lambda p: p.get("predicted_pts", 0), reverse=True)

    # ── Write JSON files ─────────────────────────────────────────────────────
    write_json(all_players, "all_players.json")
    write_json(d2_comp_sorted, "d2_comp.json")
    write_json(d2_rec_sorted, "d2_rec.json")
    write_json(top_scorers, "top_scorers.json")
    write_json(top_rebounders, "top_rebounders.json")
    write_json(top_assists, "top_assists.json")
    write_json(top_dd, "top_double_double.json")

    print(f"\nDone. {len(all_players)} total players converted.\n")
    print("Next steps:")
    print("  cd frontend && npm install && npm run dev")
    print("  Open http://localhost:3000\n")


if __name__ == "__main__":
    main()
