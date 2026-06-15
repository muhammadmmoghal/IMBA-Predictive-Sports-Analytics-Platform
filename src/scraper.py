"""
scraper.py
Scrapes historical player-game records from the imbaonline.com JSON API.

API:
  /api/games              -> full list of every league game (metadata + played status)
  /api/games/{id}         -> box score: playerStats[] with pts/reb/ast/etc per player

Approach (game-centric, per-season):
  Phase 1  For each target season, fetch /api/games?season=S         -> game IDs
             (Using per-season filter is required: the unfiltered /api/games
              endpoint returns only the 500 oldest league games globally, silently
              cutting off the two most recent seasons.)
  Phase 2  Fetch /api/games/{id} for each game, parse playerStats    -> rows

This avoids the 20-game hard cap on /api/players/{id}.recentGames that silently
truncates career history for long-tenured players.

Output: data/raw/game_logs_raw.csv
Usage:  python src/scraper.py
"""

import time
import logging
import sys
from pathlib import Path

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.imbaonline.com"
OUTPUT_PATH = Path("data/raw/game_logs_raw.csv")
REQUEST_DELAY = 0.35       # seconds between API calls
MAX_RETRIES = 3
RETRY_DELAY = 2.0

TARGET_SEASONS = {
    "D1 2025-26 Winter",
    "D2 2025-26 Winter",
    "D2 2025 Summer",
    "D1 2024 Winter",
    "D2 2024 Winter",
    "D1 2024 Summer",
}

EXCLUDE_SEASONS = {
    "D2 Comp 2026 Summer",
    "D2 Rec 2026 Summer",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

OUTPUT_COLUMNS = [
    "player_id", "player_name", "player_number", "player_position",
    "game_id", "date", "season", "tier",
    "team", "opponent", "result", "my_score", "opp_score",
    "pts", "reb", "ast", "stl", "blk",
    "turnovers", "fouls",
    "two_pt_made", "two_pt_att",
    "three_made", "three_att",
    "ft_made", "ft_att",
]

# ---------------------------------------------------------------------------
# Logging setup
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
    """GET with retry. Returns parsed JSON or None on failure."""
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
# Phase 1: collect game IDs for target seasons
# ---------------------------------------------------------------------------

def collect_game_ids(session: requests.Session) -> list[dict]:
    """
    Returns a list of game metadata dicts for every played game in TARGET_SEASONS.

    Queries /api/games?season=S once per season rather than the unfiltered
    /api/games endpoint.  The unfiltered endpoint is capped at 500 oldest games
    globally and silently drops recent seasons (D1/D2 2025-26 Winter are
    missing 23 and 34 games respectively when queried without a season param).
    """
    log.info("Phase 1 — fetching game lists per target season")
    target_games: list[dict] = []
    seen_ids: set[str] = set()

    for season in TARGET_SEASONS:
        data = _get(session, f"{BASE_URL}/api/games", params={"season": season})
        time.sleep(REQUEST_DELAY)

        if not data:
            log.warning("  No data for season: %s", season)
            continue

        played = [
            g for g in data
            if g.get("played") is True and g.get("id") not in seen_ids
        ]
        for g in played:
            seen_ids.add(g["id"])
        target_games.extend(played)
        log.info("  %-24s  %3d played games", season, len(played))

    log.info("Total played games to fetch: %d", len(target_games))
    return target_games


# ---------------------------------------------------------------------------
# Phase 2: fetch box scores and build rows
# ---------------------------------------------------------------------------

def _parse_stat(stat: dict, game: dict) -> dict:
    """
    Convert one playerStats entry + its parent game record into a flat row.
    Determines opponent, my_score, opp_score, and result from home/away metadata.
    """
    player    = stat.get("player") or {}
    team_meta = stat.get("team") or {}
    team_id   = stat.get("teamId", "")
    season    = game.get("season", "")

    # Determine home/away side for this player's team
    is_home = (team_id == game.get("homeTeamId"))
    if is_home:
        my_score   = game.get("homeScore")
        opp_score  = game.get("awayScore")
        opponent   = (game.get("awayTeam") or {}).get("name", "")
    else:
        my_score   = game.get("awayScore")
        opp_score  = game.get("homeScore")
        opponent   = (game.get("homeTeam") or {}).get("name", "")

    if my_score is not None and opp_score is not None:
        result = "W" if my_score > opp_score else ("L" if my_score < opp_score else "T")
    else:
        result = ""

    date_raw = game.get("date", "")

    return {
        "player_id":       stat.get("playerId", ""),
        "player_name":     player.get("name", ""),
        "player_number":   player.get("number"),
        "player_position": player.get("position", ""),
        "game_id":         stat.get("gameId", ""),
        "date":            date_raw[:10] if date_raw else "",
        "season":          season,
        "tier":            season.split()[0] if season else "",
        "team":            team_meta.get("name", ""),
        "opponent":        opponent,
        "result":          result,
        "my_score":        my_score,
        "opp_score":       opp_score,
        "pts":             stat.get("points"),
        "reb":             stat.get("rebounds"),
        "ast":             stat.get("assists"),
        "stl":             stat.get("steals"),
        "blk":             stat.get("blocks"),
        "turnovers":       stat.get("turnovers"),
        "fouls":           stat.get("fouls"),
        "two_pt_made":     stat.get("twoPtMade"),
        "two_pt_att":      stat.get("twoPtAtt"),
        "three_made":      stat.get("threeMade"),
        "three_att":       stat.get("threeAtt"),
        "ft_made":         stat.get("ftMade"),
        "ft_att":          stat.get("ftAtt"),
    }


def collect_box_scores(
    session: requests.Session,
    games: list[dict],
) -> tuple[list[dict], list[str]]:
    """
    Fetches /api/games/{id} for every game and extracts playerStats rows.
    Deduplicates on (player_id, game_id) in case of API duplication.
    Returns (rows, failed_game_ids).
    """
    log.info("Phase 2 — fetching box scores for %d games", len(games))
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()   # (player_id, game_id)
    failed: list[str] = []

    total = len(games)
    for idx, game_meta in enumerate(games, 1):
        game_id = game_meta["id"]

        if idx % 25 == 0 or idx == total:
            log.info("  [%d/%d] %d player-game rows collected", idx, total, len(rows))

        data = _get(session, f"{BASE_URL}/api/games/{game_id}")
        time.sleep(REQUEST_DELAY)

        if data is None:
            failed.append(game_id)
            continue

        # Merge game-level metadata not returned in /api/games/{id} outer fields
        # (date, homeScore, awayScore, homeTeamId, awayTeamId are present in the response)
        for stat in data.get("playerStats", []):
            player_id = stat.get("playerId", "")
            dedup_key = (player_id, game_id)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            rows.append(_parse_stat(stat, data))

    return rows, failed


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------

def print_audit(df: pd.DataFrame, n_games: int, failed: list[str]) -> None:
    sep = "=" * 60

    print()
    print(sep)
    print("SCRAPE COMPLETE — AUDIT REPORT")
    print(sep)
    print(f"  Games fetched:          {n_games}")
    print(f"  Failed game fetches:    {len(failed)}")
    print(f"  Player-game rows:       {len(df)}")
    print(f"  Unique players:         {df['player_id'].nunique()}")
    print(f"  Unique game IDs in CSV: {df['game_id'].nunique()}")
    print()

    print("Rows by season:")
    season_counts = df.groupby("season").size().sort_values(ascending=False)
    for s, c in season_counts.items():
        print(f"  {s:<25}  {c:>5} rows")
    print()

    print("Games covered per season (unique game IDs):")
    games_per_season = df.groupby("season")["game_id"].nunique().sort_values(ascending=False)
    for s, c in games_per_season.items():
        print(f"  {s:<25}  {c:>5} games")
    print()

    gpd = df.groupby(["player_id", "player_name"]).size().sort_values(ascending=False)
    max_games = gpd.max()
    print(f"Max games by a single player: {max_games}  (cap was 20 before fix)")
    print(f"Players with 20+ games:       {(gpd >= 20).sum()}")
    print(f"Players with 30+ games:       {(gpd >= 30).sum()}")
    print()

    print("Top 10 players by games played:")
    for (pid, pname), cnt in gpd.head(10).items():
        print(f"  {pname:<30}  {cnt:>3} games   (id: {pid[:20]}...)")
    print()

    print("Games-per-player distribution:")
    buckets = pd.cut(gpd, bins=[0, 5, 10, 15, 20, 25, 30, 40, 999],
                     labels=["1-5","6-10","11-15","16-20","21-25","26-30","31-40","40+"])
    for label, count in buckets.value_counts().sort_index().items():
        print(f"  {label:>6} games: {count:>3} players")
    print()

    print("Missing-value check:")
    null_counts = df[["pts","reb","ast","stl","blk"]].isnull().sum()
    for col, cnt in null_counts.items():
        status = "OK" if cnt == 0 else f"WARNING — {cnt} nulls"
        print(f"  {col}: {status}")

    excluded_check = df[df["season"].isin(EXCLUDE_SEASONS)]
    print(f"\nExcluded seasons present in CSV: {len(excluded_check)} rows (expect 0)")

    if failed:
        print(f"\nFailed game IDs: {failed}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    # Phase 1 — game IDs
    games = collect_game_ids(session)

    # Phase 2 — box scores
    rows, failed = collect_box_scores(session, games)

    if not rows:
        log.error("No data collected — CSV not written.")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved %d rows to %s", len(df), OUTPUT_PATH)

    print_audit(df, len(games), failed)


if __name__ == "__main__":
    main()
