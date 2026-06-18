'use client';

import { Suspense, useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import type { Player } from '@/lib/types';
import { cn, fmt, shortDivision } from '@/lib/utils';
import ConfidenceBadge from '@/components/ConfidenceBadge';
import TeamLogo from '@/components/TeamLogo';
import ProbabilityBar from '@/components/ProbabilityBar';

// ── Deduplication helpers ────────────────────────────────────────────────────

// Normalize name: lowercase, trim, collapse internal whitespace
function normalizeName(name: string): string {
  return name.toLowerCase().trim().replace(/\s+/g, ' ');
}

// Group key is always the normalized name — player_id is intentionally ignored
// because different division rows for the same player can have different player_ids
function playerKey(p: Player): string {
  return normalizeName(p.player_name);
}

const CONF_RANK: Record<string, number> = { High: 2, Medium: 1, Low: 0 };

// One representative per unique player for the search dropdown (highest confidence wins)
function dedupeForSearch(players: Player[]): Player[] {
  const best = new Map<string, Player>();
  for (const p of players) {
    const k = playerKey(p);
    const cur = best.get(k);
    if (!cur || (CONF_RANK[p.confidence_level] ?? 0) > (CONF_RANK[cur.confidence_level] ?? 0)) {
      best.set(k, p);
    }
  }
  return Array.from(best.values());
}

// All division-level records for a player, deduplicated to one per short-division
function getDivisionMap(allPlayers: Player[], rep: Player): Map<string, Player> {
  const k = playerKey(rep);
  const map = new Map<string, Player>();
  for (const p of allPlayers) {
    if (playerKey(p) !== k) continue;
    const div = shortDivision(p.division);
    const cur = map.get(div);
    if (!cur || (CONF_RANK[p.confidence_level] ?? 0) > (CONF_RANK[cur.confidence_level] ?? 0)) {
      map.set(div, p);
    }
  }
  return map;
}

// Division keys in canonical order
const DIV_ORDER = ['D2 Comp', 'D2 Rec'];
function sortedDivKeys(map: Map<string, Player>): string[] {
  return Array.from(map.keys()).sort(
    (a, b) => DIV_ORDER.indexOf(a) - DIV_ORDER.indexOf(b)
  );
}

// ── StatPill ─────────────────────────────────────────────────────────────────

interface StatPillProps {
  label: string;
  value: number;
  low: number;
  high: number;
  accent?: 'green' | 'gold' | 'default';
}

function StatPill({ label, value, low, high, accent = 'default' }: StatPillProps) {
  const valueColor = {
    green: 'text-gradient-green',
    gold: 'text-gradient-gold',
    default: 'text-white',
  }[accent];

  const border = {
    green: 'border-blue-500/20 bg-blue-500/5',
    gold: 'border-sky-500/20 bg-sky-500/5',
    default: 'border-blue-500/20 bg-[#091426]',
  }[accent];

  return (
    <div className={cn('rounded-xl border p-4 text-center flex-1 min-w-[100px]', border)}>
      <p className={cn('text-3xl font-black tabular-nums mb-0.5', valueColor)}>{fmt(value)}</p>
      <p className="text-xs font-bold tracking-widest text-slate-500 uppercase mb-2">{label}</p>
      <p className="text-xs text-slate-600 tabular-nums">
        {fmt(low)}–{fmt(high)}
      </p>
    </div>
  );
}

// ── PlayerPanel ───────────────────────────────────────────────────────────────
// Key this component from the parent so state resets on player change.

function PlayerPanel({ divMap }: { divMap: Map<string, Player> }) {
  const divKeys = sortedDivKeys(divMap);

  // Default to D2 Comp if available, otherwise the first available division
  const defaultDiv = divKeys.includes('D2 Comp') ? 'D2 Comp' : divKeys[0];

  const [selectedDiv, setSelectedDiv] = useState(defaultDiv);
  const player = divMap.get(selectedDiv) ?? divMap.get(divKeys[0])!;
  const isComp = selectedDiv === 'D2 Comp';

  return (
    <div className="animate-fade-up space-y-6">
      {/* Player Header */}
      <div className={cn('rounded-xl p-6 border', isComp ? 'card-green' : 'card-gold')}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="font-heading text-3xl font-black uppercase text-white mb-1">
              {player.player_name}
            </h2>
            <div className="flex items-center gap-2 text-slate-400">
              <TeamLogo team={player.team} size="md" />
              {player.team} &bull;{' '}
              <span className={isComp ? 'text-blue-400' : 'text-sky-400'}>{selectedDiv}</span>
            </div>
            <p className="text-slate-600 text-sm mt-1">
              {player.games_played_history} game{player.games_played_history !== 1 ? 's' : ''} played (history)
            </p>

            {/* Division toggle — always visible so user knows what divisions exist */}
            <div className="flex gap-2 mt-3">
              {divKeys.map((div) => (
                <button
                  key={div}
                  onClick={() => setSelectedDiv(div)}
                  className={cn(
                    'px-3 py-1 rounded-lg text-xs font-bold tracking-wider uppercase border transition-all',
                    selectedDiv === div
                      ? div === 'D2 Comp'
                        ? 'bg-blue-500/15 text-blue-400 border-blue-500/30'
                        : 'bg-sky-500/15 text-sky-400 border-sky-500/30'
                      : 'text-slate-500 border-blue-500/20 hover:border-blue-500/40 hover:text-slate-300'
                  )}
                >
                  {div}
                </button>
              ))}
            </div>
          </div>
          <ConfidenceBadge level={player.confidence_level} score={player.confidence_score} />
        </div>
      </div>

      {/* Main Stat Pills */}
      <div>
        <p className="text-[11px] font-bold tracking-widest text-slate-500 uppercase mb-3">
          Season Projections
        </p>
        <div className="flex flex-wrap gap-3">
          <StatPill label="PTS" value={player.predicted_pts} low={player.pts_low} high={player.pts_high} accent="green" />
          <StatPill label="REB" value={player.predicted_reb} low={player.reb_low} high={player.reb_high} accent="green" />
          <StatPill label="AST" value={player.predicted_ast} low={player.ast_low} high={player.ast_high} accent="gold" />
          <StatPill label="STL" value={player.predicted_stl} low={player.stl_low} high={player.stl_high} />
          <StatPill label="BLK" value={player.predicted_blk} low={player.blk_low} high={player.blk_high} />
        </div>
        <p className="text-xs text-slate-600 mt-2">Values shown as predicted average. Range shows prediction interval.</p>
      </div>

      {/* Probability Thresholds */}
      <div className="card rounded-xl p-6">
        <p className="text-[11px] font-bold tracking-widest text-slate-500 uppercase mb-5">
          Probability Thresholds
        </p>
        <div className="grid sm:grid-cols-2 gap-5">
          <div className="space-y-4">
            <p className="text-[11px] font-bold text-slate-600 uppercase tracking-wider">Scoring</p>
            <ProbabilityBar label="10+ Points" value={player.prob_10_plus_pts} />
            <ProbabilityBar label="15+ Points" value={player.prob_15_plus_pts} />
            <ProbabilityBar label="20+ Points" value={player.prob_20_plus_pts} />
          </div>
          <div className="space-y-4">
            <p className="text-[11px] font-bold text-slate-600 uppercase tracking-wider">Other</p>
            <ProbabilityBar label="5+ Rebounds" value={player.prob_5_plus_reb} />
            <ProbabilityBar label="10+ Rebounds" value={player.prob_10_plus_reb} />
            <ProbabilityBar label="5+ Assists" value={player.prob_5_plus_ast} />
            <ProbabilityBar label="Double-Double" value={player.prob_double_double} />
          </div>
        </div>
      </div>

      {/* Confidence Meter */}
      <div className="card rounded-xl p-6">
        <p className="text-[11px] font-bold tracking-widest text-slate-500 uppercase mb-4">
          Confidence Score
        </p>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="h-2.5 rounded-full bg-[#060E1C] overflow-hidden">
              <div
                className={cn(
                  'h-full rounded-full transition-all duration-700',
                  player.confidence_level === 'High'
                    ? 'bg-emerald-500'
                    : player.confidence_level === 'Medium'
                    ? 'bg-yellow-500'
                    : 'bg-red-500'
                )}
                style={{ width: `${player.confidence_score}%` }}
              />
            </div>
          </div>
          <span
            className={cn(
              'text-2xl font-black tabular-nums',
              player.confidence_level === 'High'
                ? 'text-emerald-400'
                : player.confidence_level === 'Medium'
                ? 'text-yellow-400'
                : 'text-red-400'
            )}
          >
            {player.confidence_score}
          </span>
          <span className="text-slate-500 text-sm">/ 100</span>
        </div>
        <p className="text-xs text-slate-600 mt-3">
          Confidence is based on the number of historical games available for this player.
          Higher game counts yield more reliable predictions.
        </p>
      </div>
    </div>
  );
}

// ── PlayersContent ────────────────────────────────────────────────────────────

function PlayersContent() {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get('q') ?? '';

  const [allPlayers, setAllPlayers] = useState<Player[]>([]);
  const [uniquePlayers, setUniquePlayers] = useState<Player[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState(initialQuery);
  const [selectedRep, setSelectedRep] = useState<Player | null>(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch('/data/all_players.json')
      .then((r) => {
        if (!r.ok) throw new Error('not found');
        return r.json();
      })
      .then((data: Player[]) => {
        const deduped = dedupeForSearch(data);
        setAllPlayers(data);
        setUniquePlayers(deduped);
        if (initialQuery) {
          const match = deduped.find(
            (p) => p.player_name.toLowerCase() === initialQuery.toLowerCase()
          );
          if (match) setSelectedRep(match);
        }
      })
      .catch(() => setError('Run convert_csv_to_json.py to generate data files.'))
      .finally(() => setLoading(false));
  }, [initialQuery]);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  // Search filters the deduplicated list — one row per unique player
  const filtered = search.length >= 1
    ? uniquePlayers
        .filter((p) => p.player_name.toLowerCase().includes(search.toLowerCase()))
        .slice(0, 8)
    : [];

  function selectPlayer(rep: Player) {
    setSelectedRep(rep);
    setSearch(rep.player_name);
    setShowDropdown(false);
  }

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-10 space-y-6 animate-pulse">
        <div className="h-8 w-64 shimmer rounded" />
        <div className="h-14 shimmer rounded-xl" />
        <div className="h-48 shimmer rounded-xl" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-20 text-center">
        <p className="text-rose-400 font-semibold mb-2">Data unavailable</p>
        <p className="text-slate-500 text-sm">{error}</p>
      </div>
    );
  }

  const divMap = selectedRep ? getDivisionMap(allPlayers, selectedRep) : null;

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="font-heading text-3xl font-black uppercase tracking-wider text-white mb-1">
          Player Predictor
        </h1>
        <p className="text-slate-400">Search any player to view their season projections and probability breakdown.</p>
      </div>

      {/* Search */}
      <div className="relative mb-8" ref={wrapperRef}>
        <div className="relative">
          <svg
            className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500"
            viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
          >
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.35-4.35" />
          </svg>
          <input
            type="text"
            placeholder="Search by player name…"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setShowDropdown(true);
              if (e.target.value === '') setSelectedRep(null);
            }}
            onFocus={() => setShowDropdown(true)}
            className="w-full bg-[#091426] border border-blue-500/20 rounded-xl pl-11 pr-4 py-3.5 text-slate-100 placeholder-slate-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20 transition-all text-base"
          />
          {search && (
            <button
              onClick={() => { setSearch(''); setSelectedRep(null); setShowDropdown(false); }}
              className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
            >
              ✕
            </button>
          )}
        </div>

        {/* Dropdown — one row per unique player */}
        {showDropdown && filtered.length > 0 && (
          <div className="absolute z-30 w-full mt-1 bg-[#091426] border border-blue-500/20 rounded-xl overflow-hidden shadow-2xl">
            {filtered.map((rep) => {
              const divs = sortedDivKeys(getDivisionMap(allPlayers, rep));
              return (
                <button
                  key={playerKey(rep)}
                  onClick={() => selectPlayer(rep)}
                  className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-[#0C1A33] transition-colors border-b border-blue-500/[0.08] last:border-0"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <TeamLogo team={rep.team} size="xs" />
                    <div className="min-w-0">
                      <p className="font-semibold text-slate-100">{rep.player_name}</p>
                      <p className="text-xs text-slate-500">
                        {rep.team} · {divs.join(' · ')}
                      </p>
                    </div>
                  </div>
                  <ConfidenceBadge level={rep.confidence_level} showScore={false} />
                </button>
              );
            })}
          </div>
        )}

        {showDropdown && search.length >= 1 && filtered.length === 0 && (
          <div className="absolute z-30 w-full mt-1 bg-[#091426] border border-blue-500/20 rounded-xl px-4 py-3 text-slate-500 text-sm">
            No players found for &quot;{search}&quot;
          </div>
        )}
      </div>

      {/* Player Panel — keyed by identity so state resets on player change */}
      {divMap ? (
        <PlayerPanel key={selectedRep ? playerKey(selectedRep) : ''} divMap={divMap} />
      ) : (
        <div className="card rounded-xl p-14 text-center">
          <div className="w-14 h-14 mx-auto mb-5 opacity-15">
            <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="50" cy="50" r="47" stroke="#3B82F6" strokeWidth="2"/>
              <path d="M50 10 L60 36 L88 38 L67 55 L74 82 L50 67 L26 82 L33 55 L12 38 L40 36 Z"
                    fill="none" stroke="#3B82F6" strokeWidth="2.5" strokeLinejoin="round"/>
            </svg>
          </div>
          <p className="text-slate-400 font-semibold mb-1">Select a player to see their projections</p>
          <p className="text-slate-600 text-sm">
            {uniquePlayers.length} players available across D2 Comp and D2 Rec
          </p>
        </div>
      )}
    </div>
  );
}

export default function PlayersPage() {
  return (
    <Suspense
      fallback={
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-10 space-y-6 animate-pulse">
          <div className="h-8 w-64 shimmer rounded" />
          <div className="h-14 shimmer rounded-xl" />
          <div className="h-48 shimmer rounded-xl" />
        </div>
      }
    >
      <PlayersContent />
    </Suspense>
  );
}
