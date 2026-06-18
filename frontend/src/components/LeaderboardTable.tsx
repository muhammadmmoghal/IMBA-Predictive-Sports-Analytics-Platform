'use client';

import { useState } from 'react';
import { cn, fmt, shortDivision } from '@/lib/utils';
import ConfidenceBadge from './ConfidenceBadge';
import TeamLogo from './TeamLogo';
import type { Player } from '@/lib/types';
import Link from 'next/link';

type SortKey = 'predicted_pts' | 'predicted_reb' | 'predicted_ast' | 'predicted_stl' | 'predicted_blk' | 'prob_double_double';
type SortDir = 'asc' | 'desc';

interface LeaderboardTableProps {
  players: Player[];
  statKey?: keyof Player;
  statLabel?: string;
  limit?: number;
}

const RANK_STYLES: Record<number, string> = {
  1: 'text-blue-400 font-black',
  2: 'text-slate-300 font-bold',
  3: 'text-blue-600 font-bold',
};

const RANK_BG: Record<number, string> = {
  1: 'bg-blue-400/10 border-blue-400/25',
  2: 'bg-slate-500/10 border-slate-500/20',
  3: 'bg-blue-800/10 border-blue-800/20',
};

const STAT_COLS: { key: SortKey; label: string }[] = [
  { key: 'predicted_pts', label: 'PTS' },
  { key: 'predicted_reb', label: 'REB' },
  { key: 'predicted_ast', label: 'AST' },
  { key: 'predicted_stl', label: 'STL' },
  { key: 'predicted_blk', label: 'BLK' },
  { key: 'prob_double_double', label: 'DD%' },
];

const EMPTY = (
  <div className="rounded-xl border border-blue-500/20 bg-[#091426] p-10 text-center text-[#8EA7C8]">
    No players to display.
  </div>
);

export default function LeaderboardTable({ players, statKey, statLabel, limit = 20 }: LeaderboardTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('predicted_pts');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  }

  // ── Multi-column sortable mode (leaderboards page) ──────────────────────
  if (!statKey) {
    const sorted = [...players]
      .sort((a, b) => {
        const av = a[sortKey] as number;
        const bv = b[sortKey] as number;
        return sortDir === 'desc' ? bv - av : av - bv;
      })
      .slice(0, limit);

    if (sorted.length === 0) return EMPTY;

    return (
      <div className="overflow-x-auto rounded-xl border border-blue-500/20">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-blue-500/20 bg-[#060E1C]">
              <th className="px-5 py-4 text-left text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase w-14">#</th>
              <th className="px-5 py-4 text-left text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase">PLAYER</th>
              <th className="px-5 py-4 text-left text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase hidden md:table-cell">TEAM</th>
              <th className="px-5 py-4 text-left text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase hidden lg:table-cell">DIVISION</th>
              {STAT_COLS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className={cn(
                    'px-4 py-4 text-right text-[10px] font-bold tracking-[0.15em] uppercase cursor-pointer select-none whitespace-nowrap transition-colors',
                    sortKey === col.key
                      ? 'text-blue-400'
                      : 'text-blue-400/80 hover:text-blue-400'
                  )}
                >
                  {col.label}
                  {sortKey === col.key && (
                    <span className="ml-1">{sortDir === 'desc' ? '↓' : '↑'}</span>
                  )}
                </th>
              ))}
              <th className="px-5 py-4 text-center text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase hidden sm:table-cell">CONFIDENCE</th>
              <th className="px-5 py-4 text-right text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase hidden md:table-cell">GP</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((player, index) => {
              const rank = index + 1;
              return (
                <tr
                  key={`${player.player_name}-${index}`}
                  className="border-b border-blue-500/[0.08] bg-[#091426] transition-colors hover:bg-[#0C1A33]"
                >
                  <td className="px-5 py-3">
                    <span
                      className={cn(
                        'inline-flex items-center justify-center w-7 h-7 rounded-lg text-xs border',
                        rank <= 3
                          ? RANK_BG[rank] + ' ' + RANK_STYLES[rank]
                          : 'text-slate-600 border-transparent'
                      )}
                    >
                      {rank}
                    </span>
                  </td>
                  <td className="px-5 py-3">
                    <Link
                      href={`/players?q=${encodeURIComponent(player.player_name)}`}
                      className="font-semibold text-slate-100 hover:text-blue-400 transition-colors"
                    >
                      {player.player_name}
                    </Link>
                  </td>
                  <td className="px-5 py-3 hidden md:table-cell">
                    <div className="flex items-center gap-2 text-[#8EA7C8]">
                      <TeamLogo team={player.team} size="sm" />
                      {player.team}
                    </div>
                  </td>
                  <td className="px-5 py-3 hidden lg:table-cell">
                    <span
                      className={cn(
                        'text-[10px] px-2.5 py-1 rounded font-bold tracking-wider uppercase border',
                        shortDivision(player.division) === 'D2 Comp'
                          ? 'bg-blue-500/10 text-blue-400 border-blue-500/25'
                          : 'bg-sky-500/10 text-sky-400 border-sky-500/25'
                      )}
                    >
                      {shortDivision(player.division)}
                    </span>
                  </td>
                  {STAT_COLS.map((col) => (
                    <td
                      key={col.key}
                      className={cn(
                        'px-4 py-3 text-right tabular-nums',
                        sortKey === col.key
                          ? 'font-black text-white text-lg'
                          : 'text-slate-300'
                      )}
                    >
                      {col.key === 'prob_double_double'
                        ? `${fmt(player[col.key] as number, 1)}%`
                        : fmt(player[col.key] as number)}
                    </td>
                  ))}
                  <td className="px-5 py-3 text-center hidden sm:table-cell">
                    <ConfidenceBadge level={player.confidence_level} showScore={false} />
                  </td>
                  <td className="px-5 py-3 text-right text-[#8EA7C8] hidden md:table-cell tabular-nums">
                    {player.games_played_history}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  // ── Single-column mode (homepage "Top Scorers" table) ───────────────────
  const displayed = players.slice(0, limit);

  if (displayed.length === 0) return EMPTY;

  return (
    <div className="overflow-x-auto rounded-xl border border-blue-500/20">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-blue-500/20 bg-[#060E1C]">
            <th className="px-5 py-4 text-left text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase w-14">#</th>
            <th className="px-5 py-4 text-left text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase">PLAYER</th>
            <th className="px-5 py-4 text-left text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase hidden md:table-cell">TEAM</th>
            <th className="px-5 py-4 text-left text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase hidden lg:table-cell">DIVISION</th>
            <th className="px-5 py-4 text-right text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase">{statLabel}</th>
            <th className="px-5 py-4 text-center text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase hidden sm:table-cell">CONFIDENCE LEVEL</th>
            <th className="px-5 py-4 text-right text-[10px] font-bold tracking-[0.15em] text-blue-400/80 uppercase hidden md:table-cell">GP</th>
          </tr>
        </thead>
        <tbody>
          {displayed.map((player, index) => {
            const rank = index + 1;
            const value = player[statKey] as number;
            return (
              <tr
                key={`${player.player_name}-${index}`}
                className="border-b border-blue-500/[0.08] bg-[#091426] transition-colors hover:bg-[#0C1A33]"
              >
                <td className="px-5 py-3">
                  <span
                    className={cn(
                      'inline-flex items-center justify-center w-7 h-7 rounded-lg text-xs border',
                      rank <= 3
                        ? RANK_BG[rank] + ' ' + RANK_STYLES[rank]
                        : 'text-slate-600 border-transparent'
                    )}
                  >
                    {rank}
                  </span>
                </td>
                <td className="px-5 py-3">
                  <Link
                    href={`/players?q=${encodeURIComponent(player.player_name)}`}
                    className="font-semibold text-slate-100 hover:text-blue-400 transition-colors"
                  >
                    {player.player_name}
                  </Link>
                </td>
                <td className="px-5 py-3 hidden md:table-cell">
                  <div className="flex items-center gap-2 text-[#8EA7C8]">
                    <TeamLogo team={player.team} size="sm" />
                    {player.team}
                  </div>
                </td>
                <td className="px-5 py-3 hidden lg:table-cell">
                  <span
                    className={cn(
                      'text-[10px] px-2.5 py-1 rounded font-bold tracking-wider uppercase border',
                      shortDivision(player.division) === 'D2 Comp'
                        ? 'bg-blue-500/10 text-blue-400 border-blue-500/25'
                        : 'bg-sky-500/10 text-sky-400 border-sky-500/25'
                    )}
                  >
                    {shortDivision(player.division)}
                  </span>
                </td>
                <td className="px-5 py-3 text-right">
                  <span className="font-black text-white tabular-nums text-lg">
                    {statKey === 'prob_double_double'
                      ? `${fmt(value, 1)}%`
                      : fmt(value)}
                  </span>
                </td>
                <td className="px-5 py-3 text-center hidden sm:table-cell">
                  <ConfidenceBadge level={player.confidence_level} showScore={false} />
                </td>
                <td className="px-5 py-3 text-right text-[#8EA7C8] hidden md:table-cell tabular-nums">
                  {player.games_played_history}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
