'use client';

import { useEffect, useState } from 'react';
import type { Player } from '@/lib/types';
import type { DivisionFilter } from '@/lib/types';
import { cn, isComp, isRec } from '@/lib/utils';
import LeaderboardTable from '@/components/LeaderboardTable';

const DIVISION_FILTERS: { key: DivisionFilter; label: string }[] = [
  { key: 'all', label: 'All Players' },
  { key: 'comp', label: 'D2 Comp' },
  { key: 'rec', label: 'D2 Rec' },
];

export default function LeaderboardsPage() {
  const [players, setPlayers] = useState<Player[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [divFilter, setDivFilter] = useState<DivisionFilter>('all');

  useEffect(() => {
    async function load() {
      try {
        const r = await fetch('/data/all_players.json');
        if (!r.ok) throw new Error('not found');
        setPlayers(await r.json());
      } catch {
        setError('Run convert_csv_to_json.py to generate data files.');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const filteredPlayers = players.filter((p) => {
    if (divFilter === 'comp') return isComp(p.division);
    if (divFilter === 'rec') return isRec(p.division);
    return true;
  });

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-5 animate-pulse">
        <div className="h-8 w-48 shimmer rounded" />
        <div className="flex gap-3">
          {[...Array(3)].map((_, i) => <div key={i} className="h-10 w-28 shimmer rounded-xl" />)}
        </div>
        <div className="h-96 shimmer rounded-xl" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center">
        <p className="text-rose-400 font-semibold mb-2">Data unavailable</p>
        <p className="text-slate-500 text-sm">{error}</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="font-heading text-3xl font-black uppercase tracking-wider text-white mb-1">
          Leaderboards
        </h1>
        <p className="text-slate-400">Top projected performers across all statistical categories.</p>
      </div>

      {/* Division Filters */}
      <div className="flex gap-2 mb-8">
        {DIVISION_FILTERS.map((f) => (
          <button
            key={f.key}
            onClick={() => setDivFilter(f.key)}
            className={cn(
              'px-3 py-2 rounded-xl text-xs font-bold tracking-wider uppercase border transition-all',
              divFilter === f.key
                ? f.key === 'rec'
                  ? 'bg-sky-500/15 text-sky-400 border-sky-500/30'
                  : 'bg-blue-500/15 text-blue-400 border-blue-500/30'
                : 'text-slate-500 border-blue-500/20 hover:border-blue-500/40 hover:text-slate-300'
            )}
          >
            {f.label}
          </button>
        ))}
      </div>

      {/* Result count */}
      <p className="text-slate-500 text-sm mb-4">
        Showing {filteredPlayers.length} player{filteredPlayers.length !== 1 ? 's' : ''}
        {divFilter !== 'all' && ` in ${divFilter === 'comp' ? 'D2 Comp' : 'D2 Rec'}`}
      </p>

      <LeaderboardTable players={filteredPlayers} limit={500} />
    </div>
  );
}
