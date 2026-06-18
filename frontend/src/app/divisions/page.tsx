'use client';

import { useEffect, useState } from 'react';
import type { Player } from '@/lib/types';
import { fmt, avg, shortDivision, cn } from '@/lib/utils';
import ConfidenceBadge from '@/components/ConfidenceBadge';
import Link from 'next/link';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface DivisionStatsProps {
  players: Player[];
  label: string;
  accent: 'green' | 'gold';
}

function DivisionStats({ players, label, accent }: DivisionStatsProps) {
  if (players.length === 0) return null;

  const avgPts = avg(players.map((p) => p.predicted_pts));
  const avgReb = avg(players.map((p) => p.predicted_reb));
  const avgAst = avg(players.map((p) => p.predicted_ast));
  const highConf = players.filter((p) => p.confidence_level === 'High').length;
  const topScorer = [...players].sort((a, b) => b.predicted_pts - a.predicted_pts)[0];

  const cardClass = accent === 'green' ? 'card-green' : 'card-gold';
  const accentText = accent === 'green' ? 'text-blue-400' : 'text-sky-400';
  const accentBg = accent === 'green' ? 'bg-blue-500/10' : 'bg-sky-500/10';
  const accentBorder = accent === 'green' ? 'border-blue-500/20' : 'border-sky-500/20';
  const gradientText = accent === 'green' ? 'text-gradient-green' : 'text-gradient-gold';

  return (
    <div className={cn('rounded-xl p-6', cardClass)}>
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className={cn(
          'px-3 py-1.5 rounded-lg border text-xs font-bold tracking-wider uppercase',
          accentBg, accentBorder, accentText
        )}>
          {label}
        </div>
        <span className="text-slate-500 text-sm">{players.length} players</span>
      </div>

      {/* Average Stats */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        {[
          { label: 'AVG PTS', value: avgPts },
          { label: 'AVG REB', value: avgReb },
          { label: 'AVG AST', value: avgAst },
        ].map((s) => (
          <div key={s.label} className="text-center">
            <p className={cn('text-2xl font-black tabular-nums', gradientText)}>{fmt(s.value)}</p>
            <p className="text-[10px] font-bold tracking-widest text-slate-600 uppercase mt-0.5">{s.label}</p>
          </div>
        ))}
      </div>

      {/* High Confidence */}
      <div className="flex items-center gap-2 mb-6 text-sm text-slate-400">
        <span className="text-emerald-400 font-bold">{highConf}</span>
        <span>high-confidence players (10+ games)</span>
      </div>

      {/* Top Scorer */}
      {topScorer && (
        <div className={cn('rounded-lg border p-4 mb-6', accentBg, accentBorder)}>
          <p className={cn('text-[10px] font-bold tracking-widest uppercase mb-2', accentText)}>
            Top Scorer
          </p>
          <div className="flex items-center justify-between">
            <div>
              <Link
                href={`/players?q=${encodeURIComponent(topScorer.player_name)}`}
                className="font-bold text-white hover:text-blue-400 transition-colors"
              >
                {topScorer.player_name}
              </Link>
              <p className="text-xs text-slate-500 mt-0.5">{topScorer.team}</p>
            </div>
            <div className="text-right">
              <p className={cn('text-2xl font-black tabular-nums', gradientText)}>
                {fmt(topScorer.predicted_pts)}
              </p>
              <p className="text-xs text-slate-600">PPG</p>
            </div>
          </div>
        </div>
      )}

      {/* Top 5 Players */}
      <div>
        <p className="text-[10px] font-bold tracking-widest text-slate-600 uppercase mb-3">
          Top 5 Projected Scorers
        </p>
        <div className="space-y-2">
          {[...players]
            .sort((a, b) => b.predicted_pts - a.predicted_pts)
            .slice(0, 5)
            .map((player, idx) => (
              <div
                key={player.player_name}
                className="flex items-center gap-3 py-2 border-b border-blue-500/[0.1] last:border-0"
              >
                <span className="text-xs text-slate-600 w-4 text-right">{idx + 1}</span>
                <div className="flex-1 min-w-0">
                  <Link
                    href={`/players?q=${encodeURIComponent(player.player_name)}`}
                    className="text-sm font-semibold text-slate-200 hover:text-blue-400 transition-colors truncate block"
                  >
                    {player.player_name}
                  </Link>
                  <p className="text-xs text-slate-600 truncate">{player.team}</p>
                </div>
                <div className="text-right shrink-0">
                  <p className="text-sm font-bold text-white tabular-nums">{fmt(player.predicted_pts)}</p>
                  <p className="text-xs text-slate-600">PTS</p>
                </div>
                <ConfidenceBadge level={player.confidence_level} showScore={false} />
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: ChartTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#091426] border border-blue-500/20 rounded-lg p-3 text-sm shadow-xl">
      <p className="text-slate-300 font-semibold mb-2">{label}</p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
          <span className="text-slate-400">{entry.name}:</span>
          <span className="font-bold text-white">{fmt(entry.value)}</span>
        </div>
      ))}
    </div>
  );
}

export default function DivisionsPage() {
  const [compPlayers, setCompPlayers] = useState<Player[]>([]);
  const [recPlayers, setRecPlayers] = useState<Player[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [r1, r2] = await Promise.all([
          fetch('/data/d2_comp.json'),
          fetch('/data/d2_rec.json'),
        ]);
        if (!r1.ok) throw new Error('not found');
        const [comp, rec] = await Promise.all([r1.json(), r2.json()]);
        setCompPlayers(comp);
        setRecPlayers(rec);
      } catch {
        setError('Run convert_csv_to_json.py to generate data files.');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-6 animate-pulse">
        <div className="h-8 w-56 shimmer rounded" />
        <div className="grid md:grid-cols-2 gap-6">
          <div className="h-[500px] shimmer rounded-xl" />
          <div className="h-[500px] shimmer rounded-xl" />
        </div>
        <div className="h-64 shimmer rounded-xl" />
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

  const compAvgPts = avg(compPlayers.map((p) => p.predicted_pts));
  const compAvgReb = avg(compPlayers.map((p) => p.predicted_reb));
  const compAvgAst = avg(compPlayers.map((p) => p.predicted_ast));
  const recAvgPts = avg(recPlayers.map((p) => p.predicted_pts));
  const recAvgReb = avg(recPlayers.map((p) => p.predicted_reb));
  const recAvgAst = avg(recPlayers.map((p) => p.predicted_ast));

  const chartData = [
    { stat: 'Points', 'D2 Comp': parseFloat(compAvgPts.toFixed(2)), 'D2 Rec': parseFloat(recAvgPts.toFixed(2)) },
    { stat: 'Rebounds', 'D2 Comp': parseFloat(compAvgReb.toFixed(2)), 'D2 Rec': parseFloat(recAvgReb.toFixed(2)) },
    { stat: 'Assists', 'D2 Comp': parseFloat(compAvgAst.toFixed(2)), 'D2 Rec': parseFloat(recAvgAst.toFixed(2)) },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      {/* Header */}
      <div className="mb-10">
        <h1 className="font-heading text-3xl font-black uppercase tracking-wider text-white mb-1">
          Division Comparison
        </h1>
        <p className="text-slate-400">
          Side-by-side breakdown of D2 Comp and D2 Rec projected statistics.
        </p>
      </div>

      {/* Division Cards */}
      <div className="grid md:grid-cols-2 gap-6 mb-10">
        <DivisionStats players={compPlayers} label="D2 Comp" accent="green" />
        <DivisionStats players={recPlayers} label="D2 Rec" accent="gold" />
      </div>

      {/* Bar Chart */}
      <div className="card rounded-xl p-6">
        <h2 className="font-heading text-base font-black uppercase tracking-widest text-white mb-6">
          Average Projected Stats by Division
        </h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} barCategoryGap="35%" barGap={6}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(59,130,246,0.12)" vertical={false} />
              <XAxis
                dataKey="stat"
                tick={{ fill: '#64748b', fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                width={35}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Legend
                wrapperStyle={{ fontSize: 12, paddingTop: 12 }}
                formatter={(v) => <span style={{ color: '#94a3b8' }}>{v}</span>}
              />
              <Bar dataKey="D2 Comp" fill="#3B82F6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="D2 Rec" fill="#0EA5E9" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
