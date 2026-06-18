'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { ChevronRight } from 'lucide-react';
import type { Player } from '@/lib/types';
import { fmt, fmtProb, shortDivision } from '@/lib/utils';
import StatCard from '@/components/StatCard';
import ConfidenceBadge from '@/components/ConfidenceBadge';
import TeamLogo from '@/components/TeamLogo';
import LeaderboardTable from '@/components/LeaderboardTable';

function LoadingSkeleton() {
  return (
    <div className="mx-auto max-w-[1440px] px-6 pb-16 lg:px-10 animate-pulse">
      <div className="pt-12 pb-4 space-y-4">
        <div className="h-4 w-36 shimmer rounded-full" />
        <div className="h-16 w-[480px] shimmer rounded-xl" />
        <div className="h-4 w-80 shimmer rounded" />
        <div className="flex gap-4 pt-2">
          <div className="h-12 w-40 shimmer rounded-lg" />
          <div className="h-12 w-44 shimmer rounded-lg" />
        </div>
      </div>
      <div className="mt-8 space-y-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => <div key={i} className="h-36 shimmer rounded-lg" />)}
        </div>
        <div className="h-96 shimmer rounded-xl" />
      </div>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="max-w-[1440px] mx-auto px-6 lg:px-10 py-20 text-center">
      <h2 className="font-heading text-2xl uppercase text-rose-400 mb-2">Data Not Found</h2>
      <p className="text-[#8EA7C8] mb-6">{message}</p>
      <div className="inline-block bg-[#091426] border border-blue-500/20 rounded-xl px-6 py-4 text-left text-sm font-mono text-slate-300">
        <p className="text-[#8EA7C8] mb-1"># Run from the project root:</p>
        <p className="text-blue-400">python frontend/scripts/convert_csv_to_json.py</p>
      </div>
    </div>
  );
}

export default function HomePage() {
  const [allPlayers, setAllPlayers] = useState<Player[]>([]);
  const [topScorers, setTopScorers] = useState<Player[]>([]);
  const [topDD, setTopDD] = useState<Player[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [r1, r2, r3] = await Promise.all([
          fetch('/data/all_players.json'),
          fetch('/data/top_scorers.json'),
          fetch('/data/top_double_double.json'),
        ]);
        if (!r1.ok || !r2.ok || !r3.ok) throw new Error('Data files not found');
        const [all, scorers, dd] = await Promise.all([r1.json(), r2.json(), r3.json()]);
        setAllPlayers(all);
        setTopScorers(scorers);
        setTopDD(dd);
      } catch {
        setError('Run the CSV conversion script to generate the required data files.');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <LoadingSkeleton />;
  if (error) return <ErrorState message={error} />;

  const compPlayers    = allPlayers.filter((p) => p.division.includes('Comp'));
  const recPlayers     = allPlayers.filter((p) => p.division.includes('Rec'));
  const highConf       = allPlayers.filter((p) => p.confidence_level === 'High');
  const proj10Comp     = compPlayers.filter((p) => p.predicted_pts >= 10);
  const proj10Rec      = recPlayers.filter((p) => p.predicted_pts >= 10);
  const proj10CompPct  = compPlayers.length > 0 ? Math.round((proj10Comp.length / compPlayers.length) * 100) : 0;
  const proj10RecPct   = recPlayers.length  > 0 ? Math.round((proj10Rec.length  / recPlayers.length)  * 100) : 0;
  const ddCandidates   = allPlayers.filter((p) => p.prob_double_double >= 50);
  const topScorer    = topScorers[0];
  const topDD0       = topDD[0];
  const topRebounder = [...allPlayers].sort((a, b) => b.predicted_reb - a.predicted_reb)[0];
  const topPlaymaker = [...allPlayers].sort((a, b) => b.predicted_ast - a.predicted_ast)[0];
  const topBlocker   = [...allPlayers].sort((a, b) => b.predicted_blk - a.predicted_blk)[0];
  const topStealer   = [...allPlayers].sort((a, b) => b.predicted_stl - a.predicted_stl)[0];

  return (
    <div className="relative">

      <div className="relative z-10 mx-auto max-w-[1440px] px-6 pb-16 lg:px-10">

        {/* ══════════════════════════════
            HERO
        ══════════════════════════════ */}
        <section className="relative overflow-hidden">

          {/* ── Hero watermark (v0 exact) ── */}
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-y-0 right-0 z-0 hidden w-[45%] select-none md:block"
          >
            <div className="absolute inset-0 flex items-center justify-center" style={{ transform: 'translateY(50px) translateX(-50px)' }}>
              {/* Spotlight glow beam */}
              <div className="absolute h-[95%] w-[70%] bg-[radial-gradient(ellipse_at_50%_40%,rgba(59,130,246,0.48),transparent_60%)] blur-2xl" />
              {/* Logo — double-masked: vertical fade + radial vignette */}
              <Image
                src="/images/imba-logo.png"
                alt=""
                width={760}
                height={760}
                priority
                className="relative h-auto w-[88%] rotate-[-8deg] opacity-[0.26] [mask-image:linear-gradient(to_bottom,transparent,black_22%,black_70%,transparent),radial-gradient(ellipse_at_center,black_62%,transparent_90%)] [mask-composite:intersect] [-webkit-mask-image:linear-gradient(to_bottom,transparent,black_22%,black_70%,transparent),radial-gradient(ellipse_at_center,black_62%,transparent_90%)] [-webkit-mask-composite:source-in]"
              />
            </div>
          </div>

          <div className="relative z-10 grid grid-cols-1 items-center gap-8 pb-4 pt-12 md:grid-cols-[1.2fr_1fr]">
          <div>
            <p className="mb-3 text-sm font-semibold tracking-[0.2em] text-blue-400">
              2026 SUMMER SEASON
            </p>
            <h1 className="font-heading text-5xl font-extrabold italic uppercase leading-[0.92] tracking-tight text-balance lg:text-7xl">
              <span className="block text-white">IMBA Predictive</span>
              <span className="block text-blue-500">Sports Analytics</span>
            </h1>
            <p className="mt-6 max-w-md text-base leading-relaxed text-[#8EA7C8]">
              Machine learning&ndash;powered player projections for the IMBA
              basketball community.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-4">
              <Link
                href="/players"
                className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-6 py-3.5 text-sm font-semibold tracking-wider text-white transition-colors hover:bg-blue-500"
                style={{ boxShadow: '0 3px 16px rgba(59,130,246,0.35)' }}
              >
                PREDICT A PLAYER <ChevronRight className="h-4 w-4" />
              </Link>
              <Link
                href="/leaderboards"
                className="inline-flex items-center gap-2 rounded-md border border-blue-500/30 bg-blue-500/[0.07] px-6 py-3.5 text-sm font-semibold tracking-wider text-white transition-colors hover:bg-blue-500/[0.12]"
              >
                VIEW LEADERBOARDS
              </Link>
            </div>
          </div>
          </div>
        </section>

        {/* ══════════════════════════════
            KPI CARDS
        ══════════════════════════════ */}
        <div className="mt-8 grid grid-cols-2 gap-4 md:grid-cols-4">
          {/* Card 1: Total Players with Comp/Rec breakdown */}
          <StatCard
            title="Total Players"
            value={allPlayers.length}
            icon={
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
            }
          >
            <div className="mt-1 flex gap-3 text-sm text-[#8EA7C8]">
              <span>Comp: {compPlayers.length}</span>
              <span>Rec: {recPlayers.length}</span>
            </div>
          </StatCard>

          {/* Card 2: Projected 10+ Point Scorers */}
          <StatCard
            title="Projected 10+ Pts"
            value={proj10Comp.length + proj10Rec.length}
            accent="green"
            icon={
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
                <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6" />
                <path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18" />
                <path d="M4 22h16" />
                <path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22" />
                <path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22" />
                <path d="M18 2H6v7a6 6 0 0 0 12 0V2Z" />
              </svg>
            }
          >
            <div className="mt-1 space-y-0.5 text-sm text-[#8EA7C8]">
              <p>Comp: {proj10Comp.length} ({proj10CompPct}%)</p>
              <p>Rec: {proj10Rec.length} ({proj10RecPct}%)</p>
            </div>
          </StatCard>

          {/* Card 3: High Confidence Players */}
          <StatCard
            title="High Confidence Players"
            value={highConf.length}
            subtitle="10+ games played"
            accent="gold"
            icon={
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
                <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
                <polyline points="17 6 23 6 23 12" />
              </svg>
            }
          />

          {/* Card 4: Double-Double Candidates */}
          <StatCard
            title="Double-Double Candidates"
            value={ddCandidates.length}
            subtitle="50%+ probability"
            accent="gold"
            icon={
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
                <circle cx="12" cy="12" r="10" />
                <path d="M12 8v4l3 3" />
              </svg>
            }
          />
        </div>

        {/* ══════════════════════════════
            TOP PROJECTED SCORERS TABLE
        ══════════════════════════════ */}
        <div className="mt-6 rounded-lg border border-blue-500/[0.2] bg-[#091426]/40 p-6">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="h-6 w-1 rounded-full bg-blue-500" />
              <h2 className="font-heading text-2xl font-bold tracking-wide text-white">
                TOP PROJECTED SCORERS
              </h2>
            </div>
            <Link
              href="/leaderboards"
              className="inline-flex items-center gap-1 text-sm font-semibold tracking-wider text-blue-400 transition-colors hover:text-blue-300"
            >
              VIEW FULL LEADERBOARD <ChevronRight className="h-4 w-4" />
            </Link>
          </div>
          <LeaderboardTable players={topScorers} statKey="predicted_pts" statLabel="PTS" limit={10} />
        </div>

        {/* ══════════════════════════════
            FEATURED PROJECTIONS
        ══════════════════════════════ */}
        {allPlayers.length > 0 && (
          <div className="mt-6 mb-14">
            <div className="flex items-center gap-3 mb-5">
              <span className="h-6 w-1 rounded-full bg-blue-500" />
              <h2 className="font-heading text-2xl font-bold tracking-wide text-white">
                FEATURED PROJECTIONS
              </h2>
            </div>
            <div className="grid md:grid-cols-3 gap-6">

              {/* ── Card 1: Top Projected Scorer ── */}
              {topScorer && (
                <div className="card-green p-6 rounded-xl">
                  <p className="text-[10px] font-bold tracking-[0.18em] text-blue-400 uppercase mb-4">
                    Top Projected Scorer
                  </p>
                  <h3 className="font-heading text-2xl uppercase text-white mb-1">
                    {topScorer.player_name}
                  </h3>
                  <div className="flex items-center gap-2 text-[#8EA7C8] text-sm mb-5">
                    <TeamLogo team={topScorer.team} size="sm" />
                    {topScorer.team} &bull; {shortDivision(topScorer.division)}
                  </div>
                  <div className="flex items-baseline gap-2 mb-1">
                    <span className="text-6xl font-black text-gradient-green tabular-nums">
                      {fmt(topScorer.predicted_pts)}
                    </span>
                    <span className="text-blue-400 text-xl font-bold">PPG</span>
                  </div>
                  <p className="text-[#8EA7C8] text-sm mb-5">
                    Range: {fmt(topScorer.pts_low)}–{fmt(topScorer.pts_high)} pts
                  </p>
                  <ConfidenceBadge level={topScorer.confidence_level} score={topScorer.confidence_score} />
                </div>
              )}

              {/* ── Card 2: Top Projected Rebounder ── */}
              {topRebounder && (
                <div className="card-gold p-6 rounded-xl">
                  <p className="text-[10px] font-bold tracking-[0.18em] text-sky-400 uppercase mb-4">
                    Top Projected Rebounder
                  </p>
                  <h3 className="font-heading text-2xl uppercase text-white mb-1">
                    {topRebounder.player_name}
                  </h3>
                  <div className="flex items-center gap-2 text-[#8EA7C8] text-sm mb-5">
                    <TeamLogo team={topRebounder.team} size="sm" />
                    {topRebounder.team} &bull; {shortDivision(topRebounder.division)}
                  </div>
                  <div className="flex items-baseline gap-2 mb-1">
                    <span className="text-6xl font-black text-gradient-gold tabular-nums">
                      {fmt(topRebounder.predicted_reb)}
                    </span>
                    <span className="text-sky-400 text-xl font-bold">RPG</span>
                  </div>
                  <p className="text-[#8EA7C8] text-sm mb-5">
                    Range: {fmt(topRebounder.reb_low)}–{fmt(topRebounder.reb_high)} reb
                  </p>
                  <ConfidenceBadge level={topRebounder.confidence_level} score={topRebounder.confidence_score} />
                </div>
              )}

              {/* ── Card 3: Top Projected Playmaker ── */}
              {topPlaymaker && (
                <div className="card-green p-6 rounded-xl">
                  <p className="text-[10px] font-bold tracking-[0.18em] text-blue-400 uppercase mb-4">
                    Top Projected Playmaker
                  </p>
                  <h3 className="font-heading text-2xl uppercase text-white mb-1">
                    {topPlaymaker.player_name}
                  </h3>
                  <div className="flex items-center gap-2 text-[#8EA7C8] text-sm mb-5">
                    <TeamLogo team={topPlaymaker.team} size="sm" />
                    {topPlaymaker.team} &bull; {shortDivision(topPlaymaker.division)}
                  </div>
                  <div className="flex items-baseline gap-2 mb-1">
                    <span className="text-6xl font-black text-gradient-green tabular-nums">
                      {fmt(topPlaymaker.predicted_ast)}
                    </span>
                    <span className="text-blue-400 text-xl font-bold">APG</span>
                  </div>
                  <p className="text-[#8EA7C8] text-sm mb-5">
                    Range: {fmt(topPlaymaker.ast_low)}–{fmt(topPlaymaker.ast_high)} ast
                  </p>
                  <ConfidenceBadge level={topPlaymaker.confidence_level} score={topPlaymaker.confidence_score} />
                </div>
              )}

              {/* ── Card 4: Top Projected Block Leader ── */}
              {topBlocker && (
                <div className="card-gold p-6 rounded-xl">
                  <p className="text-[10px] font-bold tracking-[0.18em] text-sky-400 uppercase mb-4">
                    Top Projected Block Leader
                  </p>
                  <h3 className="font-heading text-2xl uppercase text-white mb-1">
                    {topBlocker.player_name}
                  </h3>
                  <div className="flex items-center gap-2 text-[#8EA7C8] text-sm mb-5">
                    <TeamLogo team={topBlocker.team} size="sm" />
                    {topBlocker.team} &bull; {shortDivision(topBlocker.division)}
                  </div>
                  <div className="flex items-baseline gap-2 mb-1">
                    <span className="text-6xl font-black text-gradient-gold tabular-nums">
                      {fmt(topBlocker.predicted_blk)}
                    </span>
                    <span className="text-sky-400 text-xl font-bold">BPG</span>
                  </div>
                  <p className="text-[#8EA7C8] text-sm mb-5">
                    Range: {fmt(topBlocker.blk_low)}–{fmt(topBlocker.blk_high)} blk
                  </p>
                  <ConfidenceBadge level={topBlocker.confidence_level} score={topBlocker.confidence_score} />
                </div>
              )}

              {/* ── Card 5: Top Projected Steals Leader ── */}
              {topStealer && (
                <div className="card-green p-6 rounded-xl">
                  <p className="text-[10px] font-bold tracking-[0.18em] text-blue-400 uppercase mb-4">
                    Top Projected Steals Leader
                  </p>
                  <h3 className="font-heading text-2xl uppercase text-white mb-1">
                    {topStealer.player_name}
                  </h3>
                  <div className="flex items-center gap-2 text-[#8EA7C8] text-sm mb-5">
                    <TeamLogo team={topStealer.team} size="sm" />
                    {topStealer.team} &bull; {shortDivision(topStealer.division)}
                  </div>
                  <div className="flex items-baseline gap-2 mb-1">
                    <span className="text-6xl font-black text-gradient-green tabular-nums">
                      {fmt(topStealer.predicted_stl)}
                    </span>
                    <span className="text-blue-400 text-xl font-bold">SPG</span>
                  </div>
                  <p className="text-[#8EA7C8] text-sm mb-5">
                    Range: {fmt(topStealer.stl_low)}–{fmt(topStealer.stl_high)} stl
                  </p>
                  <ConfidenceBadge level={topStealer.confidence_level} score={topStealer.confidence_score} />
                </div>
              )}

              {/* ── Card 6: Top Double-Double Candidate ── */}
              {topDD0 && (
                <div className="card-gold p-6 rounded-xl">
                  <p className="text-[10px] font-bold tracking-[0.18em] text-sky-400 uppercase mb-4">
                    Top Double-Double Candidate
                  </p>
                  <h3 className="font-heading text-2xl uppercase text-white mb-1">
                    {topDD0.player_name}
                  </h3>
                  <div className="flex items-center gap-2 text-[#8EA7C8] text-sm mb-5">
                    <TeamLogo team={topDD0.team} size="sm" />
                    {topDD0.team} &bull; {shortDivision(topDD0.division)}
                  </div>
                  <div className="flex items-baseline gap-2 mb-1">
                    <span className="text-6xl font-black text-gradient-gold tabular-nums">
                      {fmtProb(topDD0.prob_double_double)}
                    </span>
                  </div>
                  <p className="text-[#8EA7C8] text-sm mb-5">
                    {fmt(topDD0.predicted_pts)} PTS / {fmt(topDD0.predicted_reb)} REB projected
                  </p>
                  <ConfidenceBadge level={topDD0.confidence_level} score={topDD0.confidence_score} />
                </div>
              )}

            </div>
          </div>
        )}

      </div>
    </div>
  );
}
