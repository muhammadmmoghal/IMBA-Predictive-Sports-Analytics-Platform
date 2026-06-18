import { cn, probBarColor, probTextColor, fmtProb } from '@/lib/utils';

interface ProbabilityBarProps {
  label: string;
  value: number;
}

export default function ProbabilityBar({ label, value }: ProbabilityBarProps) {
  const clamped = Math.min(100, Math.max(0, value));

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="text-slate-400">{label}</span>
        <span className={cn('font-bold tabular-nums', probTextColor(clamped))}>
          {fmtProb(clamped)}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-[#1a1a30] overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-700 ease-out', probBarColor(clamped))}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}
