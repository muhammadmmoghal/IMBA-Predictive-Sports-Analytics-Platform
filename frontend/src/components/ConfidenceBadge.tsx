import { cn } from '@/lib/utils';

interface ConfidenceBadgeProps {
  level: string;
  score?: number;
  showScore?: boolean;
}

const LEVEL_CONFIG: Record<string, { label: string; classes: string }> = {
  High: {
    label: 'HIGH',
    classes: 'text-emerald-400 bg-emerald-400/10 border-emerald-400/25',
  },
  Medium: {
    label: 'MEDIUM',
    classes: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/25',
  },
  Low: {
    label: 'LOW',
    classes: 'text-red-400 bg-red-400/10 border-red-400/25',
  },
};

export default function ConfidenceBadge({ level, showScore = true, score }: ConfidenceBadgeProps) {
  const config = LEVEL_CONFIG[level] ?? {
    label: level.toUpperCase(),
    classes: 'text-slate-400 bg-slate-400/10 border-slate-400/25',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded text-[11px] font-bold border tracking-wider',
        config.classes
      )}
    >
      {config.label}
      {showScore && score !== undefined && (
        <span className="opacity-50 ml-1 font-normal">· {score}</span>
      )}
    </span>
  );
}
