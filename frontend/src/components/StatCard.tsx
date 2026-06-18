import { cn } from '@/lib/utils';
import type { ReactNode } from 'react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  children?: ReactNode;
  icon?: ReactNode;
  accent?: 'green' | 'gold' | 'default';
}

export default function StatCard({ title, value, subtitle, children, icon, accent = 'default' }: StatCardProps) {
  const iconClass = {
    green: 'text-blue-400',
    gold: 'text-sky-400',
    default: 'text-slate-400',
  }[accent];

  return (
    <div className="rounded-lg border border-blue-500/[0.25] bg-[#091426]/50 p-5 backdrop-blur-sm transition-colors hover:border-blue-500/40 hover:bg-[#091426]/80">
      <div className="flex items-start justify-between">
        <p className="text-xs font-semibold tracking-wider text-[#8EA7C8] uppercase">{title}</p>
        {icon && (
          <div className={cn('h-6 w-6 shrink-0', iconClass)}>{icon}</div>
        )}
      </div>
      <p className="mt-2 font-heading text-5xl font-bold tracking-tight text-white tabular-nums">
        {value}
      </p>
      {subtitle && (
        <p className="mt-1 text-sm text-[#8EA7C8]">{subtitle}</p>
      )}
      {children}
    </div>
  );
}
