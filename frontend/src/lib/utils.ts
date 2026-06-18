import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmt(value: number, decimals = 1): string {
  return value.toFixed(decimals);
}

export function fmtProb(value: number): string {
  return `${value.toFixed(1)}%`;
}

export function shortDivision(division: string): string {
  if (division.includes('Comp')) return 'D2 Comp';
  if (division.includes('Rec')) return 'D2 Rec';
  return division;
}

export function isComp(division: string): boolean {
  return division.includes('Comp');
}

export function isRec(division: string): boolean {
  return division.includes('Rec');
}

export function probBarColor(prob: number): string {
  if (prob >= 70) return 'bg-blue-500';
  if (prob >= 40) return 'bg-sky-500';
  return 'bg-slate-600';
}

export function probTextColor(prob: number): string {
  if (prob >= 70) return 'text-blue-400';
  if (prob >= 40) return 'text-sky-400';
  return 'text-slate-500';
}

export function confidenceColors(level: string): string {
  switch (level) {
    case 'High':
      return 'text-emerald-400 bg-emerald-400/10 border-emerald-400/25';
    case 'Medium':
      return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/25';
    case 'Low':
      return 'text-red-400 bg-red-400/10 border-red-400/25';
    default:
      return 'text-slate-400 bg-slate-400/10 border-slate-400/25';
  }
}

export function avg(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}
