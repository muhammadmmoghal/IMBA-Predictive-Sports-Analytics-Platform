'use client';

import { useState } from 'react';
import { cn } from '@/lib/utils';
import { getTeamLogo } from '@/lib/team-logos';

interface TeamLogoProps {
  team: string;
  size?: 'xs' | 'sm' | 'md';
  className?: string;
}

const SIZE: Record<string, string> = {
  xs: 'w-5 h-5 text-[8px]',
  sm: 'w-6 h-6 text-[9px]',
  md: 'w-8 h-8 text-xs',
};

export default function TeamLogo({ team, size = 'sm', className }: TeamLogoProps) {
  const [imgError, setImgError] = useState(false);
  const src = getTeamLogo(team);
  const sizeClass = SIZE[size];

  if (!src || imgError) {
    return (
      <span
        className={cn(
          'inline-flex items-center justify-center rounded-full shrink-0 font-bold bg-blue-500/20 text-blue-400',
          sizeClass,
          className
        )}
      >
        {team[0]?.toUpperCase() ?? '?'}
      </span>
    );
  }

  return (
    <img
      src={src}
      alt=""
      className={cn('rounded-full object-cover shrink-0', sizeClass, className)}
      onError={() => setImgError(true)}
    />
  );
}
