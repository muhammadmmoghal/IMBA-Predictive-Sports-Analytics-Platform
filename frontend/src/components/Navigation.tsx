'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

const navLinks = [
  { href: '/', label: 'OVERVIEW' },
  { href: '/players', label: 'PLAYERS' },
  { href: '/leaderboards', label: 'LEADERBOARDS' },
  { href: '/divisions', label: 'DIVISIONS' },
];

export default function Navigation() {
  const pathname = usePathname();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-blue-500/[0.15] bg-[#03060D]/95 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1440px] items-center justify-between px-6 py-5 lg:px-10">

        {/* Brand */}
        <Link href="/" className="flex items-center gap-4">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/images/imba-logo.png"
            alt="IMBA logo"
            width={48}
            height={48}
            className="h-12 w-12 object-contain"
          />
          <div className="flex items-baseline gap-3">
            <span className="font-heading text-3xl font-bold tracking-tight text-white">
              IMBA
            </span>
            <span className="hidden font-heading text-sm font-bold tracking-[0.18em] text-blue-400 sm:inline">
              PREDICTIVE ANALYTICS
            </span>
          </div>
        </Link>

        {/* Nav links — desktop */}
        <nav className="hidden items-center gap-10 lg:flex">
          {navLinks.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  'relative py-1 text-sm font-semibold tracking-wider transition-colors duration-150',
                  isActive
                    ? 'text-white after:absolute after:bottom-0 after:left-0 after:right-0 after:h-0.5 after:rounded-full after:bg-blue-500'
                    : 'text-[#8EA7C8] hover:text-white'
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* Nav — mobile */}
        <nav className="flex items-center gap-0.5 lg:hidden">
          {navLinks.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  'px-2 py-1.5 text-[9px] font-bold tracking-wider transition-colors',
                  isActive ? 'text-white' : 'text-slate-600 hover:text-slate-300'
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
