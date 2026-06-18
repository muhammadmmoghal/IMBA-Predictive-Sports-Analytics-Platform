// Maps team name (as it appears in player data) to local logo path under /public/team-logos/
const TEAM_LOGOS: Record<string, string> = {
  // ── D2 Comp ───────────────────────────────────────────────────────────────
  'Irving OGs':      '/team-logos/irving-ogs.png',
  'Lah Bros':        '/team-logos/lah-bros.png',
  'Amel Foundation': '/team-logos/amel-foundation.png',
  'Top Class':       '/team-logos/top-class.png',
  'NSG':             '/team-logos/nsg.jpg',
  'Starz':           '/team-logos/starz.png',
  'Fear 1':          '/team-logos/fear-1.png',
  'Easy Money':      '/team-logos/easy-money.jpg',
  'Baitul Ballers':  '/team-logos/baitul-ballers.png',
  'ATX':             '/team-logos/atx.png',
  'Free Sudan':      '/team-logos/free-sudan.png',
  'Net Rippers':     '/team-logos/net-rippers.jpg',
  'Deloaders':       '/team-logos/deloaders.png',
  'The Dallas Storm':'/team-logos/the-dallas-storm.png',
  'Seljuks':         '/team-logos/seljuks.png',
  'The Rich':        '/team-logos/the-rich.png',
  // ── D2 Rec ────────────────────────────────────────────────────────────────
  'Mid-Life Prime':   '/team-logos/mid-life-prime.png',
  'Spray Dat':        '/team-logos/spray-dat.jpg',
  'Halal Hustlers':   '/team-logos/halal-hustlers.png',
  'Dallas Hoopers':   '/team-logos/dallas-hoopers.png',
  'ITX':              '/team-logos/itx.png',
  'Amoud Foundation': '/team-logos/amoud-foundation.png',
  'Iso':              '/team-logos/iso.png',
  'VRIC Ballers':     '/team-logos/vric-ballers.jpg',
  'Gauchos':          '/team-logos/gauchos.jpg',
  'The Askars':       '/team-logos/the-askars.png',
  'Deen Dynasty':     '/team-logos/deen-dynasty.jpg',
  'Core Crew':        '/team-logos/core-crew.png',
  'Ummah United':     '/team-logos/ummah-united.jpg',
  'Al Rozzy':         '/team-logos/al-rozzy.jpg',
  'Baja Blast':       '/team-logos/baja-blast.png',
  'Chachos':          '/team-logos/chachos.png',
  'TNZ':              '/team-logos/tnz.png',
  'Deen Up':          '/team-logos/deen-up.png',
  'Vela Victors':     '/team-logos/vela-victors.png',
};

export function getTeamLogo(team: string): string | null {
  return TEAM_LOGOS[team] ?? null;
}
