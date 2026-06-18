import type { Metadata } from 'next';
import { Inter, Saira } from 'next/font/google';
import './globals.css';
import Navigation from '@/components/Navigation';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const saira = Saira({
  weight: ['400', '500', '600', '700', '800', '900'],
  style: ['normal', 'italic'],
  subsets: ['latin'],
  variable: '--font-heading',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'IMBA Predictive Analytics',
  description: '2026 Season predictions and player projections for IMBA basketball.',
  icons: {
    icon: [
      { url: '/favicon.ico' },
      { url: '/icon.png', type: 'image/png' },
    ],
    shortcut: '/favicon.ico',
    apple: '/apple-icon.png',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${saira.variable}`}>
      <body className="min-h-screen bg-[#03060D] text-slate-100 antialiased">
        <Navigation />
        <main className="pt-[88px]">{children}</main>
      </body>
    </html>
  );
}
