import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          bg: '#03060D',
          card: '#091426',
          'card-hover': '#0C1A33',
          dark: '#060D1C',
          blue: '#3B82F6',
          'blue-bright': '#4DA3FF',
          'blue-light': '#93C5FD',
          'blue-dark': '#2563EB',
          sky: '#0EA5E9',
          'sky-light': '#7DD3FC',
          muted: '#8EA7C8',
        },
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'system-ui', 'sans-serif'],
        heading: ['var(--font-heading)', 'Impact', 'system-ui', 'sans-serif'],
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(14px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseGlow: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '0.9' },
        },
      },
      animation: {
        shimmer: 'shimmer 2s infinite linear',
        fadeUp: 'fadeUp 0.45s ease-out',
        pulseGlow: 'pulseGlow 2.5s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};

export default config;
