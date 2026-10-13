import type { Config } from 'tailwindcss'

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'hsl(var(--bg))',
        surface: 'hsl(var(--surface))',
        raised: 'hsl(var(--raised))',
        line: 'hsl(var(--line))',
        ink: 'hsl(var(--ink))',
        soft: 'hsl(var(--soft))',
        faint: 'hsl(var(--faint))',
        accent: { DEFAULT: 'hsl(var(--accent))', ink: 'hsl(var(--accent-ink))', wash: 'hsl(var(--accent-wash))' },
        mark: 'hsl(var(--mark))',
        ok: 'hsl(var(--ok))',
        warn: 'hsl(var(--warn))',
        bad: 'hsl(var(--bad))',
        sparse: 'hsl(var(--sparse))',
        dense: 'hsl(var(--dense))',
      },
      fontFamily: {
        sans: ['"Inter Tight"', 'Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif'],
        serif: ['"Iowan Old Style"', '"Palatino Linotype"', 'Palatino', 'Charter', 'Georgia', 'serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
      borderRadius: { xl: '14px', lg: '10px', md: '8px', sm: '6px' },
      keyframes: {
        rise: { from: { opacity: '0', transform: 'translateY(4px)' }, to: { opacity: '1', transform: 'none' } },
        grow: { from: { transform: 'scaleX(0)' }, to: { transform: 'scaleX(1)' } },
      },
      animation: { rise: 'rise .22s ease-out', grow: 'grow .5s cubic-bezier(.2,.8,.2,1) both' },
    },
  },
  plugins: [],
} satisfies Config
