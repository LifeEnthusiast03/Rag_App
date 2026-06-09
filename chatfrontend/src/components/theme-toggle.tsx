import { Sun, Moon } from 'lucide-react';
import { useTheme } from '@/components/theme-provider';

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  // Theme is strictly "dark" | "light" — no "system" variant exists in this provider
  const isDark = theme === 'dark';

  return (
    <button
      onClick={toggleTheme}
      className="w-8 h-8 rounded-md border border-[var(--color-border)] hover:bg-[var(--color-bg-elevated)] flex items-center justify-center text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {isDark ? <Sun size={14} /> : <Moon size={14} />}
    </button>
  );
}
