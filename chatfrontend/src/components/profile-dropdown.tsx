import { useState, useRef, useEffect, useCallback } from 'react';
import { LogOut, ChevronDown, Copy, Check } from 'lucide-react';
import { useAuthContext } from '@/hooks/useauth';
import { useChat } from '@/hooks/usechat';
import { useNavigate } from 'react-router';

/* ─── Avatar ─────────────────────────────────────────────────── */
function Avatar({ name, size = 'sm' }: { name: string; size?: 'sm' | 'md' }) {
  const initials = name
    ? name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2)
    : '?';

  const dim = size === 'md' ? 'w-10 h-10 text-sm' : 'w-6 h-6 text-[10px]';

  return (
    <div
      className={`${dim} rounded-full flex items-center justify-center font-semibold text-white flex-shrink-0`}
      style={{
        background: 'linear-gradient(135deg, var(--color-accent) 0%, oklch(55% 0.22 300) 100%)',
      }}
    >
      {initials}
    </div>
  );
}

/* ─── Copy-to-clipboard cell ─────────────────────────────────── */
function CopyField({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard not available — silently fail
    }
  };

  return (
    <div className="group flex items-center justify-between gap-2 px-3 py-2 rounded-md hover:bg-[var(--color-bg-surface)] transition-colors">
      <div className="min-w-0 flex-1">
        <p className="text-[10px] font-mono text-[var(--color-text-hint)] leading-none mb-0.5">{label}</p>
        <p className="text-xs text-[var(--color-text-primary)] truncate font-medium">{value}</p>
      </div>
      <button
        onClick={handleCopy}
        className="flex-shrink-0 opacity-0 group-hover:opacity-100 w-6 h-6 flex items-center justify-center rounded text-[var(--color-text-hint)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-elevated)] transition-all"
        title={copied ? 'Copied' : `Copy ${label}`}
      >
        {copied
          ? <Check size={11} className="text-[var(--color-success)]" />
          : <Copy size={11} />
        }
      </button>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   PROFILE DROPDOWN
═══════════════════════════════════════════════════════════════ */
export default function ProfileDropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  const { user, logout } = useAuthContext();
  const { userChats } = useChat();
  const navigate = useNavigate();

  /* ── Close on outside click ──────────────────────────────────── */
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  /* ── Close on Escape ─────────────────────────────────────────── */
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
        triggerRef.current?.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  /* ── Sign out ────────────────────────────────────────────────── */
  const handleLogout = useCallback(() => {
    setIsOpen(false);
    logout();
    navigate('/login');
  }, [logout, navigate]);

  /* ── Derived data ────────────────────────────────────────────── */
  const memberSince = new Date().getFullYear(); // real apps: derive from user.created_at
  const conversationCount = userChats.length;

  return (
    <div className="relative" ref={dropdownRef}>

      {/* ── Trigger ──────────────────────────────────────────────── */}
      <button
        ref={triggerRef}
        onClick={() => setIsOpen(prev => !prev)}
        aria-label="Open profile menu"
        aria-expanded={isOpen}
        aria-haspopup="menu"
        className={`flex items-center gap-2 pl-1.5 pr-2.5 py-1.5 rounded-md border transition-colors ${
          isOpen
            ? 'border-[var(--color-border-focus)] bg-[var(--color-bg-elevated)]'
            : 'border-[var(--color-border)] bg-[var(--color-bg-elevated)] hover:border-[var(--color-text-hint)]'
        }`}
      >
        <Avatar name={user.user_name} size="sm" />
        <span className="text-xs font-medium text-[var(--color-text-primary)] hidden sm:block max-w-[100px] truncate">
          {user.user_name || 'Account'}
        </span>
        <ChevronDown
          size={12}
          className={`text-[var(--color-text-hint)] transition-transform duration-200 flex-shrink-0 ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {/* ── Dropdown panel ───────────────────────────────────────── */}
      {isOpen && (
        <div
          role="menu"
          aria-label="Profile menu"
          className="absolute right-0 mt-2 w-64 bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-xl z-50 overflow-hidden"
        >
          {/* ── Identity header ──────────────────────────────────── */}
          <div className="px-4 pt-4 pb-3 border-b border-[var(--color-border)]">
            <div className="flex items-center gap-3 mb-3">
              <Avatar name={user.user_name} size="md" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-[var(--color-text-primary)] truncate leading-tight">
                  {user.user_name || '—'}
                </p>
                <p className="font-mono text-[10px] text-[var(--color-text-hint)] truncate mt-0.5">
                  {user.email || '—'}
                </p>
              </div>
              {/* Online indicator */}
              <div className="flex items-center gap-1.5 flex-shrink-0 px-2 py-1 rounded-full border border-[var(--color-border)] bg-[var(--color-bg-surface)]">
                <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-success)]" />
                <span className="font-mono text-[9px] text-[var(--color-text-hint)]">online</span>
              </div>
            </div>

            {/* ── Stats row ────────────────────────────────────────── */}
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border)] rounded-lg px-3 py-2">
                <p className="font-mono text-base font-semibold text-[var(--color-text-primary)] leading-none">
                  {conversationCount}
                </p>
                <p className="text-[10px] text-[var(--color-text-hint)] mt-1 leading-none">
                  {conversationCount === 1 ? 'conversation' : 'conversations'}
                </p>
              </div>
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border)] rounded-lg px-3 py-2">
                <p className="font-mono text-base font-semibold text-[var(--color-text-primary)] leading-none">
                  {memberSince}
                </p>
                <p className="text-[10px] text-[var(--color-text-hint)] mt-1 leading-none">member since</p>
              </div>
            </div>
          </div>

          {/* ── Account details (copyable) ────────────────────────── */}
          <div className="p-1.5 border-b border-[var(--color-border)]">
            <p className="text-[10px] font-mono text-[var(--color-text-hint)] px-3 pt-1.5 pb-1 tracking-widest uppercase">
              Account
            </p>
            <CopyField label="username" value={user.user_name} />
            <CopyField label="email" value={user.email} />
            {user.user_id > 0 && (
              <CopyField label="user id" value={String(user.user_id)} />
            )}
          </div>

          {/* ── Sign out ──────────────────────────────────────────── */}
          <div className="p-1.5">
            <button
              role="menuitem"
              onClick={handleLogout}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-xs text-red-400 hover:bg-red-950/40 hover:text-red-300 transition-colors"
            >
              <LogOut size={13} className="flex-shrink-0" />
              <span className="font-medium">Sign out</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
