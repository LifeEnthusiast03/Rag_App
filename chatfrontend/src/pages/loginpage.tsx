import { useState } from 'react';
import { useNavigate, Link } from 'react-router';
import { Mail, Lock, Eye, EyeOff, Zap, Target, MessageCircle, BookOpen, AlertCircle } from 'lucide-react';
import type { loginForm, callResponse } from '@/type/types';
import { useAuthContext } from '@/hooks/useauth';

/* ─── Shared: GitHub SVG mark ────────────────────────────────── */
const GitHubIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 0C5.37 0 0 5.373 0 12c0 5.303 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.84 1.237 1.84 1.237 1.07 1.834 2.807 1.304 3.492.997.108-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
  </svg>
);

/* ─── Shared: Spinner ────────────────────────────────────────── */
const Spinner = () => (
  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
  </svg>
);

/* ─── Shared: Form field ─────────────────────────────────────── */
interface FieldProps {
  label: string;
  children: React.ReactNode;
}
const Field = ({ label, children }: FieldProps) => (
  <div className="space-y-1.5">
    <label className="block text-[11px] font-medium text-[var(--color-text-muted)] tracking-wide uppercase">
      {label}
    </label>
    {children}
  </div>
);

const inputCls =
  'w-full bg-[var(--color-bg-base)] border border-[var(--color-border)] rounded-md text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-hint)] focus:outline-none focus:border-[var(--color-border-focus)] focus:ring-1 focus:ring-[var(--color-border-focus)]/50 transition-all duration-150';

/* ─── Left panel features data ───────────────────────────────── */
const features = [
  {
    Icon: Zap,
    title: 'AI-powered analysis',
    desc: 'Extracts meaning from any PDF — not just text, but context and intent.',
  },
  {
    Icon: Target,
    title: 'Structured every time',
    desc: 'Answers include key points, source citations, and a confidence grade.',
  },
  {
    Icon: MessageCircle,
    title: 'Conversations that build',
    desc: 'Intelligent follow-up suggestions that guide your research further.',
  },
  {
    Icon: BookOpen,
    title: 'Everything, organized',
    desc: 'Every upload and conversation, indexed and instantly recalled.',
  },
];

/* ─── Left branding panel ────────────────────────────────────── */
function BrandPanel() {
  return (
    <div
      className="hidden lg:flex flex-col justify-between h-full px-12 py-10 relative overflow-hidden"
      style={{ background: 'var(--color-bg-base)' }}
    >
      {/* Background: dot grid */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.35]"
        style={{
          backgroundImage: 'radial-gradient(circle, var(--color-border) 1px, transparent 1px)',
          backgroundSize: '28px 28px',
        }}
      />
      {/* Background: violet glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse at 60% 35%, oklch(18% 0.07 285 / 0.75) 0%, transparent 60%)',
        }}
      />

      {/* Top: wordmark */}
      <div className="relative z-10">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border)] flex items-center justify-center">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              className="text-[var(--color-accent)]">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
              <polyline points="10 9 9 9 8 9" />
            </svg>
          </div>
          <span className="text-base font-bold tracking-tight text-[var(--color-text-primary)]">
            Readwise
          </span>
        </div>
      </div>

      {/* Middle: headline + features */}
      <div className="relative z-10 space-y-8">
        <div>
          <h2 className="text-3xl font-bold -tracking-tight text-[var(--color-text-primary)] leading-[1.2]">
            Your documents,
            <br />
            <span style={{ color: 'var(--color-accent)' }}>finally understood.</span>
          </h2>
          <p className="mt-3 text-sm text-[var(--color-text-muted)] leading-relaxed max-w-xs">
            Ask anything. Get structured answers with citations,
            key points, and confidence scores.
          </p>
        </div>

        {/* Feature list */}
        <div className="space-y-4">
          {features.map(({ Icon, title, desc }) => (
            <div key={title} className="flex gap-3 items-start">
              <div className="w-7 h-7 rounded-md bg-[var(--color-bg-elevated)] border border-[var(--color-border)] flex items-center justify-center flex-shrink-0 mt-0.5">
                <Icon size={13} className="text-[var(--color-accent)]" />
              </div>
              <div>
                <p className="text-xs font-semibold text-[var(--color-text-primary)]">{title}</p>
                <p className="text-[11px] text-[var(--color-text-muted)] mt-0.5 leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Live response preview card */}
        <div className="border border-[var(--color-border)] rounded-xl overflow-hidden">
          <div className="bg-[var(--color-bg-elevated)] px-3 py-2 border-b border-[var(--color-border)] flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-success)]" />
            <span className="font-mono text-[9px] text-[var(--color-text-hint)] uppercase tracking-widest">
              Example response
            </span>
          </div>
          <div className="bg-[var(--color-bg-surface)] px-4 py-3 space-y-2.5">
            <p className="text-xs text-[var(--color-text-primary)] leading-relaxed">
              "The study shows a{' '}
              <span className="text-[var(--color-accent)] font-medium">34% improvement</span> in
              retention when using spaced repetition with active recall techniques..."
            </p>
            <div className="flex items-center gap-3 pt-1 border-t border-[var(--color-border)]">
              <span className="font-mono text-[9px] text-[var(--color-text-hint)]">
                confidence: HIGH
              </span>
              <span className="w-px h-3 bg-[var(--color-border)]" />
              <span className="text-[9px] text-[var(--color-text-muted)]">3 sources cited</span>
              <span className="w-px h-3 bg-[var(--color-border)]" />
              <span className="text-[9px] text-[var(--color-text-muted)]">5 key points</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom: attribution */}
      <div className="relative z-10">
        <p className="text-[11px] text-[var(--color-text-hint)]">
          © 2026 Readwise. Built for researchers, students, and analysts.
        </p>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   LOGIN PAGE
═══════════════════════════════════════════════════════════════ */
export default function LoginPage() {
  const navigate = useNavigate();
  const { loading, login } = useAuthContext();

  const [formData, setFormData] = useState<loginForm>({ email: '', password: '' });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) =>
    setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleSignIn = async () => {
    setError('');
    if (!formData.email || !formData.password) {
      setError('Please fill in both fields.');
      return;
    }
    try {
      const res: callResponse = await login(formData);
      if (!res.Successful) throw new Error(res.msg);
      navigate('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed. Please try again.');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSignIn();
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] flex">

      {/* ── Left: brand panel ─────────────────────────────────── */}
      <div className="lg:w-[45%] border-r border-[var(--color-border)]">
        <BrandPanel />
      </div>

      {/* ── Right: form panel ─────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-h-screen">

        {/* Top bar */}
        <div className="flex items-center justify-between px-8 py-5 border-b border-[var(--color-border)]">
          {/* Mobile-only logo (hidden on lg where left panel shows) */}
          <div className="flex items-center gap-2 lg:hidden">
            <div className="w-7 h-7 rounded-lg bg-[var(--color-bg-elevated)] border border-[var(--color-border)] flex items-center justify-center">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                className="text-[var(--color-accent)]">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
            </div>
            <span className="text-sm font-bold text-[var(--color-text-primary)]">Readwise</span>
          </div>
          <div className="hidden lg:block" />

          <p className="text-xs text-[var(--color-text-muted)]">
            Don&apos;t have an account?{' '}
            <Link
              to="/signup"
              className="text-[var(--color-accent)] hover:text-[var(--color-accent-hover)] font-medium transition-colors"
            >
              Sign up
            </Link>
          </p>
        </div>

        {/* Center: form */}
        <div className="flex-1 flex flex-col items-center justify-center px-8 py-12">
          <div className="w-full max-w-[360px]" onKeyDown={handleKeyDown}>

            {/* Heading */}
            <div className="mb-8">
              <h1 className="text-2xl font-bold -tracking-tight text-[var(--color-text-primary)]">
                Welcome back
              </h1>
              <p className="text-sm text-[var(--color-text-muted)] mt-1">
                Sign in to continue to your workspace.
              </p>
            </div>

            {/* GitHub OAuth — primary CTA position */}
            <a
              href="http://localhost:8000/githublogin"
              className="w-full flex items-center justify-center gap-2.5 px-4 py-2.5 bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-md text-sm font-medium text-[var(--color-text-primary)] hover:bg-[var(--color-bg-surface)] hover:border-[var(--color-text-hint)] transition-all duration-150"
            >
              <GitHubIcon />
              <span>Continue with GitHub</span>
            </a>

            {/* Divider */}
            <div className="relative my-5">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-[var(--color-border)]" />
              </div>
              <div className="relative flex justify-center">
                <span className="px-3 bg-[var(--color-bg-base)] text-[11px] text-[var(--color-text-hint)] font-mono">
                  or continue with email
                </span>
              </div>
            </div>

            {/* Error banner */}
            {error && (
              <div className="flex items-start gap-2.5 mb-4 px-3.5 py-3 bg-red-950/30 border border-red-900/50 rounded-lg">
                <AlertCircle size={14} className="text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-red-300 leading-relaxed">{error}</p>
              </div>
            )}

            {/* Fields */}
            <div className="space-y-4">
              <Field label="Email address">
                <div className="relative">
                  <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Mail size={13} className="text-[var(--color-text-hint)]" />
                  </span>
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="you@company.com"
                    autoComplete="email"
                    className={`${inputCls} pl-9 pr-3 py-2.5`}
                  />
                </div>
              </Field>

              <Field label="Password">
                <div className="relative">
                  <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock size={13} className="text-[var(--color-text-hint)]" />
                  </span>
                  <input
                    type={showPassword ? 'text' : 'password'}
                    name="password"
                    value={formData.password}
                    onChange={handleChange}
                    placeholder="••••••••"
                    autoComplete="current-password"
                    className={`${inputCls} pl-9 pr-10 py-2.5`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(v => !v)}
                    tabIndex={-1}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-[var(--color-text-hint)] hover:text-[var(--color-text-muted)] transition-colors"
                  >
                    {showPassword ? <EyeOff size={13} /> : <Eye size={13} />}
                  </button>
                </div>
                <div className="flex justify-end mt-1">
                  <button
                    type="button"
                    className="text-[11px] text-[var(--color-text-muted)] hover:text-[var(--color-accent)] transition-colors"
                  >
                    Forgot password?
                  </button>
                </div>
              </Field>
            </div>

            {/* Submit */}
            <button
              onClick={handleSignIn}
              disabled={loading}
              className="btn-primary w-full rounded-md py-2.5 text-sm font-semibold flex items-center justify-center gap-2 mt-6"
            >
              {loading ? <><Spinner /><span>Signing in…</span></> : <span>Sign in →</span>}
            </button>

            {/* Terms */}
            <p className="text-[11px] text-[var(--color-text-hint)] text-center mt-5 leading-relaxed">
              By signing in, you agree to our{' '}
              <span className="text-[var(--color-text-muted)] hover:text-[var(--color-accent)] cursor-pointer transition-colors">Terms of Service</span>
              {' '}and{' '}
              <span className="text-[var(--color-text-muted)] hover:text-[var(--color-accent)] cursor-pointer transition-colors">Privacy Policy</span>.
            </p>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="px-8 py-4 border-t border-[var(--color-border)] flex items-center justify-between">
          <p className="text-[11px] text-[var(--color-text-hint)]">© 2026 Readwise</p>
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-success)]" />
            <span className="font-mono text-[10px] text-[var(--color-text-hint)]">All systems operational</span>
          </div>
        </div>
      </div>
    </div>
  );
}