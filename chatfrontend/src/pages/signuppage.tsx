import { useState } from 'react';
import { useNavigate, Link } from 'react-router';
import { User, Mail, Lock, Eye, EyeOff, ShieldCheck, Layers, GitBranch, Globe, AlertCircle, CheckCircle2 } from 'lucide-react';
import type { callResponse } from '@/type/types';
import { useAuthContext } from '@/hooks/useauth';

/* ─── Shared atoms (same as loginpage) ───────────────────────── */
const GitHubIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 0C5.37 0 0 5.373 0 12c0 5.303 3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.84 1.237 1.84 1.237 1.07 1.834 2.807 1.304 3.492.997.108-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
  </svg>
);

const Spinner = () => (
  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
  </svg>
);

const inputCls =
  'w-full bg-[var(--color-bg-base)] border border-[var(--color-border)] rounded-md text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-hint)] focus:outline-none focus:border-[var(--color-border-focus)] focus:ring-1 focus:ring-[var(--color-border-focus)]/50 transition-all duration-150';

interface FieldProps { label: string; children: React.ReactNode }
const Field = ({ label, children }: FieldProps) => (
  <div className="space-y-1.5">
    <label className="block text-[11px] font-medium text-[var(--color-text-muted)] tracking-wide uppercase">
      {label}
    </label>
    {children}
  </div>
);

/* ─── Password strength meter ────────────────────────────────── */
function PasswordStrength({ password }: { password: string }) {
  if (!password) return null;

  const checks = [
    { label: 'At least 6 characters', pass: password.length >= 6 },
    { label: 'Contains a number', pass: /\d/.test(password) },
    { label: 'Contains a special character', pass: /[^a-zA-Z0-9]/.test(password) },
  ];
  const score = checks.filter(c => c.pass).length;
  const labels = ['', 'Weak', 'Fair', 'Strong'];
  const colors = ['', 'bg-red-500', 'bg-yellow-500', 'bg-[var(--color-success)]'];
  const textColors = ['', 'text-red-400', 'text-yellow-400', 'text-[var(--color-success)]'];

  return (
    <div className="mt-2 space-y-2">
      {/* Bar */}
      <div className="flex gap-1">
        {[1, 2, 3].map(i => (
          <div
            key={i}
            className={`h-1 flex-1 rounded-full transition-all duration-300 ${
              i <= score ? colors[score] : 'bg-[var(--color-border)]'
            }`}
          />
        ))}
      </div>
      {/* Label */}
      <p className={`text-[10px] font-mono ${textColors[score]}`}>
        {labels[score]} password
      </p>
      {/* Checklist */}
      <div className="space-y-1">
        {checks.map(({ label, pass }) => (
          <div key={label} className="flex items-center gap-1.5">
            <CheckCircle2
              size={11}
              className={pass ? 'text-[var(--color-success)]' : 'text-[var(--color-border)]'}
            />
            <span className={`text-[10px] ${pass ? 'text-[var(--color-text-muted)]' : 'text-[var(--color-text-hint)]'}`}>
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Left panel: signup variant ─────────────────────────────── */
const signupBenefits = [
  {
    Icon: ShieldCheck,
    title: 'Private & secure',
    desc: 'Your documents are processed securely and never shared.',
  },
  {
    Icon: Layers,
    title: 'Structured AI answers',
    desc: 'Every response includes citations, key points, and a confidence grade.',
  },
  {
    Icon: GitBranch,
    title: 'Conversation threads',
    desc: 'Ask follow-up questions. Every exchange is saved and searchable.',
  },
  {
    Icon: Globe,
    title: 'Works on any PDF',
    desc: 'Research papers, contracts, textbooks, reports — upload anything.',
  },
];

function BrandPanel() {
  return (
    <div
      className="hidden lg:flex flex-col justify-between h-full px-12 py-10 relative overflow-hidden"
      style={{ background: 'var(--color-bg-base)' }}
    >
      {/* Dot grid */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.35]"
        style={{
          backgroundImage: 'radial-gradient(circle, var(--color-border) 1px, transparent 1px)',
          backgroundSize: '28px 28px',
        }}
      />
      {/* Violet glow — top-right for variety */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse at 70% 25%, oklch(18% 0.07 285 / 0.75) 0%, transparent 60%)',
        }}
      />

      {/* Wordmark */}
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

      {/* Headline + benefits */}
      <div className="relative z-10 space-y-8">
        <div>
          <h2 className="text-3xl font-bold -tracking-tight text-[var(--color-text-primary)] leading-[1.2]">
            Start your research
            <br />
            <span style={{ color: 'var(--color-accent)' }}>smarter, not harder.</span>
          </h2>
          <p className="mt-3 text-sm text-[var(--color-text-muted)] leading-relaxed max-w-xs">
            Join researchers, students, and analysts who use Readwise
            to get more from every document.
          </p>
        </div>

        <div className="space-y-4">
          {signupBenefits.map(({ Icon, title, desc }) => (
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

        {/* Testimonial card */}
        <div className="border border-[var(--color-border)] rounded-xl bg-[var(--color-bg-surface)] px-4 py-3.5 space-y-2">
          <p className="text-xs text-[var(--color-text-primary)] leading-relaxed italic">
            "Readwise cut the time I spend reading papers in half.
            The structured answers are genuinely impressive."
          </p>
          <div className="flex items-center gap-2.5">
            <div
              className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-semibold text-white flex-shrink-0"
              style={{ background: 'linear-gradient(135deg, var(--color-accent), oklch(55% 0.22 300))' }}
            >
              AK
            </div>
            <div>
              <p className="text-[11px] font-medium text-[var(--color-text-primary)]">Alex K.</p>
              <p className="text-[10px] text-[var(--color-text-hint)]">PhD Researcher, NeurIPS 2024</p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="relative z-10">
        <p className="text-[11px] text-[var(--color-text-hint)]">
          © 2026 Readwise. Built for researchers, students, and analysts.
        </p>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   SIGNUP PAGE
═══════════════════════════════════════════════════════════════ */
export default function SignUpPage() {
  const navigate = useNavigate();
  const { loading, signup } = useAuthContext();

  const [formData, setFormData] = useState({
    user_name: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [showStrength, setShowStrength] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) =>
    setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleCreateAccount = async () => {
    setError('');
    if (!formData.user_name || !formData.email || !formData.password || !formData.confirmPassword) {
      setError('Please fill in all required fields.');
      return;
    }
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    try {
      const res: callResponse = await signup({
        user_name: formData.user_name,
        email: formData.email,
        password: formData.password,
      });
      if (!res.Successful) throw new Error(res.msg);
      navigate('/login');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Account creation failed. Please try again.');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleCreateAccount();
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg-base)] flex">

      {/* ── Left panel ────────────────────────────────────────── */}
      <div className="lg:w-[45%] border-r border-[var(--color-border)]">
        <BrandPanel />
      </div>

      {/* ── Right panel ───────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-h-screen">

        {/* Top bar */}
        <div className="flex items-center justify-between px-8 py-5 border-b border-[var(--color-border)]">
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
            Already have an account?{' '}
            <Link
              to="/login"
              className="text-[var(--color-accent)] hover:text-[var(--color-accent-hover)] font-medium transition-colors"
            >
              Sign in
            </Link>
          </p>
        </div>

        {/* Form */}
        <div className="flex-1 flex flex-col items-center justify-center px-8 py-10">
          <div className="w-full max-w-[360px]" onKeyDown={handleKeyDown}>

            <div className="mb-7">
              <h1 className="text-2xl font-bold -tracking-tight text-[var(--color-text-primary)]">
                Create your account
              </h1>
              <p className="text-sm text-[var(--color-text-muted)] mt-1">
                Free to start. No credit card required.
              </p>
            </div>

            {/* GitHub OAuth */}
            <a
              href="http://localhost:8000/githublogin"
              className="w-full flex items-center justify-center gap-2.5 px-4 py-2.5 bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-md text-sm font-medium text-[var(--color-text-primary)] hover:bg-[var(--color-bg-surface)] hover:border-[var(--color-text-hint)] transition-all duration-150"
            >
              <GitHubIcon />
              <span>Sign up with GitHub</span>
            </a>

            {/* Divider */}
            <div className="relative my-5">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-[var(--color-border)]" />
              </div>
              <div className="relative flex justify-center">
                <span className="px-3 bg-[var(--color-bg-base)] text-[11px] text-[var(--color-text-hint)] font-mono">
                  or sign up with email
                </span>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="flex items-start gap-2.5 mb-4 px-3.5 py-3 bg-red-950/30 border border-red-900/50 rounded-lg">
                <AlertCircle size={14} className="text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-red-300 leading-relaxed">{error}</p>
              </div>
            )}

            <div className="space-y-4">
              {/* Full name */}
              <Field label="Full name">
                <div className="relative">
                  <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <User size={13} className="text-[var(--color-text-hint)]" />
                  </span>
                  <input
                    type="text"
                    name="user_name"
                    value={formData.user_name}
                    onChange={handleChange}
                    placeholder="Alex Kim"
                    autoComplete="name"
                    className={`${inputCls} pl-9 pr-3 py-2.5`}
                  />
                </div>
              </Field>

              {/* Email */}
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

              {/* Password + strength meter */}
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
                    onFocus={() => setShowStrength(true)}
                    placeholder="Min. 6 characters"
                    autoComplete="new-password"
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
                {showStrength && <PasswordStrength password={formData.password} />}
              </Field>

              {/* Confirm password */}
              <Field label="Confirm password">
                <div className="relative">
                  <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <Lock size={13} className="text-[var(--color-text-hint)]" />
                  </span>
                  <input
                    type={showConfirm ? 'text' : 'password'}
                    name="confirmPassword"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    placeholder="Repeat password"
                    autoComplete="new-password"
                    className={`${inputCls} pl-9 pr-10 py-2.5 ${
                      formData.confirmPassword && formData.password !== formData.confirmPassword
                        ? 'border-red-800/70 focus:border-red-600'
                        : formData.confirmPassword && formData.password === formData.confirmPassword
                        ? 'border-[var(--color-success)]/50'
                        : ''
                    }`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirm(v => !v)}
                    tabIndex={-1}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-[var(--color-text-hint)] hover:text-[var(--color-text-muted)] transition-colors"
                  >
                    {showConfirm ? <EyeOff size={13} /> : <Eye size={13} />}
                  </button>
                </div>
                {/* Inline password match feedback */}
                {formData.confirmPassword && formData.password !== formData.confirmPassword && (
                  <p className="text-[10px] text-red-400 mt-1 font-mono">Passwords do not match</p>
                )}
                {formData.confirmPassword && formData.password === formData.confirmPassword && (
                  <p className="text-[10px] text-[var(--color-success)] mt-1 font-mono flex items-center gap-1">
                    <CheckCircle2 size={10} /> Passwords match
                  </p>
                )}
              </Field>
            </div>

            {/* Submit */}
            <button
              onClick={handleCreateAccount}
              disabled={loading}
              className="btn-primary w-full rounded-md py-2.5 text-sm font-semibold flex items-center justify-center gap-2 mt-6"
            >
              {loading
                ? <><Spinner /><span>Creating account…</span></>
                : <span>Create account →</span>
              }
            </button>

            {/* Terms */}
            <p className="text-[11px] text-[var(--color-text-hint)] text-center mt-4 leading-relaxed">
              By creating an account, you agree to our{' '}
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
