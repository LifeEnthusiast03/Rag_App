import { useState } from 'react';
import { useNavigate } from 'react-router';
import type { loginForm,callResponse } from '@/type/types';
import { useAuthContext } from '@/hooks/useauth';
export default function LoginPage() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState<loginForm>({
    email: '',
    password: ''
  });
  const {loading,login} = useAuthContext()
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError('');

    // Validation
    if (!formData.email || !formData.password) {
      setError('Please fill in all required fields');
      return;
    }

    try {
          const res:callResponse = await login(formData);
          if(!res.Successful){
              throw new Error(res.msg)
          }
          navigate('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center p-4 overflow-hidden relative">
      {/* Background Effect */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-gray-800/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-gray-700/20 rounded-full blur-3xl"></div>
      </div>

      {/* Auth Container */}
      <div className="w-full max-w-md relative z-10 animate-fade-in">
        {/* Logo/Brand */}
        <div className="text-center mb-8 animate-slide-up">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-gray-800 to-gray-700 rounded-3xl mb-4 shadow-2xl hover:scale-105 transition-transform duration-300">
            <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
          </div>
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
            Readwise
          </h1>
          <p className="text-gray-400">
            Welcome back
          </p>
        </div>

        {/* Auth Form */}
        <div className="bg-gray-900/40 backdrop-blur-xl border border-gray-800/50 rounded-3xl p-8 shadow-2xl">
          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 bg-red-900/30 backdrop-blur-sm border border-red-700/50 rounded-2xl flex items-start gap-3 animate-shake">
              <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Email Field */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                Email Address
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 12a4 4 0 10-8 0 4 4 0 008 0zm0 0v1.5a2.5 2.5 0 005 0V12a9 9 0 10-9 9m4.5-1.206a8.959 8.959 0 01-4.5 1.207" />
                  </svg>
                </div>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  className="w-full pl-12 pr-4 py-4 bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 text-white placeholder-gray-500 rounded-2xl focus:outline-none focus:ring-2 focus:ring-gray-600 focus:border-gray-600 transition-all shadow-lg hover:shadow-xl"
                  placeholder="you@example.com"
                  required
                />
              </div>
            </div>

            {/* Password Field */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                Password
              </label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                  <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                  </svg>
                </div>
                <input
                  type="password"
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleInputChange}
                  className="w-full pl-12 pr-4 py-4 bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 text-white placeholder-gray-500 rounded-2xl focus:outline-none focus:ring-2 focus:ring-gray-600 focus:border-gray-600 transition-all shadow-lg hover:shadow-xl"
                  placeholder="••••••••"
                  required
                />
              </div>
            </div>

            {/* Forgot Password */}
            <div className="flex justify-end">
              <button type="button" className="text-sm text-gray-400 hover:text-gray-200 transition-colors">
                Forgot password?
              </button>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-5 px-6 rounded-2xl font-bold text-white text-lg transition-all transform flex items-center justify-center gap-3 relative overflow-hidden group bg-gradient-to-r from-gray-800 via-gray-700 to-gray-600 hover:from-gray-700 hover:via-gray-600 hover:to-gray-500 hover:scale-[1.02] active:scale-[0.98] shadow-2xl hover:shadow-3xl disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              {!loading && (
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700"></div>
              )}
              {loading ? (
                <>
                  <svg className="animate-spin h-6 w-6" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <span className="relative z-10">Sign In</span>
                  <svg className="w-5 h-5 relative z-10 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="flex items-center gap-4 my-6">
            <div className="flex-1 h-px bg-gray-700"></div>
            <span className="text-gray-500 text-sm font-medium">OR</span>
            <div className="flex-1 h-px bg-gray-700"></div>
          </div>

          {/* GitHub Login Button */}
          <a
            href="http://localhost:8000/githublogin"
            className="w-full py-4 px-6 rounded-2xl font-semibold text-white text-base transition-all transform flex items-center justify-center gap-3 bg-gray-800/70 border border-gray-700/70 hover:bg-gray-700/70 hover:border-gray-600 hover:scale-[1.02] active:scale-[0.98] shadow-lg hover:shadow-xl"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
            <span>Continue with GitHub</span>
          </a>

          {/* Sign Up Link */}
          <div className="text-center mt-6">
            <p className="text-sm text-gray-400">
              Don't have an account?{' '}
              <button
                type="button"
                onClick={() => navigate('/signup')}
                className="text-gray-200 hover:text-white font-semibold underline transition-colors"
              >
                Sign up
              </button>
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-6 text-sm text-gray-500">
          <p>© 2026 PDF Chat. All rights reserved.</p>
        </div>
      </div>
    </div>
  );
}