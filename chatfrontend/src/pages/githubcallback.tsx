import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router';
import { useAuthContext } from '@/hooks/useauth';

export default function GitHubCallback() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { setUser, setToken, setLoggedIn } = useAuthContext();
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const handleCallback = async () => {
      try {
        // Get token from URL parameters
        const token = searchParams.get('token');
        const userId = searchParams.get('user_id');
        const userName = searchParams.get('user_name');
        const email = searchParams.get('email');
        const errorMsg = searchParams.get('error');

        if (errorMsg) {
          throw new Error(errorMsg);
        }

        if (!token || !userId || !userName || !email) {
          throw new Error('Missing authentication data');
        }

        // Set user data in context
        const user = {
          user_id: parseInt(userId),
          user_name: userName,
          email: email,
          token: token
        };

        setUser(user);
        setToken(token);
        setLoggedIn(true);

        // Store in localStorage
        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(user));

        // Redirect to home page immediately
        setLoading(false);
        navigate('/');
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Authentication failed');
        setLoading(false);
      }
    };

    handleCallback();
  }, [searchParams, navigate, setUser, setToken, setLoggedIn]);

  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center p-4 overflow-hidden relative">
      {/* Background Effect */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-gray-800/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-gray-700/20 rounded-full blur-3xl"></div>
      </div>

      {/* Content Container */}
      <div className="w-full max-w-md relative z-10 animate-fade-in">
        <div className="bg-gray-900/40 backdrop-blur-xl border border-gray-800/50 rounded-3xl p-8 shadow-2xl">
          {loading && !error ? (
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-gray-800 to-gray-700 rounded-3xl mb-4 shadow-2xl">
                <svg className="animate-spin h-10 w-10" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              </div>
              <h2 className="text-2xl font-bold mb-2">Authenticating with GitHub</h2>
              <p className="text-gray-400">Please wait while we complete your login...</p>
            </div>
          ) : error ? (
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-red-900/30 rounded-3xl mb-4">
                <svg className="w-10 h-10 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold mb-2 text-red-400">Authentication Failed</h2>
              <p className="text-gray-300 mb-6">{error}</p>
              <button
                onClick={() => navigate('/login')}
                className="py-3 px-6 rounded-2xl font-medium text-white bg-gradient-to-r from-gray-800 via-gray-700 to-gray-600 hover:from-gray-700 hover:via-gray-600 hover:to-gray-500 transition-all"
              >
                Back to Login
              </button>
            </div>
          ) : (
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-green-900/30 rounded-3xl mb-4">
                <svg className="w-10 h-10 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold mb-2 text-green-400">Success!</h2>
              <p className="text-gray-400">Redirecting to your dashboard...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
