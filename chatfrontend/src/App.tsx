 import { useState } from "react"
 import { Calendar } from "@/components/ui/calendar"
function App() {
  const [count, setCount] = useState(0)
  const [isDark, setIsDark] = useState(false)
  const [date, setDate] = useState<Date | undefined>(new Date())
 

  return (
    <>
    <div className={`min-h-screen ${isDark ? 'bg-gray-900' : 'bg-gradient-to-br from-purple-400 via-pink-500 to-red-500'} transition-all duration-500`}>
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4 drop-shadow-lg">
            🎨 Random Test Component
          </h1>
          <p className="text-white text-lg opacity-90">
            Testing Tailwind CSS with random JSX
          </p>
        </header>
        <Calendar
    mode="single"
    selected={date}
    onSelect={setDate}
    className="rounded-lg border"
  />
  

        {/* Main Content */}
        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {/* Card 1 - Counter */}
          <div className="bg-white rounded-2xl shadow-2xl p-8 transform hover:scale-105 transition-transform duration-300">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Counter Test</h2>
              <span className="text-4xl">🔢</span>
            </div>
            <div className="text-center">
              <div className="text-6xl font-bold text-purple-600 mb-6">
                {count}
              </div>
              <div className="flex gap-4 justify-center">
                <button
                  onClick={() => setCount(count - 1)}
                  className="bg-red-500 hover:bg-red-600 text-white font-semibold py-3 px-6 rounded-lg shadow-md transition-colors duration-200"
                >
                  Decrease
                </button>
                <button
                  onClick={() => setCount(count + 1)}
                  className="bg-green-500 hover:bg-green-600 text-white font-semibold py-3 px-6 rounded-lg shadow-md transition-colors duration-200"
                >
                  Increase
                </button>
              </div>
              <button
                onClick={() => setCount(0)}
                className="mt-4 bg-gray-500 hover:bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition-colors duration-200"
              >
                Reset
              </button>
            </div>
          </div>

          {/* Card 2 - Theme Toggle */}
          <div className="bg-white rounded-2xl shadow-2xl p-8 transform hover:scale-105 transition-transform duration-300">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Theme Toggle</h2>
              <span className="text-4xl">{isDark ? '🌙' : '☀️'}</span>
            </div>
            <div className="text-center">
              <div className="mb-6">
                <div className={`w-32 h-32 mx-auto rounded-full ${isDark ? 'bg-gray-800' : 'bg-yellow-400'} shadow-lg flex items-center justify-center text-6xl transition-colors duration-500`}>
                  {isDark ? '🌙' : '☀️'}
                </div>
              </div>
              <button
                onClick={() => setIsDark(!isDark)}
                className="bg-indigo-500 hover:bg-indigo-600 text-white font-semibold py-3 px-8 rounded-lg shadow-md transition-colors duration-200"
              >
                Toggle {isDark ? 'Light' : 'Dark'} Mode
              </button>
              <p className="mt-4 text-gray-600">
                Current theme: <span className="font-bold">{isDark ? 'Dark' : 'Light'}</span>
              </p>
            </div>
          </div>
        </div>

        {/* Feature Cards */}
        <div className="mt-12 grid md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          <div className="bg-white bg-opacity-20 backdrop-blur-lg rounded-xl p-6 text-white">
            <div className="text-4xl mb-3">⚡</div>
            <h3 className="text-xl font-bold mb-2">Fast</h3>
            <p className="opacity-90">Lightning-fast development with Tailwind CSS</p>
          </div>
          <div className="bg-white bg-opacity-20 backdrop-blur-lg rounded-xl p-6 text-white">
            <div className="text-4xl mb-3">🎯</div>
            <h3 className="text-xl font-bold mb-2">Precise</h3>
            <p className="opacity-90">Utility-first CSS for precise styling</p>
          </div>
          <div className="bg-white bg-opacity-20 backdrop-blur-lg rounded-xl p-6 text-white">
            <div className="text-4xl mb-3">🚀</div>
            <h3 className="text-xl font-bold mb-2">Modern</h3>
            <p className="opacity-90">Modern design patterns and best practices</p>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-white opacity-80">
          <p className="text-sm">Made with React + TypeScript + Tailwind CSS</p>
        </footer>
      </div>
    </div>
     
    </>
  )
    
}

export default App
