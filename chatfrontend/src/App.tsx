import { ThemeProvider } from "@/components/theme-provider"
import LoginPage from "./pages/loginpage"

function App() {
  // Set dark theme immediately
  if (typeof window !== "undefined") {
    document.documentElement.classList.add("dark")
  }

  return (
    <ThemeProvider defaultTheme="dark">
      <div className="min-h-screen bg-background text-foreground">
        <LoginPage />
      </div>
    </ThemeProvider>
  )
}

export default App
