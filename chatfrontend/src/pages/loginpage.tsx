import { GalleryVerticalEnd } from "lucide-react"

import { LoginForm } from "@/components/login-form"
import { ThemeToggle } from "@/components/theme-toggle"

export default function LoginPage() {
  return (
    <div className="grid min-h-screen lg:grid-cols-2 bg-background">
      <div className="flex flex-col gap-6 p-8 md:p-12">
        <div className="flex justify-between items-center">
          <div className="flex justify-center gap-2 md:justify-start">
            <a href="#" className="flex items-center gap-2 font-semibold text-lg hover:opacity-80 transition-opacity">
              <div className="bg-primary text-primary-foreground flex size-8 items-center justify-center rounded-lg">
                <GalleryVerticalEnd className="size-5" />
              </div>
              Acme Inc.
            </a>
          </div>
          <ThemeToggle />
        </div>
        <div className="flex flex-1 items-center justify-center">
          <div className="w-full max-w-md">
            <LoginForm />
          </div>
        </div>
      </div>
      <div className="relative hidden lg:flex items-center justify-center bg-muted/50 border-l border-border">
        <div className="text-center p-12 max-w-md">
          <h2 className="text-3xl font-bold mb-4">Welcome Back</h2>
          <p className="text-muted-foreground text-lg">
            Access your account to continue with our amazing services.
          </p>
        </div>
      </div>
    </div>
  )
}
