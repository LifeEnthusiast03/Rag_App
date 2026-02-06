import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldSeparator,
} from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Github } from "lucide-react"

export function LoginForm({
  className,
  ...props
}: React.ComponentProps<"form">) {
  return (
    <form className={cn("flex flex-col gap-8", className)} {...props}>
      <FieldGroup>
        <div className="flex flex-col gap-2 text-center">
          <h1 className="text-3xl font-bold tracking-tight">Welcome back</h1>
          <p className="text-muted-foreground text-base">
            Enter your credentials to access your account
          </p>
        </div>
        <Field>
          <FieldLabel htmlFor="email" className="text-base">Email</FieldLabel>
          <Input 
            id="email" 
            type="email" 
            placeholder="name@example.com" 
            required 
            className="h-11 text-base"
          />
        </Field>
        <Field>
          <div className="flex items-center justify-between mb-2">
            <FieldLabel htmlFor="password" className="text-base">Password</FieldLabel>
            <a
              href="#"
              className="text-sm text-primary hover:text-primary/80 underline-offset-4 hover:underline transition-colors"
            >
              Forgot password?
            </a>
          </div>
          <Input 
            id="password" 
            type="password" 
            required 
            className="h-11 text-base"
            placeholder="Enter your password"
          />
        </Field>
        <Field>
          <Button type="submit" className="h-11 text-base font-semibold">Sign in</Button>
        </Field>
        <FieldSeparator>Or continue with</FieldSeparator>
        <Field>
          <Button variant="outline" type="button" className="h-11 gap-2">
            <Github className="size-5" />
            Continue with GitHub
          </Button>
          <FieldDescription className="text-center text-base mt-4">
            Don&apos;t have an account?{" "}
            <a href="#" className="text-primary font-medium hover:text-primary/80 underline-offset-4 hover:underline transition-colors">
              Create account
            </a>
          </FieldDescription>
        </Field>
      </FieldGroup>
    </form>
  )
}
