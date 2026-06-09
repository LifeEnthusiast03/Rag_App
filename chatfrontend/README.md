# Readwise — PDF Chat Frontend

> **Your documents, finally understood.**  
> Upload any PDF and have an intelligent, structured conversation about its contents — powered by AI.

---

## Overview

Readwise is a dark-mode-first SaaS web application that lets users upload PDF documents and interact with them through an AI-powered chat interface. Responses are structured with key points, source citations, confidence grades, and follow-up suggestions.

**Design language:** *Ink & Glass* — dark, precise, editorial. Built to feel like a research terminal designed by a design-conscious team.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | React 19 + TypeScript |
| Build tool | Vite 7 |
| Styling | Tailwind CSS v4 (utility-first + custom CSS vars) |
| Icons | Lucide React |
| Routing | React Router v7 |
| Fonts | Inter (UI) · JetBrains Mono (metadata, badges, timestamps) |
| Deployment | Vercel (SPA rewrites via `vercel.json`) |

---

## Project Structure

```
src/
├── App.css                    # Design tokens, shimmer animation, global styles
├── main.tsx                   # Entry point — context providers + router
├── App.tsx                    # Route definitions
│
├── pages/
│   ├── loginpage.tsx          # Split-panel login (branding left, form right)
│   ├── signuppage.tsx         # Split-panel signup with password strength meter
│   ├── uploadchat.tsx         # Main dashboard: sidebar + upload + chat interface
│   └── githubcallback.tsx     # GitHub OAuth callback handler
│
├── components/
│   ├── profile-dropdown.tsx   # User menu with stats, copyable fields, sign-out
│   ├── theme-toggle.tsx       # Sun/Moon toggle (dark ↔ light)
│   ├── theme-provider.tsx     # React context for dark/light theme
│   └── login-form.tsx         # Reusable form component
│
├── context/
│   ├── authcontext.tsx        # Auth state (token, user), login/signup/logout
│   └── chatcontext.tsx        # Chat list state, conversation loading
│
├── service/
│   ├── authservice.ts         # fetch() wrappers → POST /login, POST /signup
│   └── chatservice.ts         # fetch() wrappers → chat CRUD + upload
│
├── hooks/
│   ├── useauth.ts             # useContext(AuthContext)
│   └── usechat.ts             # useContext(ChatContext)
│
├── routes/
│   └── protectedroute.tsx     # Redirects to /login if not authenticated
│
└── type/
    └── types.ts               # All TypeScript interfaces
```

---

## Features

### Auth
- **Email / password** login and signup
- **GitHub OAuth** — redirects to backend, handled via `/auth/github/callback`
- Persistent session via `localStorage` (token + user object)
- Password strength meter on signup (length, number, special char)
- Real-time confirm-password match indicator

### Main Dashboard
- **PDF upload** — click-to-browse or drag-and-drop, up to 50 MB
- **Sidebar** — conversation history with hover-reveal delete, active item accent
- **Structured AI responses** parsed from JSON:
  - `answer` — main response text
  - `key_points` — collapsible bullet list
  - `confidence_level` — monospace badge (LOW / MEDIUM / HIGH)
  - `sources_cited` — numbered source list
  - `follow_up_suggestions` — horizontal-scroll chips; clicking sends the suggestion
- **Auto-growing textarea** (1–6 rows), Enter to send, Shift+Enter for newline
- Typing indicator while AI is responding

### Profile Dropdown
- Gradient avatar with initials
- Stats: conversation count + member year
- Copyable username, email, user ID (clipboard icon reveals on hover, Check on success)
- Escape key closes and returns focus to trigger

---

## Design System — *Ink & Glass*

All design tokens are CSS custom properties defined in `src/App.css`:

```css
--color-bg-base:      oklch(8% 0.01 270)   /* #0A0A10 near-black */
--color-bg-surface:   oklch(11% 0.012 270) /* panel backgrounds  */
--color-bg-elevated:  oklch(15% 0.014 270) /* inputs, cards      */
--color-border:       oklch(22% 0.012 270) /* all borders        */
--color-border-focus: oklch(62% 0.18 285)  /* focus ring (violet)*/
--color-accent:       oklch(60% 0.2 285)   /* #6B5CE7 muted violet*/
--color-text-primary: oklch(94% 0.006 270)
--color-text-muted:   oklch(58% 0.01 270)
--color-text-hint:    oklch(40% 0.008 270)
--color-success:      oklch(72% 0.18 150)  /* green dot           */
```

**Rules:**
- No `box-shadow` anywhere — depth via borders only
- No background gradients on surfaces — flat dark panels
- `font-mono` → JetBrains Mono (filenames, timestamps, badges, IDs)
- `.btn-primary` → 3-second violet shimmer animation (`@keyframes shimmer`)

---

## Backend API

The frontend communicates with a FastAPI backend running at `http://localhost:8000`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/login` | POST | Email/password authentication |
| `/signup` | POST | User registration |
| `/githublogin` | GET | GitHub OAuth redirect |
| `/upload-pdfs` | POST | Upload PDF → returns `chat_id` |
| `/chat` | POST | Send question → structured JSON response |
| `/getchat` | GET | List all chats for the user |
| `/getchatconversation?chatid=` | GET | Load message history |
| `/deletechat?chatid=` | DELETE | Remove a conversation |

---

## Getting Started

### Prerequisites
- Node.js 18+
- Backend API running at `http://localhost:8000`

### Install & run

```bash
# Install dependencies
npm install

# Start development server (http://localhost:5173)
npm run dev

# Type check
npx tsc --noEmit

# Production build
npm run build
```

### Environment
No `.env` file needed for the frontend — the backend URL is hardcoded to `http://localhost:8000`. Update `src/service/` files if deploying against a different API base URL.

---

## Deployment

The project includes a `vercel.json` that rewrites all routes to `index.html` for client-side routing:

```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```

Deploy with:
```bash
npm run build
# then deploy the dist/ folder to Vercel, Netlify, or any static host
```

---

## Routes

| Path | Component | Auth required |
|---|---|---|
| `/` | `PDFChatInterface` | Yes |
| `/login` | `LoginPage` | No |
| `/signup` | `SignUpPage` | No |
| `/auth/github/callback` | `GitHubCallback` | No |

---

## License

Private — all rights reserved.
