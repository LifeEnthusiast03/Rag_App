# RAG Chat Application - Full Stack

> A modern full-stack RAG (Retrieval-Augmented Generation) application for intelligent conversations with PDF documents. Built with React 19, TypeScript, FastAPI, PostgreSQL, and FAISS vector search. Upload PDFs and have AI-powered conversations grounded in your document content with complete authentication and conversation management.

## 🎯 Quick Overview

This is a production-ready full-stack RAG (Retrieval-Augmented Generation) chat application that enables users to have intelligent conversations with their PDF documents. The frontend is built with React 19, TypeScript, and Tailwind CSS, integrated with a FastAPI backend using PostgreSQL and FAISS vector search.

**Key Capabilities**:
- 📄 Upload PDF documents with automatic processing
- 💬 Chat with PDFs using AI-powered RAG responses
- 🔐 Secure authentication with JWT tokens
- 📚 Manage multiple conversations seamlessly
- 🗑️ Delete conversations with confirmation dialogs
- 💾 Persistent chat history in PostgreSQL database
- 🎨 Modern, responsive dark UI with smooth interactions
- 🔍 FAISS vector similarity search for accurate context retrieval

**Tech Stack**: React 19 • TypeScript 5.9 • Vite 7.2 • Tailwind CSS 4.1 • React Router 7.13 • FastAPI • PostgreSQL • FAISS

**Status**: ✅ Core features complete | 🚀 Active development | ✨ Production-ready frontend

## � Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Development](#-development)
- [Testing & Quality](#-testing--quality)
- [API Integration](#-api-integration)
- [Application Architecture](#-application-architecture)
- [Customization](#-customization)
- [Authentication Flow](#-authentication-flow)
- [Data Persistence & Storage](#-data-persistence--storage)
- [Features & Functionality](#-features--functionality)
- [Troubleshooting](#-troubleshooting)
- [Build & Deployment](#-build--deployment)
- [Contributing](#-contributing)
- [Current State & Future Enhancements](#-current-state--future-enhancements)
- [Project Status](#-project-status)
- [Support & Contact](#-support--contact)

## �🚀 Features

### Authentication System
- **User Registration & Login**: Secure JWT-based authentication
- **Protected Routes**: Route guards preventing unauthorized access to chat features
- **Persistent Sessions**: Automatic login restoration using localStorage
- **Dual Context Architecture**: Separate AuthContext and ChatContext for clean state management
- **Secure Logout**: Complete session cleanup on logout

### PDF Document Management
- **PDF Upload**: Upload PDF files with validation
- **File Type Validation**: Client-side PDF type checking
- **Multi-file Support**: Handle multiple PDF documents
- **Upload Progress**: Visual feedback during file uploads
- **Automatic Chat Creation**: Each PDF upload creates a new chat session

### Chat Interface
- **Interactive Q&A**: Ask questions and get AI-powered answers about uploaded documents
- **Chat History Persistence**: All conversations saved to backend database
- **Multi-conversation Support**: Manage and switch between multiple PDF chat sessions
- **Conversation Sidebar**: Collapsible sidebar displaying all user chat sessions
- **Load Previous Chats**: Retrieve and continue past conversations
- **Delete Conversations**: Remove unwanted chats with confirmation dialog
- **Real-time Messaging**: Instant AI responses with loading indicators
- **Message Display**: Clean message interface with role-based styling (user/assistant)
- **Smart State Management**: Automatic UI updates when deleting active conversations

### User Interface & Experience
- **Modern Dark Theme**: Sleek, gradient-based dark UI with backdrop blur effects
- **Responsive Design**: Mobile-first design working seamlessly across all devices
- **Collapsible Sidebar**: Toggleable conversation history sidebar
- **Theme Support**: Theme provider infrastructure for light/dark modes
- **Smooth Animations**: Polished transitions and hover effects
- **Hover Actions**: Context-sensitive delete buttons appearing on hover
- **Auto-scroll**: Automatic scroll to latest messages
- **Loading States**: Visual feedback for uploads, chat responses, and data loading
- **Error Handling**: User-friendly error messages and validation
- **Keyboard Shortcuts**: Enter to send messages, easy navigation
- **Confirmation Dialogs**: Safety prompts for destructive actions

### State Management Architecture
- **AuthContext**: Centralized authentication state (user, token, login/logout)
- **ChatContext**: Chat-specific state (current chat, chat list, conversation loading)
- **Custom Hooks**: `useauth` and `usechat` for clean component integration
- **localStorage Integration**: Persistent auth tokens and user data
- **Auto-restore Sessions**: Automatic login on app reload if token exists

## 🛠️ Tech Stack

### Core
- **Framework**: React 19.2 + TypeScript 5.9
- **Build Tool**: Vite 7.2
- **Routing**: React Router 7.13
- **State Management**: React Context API + Custom Hooks

### UI & Styling
- **Styling**: Tailwind CSS 4.1 with @tailwindcss/vite plugin
- **UI Components**: Custom components built with Radix UI primitives
- **Component Library**: shadcn/ui pattern with custom styling
- **Icons**: Lucide React
- **Utilities**: 
  - clsx & tailwind-merge for className management
  - class-variance-authority for component variants
  - tw-animate-css for animations

### Development
- **Type Safety**: TypeScript with strict type checking
- **Linting**: ESLint 9 with React-specific rules
- **Code Quality**: typescript-eslint for TypeScript linting
- **Hot Reload**: Vite's lightning-fast HMR

### Additional Libraries
- **Date Handling**: date-fns 4.1
- **Calendar**: react-day-picker 9.13
- **HTTP Client**: Native Fetch API

### Backend Stack (Integrated)
- **Framework**: FastAPI (Python)
- **Database**: PostgreSQL for user data, chat sessions, and message history
- **Vector Store**: FAISS (Facebook AI Similarity Search) for document embeddings
- **RAG Pipeline**: LangChain integration for retrieval-augmented generation
- **Authentication**: JWT (JSON Web Tokens) for secure session management
- **Document Processing**: PDF parsing and text extraction
- **Embeddings**: Vector embeddings for semantic search

## 📁 Project Structure

```
e:\rag/
├── chatfrontend/                # Frontend application
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── ui/              # shadcn/ui style components
│   │   │   │   ├── button.tsx
│   │   │   │   ├── input.tsx
│   │   │   │   ├── label.tsx
│   │   │   │   ├── calendar.tsx
│   │   │   │   ├── field.tsx
│   │   │   │   └── separator.tsx
│   │   │   ├── login-form.tsx
│   │   │   ├── theme-provider.tsx
│   │   │   └── theme-toggle.tsx
│   │   ├── context/             # React Context providers
│   │   │   ├── authcontext.tsx  # Authentication state & logic
│   │   │   └── chatcontext.tsx  # Chat state & conversation management
│   │   ├── hooks/               # Custom React hooks
│   │   │   ├── useauth.ts       # Hook to access AuthContext
│   │   │   └── usechat.ts       # Hook to access ChatContext
│   │   ├── pages/               # Page components
│   │   │   ├── loginpage.tsx    # Login page
│   │   │   ├── signuppage.tsx   # User registration page
│   │   │   └── uploadchat.tsx   # Main chat interface
│   │   ├── routes/              # Route configuration
│   │   │   └── protectedroute.tsx  # Route guard for authentication
│   │   ├── service/             # API service layer
│   │   │   ├── authservice.ts   # Login/signup API calls
│   │   │   └── chatservice.ts   # Chat, upload, conversation APIs
│   │   ├── type/                # TypeScript type definitions
│   │   │   └── types.ts         # All interfaces and types
│   │   ├── lib/                 # Utility functions
│   │   │   └── utils.ts         # Helper functions (cn, etc.)
│   │   ├── assets/              # Static assets
│   │   ├── App.tsx              # Main app with routing
│   │   ├── App.css              # Application styles
│   │   ├── main.tsx             # Entry point with context providers
│   │   └── index.css            # Global styles & Tailwind
│   ├── public/                  # Static public assets
│   ├── components.json          # shadcn/ui configuration
│   ├── vite.config.ts           # Vite configuration
│   ├── tsconfig.json            # TypeScript config
│   ├── tsconfig.app.json        # App-specific TS config
│   ├── tsconfig.node.json       # Node-specific TS config
│   ├── eslint.config.js         # ESLint configuration
│   ├── package.json             # Dependencies & scripts
│   └── index.html               # HTML entry point
├── README.md                    # This file
├── LICENSE                      # License file
├── doubt.txt                    # Development notes
└── nextjob.txt                  # Future tasks/todos
```

## 🚦 Getting Started

### Prerequisites

**Frontend Requirements**:
- **Node.js**: 18+ (with npm/yarn/pnpm)
- Modern web browser (Chrome, Firefox, Safari, Edge)

**Backend Requirements** (must be running):
- **Python**: 3.8+ with FastAPI installed
- **PostgreSQL**: Database server running and configured
- **Backend API**: FastAPI server running on `http://localhost:8000`
- **Dependencies**: 
  - FastAPI, Uvicorn (ASGI server)
  - SQLAlchemy (ORM)
  - psycopg2 (PostgreSQL adapter)
  - FAISS for vector similarity search
  - LangChain for RAG pipeline
  - PyPDF2 or similar for PDF processing
  - python-jose for JWT tokens

### Backend Setup (Required First)

1. **Set up PostgreSQL database**
   ```bash
   # Create database for the application
   createdb rag_chat_db
   ```

2. **Configure backend environment**
   ```bash
   # Create .env file with database credentials
   DATABASE_URL=postgresql://user:password@localhost:5432/rag_chat_db
   SECRET_KEY=your-secret-key-for-jwt
   ```

3. **Start backend server**
   ```bash
   # Navigate to backend directory (if separate)
   cd backend
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run FastAPI server
   uvicorn main:app --reload
   ```
   
   Backend should be running at `http://localhost:8000`

### Frontend Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag/chatfrontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   
   Navigate to `http://localhost:5173` (default Vite port)

### Available Scripts

- `npm run dev` - Start development server with hot module reload
- `npm run build` - Build optimized production bundle
  - Runs TypeScript compiler check first
  - Outputs to `dist/` directory
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint for code quality checks

### Environment Setup

The application expects the backend API at:
```
http://localhost:8000
```

To change the API URL, update the fetch calls in:
- [src/service/authservice.ts](chatfrontend/src/service/authservice.ts)
- [src/service/chatservice.ts](chatfrontend/src/service/chatservice.ts)
- [src/pages/uploadchat.tsx](chatfrontend/src/pages/uploadchat.tsx)

**Recommendation**: Create a `.env` file support for API_BASE_URL to make this configurable.

### First Run

**Complete Startup Sequence**:

1. **Start PostgreSQL database** (if not already running)
   ```bash
   # Linux/Mac
   sudo service postgresql start
   
   # Windows
   # Start PostgreSQL service from Services
   ```

2. **Start backend FastAPI server**
   ```bash
   cd backend
   uvicorn main:app --reload
   ```
   Verify backend is running at `http://localhost:8000`

3. **Start frontend development server**
   ```bash
   cd chatfrontend
   npm run dev
   ```
   Frontend will be available at `http://localhost:5173`

4. **Create your first account**
   - Navigate to `http://localhost:5173/signup`
   - Register with username, email, and password
   - Login with your credentials

5. **Start chatting with PDFs!**
   - Upload a PDF document
   - Wait for processing to complete
   - Ask questions about the document content
   - Get AI-powered answers using RAG

**System Architecture Flow**:
```
User → Frontend (React) → Backend (FastAPI) → PostgreSQL (data)
                                            → FAISS (vectors)
                                            → LLM (AI responses)
```

## 🛠️ Development

### Project Setup

The project uses:
- **Vite** for development server and building
- **TypeScript** for type safety
- **ESLint** for code linting
- **Tailwind CSS** for styling

### Code Organization

#### Services Layer (`src/service/`)
- Handles all API communication
- Centralizes fetch logic
- Returns typed responses

#### Context Layer (`src/context/`)
- Global state management
- Persistent data handling
- Side effects coordination

#### Hooks Layer (`src/hooks/`)
- Clean context consumption
- Shared logic extraction
- Type-safe context access

#### Pages Layer (`src/pages/`)
- Route-level components
- Feature composition
- User workflows

#### Components Layer (`src/components/`)
- Reusable UI pieces
- Styled with Tailwind
- Type-safe props

### Adding New Features

1. **New API endpoint**: Add to service layer
2. **New state**: Extend relevant context
3. **New page**: Create in pages/, add route in App.tsx
4. **New component**: Add to components/ or components/ui/
5. **New types**: Define in type/types.ts

## 🧪 Testing & Quality

### Current Status
- No automated tests implemented yet
- Manual testing for all features
- TypeScript provides compile-time type safety
- ESLint for code quality

### Recommended Testing Stack
- **Unit Tests**: Vitest + React Testing Library
- **E2E Tests**: Playwright or Cypress
- **Type Checking**: TypeScript (already in use)
- **Linting**: ESLint (already configured)

## 🔌 API Integration

The application integrates with a FastAPI backend at `http://localhost:8000` with the following endpoints:

### Authentication Endpoints
- **POST /login** - User authentication with email and password
- **POST /signup** - User registration with username, email, and password

### Chat & Document Endpoints
- **POST /upload-pdfs** - Upload PDF files and create new chat session
- **POST /chat** - Send messages and receive AI-generated responses
- **GET /getchat** - Retrieve all user's chat sessions
- **GET /getchatconversation?chatid={id}** - Get full conversation history for specific chat
- **DELETE /deletechat?chatid={id}** - Delete a chat conversation permanently

### Expected API Response Formats

**Login Response:**
```typescript
{
  Successful: boolean,
  message: string,
  User: {
    user_id: number,
    user_name: string,
    email: string,
    token: string
  }
}
```

**Signup Response:**
```typescript
{
  Successful: boolean,
  message: string,
  User: {
    user_id: number,
    user_name: string,
    email: string
  }
}
```

**Upload PDF Response:**
```typescript
{
  chat_id: number,
  chat_name: string,
  message?: string
}
```

**Chat Request Format:**
```typescript
{
  chat_id: number,
  question: string,
  chat_history: Array<{
    role: string,
    content: string
  }>
}
```

**Chat Response:**
```typescript
{
  response: string,
  Successful: boolean
}
```

**Get Chats Response:**
```typescript
{
  chats: Array<{
    chat_id: number,
    chat_name: string
  }>,
  Successful: boolean
}
```

**Get Conversation Response:**
```typescript
{
  messages: Array<{
    role: string,      // "user" or "assistant"
    content: string
  }>,
  Successful: boolean
}
```

**Delete Chat Response:**
```typescript
{
  Successful: boolean,
  message: string
}
```

### API Authentication

All protected endpoints require JWT token in Authorization header:
```
Authorization: Bearer <token>
```

## �️ Application Architecture

### Context Providers Hierarchy
```tsx
<StrictMode>
  <AuthContextProvider>        {/* Authentication state */}
    <BrowserRouter>             {/* Routing */}
      <ChatContextProvider>     {/* Chat state */}
        <App />                 {/* Routes & pages */}
      </ChatContextProvider>
    </BrowserRouter>
  </AuthContextProvider>
</StrictMode>
```

### Data Flow

**Authentication Flow:**
```
LoginPage → login() → authservice.ts → Backend API
                ↓
         AuthContext (token, user)
                ↓
         localStorage (persistence)
                ↓
         ProtectedRoute → PDFChatInterface
```

**Chat Flow:**
```
PDFChatInterface → upload PDF → chatservice.ts → Backend API
                                       ↓
                                  chat_id returned
                                       ↓
                                 ChatContext updated
                                       ↓
                            User sends message
                                       ↓
                            chatreq() → Backend RAG
                                       ↓
                            AI response displayed
```

**Conversation Loading:**
```
Sidebar Chat Click → getChatConversation() → Backend DB
                            ↓
                    message[] returned
                            ↓
                    setChatHistory()
                            ↓
                Display full conversation
```

**Delete Conversation:**
```
Delete Button Click → Confirmation Dialog
                            ↓
                      User confirms
                            ↓
              deleteChat() → Backend API
                            ↓
                    Database deletion
                            ↓
            ChatContext state updated
                            ↓
      UI refreshed (chat removed from sidebar)
                            ↓
    If active chat deleted → reset to empty state
```

### Type Safety

All API requests and responses are fully typed with TypeScript interfaces defined in [src/type/types.ts](chatfrontend/src/type/types.ts):

- `loginForm`, `signupForm` - Input forms
- `loggedUser`, `registereduser` - User entities
- `LoginResponse`, `SignupResponse` - Auth responses
- `chat`, `message` - Chat entities
- `chatRequestFormat`, `chatResponseFormat` - Chat API
- `getAllChatResponse`, `conversationResponse` - Conversation API
- `deletechatResponse` - Delete conversation response
- `AuthContextType`, `chatcontextType` - Context types

## 🎨 Customization

### Styling & Theming

**Global Styles**: [src/index.css](chatfrontend/src/index.css)
- Tailwind CSS base, components, utilities
- CSS variables for theming
- Global CSS animations and transitions

**Component Styles**: [src/App.css](chatfrontend/src/App.css)
- Application-specific styles
- Custom gradient backgrounds
- Chat interface styling

**Inline Styling**: Components use Tailwind utility classes
```tsx
<div className="flex items-center gap-2 p-4 bg-gray-900/50">
```

### Theme System

**Theme Provider**: [src/components/theme-provider.tsx](chatfrontend/src/components/theme-provider.tsx)
- React Context for theme state
- System, light, dark theme options
- localStorage persistence

**Theme Toggle**: [src/components/theme-toggle.tsx](chatfrontend/src/components/theme-toggle.tsx)
- UI component to switch themes
- Integrates with theme provider

### Tailwind Configuration

**Config Location**: [vite.config.ts](chatfrontend/vite.config.ts)
```typescript
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()]
})
```

### UI Components

Components follow the shadcn/ui pattern in [src/components/ui/](chatfrontend/src/components/ui/):
- Built on Radix UI primitives
- Customizable with className prop
- Type-safe with TypeScript
- Accessible by default

**Example Customization:**
```tsx
import { Button } from "@/components/ui/button"

<Button 
  variant="default" 
  size="lg"
  className="bg-gradient-to-r from-blue-500 to-purple-600"
>
  Custom Button
</Button>
```

## 🔐 Authentication Flow

1. **Registration**: User signs up with username, email, and password
   - Data sent to `/signup` endpoint
   - Returns user information (without token)
   - User redirected to login page

2. **Login**: User authenticates with email and password
   - Credentials sent to `/login` endpoint
   - Backend validates and returns JWT token
   - Token and user data stored in localStorage
   - AuthContext updated with user state
   - User redirected to main chat interface

3. **Session Persistence**: On app reload
   - App checks localStorage for existing token
   - If found, user is auto-logged in
   - AuthContext restored with saved user data
   - ChatContext automatically fetches user's chat list

4. **Protected Routes**: ProtectedRoute component
   - Checks authentication status from AuthContext
   - Redirects to login if not authenticated
   - Allows access to chat interface if authenticated

5. **Logout**: User ends session
   - Token removed from localStorage
   - User data cleared from localStorage
   - All context state reset
   - User redirected to login page

## 💾 Data Persistence & Storage

### Client-Side (Browser - localStorage)
- **JWT Token**: Persistent authentication token for sessions
- **User Data**: User ID, username, email
- **Auto-restore**: Automatic session recovery on page reload

### Server-Side Storage

**PostgreSQL Database**:
- **User Accounts**: Username, email, hashed passwords
- **Chat Sessions**: All conversation entries with unique chat IDs
- **Messages**: Complete conversation history (user questions + AI responses)
- **PDF Metadata**: Uploaded file names, upload timestamps, chat associations

**FAISS Vector Store** (File System):
- **Document Embeddings**: Vector representations of PDF content chunks
- **Similarity Indexes**: Optimized indexes for fast semantic search
- **Chunk Mappings**: Links between vectors and original text chunks
- **Per-Chat Indexes**: Separate FAISS index for each PDF/chat session

**File Storage**:
- **Uploaded PDFs**: Original PDF files stored on server file system
- **Organized Structure**: Files organized by user and chat ID

### State Management
- **AuthContext**: Manages authentication state, persisted via localStorage
- **ChatContext**: Manages chat sessions, fetched from backend on login
- **Component State**: Local UI state (current message, loading indicators, etc.)

### Data Flow
```
User uploads PDF → Database entry created → PDF saved to disk → 
FAISS index generated → User chats → FAISS retrieval → 
Answer generated → Saved to database → Displayed in UI
```

## 📱 Features & Functionality

### Main Chat Interface (uploadchat.tsx)

**PDF Management**
- Upload PDF files with file type validation
- Visual feedback during upload process
- Automatic chat session creation on successful upload
- Display uploaded file name
- Clear file selection after upload

**Chat Functionality**
- Send questions about uploaded PDFs
- Receive AI-generated responses using RAG
- Maintain conversation context with chat history
- Loading indicators during AI response generation
- Error handling with user-friendly messages
- Auto-scroll to latest messages

**Conversation Management**
- View all user's chat sessions in collapsible sidebar
- Switch between different chat conversations
- Load previous conversation history
- Delete conversations with confirmation dialog
- Automatic UI refresh after deletion
- Display chat names for easy identification
- Real-time sidebar updates when new chats are created
- Hover-activated delete buttons for each chat

**User Interface Elements**
- Collapsible sidebar (toggle open/close)
- Message display with role-based styling
- Input field with send button
- Logout functionality
- Responsive layout for all screen sizes
- Loading states for conversations and messages

### Authentication Pages

**Login Page (loginpage.tsx)**
- Email and password input fields
- Form validation and error display
- Secure JWT token-based authentication
- Redirect to main chat on successful login
- Link to signup page for new users

**Signup Page (signuppage.tsx)**
- Username, email, and password registration
- Form validation
- Success/error message display
- Redirect to login after successful registration

### Route Protection

**Protected Route (protectedroute.tsx)**
- Checks authentication status before rendering
- Redirects unauthenticated users to login
- Wraps main chat interface
- Integrates with AuthContext for auth state

### Context-Based Architecture

**AuthContext**
- Global authentication state management
- Login/signup/logout functionality
- Token and user data persistence
- Loading state during authentication
- Automatic session restoration

**ChatContext**
- Global chat state management
- Current chat ID tracking
- User's chat list management
- Fetch all chats on login
- Load specific conversation history
- Delete conversations with automatic state updates
- Reset active chat when deleted
- Conversation loading states

## 🔧 Troubleshooting

### Common Issues

**Backend Not Running / Connection Refused**
- Verify backend is running: `curl http://localhost:8000`
- Check backend logs for startup errors
- Ensure PostgreSQL database is running and accessible
- Verify database connection string in backend .env file

**"Failed to fetch" or Network Errors**
- Ensure backend is running on `http://localhost:8000`
- Check browser console for CORS errors
- Verify backend CORS configuration allows frontend origin (`http://localhost:5173`)
- Check if firewall is blocking connections

**Database Connection Errors**
- Verify PostgreSQL is running: `pg_isready`
- Check database credentials in backend configuration
- Ensure database exists: `psql -l | grep rag_chat_db`
- Check PostgreSQL logs for connection issues

**Not Logged In / Redirected to Login**
- Check if token exists in localStorage (F12 → Application → Local Storage)
- Token may have expired - try logging in again
- Clear localStorage and login fresh if issues persist
- Verify backend JWT secret key is consistent

**Chat History Not Loading**
- Check network tab for failed API requests
- Verify token is being sent in Authorization header
- Check backend logs for database query errors
- Ensure chat records exist in database

**PDF Upload Fails**
- Check file is valid PDF format
- Verify backend upload endpoint is working
- Check file size limits on backend
- Look for errors in browser console
- Check backend logs for PDF processing errors
- Verify FAISS index directory is writable

**PDF Processing Takes Too Long**
- Large PDFs may take time to process
- Check backend logs for progress indicators
- Verify LLM service is responding
- Consider implementing progress indicators

**AI Responses Not Working**
- Check if LLM service is configured correctly
- Verify FAISS index was created successfully
- Check backend logs for RAG pipeline errors
- Ensure embeddings model is loaded

**Delete Chat Not Working**
- Ensure backend has DELETE endpoint configured
- Check if confirmation dialog is appearing
- Verify token authorization in DELETE request
- Check browser console for error messages
- Confirm chat is removed from database on backend

**UI Issues / Styling Problems**
- Clear browser cache
- Hard refresh (Ctrl+Shift+R / Cmd+Shift+R)
- Check if Tailwind CSS is properly loaded
- Verify dev server is running without errors

### Debug Mode

**Frontend Debugging**:
- Open browser DevTools (F12)
- Check Console tab for errors
- Check Network tab for failed requests
- React DevTools extension helpful for state inspection

**Backend Debugging**:
```bash
# Run backend with debug logging
uvicorn main:app --reload --log-level debug

# Check database
psql -d rag_chat_db -c "SELECT * FROM users;"
psql -d rag_chat_db -c "SELECT * FROM chats;"
```

### Reset Application State

**Frontend Reset**:
```javascript
// In browser console
localStorage.clear()
location.reload()
```

**Backend Reset** (⚠️ This will delete all data):
```bash
# Drop and recreate database
dropdb rag_chat_db
createdb rag_chat_db

# Run database migrations
python -m alembic upgrade head
```

## 📦 Build & Deployment

### Production Build

```bash
# Build the application
npm run build

# Preview production build
npm run preview
```

Build output location: `chatfrontend/dist/`

### Deployment Checklist

1. Update API URLs for production environment
2. Configure environment variables
3. Build application: `npm run build`
4. Test production build locally: `npm run preview`
5. Deploy `dist/` folder to hosting service
6. Configure CORS on backend for production domain
7. Set up HTTPS for secure token transmission

### Deployment Platforms

Compatible with:
- **Vercel**: Zero-config React deployment
- **Netlify**: Simple drag-and-drop deployment
- **GitHub Pages**: Free static hosting
- **AWS S3 + CloudFront**: Scalable hosting
- **nginx**: Traditional web server deployment

### Environment Variables (Recommended)

Create `.env` file support:
```bash
VITE_API_BASE_URL=http://localhost:8000
```

Update service files to use:
```typescript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
```

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Test thoroughly
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Standards

- Follow existing code style
- Use TypeScript for type safety
- Write meaningful commit messages
- Add comments for complex logic
- Ensure ESLint passes: `npm run lint`
- Test all changes before committing

### Pull Request Guidelines

- Describe what changes you made and why
- Reference any related issues
- Include screenshots for UI changes
- Ensure build succeeds: `npm run build`
- Keep PRs focused on single feature/fix

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🐛 Current State & Future Enhancements

### ✅ Implemented Features
- Complete authentication system (login/signup/logout)
- JWT token-based session management
- PDF upload and processing
- AI-powered chat with RAG integration
- Multiple conversation management
- Conversation history loading
- **Delete conversations** with confirmation dialog
- Automatic state updates after deletion
- Persistent authentication state
- Protected routing
- Responsive UI with modern design
- Hover-based context actions
- Error handling and loading states

### 🔄 Planned Enhancements

**High Priority** (From nextjob.txt):
- **Multiple PDF Upload**: Support uploading multiple PDFs in a single chat session
- **Improved LLM Output**: Enhanced response formatting and quality
- **Auth Service Refinements**: Additional authentication features and security improvements

**Medium Priority**:
- **Message Features**: Copy to clipboard, message timestamps, edit messages
- **Conversation Management**: Rename chats, archive conversations, conversation search
- **Export Functionality**: Export chat history as PDF/TXT
- **Search**: Search through conversations and messages
- **File Management**: View uploaded PDFs, PDF preview within chat interface
- **Rich Text**: Support for formatted responses (markdown, code blocks)

**Future Considerations**:
- **Real-time Updates**: WebSocket support for live updates
- **Accessibility**: Enhanced ARIA labels and keyboard navigation
- **Testing**: Unit tests, integration tests, E2E tests
- **Performance**: Implement virtual scrolling for long conversations
- **User Profile**: Profile page, settings, password change, avatar upload
- **Theme Customization**: Complete light mode implementation
- **Analytics**: Track usage patterns and user interactions
- **Undo/Redo**: Undo delete operations with toast notifications
- **Bulk Operations**: Select and delete multiple conversations

## � Project Status

**Current Version**: 0.0.0 (Active Development - February 2026)

**Status**: ✅ Core features implemented and fully functional
- ✅ Authentication system complete with JWT tokens
- ✅ PDF upload and RAG-powered chat working
- ✅ Conversation management operational (create, load, delete)
- ✅ UI responsive and polished with modern design
- ✅ Error handling and user feedback implemented
- ✅ PostgreSQL database configured and operational
- ✅ FAISS vector indexing for document retrieval
- ✅ Protected routes and session persistence

**Backend Stack** (Integrated):
- FastAPI for REST API
- PostgreSQL for data persistence
- FAISS for vector similarity search
- LangChain/LLM integration for RAG responses
- JWT authentication middleware

**Recent Updates** (February 2026):
- ✅ Implemented delete chat functionality with confirmation dialogs
- ✅ Added automatic state synchronization on deletion
- ✅ Fixed error messages in chat service
- ✅ Enhanced user experience with hover-based actions
- ✅ Complete authentication system with JWT tokens
- ✅ Full PDF upload and RAG-powered chat integration
- ✅ Multi-conversation management with history loading

**Active Development**: Yes
- Frontend fully functional with all core features
- Backend RAG integration operational
- PostgreSQL database configured
- See `nextjob.txt` for upcoming enhancements

## 📞 Support & Contact

### Getting Help

- **Issues**: Open an issue in the repository for bugs or feature requests
- **Questions**: Use discussion board for general questions
- **Documentation**: This README and inline code comments

### Useful Resources

- [React Documentation](https://react.dev/)
- [TypeScript Documentation](https://www.typescriptlang.org/)
- [Vite Documentation](https://vite.dev/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)
- [Radix UI Documentation](https://www.radix-ui.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [FAISS Documentation](https://faiss.ai/)
- [LangChain Documentation](https://python.langchain.com/)

## 🧠 RAG System Architecture

### How the RAG Pipeline Works

This application implements a **Retrieval-Augmented Generation (RAG)** system that allows users to have intelligent conversations with their PDF documents.

#### 1. Document Ingestion Phase (PDF Upload)

When a user uploads a PDF:
```
PDF File → Text Extraction → Text Chunking → Embeddings → FAISS Vector Store
```

- **Text Extraction**: PDF content is extracted as plain text
- **Chunking**: Text is split into manageable chunks (with overlap for context)
- **Embedding Generation**: Each chunk is converted to a vector embedding
- **FAISS Indexing**: Vectors are stored in FAISS for efficient similarity search
- **Database Storage**: Chat session created in PostgreSQL with reference to FAISS index

#### 2. Query Phase (User Asks Question)

When a user asks a question:
```
Question → Embedding → FAISS Search → Context Retrieval → LLM Prompt → AI Answer
```

- **Question Embedding**: User's question is converted to vector
- **Semantic Search**: FAISS finds most similar document chunks
- **Context Assembly**: Retrieved chunks combined with chat history
- **LLM Generation**: Language model generates answer using context
- **Response Delivery**: Answer sent to frontend and saved to database

#### 3. Key Benefits of RAG

✅ **Accurate**: Answers grounded in actual document content  
✅ **Contextual**: Maintains conversation flow across questions  
✅ **Efficient**: FAISS enables fast similarity search  
✅ **Scalable**: Can handle multiple documents per user  
✅ **Transparent**: Responses based on retrievable source content  

### Technology Stack Summary

**Frontend**: React 19 + TypeScript + Tailwind CSS + Vite  
**Backend**: FastAPI + Python  
**Database**: PostgreSQL (user data, chat sessions, messages)  
**Vector Store**: FAISS (document embeddings, similarity search)  
**RAG Framework**: LangChain + LLM integration  
**Authentication**: JWT tokens  

## 🙏 Acknowledgments

**Frontend**:
- **React** team for the amazing framework
- **Vite** for lightning-fast development experience
- **Tailwind CSS** for utility-first styling
- **Radix UI** for accessible component primitives
- **shadcn/ui** for component patterns and inspiration

**Backend & AI**:
- **FastAPI** for modern Python web framework
- **PostgreSQL** for reliable data persistence
- **FAISS** (Facebook AI) for efficient vector similarity search
- **LangChain** for RAG pipeline orchestration
- **OpenAI/LLM providers** for natural language generation

---

**Built with ❤️ using React, TypeScript, FastAPI, and AI**
