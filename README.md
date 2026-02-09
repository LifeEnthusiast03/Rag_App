# PDF Chat Frontend

A modern, full-featured React application for chatting with PDF documents using AI. Upload your PDFs and have intelligent conversations about their content.

## 🚀 Features

### Authentication System
- **User Registration & Login**: Secure authentication with JWT tokens
- **Protected Routes**: Authenticated access to main application features
- **Persistent Sessions**: Auto-login with localStorage token management
- **Context-based State**: Global authentication state using React Context API

### PDF Chat Interface
- **PDF Upload**: Upload PDF files (up to 50MB) for analysis
- **Interactive Chat**: Ask questions and get AI-powered answers about your documents
- **Chat History**: Automatic saving of conversations to localStorage
- **Message Management**: View, copy, and manage chat messages
- **Real-time Updates**: Live chat interface with message streaming
- **Multi-conversation Support**: Switch between multiple PDF chat sessions

### User Interface
- **Modern Dark Theme**: Sleek, gradient-based dark UI with backdrop blur effects
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Sidebar Navigation**: Collapsible sidebar with conversation history
- **Theme Toggle**: Light/dark mode support
- **Smooth Animations**: Polished transitions and hover effects
- **File Management**: Visual feedback for uploads and file handling

## 🛠️ Tech Stack

- **Framework**: React 19.2 + TypeScript 5.9
- **Build Tool**: Vite 7.2
- **Styling**: Tailwind CSS 4.1 with custom animations
- **Routing**: React Router 7.13
- **UI Components**: Custom components built with Radix UI primitives
- **State Management**: React Context API
- **Icons**: Lucide React
- **Date Handling**: date-fns
- **HTTP Client**: Native Fetch API

## 📁 Project Structure

```
chatfrontend/
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── ui/           # shadcn/ui style components
│   │   │   ├── button.tsx
│   │   │   ├── input.tsx
│   │   │   ├── label.tsx
│   │   │   ├── calendar.tsx
│   │   │   ├── field.tsx
│   │   │   └── separator.tsx
│   │   ├── login-form.tsx
│   │   ├── theme-provider.tsx
│   │   └── theme-toggle.tsx
│   ├── context/          # React Context providers
│   │   └── authcontext.tsx
│   ├── hooks/            # Custom React hooks
│   │   └── useauth.ts
│   ├── pages/            # Page components
│   │   ├── loginpage.tsx
│   │   ├── signuppage.tsx
│   │   └── uploadchat.tsx
│   ├── routes/           # Route configuration
│   │   └── protectedroute.tsx
│   ├── service/          # API service layer
│   │   └── authservice.ts
│   ├── type/             # TypeScript type definitions
│   │   └── types.ts
│   ├── lib/              # Utility functions
│   │   └── utils.ts
│   ├── App.tsx           # Main application component
│   └── main.tsx          # Application entry point
├── public/               # Static assets
├── components.json       # shadcn/ui configuration
├── vite.config.ts        # Vite configuration
├── tsconfig.json         # TypeScript configuration
└── package.json          # Project dependencies
```

## 🚦 Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- Backend API running on `http://localhost:8000`

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chatfrontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   
   Navigate to `http://localhost:5173` (default Vite port)

### Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint for code quality checks

## 🔌 API Integration

The application integrates with a backend API at `http://localhost:8000` with the following endpoints:

- `POST /login` - User authentication
- `POST /signup` - User registration
- `POST /upload-pdfs` - Upload PDF files
- `POST /chat` - Send chat messages and receive AI responses

### Expected API Response Format

**Login/Signup Response:**
```typescript
{
  Successful: boolean,
  message: string,
  User?: {
    user_id: number,
    user_name: string,
    email: string,
    token: string
  }
}
```

**Upload Response:**
```typescript
{
  chat_id: string,
  message: string
}
```

**Chat Response:**
```typescript
{
  answer: string,
  response: string
}
```

## 🎨 Customization

### Styling

The project uses Tailwind CSS 4 with custom configurations. Modify styles in:
- `src/index.css` - Global styles and Tailwind directives
- `src/App.css` - Application-specific styles
- Component files - Inline Tailwind classes

### Theme

Theme configuration is managed through:
- `src/components/theme-provider.tsx` - Theme context provider
- `src/components/theme-toggle.tsx` - Theme switch component

### Components

UI components follow the shadcn/ui pattern with custom styling and can be modified in `src/components/ui/`.

## 🔐 Authentication Flow

1. User signs up with username, email, and password
2. User logs in with email and password
3. JWT token is stored in localStorage
4. Token is included in subsequent API requests
5. Protected routes check for valid token
6. Logout clears token and redirects to login

## 💾 Data Persistence

- **Authentication**: JWT tokens and user data stored in localStorage
- **Chat History**: Conversations saved locally with localStorage
- **Session Management**: Automatic session restoration on page reload

## 📱 Features & Functionality

### Chat Interface

- Upload PDF documents for analysis
- Real-time question and answer interface
- Message history with timestamps
- Copy messages to clipboard
- Delete individual conversations
- Load previous conversations
- Visual indicators for message status
- Error handling with user feedback

### User Experience

- Smooth animations and transitions
- Loading states for async operations
- Error messages and validation
- Responsive design for all screen sizes
- Keyboard shortcuts (Enter to send)
- Auto-scroll to latest messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is private and proprietary.

## 🐛 Known Issues & Future Enhancements

- Add file type validation for PDF uploads
- Implement real-time collaboration features
- Add export chat history functionality
- Enhance error handling and retry logic
- Add comprehensive test coverage
- Implement rate limiting on frontend

## 📞 Support

For issues and questions, please open an issue in the repository or contact the development team.
