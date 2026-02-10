import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter} from 'react-router'
import './index.css'
import App from './App.tsx'
import { AuthContextProvider } from './context/authcontext.tsx'
import { ChatContextProvider } from './context/chatcontext.tsx'
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AuthContextProvider>
      <BrowserRouter>
        <ChatContextProvider>
            <App/>
        </ChatContextProvider>
      </BrowserRouter>
    </AuthContextProvider>
  </StrictMode>,
)
