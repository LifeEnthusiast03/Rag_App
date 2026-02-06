import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import PDFUploadTest from './upload'
import AuthPage from './pages/AuthPage'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AuthPage />
    <PDFUploadTest/>
  </StrictMode>,
)
