import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import PDFUploadTest from './upload'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <PDFUploadTest />
  </StrictMode>,
)
