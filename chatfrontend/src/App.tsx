

import PDFChatInterface from "./pages/uploadchat"
import LoginPage from "./pages/loginpage"
import SignUpPage from "./pages/signuppage"
import { Routes,Route } from "react-router"
import { ProtectedRoute } from "./routes/protectedroute"
function App() {
  

  return (
          <Routes>
            
            <Route path="/" element={<ProtectedRoute>
                                      <PDFChatInterface/>
                                    </ProtectedRoute>}/>
            <Route path="/login" element={<LoginPage/>}/>
            <Route path="/signup" element={<SignUpPage/>}/>
          </Routes>
          
  )
}

export default App
