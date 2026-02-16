
import type { ReactNode } from "react"
interface loginForm {
    email:string
    password:string
}
interface signupForm{
    user_name:string
    email:string
    password:string
}

interface registereduser{
    user_id:number
    user_name:string
    email:string
}
interface loggedUser extends registereduser{
    token:string
}

interface ApiResponse {
    User: loggedUser | registereduser
    Successful: boolean
    message: string
}

interface LoginResponse {
    User: loggedUser
    Successful: boolean
    message: string
}

interface SignupResponse {
    User: registereduser
    Successful: boolean
    message: string
}
interface callResponse{
        Successful:boolean,
        msg:string
}
interface AuthContextType {
    token:string
    setToken:(token:string)=>void
    loading:boolean
    setLoading:(load:boolean)=>void
    user:loggedUser
    setUser:(user:loggedUser)=>void
    loggedIn:boolean,
    setLoggedIn:(logged:boolean)=>void
    login:(userdata:loginForm)=>Promise<callResponse>
    signup:(userData:signupForm)=>Promise<callResponse>
    logout:()=>void
}

interface AuthProviderProps{
    children:ReactNode
}
interface chat{
    chat_id:number
    chat_name:string
}
interface getAllChatResponse{
    chats :chat[]
    Successful:boolean
}
interface message{
    role:string
    content:string
}

interface structuredChatResponse{
    answer: string
    key_points: string[]
    confidence_level: string
    sources_cited: string[]
    needs_clarification: boolean
    clarification_needed: string | null
    follow_up_suggestions: string[]
}

interface conversationResponse{
    messages:message[]
    Successful:boolean
}

interface chatRequestFormat{
       chat_id :number
       question:string
       chat_history:message[]

}
interface chatResponseFormat{
        success: boolean
        chat_id: number
        response: string  // JSON stringified structured response
        role: string
        timestamp: string
        sources_used: number
        error_message: string | null
}
interface deletechatResponse{
            Successful:boolean
            message:string
}
export type {loginForm,
            signupForm,
            loggedUser,
            registereduser,
            ApiResponse,
            LoginResponse,
            SignupResponse,
            callResponse,
            AuthContextType,
            AuthProviderProps,
            chat,
            getAllChatResponse,
            message,
            structuredChatResponse,
            conversationResponse,
            chatRequestFormat,
            chatResponseFormat,
            deletechatResponse}