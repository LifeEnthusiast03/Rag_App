
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
export type {loginForm,signupForm,loggedUser,registereduser,ApiResponse,LoginResponse,SignupResponse,callResponse,AuthContextType,AuthProviderProps}