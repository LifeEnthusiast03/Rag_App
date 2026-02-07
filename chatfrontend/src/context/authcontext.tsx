import { createContext, useState, useEffect} from "react";
import { loginReq,signupReq } from "@/service/authservice";
import type { loggedUser,loginForm,signupForm,callResponse,AuthContextType,AuthProviderProps } from "@/type/types";

export const AuthContext = createContext<AuthContextType|undefined>(undefined)

export const AuthContextProvider=({children}:AuthProviderProps)=>{
            const userNone:loggedUser = {
                    user_id:-1,
                    user_name:"",
                    email:"",
                    token:"",
            }
            
            // Initialize from localStorage
            const [token,setToken]= useState<string>(() => {
                return localStorage.getItem("token") || "";
            });
            const [loading,setLoading]=useState(false)
            const [user,setUser] = useState<loggedUser>(() => {
                const savedUser = localStorage.getItem("user");
                return savedUser ? JSON.parse(savedUser) : userNone;
            });
            const [loggedIn,setLoggedIn] =useState<boolean>(() => {
                return !!localStorage.getItem("token");
            });
            
            // Persist token to localStorage
            useEffect(() => {
                if (token) {
                    localStorage.setItem("token", token);
                } else {
                    localStorage.removeItem("token");
                }
            }, [token]);
            
            // Persist user to localStorage
            useEffect(() => {
                if (user.user_id !== -1) {
                    localStorage.setItem("user", JSON.stringify(user));
                } else {
                    localStorage.removeItem("user");
                }
            }, [user]);
            
            const login = async(userData:loginForm):Promise<callResponse>=>{
                 try{
                    setLoading(true)
                    const loginreq =  await loginReq(userData);
                    if(!loginreq.Successful){
                        throw new Error(loginreq.message || "login failed");
                    }
                    const curUser = loginreq.User;
                    setUser(curUser)
                    setToken(curUser.token)
                    setLoggedIn(true)
                    const res:callResponse ={
                        Successful:true,
                        msg:loginreq.message
                    }
                    return res;
                 }
                 catch(e){
                        console.log(`Login failed ${e}`);
                        const errorMsg = e instanceof Error ? e.message : "Login failed";
                        const res:callResponse ={
                        Successful:false,
                        msg:errorMsg
                    }
                    return res
                 }
                 finally{
                    setLoading(false)
                 }
            }
            const signup = async(userData:signupForm):Promise<callResponse>=>{
                    try{
                        setLoading(true)
                        const signupreq = await signupReq(userData)
                        if(!signupreq.Successful){
                            throw new Error(signupreq.message || "User creation failed")
                        }
                        const res:callResponse ={
                        Successful:true,
                        msg:signupreq.message
                    }
                    return res
                    }
                    catch(e){
                        console.log(`signup failed ${e}`);
                        const errorMsg = e instanceof Error ? e.message : "Signup Failed";
                        const res:callResponse ={
                        Successful:false,
                        msg:errorMsg
                    }
                    return res
                }
                finally{
                    setLoading(false)
                }
        }
        
        const logout = () => {
            setToken("");
            setUser(userNone);
            setLoggedIn(false);
            localStorage.removeItem("token");
            localStorage.removeItem("user");
        }
        
        return (
           < AuthContext.Provider value={{token,setToken,loading,setLoading,user,setUser,loggedIn,setLoggedIn,login,signup,logout}}>
                {children}
           </AuthContext.Provider>
        )
}
