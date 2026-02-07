import { useAuthContext } from "@/hooks/useauth";
import { Navigate } from "react-router";
import type { ReactNode } from "react";

interface childrentype{
    children:ReactNode
}
export const ProtectedRoute = ({children}:childrentype)=>{
    const {loggedIn} = useAuthContext();
    
    if(!loggedIn){
        return <Navigate to="/login" replace />
    }
    
    return <>{children}</>
}