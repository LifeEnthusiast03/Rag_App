import { AuthContext } from "@/context/authcontext";
import { useContext } from "react";
export const useAuthContext = ()=>{
        const authContext= useContext(AuthContext)
        if(!authContext) throw new Error("Can't find AuthContext")
        return authContext
}