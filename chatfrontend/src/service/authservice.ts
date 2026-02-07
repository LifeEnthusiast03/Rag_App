import type { loginForm,signupForm,LoginResponse,SignupResponse } from "@/type/types"
const loginReq = async (loginData:loginForm):Promise<LoginResponse>=>{
      try{
            const response = await fetch("http://localhost:8000/login",{
            method:"POST",
            headers:{
                "content-type":"application/json"
            },
            body:JSON.stringify({
                email:loginData.email,
                password:loginData.password
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || errorData.message || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        return data;

      }
      catch(e){
            console.error("Login request failed:", e);
            throw e;
      }
}
const signupReq = async(signupData:signupForm):Promise<SignupResponse>=>{
      try{
            const response = await fetch("http://localhost:8000/signup",{
                method:"POST",
                headers:{
                    "content-type":"application/json"
                },
                body:JSON.stringify({
                    user_name:signupData.user_name,
                    email:signupData.email,
                    password:signupData.password
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || errorData.message || `HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data;
      }
      catch(e){
            console.error("Signup request failed:", e);
            throw e;
      }
}
export {loginReq,signupReq}