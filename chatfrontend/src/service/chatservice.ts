
import type { getAllChatResponse,conversationResponse,chatRequestFormat,chatResponseFormat,deletechatResponse } from "@/type/types";
const getAllChats = async(token:string):Promise<getAllChatResponse>=>{
    try{
        const authHeader = "Bearer "+token
        const response = await fetch("http://localhost:8000/getchat",{
            method:"GET",
            headers:{
                "content-type":"application/json",
                "Authorization":authHeader
            }
        });
       if(!(response.ok)){
            throw new Error("failed to fetch chats")
       }
       const chats = await response.json()
       return chats
    }
    catch(e){
        console.error("Failed to get chats", e);
        throw e;
        
    }
    
}
const getChatConversation = async(token:string,chatid:number):Promise<conversationResponse>=>{
    try{
        const authHeader = "Bearer "+token
        const response = await fetch(`http://localhost:8000/getchatconversation?chatid=${chatid}`,{
            method:"GET",
            headers:{
                "content-type":"application/json",
                "Authorization":authHeader
            }
        });
       if(!(response.ok)){
            throw new Error("failed to fetch chat conversation")
       }
       const data =await response.json();
        return data
    }
    catch(e){
            console.error("failed to fetch chat conversation")
            throw e
    }
    
}
const chatreq = async(req:chatRequestFormat,token:string):Promise<chatResponseFormat>=>{
        try{
            const authHeader = "Bearer "+token
            const response = await fetch(`http://localhost:8000/chat`,{
                method:"POST",
                headers:{
                    "content-type":"application/json",
                    "Authorization":authHeader
                },
                body:JSON.stringify(req)
            });
            if(!(response.ok)){
                    throw new Error("failed to fetch chat response ")
            }
            const data = await response.json()
            return data
        }
        catch(e){
                console.error("failed to get chat response")
                throw e
        }
}

const deletechat = async(token:string,chatid:number):Promise<deletechatResponse>=>{
        try{
                const authHeader = "Bearer "+token
                const response = await fetch(`http://localhost:8000/deletechat?chatid=${chatid}`,{
                    method:"DELETE",
                    headers:{
                        "content-type":"application/json",
                        "Authorization":authHeader
                    }
                });
            if(!(response.ok)){
                    throw new Error("failed to delete conversation")
            }
            const data =await response.json();
                return data
        }
        catch(e){
                console.error("failed to delete chat")
                throw e
        }
}
export {getAllChats,getChatConversation,chatreq,deletechat}