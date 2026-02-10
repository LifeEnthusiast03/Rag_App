import { ChatContext } from "@/context/chatcontext";
import { useContext } from "react";
export const useChat = ()=>{
        const chatcontext = useContext(ChatContext)
        if(!chatcontext)throw new Error("No chat context found")
        return chatcontext
}