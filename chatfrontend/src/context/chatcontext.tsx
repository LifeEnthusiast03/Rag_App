import { getAllChats,getChatConversation } from "@/service/chatservice";
import {createContext,useEffect, useState} from "react"
import type { chat,message } from "@/type/types";
import type { ReactNode } from "react";
import { useAuthContext } from "@/hooks/useauth";

interface chatcontextType{
    curChatId:number
    setCurChatId:(id:number)=>void
    userChats:chat[]
    setUserChats:(chats:chat[])=>void
    getChatconversation:(chatid:number)=>Promise<message[]>
    loading:boolean
}
interface chatProviderProp{
    children:ReactNode
}
export const ChatContext = createContext<chatcontextType|undefined>(undefined)
export const ChatContextProvider = ({children}:chatProviderProp)=>{
            
            const [curChatId,setCurChatId]= useState<number>(0)
            const [userChats,setUserChats]=useState<chat[]>([])
            const [loading,setLoading]=useState<boolean>(false)
            const {token, loggedIn} = useAuthContext()

            useEffect(()=>{
                const fetchChats = async () => {
                    console.log("ChatContext: useEffect triggered", {token: token ? "exists" : "none", loggedIn})
                    
                    if(!token || !loggedIn) {
                        console.log("ChatContext: Skipping fetch - no token or not logged in")
                        return;
                    }
                    
                    try{   
                        console.log("ChatContext: Fetching chats...")
                        setLoading(true)
                        const data = await getAllChats(token)
                        if(data.Successful){
                            setUserChats(data.chats)
                        }
                    }
                    catch(e){
                        console.error("ChatContext: Failed to fetch chats:", e)
                        setUserChats([])
                    }
                    finally{
                        setLoading(false)
                    }
                }
                
                fetchChats()
            }, [token, loggedIn])

            const getChatconversation = async (chatid:number):Promise<message[]> => {
                try{
                    const data = await getChatConversation(token, chatid)
                    if(data.Successful){
                        return data.messages
                    }
                    return []
                }
                catch(e){
                    console.error("Failed to fetch chat conversation:", e)
                    return []
                }
            }

            return (
                <ChatContext.Provider value={{
                    curChatId,
                    setCurChatId,
                    userChats,
                    setUserChats,
                    getChatconversation,
                    loading
                }}>
                    {children}
                </ChatContext.Provider>
            )
}