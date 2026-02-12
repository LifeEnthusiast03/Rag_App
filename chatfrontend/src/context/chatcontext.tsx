import { getAllChats,getChatConversation,deletechat } from "@/service/chatservice";
import {createContext,useEffect, useState} from "react"
import type { chat,message,deletechatResponse } from "@/type/types";
import type { ReactNode } from "react";
import { useAuthContext } from "@/hooks/useauth";

interface chatcontextType{
    curChatId:number
    setCurChatId:(id:number)=>void
    userChats:chat[]
    setUserChats:(chats:chat[])=>void
    getChatconversation:(chatid:number)=>Promise<message[]>
    deleteChat:(chatid:number)=>Promise<deletechatResponse>
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
            const deleteChat= async(chatid:number):Promise<deletechatResponse>=>{
                    try{
                        const data = await deletechat(token,chatid);
                        if(data.Successful){
                            // Update userChats state by removing the deleted chat
                            setUserChats(prevChats => prevChats.filter(chat => chat.chat_id !== chatid))
                            // Reset curChatId if the deleted chat is the current one
                            if(curChatId === chatid){
                                setCurChatId(0)
                            }
                            return data
                        }
                        return {
                            Successful: false,
                            message: data.message || "Failed to delete chat"
                        }
                    }
                    catch(e){
                            console.error("Failed to delete chat:", e)
                            return {
                                Successful:false,
                                message:"Failed to delete chat"
                            }
                    }
            }
            return (
                <ChatContext.Provider value={{
                    curChatId,
                    setCurChatId,
                    userChats,
                    setUserChats,
                    getChatconversation,
                    deleteChat,
                    loading
                }}>
                    {children}
                </ChatContext.Provider>
            )
}