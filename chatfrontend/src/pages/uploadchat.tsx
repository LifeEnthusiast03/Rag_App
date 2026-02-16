import { useState, useRef, useEffect } from 'react';
import { useAuthContext } from '@/hooks/useauth';
import { useChat } from '@/hooks/usechat';
import { chatreq } from '@/service/chatservice';
import type { message, structuredChatResponse } from '@/type/types';

export default function PDFChatInterface() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<message[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const { logout, token } = useAuthContext();
  const {curChatId,setCurChatId,userChats,getChatconversation,setUserChats,deleteChat} = useChat()

  // Helper function to check if response is structured
  const isStructuredResponse = (content: string): boolean => {
    try {
      const parsed = JSON.parse(content);
      return parsed && typeof parsed === 'object' && 'answer' in parsed;
    } catch {
      return false;
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    const pdfFiles = selectedFiles.filter((file: File) => file.type === 'application/pdf');
    
    if (pdfFiles.length === 0) {
      setError('Please select a valid PDF file');
      setFiles([]);
    } else {
      setFiles(pdfFiles);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const headers: HeadersInit = {};
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const res = await fetch('http://localhost:8000/upload-pdfs', {
        method: 'POST',
        headers,
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await res.json();
      
      if (data.chat_id) {
        setCurChatId(data.chat_id);
        setUploadedFileName(files[0].name);
        setChatHistory([]);
        
        // Add new chat to userChats
        const newChat = {
          chat_id: data.chat_id,
          chat_name: data.chat_name
        };
        setUserChats([newChat, ...userChats]);
        
        setFiles([]);
        const fileInput = document.getElementById('fileInput') as HTMLInputElement;
        if (fileInput) fileInput.value = '';
      } else {
        setError('No chat_id received from backend');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!currentQuestion.trim() || !curChatId) return;
    const userMessage = currentQuestion.trim();
    setCurrentQuestion('');
    setIsChatting(true);
    const newUserMessage: message = { 
      role: 'user', 
      content: userMessage
    };
    const updatedHistory = [...chatHistory, newUserMessage];
    setChatHistory(updatedHistory);

    try {
      const data = await chatreq({
        chat_id: curChatId,
        question: userMessage,
        chat_history: chatHistory
      }, token);
      
      if (data.success) {
        // The response field contains the JSON stringified structured response
        const assistantMessage: message = { 
          role: data.role || 'assistant', 
          content: data.response  // This is already a JSON string from backend
        };
        setChatHistory([...updatedHistory, assistantMessage]);
        setError(null);
      } else {
        setError(data.error_message || 'Failed to get response');
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Chat request failed';
      setError(errorMsg);
    } finally {
      setIsChatting(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const resetChat = () => {
    setCurChatId(0);
    setChatHistory([]);
    setUploadedFileName('');
    setError(null);
  };

  const loadChat = async (chat_id: number, chat_name: string) => {
    if (chat_id === curChatId) return;
    
    setLoadingConversation(true);
    setError(null);
    try {
      setCurChatId(chat_id);
      setUploadedFileName(chat_name);
      const messages = await getChatconversation(chat_id);
      setChatHistory(messages);
    } catch (err) {
      setError('Failed to load conversation');
      setChatHistory([]);
    } finally {
      setLoadingConversation(false);
    }
  };

  const handleDeleteChat = async (e: React.MouseEvent, chat_id: number) => {
    e.stopPropagation(); // Prevent triggering loadChat
    
    if (!confirm('Are you sure you want to delete this chat?')) return;
    
    try {
      const result = await deleteChat(chat_id);
      if (result.Successful) {
        // If deleted chat was the current one, reset the view
        if (curChatId === chat_id) {
          resetChat();
        }
      } else {
        setError(result.message || 'Failed to delete chat');
      }
    } catch (err) {
      setError('Failed to delete chat');
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="flex h-screen bg-black text-white overflow-hidden">
      {/* Background Effect */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-gray-800/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-gray-700/20 rounded-full blur-3xl"></div>
      </div>

      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 ease-in-out bg-gray-950/95 backdrop-blur-xl border-r border-gray-800/50 flex flex-col overflow-hidden relative z-10 shadow-2xl`}>
        {/* Sidebar Header */}
        <div className="p-4 border-b border-gray-800/50 bg-gradient-to-b from-gray-900/50 to-transparent">
          <button
            onClick={resetChat}
            className="w-full px-4 py-3.5 bg-gradient-to-r from-gray-800 to-gray-700 hover:from-gray-700 hover:to-gray-600 rounded-xl font-semibold transition-all flex items-center justify-center gap-2 shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98] group"
          >
            <svg className="w-5 h-5 group-hover:rotate-90 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>New Chat</span>
          </button>
        </div>
        
        {/* Chat History List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2 scrollbar-thin scrollbar-thumb-gray-800 scrollbar-track-transparent">
          {userChats.length === 0 ? (
            <div className="text-center text-gray-500 mt-12 px-4 animate-fade-in">
              <div className="w-16 h-16 bg-gray-800/50 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
              </div>
              <p className="text-sm font-medium">No conversations yet</p>
              <p className="text-xs mt-2 text-gray-600">Upload a PDF to start chatting</p>
            </div>
          ) : (
            userChats.map((chat) => (
              <div
                key={chat.chat_id}
                className={`group relative p-3.5 rounded-xl cursor-pointer transition-all duration-200 ${
                  curChatId === chat.chat_id
                    ? 'bg-gradient-to-r from-gray-800 to-gray-800/80 border border-gray-700 shadow-lg scale-[1.02]'
                    : 'hover:bg-gray-900/50 hover:scale-[1.01]'
                }`}
                onClick={() => loadChat(chat.chat_id, chat.chat_name)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                      <p className="text-sm font-medium truncate">{chat.chat_name}</p>
                    </div>
                  </div>
                  <button
                    onClick={(e) => handleDeleteChat(e, chat.chat_id)}
                    className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-900/50 rounded-lg transition-all flex-shrink-0 hover:scale-110 active:scale-95"
                    title="Delete chat"
                  >
                    <svg className="w-4 h-4 text-gray-400 hover:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-gray-800/50 bg-gradient-to-t from-gray-900/50 to-transparent">
          <div className="text-xs text-gray-500 text-center space-y-1">
            <div className="flex items-center justify-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Connected</span>
            </div>
            <p className="font-medium">{userChats.length} conversation{userChats.length !== 1 ? 's' : ''}</p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden relative z-10">
        {/* Top Bar */}
        <div className="h-16 border-b border-gray-800/50 flex items-center px-6 bg-gray-950/80 backdrop-blur-xl shadow-lg">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2.5 hover:bg-gray-800/50 rounded-xl transition-all mr-4 group hover:scale-105 active:scale-95"
          >
            <svg className="w-6 h-6 group-hover:text-gray-300 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          
          {curChatId !== 0 && (
            <div className="flex-1 flex items-center gap-3 animate-fade-in">
              <div className="flex items-center gap-2 px-4 py-2 bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 shadow-lg">
                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
                <span className="text-sm font-medium truncate max-w-md">{uploadedFileName}</span>
              </div>
              <button
                onClick={resetChat}
                className="px-4 py-2 bg-gray-800/50 hover:bg-gray-700/50 rounded-xl transition-all flex items-center gap-2 border border-gray-700/50 hover:scale-105 active:scale-95"
              >
                <span className="text-sm font-medium">New Chat</span>
              </button>
            </div>
          )}
          
          {/* Logout Button */}
          <button
            onClick={logout}
            className="ml-auto px-4 py-2 bg-gray-800/50 hover:bg-red-900/50 rounded-xl transition-all flex items-center gap-2 border border-gray-700/50 hover:border-red-700/50 group hover:scale-105 active:scale-95 shadow-lg"
          >
            <svg className="w-5 h-5 group-hover:text-red-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
            <span className="text-sm font-medium">Logout</span>
          </button>
        </div>
        {/* Content Area */}
        <div className="flex-1 overflow-hidden">
          {curChatId === 0 ? (
            // Upload Interface
            <div className="h-full flex items-center justify-center p-8 animate-fade-in">
              <div className="max-w-2xl w-full">
                <div className="text-center mb-12 animate-slide-up">
                  <div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-br from-gray-800 to-gray-700 rounded-3xl mb-6 shadow-2xl hover:scale-105 transition-transform duration-300">
                    <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
                    Chat with Your PDFs
                  </h1>
                  <p className="text-gray-400 text-xl">
                    Upload a document and start an intelligent conversation
                  </p>
                </div>

                <div className="bg-gray-900/40 backdrop-blur-xl border border-gray-800/50 rounded-3xl p-8 shadow-2xl hover:shadow-3xl transition-shadow duration-300">
                  <div className="border-2 border-dashed border-gray-700 rounded-2xl p-12 mb-6 hover:border-gray-500 hover:bg-gray-800/30 transition-all duration-300 cursor-pointer group relative overflow-hidden">
                    <div className="absolute inset-0 bg-gradient-to-r from-gray-800/0 via-gray-700/5 to-gray-800/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
                    <label htmlFor="fileInput" className="cursor-pointer block relative z-10">
                      <div className="text-center">
                        <div className="inline-flex items-center justify-center w-20 h-20 bg-gray-800 group-hover:bg-gray-700 rounded-2xl mb-4 transition-all duration-300 group-hover:scale-110 group-hover:rotate-3">
                          <svg className="w-10 h-10 text-gray-400 group-hover:text-gray-200 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                          </svg>
                        </div>
                        <p className="text-white font-semibold mb-2">Drop your PDF here</p>
                        <p className="text-gray-400 text-sm">or click to browse</p>
                        <p className="text-gray-600 text-xs mt-2">Supports PDF files up to 50MB</p>
                      </div>
                      <input
                        id="fileInput"
                        type="file"
                        accept=".pdf"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                    </label>
                  </div>

                  {files.length > 0 && (
                    <div className="mb-6 p-6 bg-gray-800/40 backdrop-blur-sm border border-gray-700/50 rounded-2xl shadow-xl animate-slide-up">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4 flex-1 min-w-0">
                          <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-gray-700 to-gray-600 rounded-xl flex items-center justify-center shadow-lg">
                            <svg className="w-7 h-7 text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-white font-medium truncate">{files[0].name}</p>
                            <p className="text-gray-400 text-sm">{formatBytes(files[0].size)}</p>
                          </div>
                        </div>
                        <button
                          onClick={() => {
                            setFiles([]);
                            const fileInput = document.getElementById('fileInput') as HTMLInputElement;
                            if (fileInput) fileInput.value = '';
                          }}
                          className="flex-shrink-0 p-2.5 hover:bg-red-900/40 rounded-xl transition-all hover:scale-110 active:scale-95 group"
                        >
                          <svg className="w-5 h-5 text-red-400 group-hover:text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  )}

                  <button
                    onClick={handleUpload}
                    disabled={files.length === 0 || uploading}
                    className={`w-full py-5 px-6 rounded-2xl font-bold text-white text-lg transition-all transform flex items-center justify-center gap-3 relative overflow-hidden group ${
                      files.length === 0 || uploading
                        ? 'bg-gray-800 cursor-not-allowed opacity-50'
                        : 'bg-gradient-to-r from-gray-800 via-gray-700 to-gray-600 hover:from-gray-700 hover:via-gray-600 hover:to-gray-500 hover:scale-[1.02] active:scale-[0.98] shadow-2xl hover:shadow-3xl'
                    }`}
                  >
                    {!uploading && (
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700"></div>
                    )}
                    {uploading ? (
                      <>
                        <svg className="animate-spin h-6 w-6" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Uploading...</span>
                      </>
                    ) : (
                      <>
                        <svg className="w-6 h-6 group-hover:scale-110 group-hover:rotate-12 transition-transform relative z-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        <span className="relative z-10">Upload & Start Chatting</span>
                      </>
                    )}
                  </button>

                  {error && (
                    <div className="mt-6 p-5 bg-red-900/30 backdrop-blur-sm border border-red-700/50 rounded-2xl flex items-start gap-3 shadow-xl animate-shake">
                      <svg className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <p className="text-red-300 text-sm flex-1">{error}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            // Chat Interface
            <div className="h-full flex flex-col">
              {/* Messages Container */}
              <div className="flex-1 overflow-y-auto px-4 py-6 scrollbar-thin scrollbar-thumb-gray-800 scrollbar-track-transparent">
                <div className="max-w-4xl mx-auto space-y-6">
                  {loadingConversation ? (
                    <div className="text-center mt-20 animate-fade-in">
                      <div className="inline-flex items-center justify-center w-20 h-20 bg-gray-800/50 backdrop-blur-sm rounded-3xl mb-6 shadow-xl">
                        <svg className="animate-spin h-10 w-10 text-gray-600" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      </div>
                      <h3 className="text-2xl font-semibold mb-2">Loading conversation...</h3>
                    </div>
                  ) : chatHistory.length === 0 ? (
                    <div className="text-center mt-20 animate-fade-in">
                      <div className="inline-flex items-center justify-center w-20 h-20 bg-gray-800/50 backdrop-blur-sm rounded-3xl mb-6 shadow-xl">
                        <svg className="w-10 h-10 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                      </div>
                      <h3 className="text-2xl font-semibold mb-2">Start a Conversation</h3>
                      <p className="text-gray-400 text-lg">Ask me anything about your PDF</p>
                    </div>
                  ) : null}

                  {!loadingConversation && chatHistory.map((message, idx) => (
                    <div
                      key={idx}
                      className={`flex gap-4 animate-slide-up ${
                        message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                      }`}
                    >
                      <div className={`flex-shrink-0 w-11 h-11 rounded-2xl flex items-center justify-center shadow-lg ${
                        message.role === 'user'
                          ? 'bg-gradient-to-br from-gray-700 to-gray-600 ring-2 ring-gray-600/20'
                          : 'bg-gray-800 border border-gray-700 ring-2 ring-gray-700/20'
                      }`}>
                        {message.role === 'user' ? (
                          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                          </svg>
                        ) : (
                          <svg className="w-6 h-6 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                          </svg>
                        )}
                      </div>

                      <div className="flex-1 max-w-3xl">
                        {message.role === 'user' ? (
                          <div className="bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-gray-700/50 shadow-xl">
                            <p className="text-gray-100 whitespace-pre-wrap leading-relaxed">{message.content}</p>
                          </div>
                        ) : (
                          // Check if it's a structured response
                          (() => {
                            if (isStructuredResponse(message.content)) {
                              const structured: structuredChatResponse = JSON.parse(message.content);
                              return (
                                <div className="bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-gray-700/50 shadow-xl space-y-5">
                                  {/* Main Answer */}
                                  <div>
                                    <p className="text-gray-100 whitespace-pre-wrap leading-relaxed">{structured.answer}</p>
                                  </div>

                                  {/* Key Points */}
                                  {structured.key_points && structured.key_points.length > 0 && (
                                    <div className="border-t border-gray-700/50 pt-4">
                                      <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                        </svg>
                                        Key Points
                                      </h4>
                                      <ul className="space-y-2">
                                        {structured.key_points.map((point, i) => (
                                          <li key={i} className="flex gap-3 text-gray-200">
                                            <span className="text-gray-500 flex-shrink-0">•</span>
                                            <span>{point}</span>
                                          </li>
                                        ))}
                                      </ul>
                                    </div>
                                  )}

                                  {/* Sources & Confidence */}
                                  <div className="flex flex-wrap gap-4 border-t border-gray-700/50 pt-4">
                                    {/* Confidence Level */}
                                    {structured.confidence_level && (
                                      <div className="flex items-center gap-2">
                                        <span className="text-xs text-gray-400">Confidence:</span>
                                        <span className={`text-xs font-medium px-3 py-1 rounded-full ${
                                          structured.confidence_level.toLowerCase() === 'high' 
                                            ? 'bg-green-900/30 text-green-400 border border-green-700/50'
                                            : structured.confidence_level.toLowerCase() === 'medium'
                                            ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-700/50'
                                            : 'bg-red-900/30 text-red-400 border border-red-700/50'
                                        }`}>
                                          {structured.confidence_level}
                                        </span>
                                      </div>
                                    )}

                                    {/* Sources */}
                                    {structured.sources_cited && structured.sources_cited.length > 0 && (
                                      <div className="flex items-center gap-2">
                                        <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                                        </svg>
                                        <span className="text-xs text-gray-400">{structured.sources_cited.length} source{structured.sources_cited.length !== 1 ? 's' : ''}</span>
                                      </div>
                                    )}
                                  </div>

                                  {/* Sources List */}
                                  {structured.sources_cited && structured.sources_cited.length > 0 && (
                                    <div>
                                      <details className="group">
                                        <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-300 flex items-center gap-2 transition-colors">
                                          <svg className="w-3 h-3 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                          </svg>
                                          View sources
                                        </summary>
                                        <ul className="mt-3 space-y-1.5 pl-5">
                                          {structured.sources_cited.map((source, i) => (
                                            <li key={i} className="text-xs text-gray-400">
                                              {i + 1}. {source}
                                            </li>
                                          ))}
                                        </ul>
                                      </details>
                                    </div>
                                  )}

                                  {/* Clarification Needed */}
                                  {structured.needs_clarification && structured.clarification_needed && (
                                    <div className="border border-yellow-700/50 bg-yellow-900/20 rounded-2xl p-4">
                                      <div className="flex items-start gap-3">
                                        <svg className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                        </svg>
                                        <div>
                                          <h4 className="text-sm font-semibold text-yellow-300 mb-1">Clarification Needed</h4>
                                          <p className="text-sm text-yellow-200/90">{structured.clarification_needed}</p>
                                        </div>
                                      </div>
                                    </div>
                                  )}

                                  {/* Follow-up Suggestions */}
                                  {structured.follow_up_suggestions && structured.follow_up_suggestions.length > 0 && (
                                    <div className="border-t border-gray-700/50 pt-4">
                                      <h4 className="text-xs font-semibold text-gray-400 mb-3">Follow-up questions:</h4>
                                      <div className="space-y-2">
                                        {structured.follow_up_suggestions.map((suggestion, i) => (
                                          <button
                                            key={i}
                                            onClick={() => {
                                              setCurrentQuestion(suggestion);
                                            }}
                                            className="w-full text-left text-sm text-gray-300 bg-gray-700/30 hover:bg-gray-700/50 px-4 py-2.5 rounded-xl transition-all hover:scale-[1.01] active:scale-[0.99] border border-gray-700/50 hover:border-gray-600/50"
                                          >
                                            {suggestion}
                                          </button>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              );
                            } else {
                              // Regular text response
                              return (
                                <div className="bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-gray-700/50 shadow-xl">
                                  <p className="text-gray-100 whitespace-pre-wrap leading-relaxed">{message.content}</p>
                                </div>
                              );
                            }
                          })()
                        )}
                      </div>
                    </div>
                  ))}

                  {isChatting && (
                    <div className="flex gap-4 animate-fade-in">
                      <div className="flex-shrink-0 w-11 h-11 rounded-2xl bg-gray-800 border border-gray-700 ring-2 ring-gray-700/20 flex items-center justify-center shadow-lg">
                        <svg className="w-6 h-6 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                        </svg>
                      </div>
                      <div className="flex-1 max-w-3xl">
                        <div className="bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-gray-700/50 shadow-xl">
                          <div className="flex gap-2.5">
                            <div className="w-3 h-3 bg-gray-500 rounded-full animate-bounce shadow-lg"></div>
                            <div className="w-3 h-3 bg-gray-500 rounded-full animate-bounce shadow-lg" style={{ animationDelay: '0.2s' }}></div>
                            <div className="w-3 h-3 bg-gray-500 rounded-full animate-bounce shadow-lg" style={{ animationDelay: '0.4s' }}></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div ref={chatEndRef} />
                </div>
              </div>

              {/* Error Display */}
              {error && (
                <div className="px-4 pb-3 animate-shake">
                  <div className="max-w-4xl mx-auto p-5 bg-red-900/30 backdrop-blur-sm border border-red-700/50 rounded-2xl flex items-start gap-3 shadow-xl">
                    <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-red-300 text-sm flex-1">{error}</p>
                  </div>
                </div>
              )}

              {/* Input Area */}
              <div className="border-t border-gray-800/50 bg-gray-950/80 backdrop-blur-xl p-6 shadow-2xl">
                <div className="max-w-4xl mx-auto">
                  <div className="flex gap-4 items-end">
                    <div className="flex-1 relative">
                      <textarea
                        value={currentQuestion}
                        onChange={(e) => setCurrentQuestion(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask a question about your PDF..."
                        rows={1}
                        className="w-full px-6 py-5 bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 text-white placeholder-gray-500 rounded-3xl focus:outline-none focus:ring-2 focus:ring-gray-600 focus:border-gray-600 transition-all resize-none shadow-xl hover:shadow-2xl"
                        style={{ minHeight: '64px', maxHeight: '200px' }}
                        disabled={isChatting}
                      />
                    </div>
                    <button
                      onClick={handleSendMessage}
                      disabled={!currentQuestion.trim() || isChatting}
                      className={`flex-shrink-0 p-5 rounded-3xl font-semibold transition-all duration-300 group relative overflow-hidden ${
                        !currentQuestion.trim() || isChatting
                          ? 'bg-gray-800 cursor-not-allowed opacity-50'
                          : 'bg-gradient-to-r from-gray-800 via-gray-700 to-gray-600 hover:from-gray-700 hover:via-gray-600 hover:to-gray-500 shadow-2xl hover:shadow-3xl hover:scale-110 active:scale-95'
                      }`}
                    >
                      {!isChatting && (
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700"></div>
                      )}
                      {isChatting ? (
                        <svg className="animate-spin h-7 w-7 text-white relative z-10" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      ) : (
                        <svg className="w-7 h-7 text-white relative z-10 group-hover:rotate-45 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                        </svg>
                      )}
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-4 text-center flex items-center justify-center gap-2">
                    <kbd className="px-2 py-1 bg-gray-800/50 rounded text-xs border border-gray-700/50">Enter</kbd>
                    <span>to send</span>
                    <span className="text-gray-700">•</span>
                    <kbd className="px-2 py-1 bg-gray-800/50 rounded text-xs border border-gray-700/50">Shift + Enter</kbd>
                    <span>for new line</span>
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}