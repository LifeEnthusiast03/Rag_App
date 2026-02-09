import { useState, useRef, useEffect } from 'react';
import { useAuthContext } from '@/hooks/useauth';
interface Message {
  role: string;
  content: string;
  timestamp: string;
}

interface Chat {
  id: string;
  title: string;
  fileName: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

export default function PDFChatInterface() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatId, setChatId] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [savedChats, setSavedChats] = useState<Chat[]>([]);
  const [currentChatIndex, setCurrentChatIndex] = useState<number | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const  {logout} = useAuthContext()
  // Load saved chats from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('pdfChats');
    if (stored) {
      try {
        setSavedChats(JSON.parse(stored));
      } catch (e) {
        console.error('Failed to load chats:', e);
      }
    }
  }, []);

  // Save chats to localStorage whenever they change
  useEffect(() => {
    if (savedChats.length > 0) {
      localStorage.setItem('pdfChats', JSON.stringify(savedChats));
    }
  }, [savedChats]);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    const pdfFiles = selectedFiles.filter((file: File) => file.type === 'application/pdf');
    
    if (pdfFiles.length === 0) {
      setError('Please select valid PDF files');
      setFiles([]);
    } else if (pdfFiles.length !== selectedFiles.length) {
      setError(`Only ${pdfFiles.length} PDF file(s) selected. Non-PDF files were ignored.`);
      setFiles(pdfFiles);
    } else {
      setFiles(pdfFiles);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select at least one file first');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });

      const token = localStorage.getItem('token');
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
      
      // Expecting chat_id from backend
      if (data.chat_id) {
        const newChatHistory = [{
          role: 'system',
          content: `PDF "${files[0].name}" uploaded successfully! You can now ask questions about it.`,
          timestamp: new Date().toISOString()
        }];
        
        setChatId(data.chat_id);
        setUploadedFileName(files[0].name);
        setChatHistory(newChatHistory);
        
        // Save to chat history
        const newChat = {
          id: data.chat_id,
          title: files[0].name,
          fileName: files[0].name,
          messages: newChatHistory,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        };
        
        const updatedChats = [newChat, ...savedChats];
        setSavedChats(updatedChats);
        setCurrentChatIndex(0);
        
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
    if (!currentQuestion.trim() || !chatId) return;

    const userMessage = currentQuestion.trim();
    setCurrentQuestion('');
    setIsChatting(true);
    
    // Add user message to chat history
    const newUserMessage = { 
      role: 'user', 
      content: userMessage,
      timestamp: new Date().toISOString()
    };
    const updatedHistory = [...chatHistory, newUserMessage];
    setChatHistory(updatedHistory);

    try {
      const token = localStorage.getItem('token');
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          chat_id: chatId,
          question: userMessage
        }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Chat request failed');
      }

      const data = await res.json();
      
      // Add assistant response to chat history
      const assistantMessage = { 
        role: 'assistant', 
        content: data.answer || data.response || 'No response received',
        timestamp: new Date().toISOString()
      };
      
      const finalHistory = [...updatedHistory, assistantMessage];
      setChatHistory(finalHistory);
      
      // Update saved chats
      if (currentChatIndex !== null) {
        const updatedChats = [...savedChats];
        updatedChats[currentChatIndex] = {
          ...updatedChats[currentChatIndex],
          messages: finalHistory,
          updatedAt: new Date().toISOString()
        };
        setSavedChats(updatedChats);
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Chat request failed';
      setError(errorMsg);
      // Add error message to chat
      const errorMessage = { 
        role: 'error', 
        content: `Error: ${errorMsg}`,
        timestamp: new Date().toISOString()
      };
      setChatHistory(prev => [...prev, errorMessage]);
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
    setChatId(null);
    setChatHistory([]);
    setUploadedFileName('');
    setError(null);
    setCurrentChatIndex(null);
  };

  const loadChat = (index: number) => {
    const chat = savedChats[index];
    setChatId(chat.id);
    setChatHistory(chat.messages);
    setUploadedFileName(chat.fileName);
    setCurrentChatIndex(index);
    setError(null);
  };

  const deleteChat = (index: number) => {
    const updatedChats = savedChats.filter((_, i) => i !== index);
    setSavedChats(updatedChats);
    
    if (index === currentChatIndex) {
      resetChat();
    } else if (currentChatIndex !== null && index < currentChatIndex) {
      setCurrentChatIndex(currentChatIndex - 1);
    }
    
    // Update localStorage
    if (updatedChats.length === 0) {
      localStorage.removeItem('pdfChats');
    }
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const formatTimestamp = (timestamp: string) => {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
    
    return date.toLocaleDateString();
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
          {savedChats.length === 0 ? (
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
            savedChats.map((chat, index) => (
              <div
                key={chat.id}
                className={`group relative p-3.5 rounded-xl cursor-pointer transition-all duration-200 ${
                  currentChatIndex === index
                    ? 'bg-gradient-to-r from-gray-800 to-gray-800/80 border border-gray-700 shadow-lg scale-[1.02]'
                    : 'hover:bg-gray-900/50 hover:scale-[1.01]'
                }`}
                onClick={() => loadChat(index)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                      </svg>
                      <p className="text-sm font-medium truncate">{chat.title}</p>
                    </div>
                    <p className="text-xs text-gray-500">{formatTimestamp(chat.updatedAt)}</p>
                    <p className="text-xs text-gray-600 mt-1">{chat.messages.length} messages</p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteChat(index);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-900/40 rounded-lg transition-all hover:scale-110 active:scale-95"
                  >
                    <svg className="w-4 h-4 text-red-400 hover:text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
            <p className="font-medium">{savedChats.length} conversation{savedChats.length !== 1 ? 's' : ''}</p>
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
          
          {chatId && (
            <div className="flex-1 flex items-center gap-3 animate-fade-in">
              <div className="flex items-center gap-2 px-4 py-2 bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 shadow-lg">
                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
                <span className="text-sm font-medium truncate max-w-md">{uploadedFileName}</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-500 bg-gray-900/50 px-3 py-1.5 rounded-lg border border-gray-800/50">
                <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>
                <span>ID: {chatId}</span>
              </div>
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
          {!chatId ? (
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
                  {chatHistory.length === 0 && (
                    <div className="text-center mt-20 animate-fade-in">
                      <div className="inline-flex items-center justify-center w-20 h-20 bg-gray-800/50 backdrop-blur-sm rounded-3xl mb-6 shadow-xl">
                        <svg className="w-10 h-10 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                        </svg>
                      </div>
                      <h3 className="text-2xl font-semibold mb-2">Start a Conversation</h3>
                      <p className="text-gray-400 text-lg">Ask me anything about your PDF</p>
                    </div>
                  )}

                  {chatHistory.map((message, idx) => (
                    <div
                      key={idx}
                      className={`flex gap-4 animate-slide-up ${
                        message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                      } ${message.role === 'system' ? 'justify-center' : ''}`}
                    >
                      {message.role !== 'system' && (
                        <div className={`flex-shrink-0 w-11 h-11 rounded-2xl flex items-center justify-center shadow-lg ${
                          message.role === 'user'
                            ? 'bg-gradient-to-br from-gray-700 to-gray-600 ring-2 ring-gray-600/20'
                            : message.role === 'assistant'
                            ? 'bg-gray-800 border border-gray-700 ring-2 ring-gray-700/20'
                            : 'bg-red-900/30 border border-red-700'
                        }`}>
                          {message.role === 'user' ? (
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                          ) : message.role === 'assistant' ? (
                            <svg className="w-6 h-6 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                            </svg>
                          ) : (
                            <svg className="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          )}
                        </div>
                      )}

                      <div className={`flex-1 ${message.role === 'system' ? 'max-w-md' : 'max-w-3xl'}`}>
                        <div className={`group relative transition-all duration-300 hover:scale-[1.01] ${
                          message.role === 'system'
                            ? 'bg-green-900/20 backdrop-blur-sm border border-green-800/50 rounded-2xl p-5 text-center shadow-lg'
                            : message.role === 'error'
                            ? 'bg-red-900/20 backdrop-blur-sm border border-red-800/50 rounded-2xl p-5 shadow-lg'
                            : 'bg-gray-800/40 backdrop-blur-sm rounded-3xl p-6 border border-gray-700/50 shadow-xl'
                        }`}>
                          <div className={`prose prose-invert max-w-none ${
                            message.role === 'system' ? 'text-green-300 text-sm' :
                            message.role === 'error' ? 'text-red-300' :
                            'text-gray-100'
                          }`}>
                            <p className="whitespace-pre-wrap leading-relaxed m-0">{message.content}</p>
                          </div>
                          
                          {message.role === 'assistant' && (
                            <div className="flex items-center gap-2 mt-4 pt-4 border-t border-gray-700/50">
                              <button
                                onClick={() => copyMessage(message.content)}
                                className="p-2.5 hover:bg-gray-700/50 rounded-xl transition-all duration-200 flex items-center gap-2 text-xs text-gray-400 hover:text-gray-200 hover:scale-105 active:scale-95"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                </svg>
                                Copy
                              </button>
                              {message.timestamp && (
                                <span className="text-xs text-gray-500 ml-auto">
                                  {formatTimestamp(message.timestamp)}
                                </span>
                              )}
                            </div>
                          )}
                        </div>
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