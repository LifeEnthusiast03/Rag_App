import { useState, useRef, useEffect } from 'react';

export default function PDFChatInterface() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [chatId, setChatId] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    const pdfFiles = selectedFiles.filter(file => file.type === 'application/pdf');
    
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

      const res = await fetch('http://localhost:8000/upload-pdfs', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await res.json();
      
      // Expecting chat_id from backend
      if (data.chat_id) {
        setChatId(data.chat_id);
        setUploadedFileName(files[0].name);
        setChatHistory([{
          role: 'system',
          content: `PDF "${files[0].name}" uploaded successfully! You can now ask questions about it.`
        }]);
        setFiles([]);
        document.getElementById('fileInput').value = '';
      } else {
        setError('No chat_id received from backend');
      }
    } catch (err) {
      setError(err.message);
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
    const updatedHistory = [...chatHistory, { role: 'user', content: userMessage }];
    setChatHistory(updatedHistory);

    try {
      // Prepare chat history for API (excluding system messages)
      const historyForAPI = updatedHistory
        .filter(msg => msg.role !== 'system')
        .map(msg => ({ role: msg.role, content: msg.content }));
      //  const historyForAPI = updatedHistory
      //   .filter(msg => msg.role !== 'system' && msg.role !== 'error')
      //   .slice(0, -1) // Exclude the current user message we just added
      //   .map(msg => msg.content);

      const res = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chat_id: chatId,
          question: userMessage,
          chat_history: historyForAPI
        }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Chat request failed');
      }

      const data = await res.json();
      
      // Add assistant response to chat history
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer || data.response || 'No response received' 
      }]);
    } catch (err) {
      setError(err.message);
      // Add error message to chat
      setChatHistory(prev => [...prev, { 
        role: 'error', 
        content: `Error: ${err.message}` 
      }]);
    } finally {
      setIsChatting(false);
    }
  };

  const handleKeyPress = (e) => {
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
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {!chatId ? (
          // Upload Section
          <div className="bg-white rounded-lg shadow-lg p-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              📄 PDF Chat Interface
            </h1>
            <p className="text-gray-600 mb-6">
              Upload a PDF and start chatting with it
            </p>

            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 mb-6 hover:border-indigo-400 transition-colors">
              <div className="text-center">
                <svg
                  className="mx-auto h-12 w-12 text-gray-400 mb-4"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <label
                  htmlFor="fileInput"
                  className="cursor-pointer text-indigo-600 hover:text-indigo-500 font-medium"
                >
                  Choose a PDF file
                </label>
                <input
                  id="fileInput"
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <p className="text-sm text-gray-500 mt-2">Select a PDF to chat with</p>
              </div>
            </div>

            {files.length > 0 && (
              <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                <p className="text-sm font-medium text-gray-700 mb-2">
                  Selected file:
                </p>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-indigo-600 font-medium truncate">{files[0].name}</span>
                  <span className="text-gray-500 ml-2">{formatBytes(files[0].size)}</span>
                </div>
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={files.length === 0 || uploading}
              className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-colors ${
                files.length === 0 || uploading
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-700'
              }`}
            >
              {uploading ? 'Uploading...' : 'Upload & Start Chat'}
            </button>

            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-800 text-sm font-medium">Error: {error}</p>
              </div>
            )}
          </div>
        ) : (
          // Chat Interface Section
          <div className="bg-white rounded-lg shadow-lg h-[calc(100vh-2rem)] flex flex-col">
            {/* Chat Header */}
            <div className="border-b border-gray-200 p-4 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold text-gray-800">💬 Chat with PDF</h2>
                <p className="text-sm text-gray-600 truncate max-w-md">
                  {uploadedFileName}
                </p>
                <p className="text-xs text-gray-500">Chat ID: {chatId}</p>
              </div>
              <button
                onClick={resetChat}
                className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm font-medium text-gray-700 transition-colors"
              >
                New Upload
              </button>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {chatHistory.map((message, idx) => (
                <div
                  key={idx}
                  className={`flex ${
                    message.role === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`max-w-[75%] rounded-lg p-4 ${
                      message.role === 'user'
                        ? 'bg-indigo-600 text-white'
                        : message.role === 'assistant'
                        ? 'bg-gray-100 text-gray-800'
                        : message.role === 'system'
                        ? 'bg-green-50 text-green-800 border border-green-200'
                        : 'bg-red-50 text-red-800 border border-red-200'
                    }`}
                  >
                    <div className="text-sm whitespace-pre-wrap">{message.content}</div>
                    {message.role === 'user' && (
                      <div className="text-xs mt-2 opacity-75">You</div>
                    )}
                    {message.role === 'assistant' && (
                      <div className="text-xs mt-2 opacity-75">AI Assistant</div>
                    )}
                  </div>
                </div>
              ))}
              {isChatting && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 rounded-lg p-4">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Error Display in Chat */}
            {error && (
              <div className="px-4 pb-2">
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-800 text-sm">{error}</p>
                </div>
              </div>
            )}

            {/* Chat Input */}
            <div className="border-t border-gray-200 p-4">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={currentQuestion}
                  onChange={(e) => setCurrentQuestion(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask a question about your PDF..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  disabled={isChatting}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!currentQuestion.trim() || isChatting}
                  className={`px-6 py-3 rounded-lg font-medium text-white transition-colors ${
                    !currentQuestion.trim() || isChatting
                      ? 'bg-gray-300 cursor-not-allowed'
                      : 'bg-indigo-600 hover:bg-indigo-700'
                  }`}
                >
                  {isChatting ? (
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  ) : (
                    'Send'
                  )}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}