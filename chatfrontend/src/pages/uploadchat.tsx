import { useState, useRef, useEffect } from 'react';
import {
  Plus, FileText, Trash2, Upload, Zap, ArrowUp, X,
  ChevronDown, Menu, Copy, Check,
} from 'lucide-react';
import { useAuthContext } from '@/hooks/useauth';
import { useChat } from '@/hooks/usechat';
import { chatreq } from '@/service/chatservice';
import type { message, structuredChatResponse } from '@/type/types';
import ProfileDropdown from '@/components/profile-dropdown';

/* ─── Structured AI response renderer ───────────────────────── */
interface StructuredAIMessageProps {
  content: string;
  onSuggestionClick: (text: string) => void;
}

function StructuredAIMessage({ content, onSuggestionClick }: StructuredAIMessageProps) {
  const [keyPointsOpen, setKeyPointsOpen] = useState(false);

  let structured: structuredChatResponse | null = null;
  try {
    const parsed = JSON.parse(content);
    if (parsed && typeof parsed === 'object' && 'answer' in parsed) {
      structured = parsed as structuredChatResponse;
    }
  } catch {
    // fall through to plain text
  }

  if (!structured) {
    return (
      <p className="text-sm leading-relaxed text-[var(--color-text-primary)] whitespace-pre-wrap">
        {content}
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {/* Answer */}
      <p className="text-sm leading-relaxed text-[var(--color-text-primary)] whitespace-pre-wrap">
        {structured.answer}
      </p>

      {/* Confidence badge */}
      {structured.confidence_level && (
        <span className="font-mono text-[10px] text-[var(--color-text-hint)] inline-block">
          confidence: {structured.confidence_level.toUpperCase()}
        </span>
      )}

      {/* Key Points collapsible */}
      {structured.key_points && structured.key_points.length > 0 && (
        <div>
          <button
            onClick={() => setKeyPointsOpen(!keyPointsOpen)}
            className="flex items-center gap-1 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
          >
            <ChevronDown
              size={12}
              className={`transition-transform duration-200 ${keyPointsOpen ? 'rotate-180' : ''}`}
            />
            Key points ({structured.key_points.length})
          </button>
          {keyPointsOpen && (
            <ul className="mt-2 space-y-1.5 pl-3 border-l border-[var(--color-border)]">
              {structured.key_points.map((point, i) => (
                <li key={i} className="flex gap-2 text-xs text-[var(--color-text-muted)]">
                  <span className="text-[var(--color-text-hint)] flex-shrink-0">•</span>
                  <span>{point}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* Clarification needed */}
      {structured.needs_clarification && structured.clarification_needed && (
        <div className="border border-[var(--color-border)] rounded-lg px-3 py-2.5">
          <p className="text-xs text-[var(--color-text-muted)]">
            <span className="font-medium text-[var(--color-text-primary)]">Clarification: </span>
            {structured.clarification_needed}
          </p>
        </div>
      )}

      {/* Sources */}
      {structured.sources_cited && structured.sources_cited.length > 0 && (
        <div className="space-y-0.5">
          {structured.sources_cited.map((source, i) => (
            <p key={i} className="text-xs text-[var(--color-text-muted)]">
              {i + 1}. {source}
            </p>
          ))}
        </div>
      )}

      {/* Follow-up suggestion chips */}
      {structured.follow_up_suggestions && structured.follow_up_suggestions.length > 0 && (
        <div className="flex gap-2 overflow-x-auto scrollbar-none pb-0.5 pt-1">
          {structured.follow_up_suggestions.map((s, i) => (
            <button
              key={i}
              onClick={() => onSuggestionClick(s)}
              className="flex-shrink-0 bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-full px-3 py-1 text-xs text-[var(--color-text-muted)] hover:border-[var(--color-accent)] hover:text-[var(--color-text-primary)] transition-colors whitespace-nowrap"
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Typing indicator ───────────────────────────────────────── */
function TypingIndicator() {
  return (
    <div className="flex gap-1 items-center py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-[var(--color-text-hint)] inline-block animate-bounce"
          style={{ animationDelay: `${i * 0.15}s`, animationDuration: '0.9s' }}
        />
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN COMPONENT
═══════════════════════════════════════════════════════════════ */
export default function PDFChatInterface() {
  /* ── state / refs (unchanged from original) ─────────────────── */
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatHistory, setChatHistory] = useState<message[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loadingConversation, setLoadingConversation] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { token } = useAuthContext();
  const { curChatId, setCurChatId, userChats, getChatconversation, setUserChats, deleteChat } = useChat();

  /* ── scroll to bottom ────────────────────────────────────────── */
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isChatting]);

  /* ── textarea auto-grow ──────────────────────────────────────── */
  const growTextarea = (el: HTMLTextAreaElement) => {
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 144)}px`;
  };

  /* ── helpers (unchanged) ─────────────────────────────────────── */
  const isStructuredResponse = (content: string): boolean => {
    try {
      const parsed = JSON.parse(content);
      return parsed && typeof parsed === 'object' && 'answer' in parsed;
    } catch {
      return false;
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  /* ── handlers (unchanged logic) ──────────────────────────────── */
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    const pdfFiles = selectedFiles.filter((f: File) => f.type === 'application/pdf');
    const skipped = selectedFiles.length - pdfFiles.length;
    if (pdfFiles.length === 0) {
      setError('Please select valid PDF files');
      setFiles([]);
    } else {
      setFiles(prev => {
        // Merge new files, avoiding duplicates by name+size
        const existing = new Set(prev.map(f => `${f.name}-${f.size}`));
        const merged = [...prev, ...pdfFiles.filter(f => !existing.has(`${f.name}-${f.size}`))];
        return merged;
      });
      setError(skipped > 0 ? `${skipped} non-PDF file${skipped > 1 ? 's' : ''} were ignored` : null);
    }
    // Reset input so the same file can be re-added after removal
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (files.length === 0) { setError('Please select a file first'); return; }
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      const headers: HeadersInit = {};
      if (token) headers['Authorization'] = `Bearer ${token}`;
      const res = await fetch('http://localhost:8000/upload-pdfs', { method: 'POST', headers, body: formData });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Upload failed');
      }
      const data = await res.json();
      if (data.chat_id) {
        setCurChatId(data.chat_id);
        const firstName = files[0].name;
        const extra = files.length - 1;
        setUploadedFileName(extra > 0 ? `${firstName} +${extra} more` : firstName);
        setChatHistory([]);
        setUserChats([{ chat_id: data.chat_id, chat_name: data.chat_name }, ...userChats]);
        setFiles([]);
        if (fileInputRef.current) fileInputRef.current.value = '';
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
    if (textareaRef.current) { textareaRef.current.style.height = 'auto'; }
    setIsChatting(true);
    const newUserMessage: message = { role: 'user', content: userMessage };
    const updatedHistory = [...chatHistory, newUserMessage];
    setChatHistory(updatedHistory);
    try {
      const data = await chatreq({ chat_id: curChatId, question: userMessage, chat_history: chatHistory }, token);
      if (data.success) {
        setChatHistory([...updatedHistory, { role: data.role || 'assistant', content: data.response, sources: data.sources ?? [] }]);
        setError(null);
      } else {
        setError(data.error_message || 'Failed to get response');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chat request failed');
    } finally {
      setIsChatting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); }
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
    } catch {
      setError('Failed to load conversation');
      setChatHistory([]);
    } finally {
      setLoadingConversation(false);
    }
  };

  const handleDeleteChat = async (e: React.MouseEvent, chat_id: number) => {
    e.stopPropagation();
    if (!confirm('Delete this conversation?')) return;
    try {
      const result = await deleteChat(chat_id);
      if (result.Successful) { if (curChatId === chat_id) resetChat(); }
      else setError(result.message || 'Failed to delete chat');
    } catch {
      setError('Failed to delete chat');
    }
  };

  /* ─── Drag-and-drop ──────────────────────────────────────────── */
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragOver(true); };
  const handleDragLeave = () => setIsDragOver(false);
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const dropped = Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf');
    if (dropped.length) {
      setFiles(prev => {
        const existing = new Set(prev.map(f => `${f.name}-${f.size}`));
        return [...prev, ...dropped.filter(f => !existing.has(`${f.name}-${f.size}`))];
      });
      setError(null);
    } else setError('Please drop valid PDF files');
  };

  /* ─── Sidebar item ────────────────────────────────────────────── */
  const SidebarItem = ({ chat }: { chat: { chat_id: number; chat_name: string } }) => {
    const isActive = chat.chat_id === curChatId;
    return (
      <div
        onClick={() => loadChat(chat.chat_id, chat.chat_name)}
        className={`group relative flex items-center gap-2.5 px-3 py-2.5 mx-1 rounded-md cursor-pointer transition-colors ${
          isActive
            ? 'bg-[var(--color-bg-elevated)] border-l-2 border-[var(--color-accent)] rounded-l-none pl-[10px]'
            : 'hover:bg-[var(--color-bg-elevated)] border-l-2 border-transparent'
        }`}
      >
        <FileText
          size={14}
          className={`flex-shrink-0 ${isActive ? 'text-[var(--color-accent)]' : 'text-[var(--color-text-muted)]'}`}
        />
        <span className="font-mono text-xs text-[var(--color-text-primary)] truncate flex-1 min-w-0">
          {chat.chat_name}
        </span>
        <button
          onClick={(e) => handleDeleteChat(e, chat.chat_id)}
          className="opacity-0 group-hover:opacity-100 flex-shrink-0 text-[var(--color-text-hint)] hover:text-red-400 transition-all p-0.5 rounded"
          title="Delete conversation"
        >
          <Trash2 size={12} />
        </button>
      </div>
    );
  };



  /* ═══════════════════════════════════════════════════════════════
     RENDER
  ═══════════════════════════════════════════════════════════════ */
  return (
    <div className="flex h-screen overflow-hidden bg-[var(--color-bg-base)]">

      {/* ── Sidebar ─────────────────────────────────────────────── */}
      <aside
        className={`${sidebarOpen ? 'w-[260px]' : 'w-0'} flex-shrink-0 bg-[var(--color-bg-surface)] border-r border-[var(--color-border)] flex flex-col overflow-hidden transition-all duration-300`}
      >
        {/* New Chat button */}
        <div className="m-4 mb-2">
          <button
            onClick={resetChat}
            className="btn-primary w-full rounded-md text-sm font-medium py-2 px-4 flex items-center justify-center gap-2"
          >
            <Plus size={15} />
            New Chat
          </button>
        </div>

        {/* RECENT label */}
        <p className="text-[10px] font-semibold tracking-widest text-[var(--color-text-hint)] uppercase px-4 pb-1 mt-2">
          Recent
        </p>

        {/* Chat list */}
        <nav className="flex-1 overflow-y-auto py-1 space-y-0.5">
          {userChats.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <p className="text-xs text-[var(--color-text-hint)]">No conversations yet</p>
              <p className="text-[10px] text-[var(--color-text-hint)] mt-1 font-mono">Upload a PDF to start</p>
            </div>
          ) : (
            userChats.map((chat) => <SidebarItem key={chat.chat_id} chat={chat} />)
          )}
        </nav>

        {/* Sidebar footer — profile */}
        <div className="mt-auto border-t border-[var(--color-border)] p-3">
          <ProfileDropdown placement="sidebar" />
        </div>
      </aside>

      {/* ── Main area ────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">

        {/* ── Shared topbar (always visible) ─────────────────────── */}
        <header className="h-14 border-b border-[var(--color-border)] bg-[var(--color-bg-surface)] px-4 flex items-center gap-3 flex-shrink-0">
          {/* Hamburger */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="w-8 h-8 rounded-md border border-[var(--color-border)] hover:bg-[var(--color-bg-elevated)] flex items-center justify-center text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors flex-shrink-0"
            aria-label="Toggle sidebar"
          >
            <Menu size={15} />
          </button>

          {/* File info when chat is active */}
          {curChatId !== 0 && (
            <div className="flex items-center gap-2.5 flex-1 min-w-0">
              <FileText size={14} className="text-[var(--color-text-muted)] flex-shrink-0" />
              <span className="font-mono text-sm text-[var(--color-text-primary)] truncate">
                {uploadedFileName}
              </span>
            </div>
          )}

          {curChatId === 0 && <div className="flex-1" />}

          {/* Right side */}
          <div className="flex items-center gap-2 flex-shrink-0">
            {curChatId !== 0 && (
              <button
                onClick={resetChat}
                className="w-7 h-7 rounded-md border border-[var(--color-border)] hover:bg-[var(--color-bg-elevated)] flex items-center justify-center text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors"
                title="Close document"
              >
                <X size={13} />
              </button>
            )}
          </div>
        </header>

        {/* ── Content area ─────────────────────────────────────────── */}
        <div className="flex-1 overflow-hidden flex flex-col">

          {curChatId === 0 ? (
            /* ══ UPLOAD STATE ════════════════════════════════════════ */
            <div className="flex-1 flex flex-col items-center justify-center p-8 overflow-y-auto">

              {/* App icon + heading */}
              <div className="text-center mb-8">
                <div className="w-12 h-12 rounded-xl bg-[var(--color-bg-elevated)] border border-[var(--color-border)] flex items-center justify-center mx-auto mb-4">
                  <FileText size={22} className="text-[var(--color-accent)]" />
                </div>
                <h1 className="text-2xl font-bold -tracking-tight text-[var(--color-text-primary)]">Readwise</h1>
                <p className="text-sm text-[var(--color-text-muted)] mt-1">
                  Upload a document and start an intelligent conversation
                </p>
              </div>

              {/* Dropzone */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-12 w-full max-w-lg flex flex-col items-center gap-3 cursor-pointer transition-all ${
                  isDragOver
                    ? 'border-[var(--color-accent)] bg-[oklch(60%_0.2_285_/_0.05)]'
                    : 'border-[var(--color-border)] hover:border-[var(--color-border-focus)] hover:bg-[var(--color-bg-elevated)]'
                }`}
              >
                <input
                  ref={fileInputRef}
                  id="fileInput"
                  type="file"
                  accept=".pdf"
                  multiple
                  onChange={handleFileChange}
                  className="hidden"
                />
                <Upload size={32} className="text-[var(--color-accent)]" />
                <div className="text-center">
                  <p className="text-sm font-medium text-[var(--color-text-primary)]">Drop your PDFs here</p>
                  <p className="text-sm text-[var(--color-text-muted)] mt-0.5">or click to browse · multiple files supported</p>
                </div>
                <p className="font-mono text-xs text-[var(--color-text-hint)]">Supports multiple PDF files up to 50MB each</p>
              </div>

              {/* Selected files list */}
              {files.length > 0 && (
                <div className="w-full max-w-lg mt-3 bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-lg overflow-hidden">
                  {/* Header row */}
                  <div className="flex items-center justify-between px-3 py-2 border-b border-[var(--color-border)]">
                    <span className="text-[10px] font-mono text-[var(--color-text-hint)] uppercase tracking-widest">
                      {files.length} file{files.length > 1 ? 's' : ''} selected
                    </span>
                    <button
                      onClick={(e) => { e.stopPropagation(); setFiles([]); }}
                      className="text-[10px] font-mono text-[var(--color-text-hint)] hover:text-red-400 transition-colors"
                    >
                      Clear all
                    </button>
                  </div>
                  {/* File rows — scrollable if many */}
                  <div className="max-h-40 overflow-y-auto divide-y divide-[var(--color-border)]">
                    {files.map((file, i) => (
                      <div key={`${file.name}-${file.size}`} className="flex items-center gap-2.5 px-3 py-2">
                        <FileText size={13} className="text-[var(--color-accent)] flex-shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium text-[var(--color-text-primary)] truncate">{file.name}</p>
                          <p className="font-mono text-[10px] text-[var(--color-text-hint)]">{formatBytes(file.size)}</p>
                        </div>
                        <button
                          onClick={(e) => { e.stopPropagation(); removeFile(i); }}
                          className="flex-shrink-0 text-[var(--color-text-hint)] hover:text-red-400 transition-colors p-0.5 rounded"
                          title="Remove file"
                        >
                          <X size={12} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Upload CTA */}
              <button
                onClick={handleUpload}
                disabled={files.length === 0 || uploading}
                className="btn-primary rounded-md w-full max-w-lg mt-4 py-3 text-sm font-medium flex items-center justify-center gap-2"
              >
                {uploading ? (
                  <>
                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span>Uploading...</span>
                  </>
                ) : (
                  <>
                    <Zap size={15} />
                    <span>Upload &amp; Start Chatting</span>
                  </>
                )}
              </button>

              {/* Error */}
              {error && (
                <div className="w-full max-w-lg mt-3 px-3 py-2.5 bg-[var(--color-bg-elevated)] border border-red-900/60 rounded-lg">
                  <p className="text-xs text-red-400">{error}</p>
                </div>
              )}
            </div>

          ) : (
            /* ══ CHAT STATE ══════════════════════════════════════════ */
            <div className="flex-1 flex flex-col overflow-hidden">

              {/* Messages area */}
              <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
                <div className="max-w-3xl mx-auto w-full space-y-6">

                  {/* Loading conversation */}
                  {loadingConversation && (
                    <div className="flex justify-center pt-16">
                      <div className="flex flex-col items-center gap-3">
                        <svg className="animate-spin h-6 w-6 text-[var(--color-text-hint)]" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        <p className="text-xs text-[var(--color-text-hint)]">Loading conversation...</p>
                      </div>
                    </div>
                  )}

                  {/* Empty chat prompt */}
                  {!loadingConversation && chatHistory.length === 0 && (
                    <div className="flex justify-center pt-16">
                      <div className="text-center">
                        <div className="w-10 h-10 rounded-xl bg-[var(--color-bg-elevated)] border border-[var(--color-border)] flex items-center justify-center mx-auto mb-3">
                          <FileText size={18} className="text-[var(--color-text-muted)]" />
                        </div>
                        <p className="text-sm font-medium text-[var(--color-text-primary)]">Ask anything about your document</p>
                        <p className="text-xs text-[var(--color-text-hint)] mt-1 font-mono">{uploadedFileName}</p>
                      </div>
                    </div>
                  )}

                  {/* Message list */}
                  {!loadingConversation && chatHistory.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      {msg.role === 'user' ? (
                        /* User bubble */
                        <div className="bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-2xl rounded-tr-sm px-4 py-2.5 max-w-[75%]">
                          <p className="text-sm text-[var(--color-text-primary)] whitespace-pre-wrap leading-relaxed">
                            {msg.content}
                          </p>
                        </div>
                      ) : (
                        /* AI — no bubble */
                        <div className="group/msg max-w-[80%]">
                          <p className="text-[10px] font-mono text-[var(--color-text-hint)] mb-1.5 uppercase tracking-widest">
                            Readwise
                          </p>
                          {isStructuredResponse(msg.content) ? (
                            <StructuredAIMessage
                              content={msg.content}
                              onSuggestionClick={(text) => setCurrentQuestion(text)}
                            />
                          ) : (
                            <p className="text-sm leading-relaxed text-[var(--color-text-primary)] whitespace-pre-wrap">
                              {msg.content}
                            </p>
                          )}
                          {/* Copy button */}
                          <button
                            onClick={async () => {
                              try {
                                // For structured responses, copy only the plain answer text
                                let textToCopy = msg.content;
                                try {
                                  const parsed = JSON.parse(msg.content);
                                  if (parsed?.answer) textToCopy = parsed.answer;
                                } catch { /* not JSON, use raw */ }
                                await navigator.clipboard.writeText(textToCopy);
                                setCopiedIndex(idx);
                                setTimeout(() => setCopiedIndex(null), 2000);
                              } catch { /* clipboard unavailable */ }
                            }}
                            className="mt-1.5 opacity-0 group-hover/msg:opacity-100 flex items-center gap-1 text-[var(--color-text-hint)] hover:text-[var(--color-text-primary)] transition-all duration-150"
                            title={copiedIndex === idx ? 'Copied!' : 'Copy message'}
                            aria-label="Copy message"
                          >
                            {copiedIndex === idx
                              ? <Check size={12} className="text-[var(--color-success)]" />
                              : <Copy size={12} />
                            }
                            <span className="text-[10px] font-mono">
                              {copiedIndex === idx ? 'Copied' : 'Copy'}
                            </span>
                          </button>
                          {/* Source citation pills */}
                          {msg.sources && msg.sources.length > 0 && (
                            <div className="flex flex-wrap gap-2 mt-2 ml-2">
                              {msg.sources.map((src, i) => (
                                <span
                                  key={i}
                                  className="flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-gray-800 text-gray-400 border border-gray-700"
                                >
                                  <FileText size={12} aria-hidden="true" />
                                  {src.filename} • p.{src.page}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}

                  {/* Typing indicator */}
                  {isChatting && (
                    <div className="flex justify-start">
                      <div className="max-w-[80%]">
                        <p className="text-[10px] font-mono text-[var(--color-text-hint)] mb-1.5 uppercase tracking-widest">
                          Readwise
                        </p>
                        <TypingIndicator />
                      </div>
                    </div>
                  )}

                  <div ref={chatEndRef} />
                </div>
              </div>

              {/* Chat error */}
              {error && (
                <div className="px-6 pb-2">
                  <div className="max-w-3xl mx-auto px-3 py-2.5 bg-[var(--color-bg-elevated)] border border-red-900/60 rounded-lg">
                    <p className="text-xs text-red-400">{error}</p>
                  </div>
                </div>
              )}

              {/* Input bar */}
              <div className="border-t border-[var(--color-border)] bg-[var(--color-bg-surface)] p-4 flex-shrink-0">
                <div className="max-w-3xl mx-auto">
                  <div
                    className="bg-[var(--color-bg-elevated)] border border-[var(--color-border)] rounded-2xl flex items-end gap-2 px-4 py-3 focus-within:border-[var(--color-border-focus)] transition-colors"
                  >
                    <textarea
                      ref={textareaRef}
                      value={currentQuestion}
                      onChange={(e) => {
                        setCurrentQuestion(e.target.value);
                        growTextarea(e.target);
                      }}
                      onKeyDown={handleKeyDown}
                      placeholder="Ask anything about this document..."
                      rows={1}
                      disabled={isChatting}
                      className="flex-1 bg-transparent text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-hint)] outline-none resize-none leading-relaxed disabled:opacity-50"
                      style={{ minHeight: '24px', maxHeight: '144px' }}
                    />
                    <button
                      onClick={handleSendMessage}
                      disabled={!currentQuestion.trim() || isChatting}
                      className="flex-shrink-0 w-8 h-8 rounded-full bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] flex items-center justify-center text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                      aria-label="Send message"
                    >
                      <ArrowUp size={14} />
                    </button>
                  </div>
                  <p className="text-[10px] text-[var(--color-text-hint)] text-center mt-2 font-mono">
                    Enter to send · Shift+Enter for new line
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