import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { Sparkles, MessageSquare, AlertCircle, FileSearch, HelpCircle } from 'lucide-react';

export const ChatView = () => {
  const { token } = useAuth();
  const {
    activeSessionId,
    messages,
    setMessages,
    recordLatency,
    loadSessions,
    loadMetrics,
    addToast,
    loadingMessages,
  } = useApp();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async ({ prompt }) => {
    if (!prompt.trim() || loading) return;

    setError(null);
    const userMsg = {
      role: 'user',
      content: prompt,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const data = await api.query(
        {
          question: prompt,
          sessionId: activeSessionId,
          useCache: true,
        },
        token
      );

      const assistantMsg = {
        role: 'assistant',
        content: data.answer || 'No answer returned from RAG engine.',
        sources: data.sources || [],
        contexts: data.contexts || [],
        expanded_queries: data.expanded_queries || [],
        retrieval_decision: data.retrieval_decision || 'answer',
        evidence_confidence: data.evidence_confidence,
        metrics: data.metrics || {},
        cached: data.cached || false,
        latency_ms: data.latency_ms,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMsg]);

      if (data.metrics) {
        recordLatency({
          total_latency: data.metrics.total_latency || 0,
          retrieval_time: data.metrics.retrieval_time || 0,
          generation_time: data.metrics.generation_time || 0,
          faithfulness_score: data.metrics.faithfulness_score,
          cached: data.cached || false,
          question: prompt,
        });
      }

      // Refresh dashboard counters without holding the answer spinner open for
      // a second metrics request after the query itself has completed.
      void loadMetrics();
      setTimeout(() => loadSessions(), 1000);
    } catch (err) {
      console.error('Query error:', err);
      setError(err.message || 'Failed to complete query. Ensure documents are indexed.');
      addToast(`Query error: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const starterPrompts = [
    'What are the key conclusions and insights from the indexed documents?',
    'Explain the high-level architecture and components described in the text.',
    'List all core methodologies, datasets, or evaluation metrics mentioned.',
    'Summarize the security, guardrails, and compliance guidelines.',
  ];

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-4rem)] overflow-hidden bg-slate-950">
      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
        {loadingMessages ? (
          <div className="h-full flex items-center justify-center text-slate-400">
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 border-2 border-brand-500/30 border-t-brand-500 rounded-full animate-spin" />
              <span className="text-sm font-medium">Loading conversation history...</span>
            </div>
          </div>
        ) : messages.length === 0 ? (
          /* Empty Chat View with Starter Suggestions */
          <div className="h-full max-w-2xl mx-auto flex flex-col items-center justify-center text-center px-4 py-8">
            <div className="w-16 h-16 rounded-3xl bg-gradient-to-tr from-brand-600/30 to-cyan-500/30 border border-brand-500/30 flex items-center justify-center mb-6 shadow-xl shadow-brand-500/10">
              <FileSearch className="w-8 h-8 text-brand-400" />
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-white mb-2">
              Enterprise Document Intelligence
            </h2>
            <p className="text-sm text-slate-400 mb-8 max-w-md">
              Ask questions across indexed PDFs, Word and Office documents, and supported images. Answers are grounded in indexed text using hybrid retrieval.
            </p>

            <div className="w-full grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-left">
              {starterPrompts.map((promptText, i) => (
                <button
                  key={i}
                  onClick={() => handleSend({ prompt: promptText })}
                  className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-brand-500/50 hover:bg-slate-800/60 transition-all text-xs text-slate-300 hover:text-white flex items-start gap-2.5 group"
                >
                  <Sparkles className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5 group-hover:scale-110 transition-transform" />
                  <span className="leading-snug">{promptText}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* Rendered Message List */
          <div className="max-w-4xl mx-auto space-y-5">
            {messages.map((msg, index) => (
              <ChatMessage key={index} message={msg} />
            ))}

            {loading && !messages.some((m) => m.streaming) && (
              <div className="py-4 px-6 rounded-2xl bg-slate-900/80 border border-slate-800 mr-auto max-w-sm flex items-center gap-3">
                <div className="w-4 h-4 border-2 border-brand-500/30 border-t-brand-500 rounded-full animate-spin" />
                <span className="text-xs text-slate-400">Searching index & synthesizing response...</span>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="max-w-4xl mx-auto p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-3">
            <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
            <div className="flex-1">{error}</div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <ChatInput onSend={handleSend} loading={loading} />
    </div>
  );
};
