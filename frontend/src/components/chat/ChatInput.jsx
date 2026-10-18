import React, { useEffect, useRef, useState } from 'react';
import { Send } from 'lucide-react';

export const ChatInput = ({ onSend, loading, disabled }) => {
  const [prompt, setPrompt] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [prompt]);

  const handleSubmit = (event) => {
    event?.preventDefault();
    if (!prompt.trim() || loading || disabled) return;
    onSend({ prompt: prompt.trim() });
    setPrompt('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-4">
      <div className="flex items-center justify-between px-2 pb-2 text-[11px] text-slate-500">
        <span>Context size adapts to relevance scores · answers are cached</span>
        <span className="hidden sm:block">
          <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">Enter</kbd> to send ·{' '}
          <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">Shift+Enter</kbd> for newline
        </span>
      </div>

      <div className="relative flex items-end gap-2 bg-slate-900/90 border border-slate-700/80 rounded-2xl p-2.5 shadow-2xl focus-within:border-brand-500/80 focus-within:ring-2 focus-within:ring-brand-500/20 transition-all">
        <textarea
          ref={textareaRef}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about your documents..."
          rows={1}
          disabled={loading || disabled}
          className="flex-1 max-h-44 min-h-[44px] bg-transparent resize-none border-0 px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-0 leading-relaxed font-sans"
        />
        <button
          onClick={handleSubmit}
          disabled={!prompt.trim() || loading || disabled}
          className={`p-3 rounded-xl flex items-center justify-center transition-all ${
            prompt.trim() && !loading && !disabled
              ? 'bg-gradient-to-r from-brand-600 to-cyan-600 text-white shadow-lg shadow-brand-500/25 hover:from-brand-500 hover:to-cyan-500 active:scale-95'
              : 'bg-slate-800 text-slate-500 cursor-not-allowed'
          }`}
        >
          {loading ? (
            <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  );
};
