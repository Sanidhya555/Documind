import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp, Hash, Percent } from 'lucide-react';

export const SourceCard = ({ source, contextText }) => {
  const [isOpen, setIsOpen] = useState(false);

  const filename = source?.filename || 'Document';
  const chunkId = source?.chunk_id ?? 0;
  const score = source?.score !== undefined ? Number(source.score).toFixed(3) : null;

  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-900/60 overflow-hidden text-xs transition-all">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-slate-800/50 transition-colors"
      >
        <div className="flex items-center gap-2 truncate">
          <FileText className="w-3.5 h-3.5 text-brand-400 flex-shrink-0" />
          <span className="font-medium text-slate-200 truncate">{filename}</span>
          <span className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-slate-400 font-mono flex items-center gap-0.5">
            <Hash className="w-2.5 h-2.5" />
            <span>{chunkId}</span>
          </span>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          {score !== null && (
            <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 text-[10px] font-mono">
              Score: {score}
            </span>
          )}
          {isOpen ? (
            <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          )}
        </div>
      </button>

      {isOpen && contextText && (
        <div className="p-3 bg-slate-950/60 border-t border-slate-800 text-slate-300 font-mono text-[11px] leading-relaxed max-h-48 overflow-y-auto whitespace-pre-wrap select-text">
          {contextText}
        </div>
      )}
    </div>
  );
};
