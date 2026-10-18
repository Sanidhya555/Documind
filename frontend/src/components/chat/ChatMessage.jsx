import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  User,
  Bot,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Zap,
  ShieldCheck,
  Search,
  Sparkles,
  Layers,
  AlertTriangle,
  HelpCircle,
  XCircle,
  CheckCircle2,
} from 'lucide-react';
import { SourceCard } from './SourceCard';

// Decision badge styling
const DECISION_CONFIG = {
  answer:        { label: 'Answered',    color: 'emerald', Icon: CheckCircle2 },
  abstain:       { label: 'Abstained',   color: 'amber',   Icon: AlertTriangle },
  clarify:       { label: 'Clarify',     color: 'blue',    Icon: HelpCircle },
  retrieve_more: { label: 'Expanded',    color: 'purple',  Icon: Search },
  blocked_by_guardrails: { label: 'Blocked', color: 'rose', Icon: XCircle },
};

const DecisionBadge = ({ decision }) => {
  const cfg = DECISION_CONFIG[decision] || DECISION_CONFIG['answer'];
  const { label, color, Icon } = cfg;
  const colorMap = {
    emerald: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300',
    amber:   'bg-amber-500/10  border-amber-500/20  text-amber-300',
    blue:    'bg-blue-500/10   border-blue-500/20   text-blue-300',
    purple:  'bg-purple-500/10 border-purple-500/20 text-purple-300',
    rose:    'bg-rose-500/10   border-rose-500/20   text-rose-300',
  };
  return (
    <span className={`px-2 py-0.5 rounded-full border text-[10px] font-medium flex items-center gap-1 ${colorMap[color]}`}>
      <Icon className="w-2.5 h-2.5" />
      <span>{label}</span>
    </span>
  );
};

export const ChatMessage = ({ message }) => {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);
  const [showSources, setShowSources] = useState(false);

  const copyContent = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const sources = message.sources || [];
  const contexts = message.contexts || [];
  const expandedQueries = message.expanded_queries || [];

  return (
    <div
      className={`py-4 px-4 sm:px-6 rounded-2xl transition-all ${
        isUser
          ? 'bg-slate-900/40 border border-slate-800/80 ml-auto max-w-3xl'
          : 'bg-slate-900/80 border border-slate-800 mr-auto max-w-4xl shadow-lg'
      }`}
    >
      {/* Header: Role icon & badges */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div
            className={`w-7 h-7 rounded-lg flex items-center justify-center text-white ${
              isUser
                ? 'bg-gradient-to-tr from-brand-600 to-cyan-600'
                : 'bg-gradient-to-tr from-purple-600 to-indigo-600'
            }`}
          >
            {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
          </div>
          <span className="text-xs font-semibold text-slate-300">
            {isUser ? 'You' : 'Document Intelligence AI'}
          </span>
          {message.cached && (
            <span className="px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20 text-[10px] font-medium flex items-center gap-1">
              <Zap className="w-2.5 h-2.5 fill-amber-400" />
              <span>Cached</span>
            </span>
          )}
          {!isUser && message.retrieval_decision && (
            <DecisionBadge decision={message.retrieval_decision} />
          )}
        </div>

        {/* Copy button */}
        {!isUser && (
          <button
            onClick={copyContent}
            title="Copy message"
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        )}
      </div>

      {/* Message Content with Markdown */}
      <div className="prose prose-invert prose-sm max-w-none text-slate-200 leading-relaxed overflow-x-auto">
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({ node, ...props }) => (
                <div className="overflow-x-auto my-3 rounded-lg border border-slate-800">
                  <table className="min-w-full divide-y divide-slate-800 text-xs text-left" {...props} />
                </div>
              ),
              th: ({ node, ...props }) => (
                <th className="bg-slate-800/80 px-3 py-2 font-semibold text-slate-200" {...props} />
              ),
              td: ({ node, ...props }) => (
                <td className="px-3 py-2 text-slate-300 border-t border-slate-800/60" {...props} />
              ),
              pre: ({ node, ...props }) => (
                <pre className="bg-slate-950 p-4 rounded-xl border border-slate-800 overflow-x-auto font-mono text-xs my-3" {...props} />
              ),
              code: ({ node, inline, ...props }) =>
                inline ? (
                  <code className="bg-slate-800 px-1.5 py-0.5 rounded text-cyan-300 font-mono text-xs" {...props} />
                ) : (
                  <code {...props} />
                ),
            }}
          >
            {message.content}
          </ReactMarkdown>
        )}
      </div>

      {/* Assistant Extra: Query Expansions */}
      {!isUser && expandedQueries.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-800/60">
          <div className="text-[11px] text-slate-400 font-medium mb-1.5 flex items-center gap-1">
            <Search className="w-3 h-3 text-cyan-400" />
            <span>Multi-Query Expansions ({expandedQueries.length}):</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {expandedQueries.map((q, idx) => (
              <span
                key={idx}
                className="px-2 py-0.5 rounded-lg bg-slate-800/60 border border-slate-700/50 text-[11px] text-slate-300 font-mono"
              >
                "{q}"
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Assistant Extra: Retrieved Sources */}
      {!isUser && sources.length > 0 && (
        <div className="mt-2 pt-2 border-t border-slate-800/60">
          <button
            type="button"
            onClick={() => setShowSources(!showSources)}
            className="flex items-center justify-between w-full text-xs text-slate-400 hover:text-white transition-colors py-1"
          >
            <span className="flex items-center gap-1.5 font-medium">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Retrieved Sources ({sources.length})</span>
            </span>
            {showSources ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>

          {showSources && (
            <div className="mt-2 space-y-2">
              {sources.map((src, i) => (
                <SourceCard key={i} source={src} contextText={contexts[i]} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
