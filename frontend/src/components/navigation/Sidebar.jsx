import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import {
  MessageSquarePlus,
  MessageSquare,
  Files,
  BarChart3,
  Trash2,
  Edit2,
  Check,
  X,
  RefreshCw,
  Search,
  Clock,
  ChevronRight,
} from 'lucide-react';

export const Sidebar = () => {
  const {
    activeTab,
    setActiveTab,
    activeSessionId,
    sessions,
    createNewSession,
    loadSessionMessages,
    deleteSession,
    renameSession,
    loadSessions,
    loadingSessions,
  } = useApp();

  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [deletingId, setDeletingId] = useState(null);

  const handleStartEdit = (session, e) => {
    e.stopPropagation();
    setEditingId(session.session_id);
    setEditTitle(session.title || 'New Chat');
  };

  const handleSaveEdit = (sessionId, e) => {
    e.stopPropagation();
    if (editTitle.trim()) {
      renameSession(sessionId, editTitle);
    }
    setEditingId(null);
  };

  const handleCancelEdit = (e) => {
    e.stopPropagation();
    setEditingId(null);
  };

  const handleDelete = (sessionId, e) => {
    e.stopPropagation();
    if (confirm('Are you sure you want to delete this chat session?')) {
      deleteSession(sessionId);
    }
  };

  const filteredSessions = sessions.filter((s) => {
    const q = searchTerm.toLowerCase();
    const title = (s.title || 'New Chat').toLowerCase();
    const id = (s.session_id || '').toLowerCase();
    return title.includes(q) || id.includes(q);
  });

  return (
    <aside className="w-80 h-[calc(100vh-4rem)] bg-slate-900/60 border-r border-slate-800 flex flex-col flex-shrink-0 z-20">
      {/* Top Action: New Chat Button */}
      <div className="p-4 border-b border-slate-800/80">
        <button
          onClick={createNewSession}
          className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-brand-600 to-cyan-600 hover:from-brand-500 hover:to-cyan-500 text-white font-medium text-sm shadow-lg shadow-brand-500/20 flex items-center justify-center gap-2 transition-all active:scale-[0.99]"
        >
          <MessageSquarePlus className="w-4 h-4" />
          <span>New Chat</span>
        </button>

        {/* Primary Navigation Tabs */}
        <div className="mt-4 grid grid-cols-3 gap-1 bg-slate-950/60 p-1 rounded-xl border border-slate-800/80">
          <button
            onClick={() => setActiveTab('chat')}
            className={`flex flex-col items-center justify-center py-2 px-1 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'chat'
                ? 'bg-brand-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <MessageSquare className="w-4 h-4 mb-1" />
            <span>Chat</span>
          </button>

          <button
            onClick={() => setActiveTab('docs')}
            className={`flex flex-col items-center justify-center py-2 px-1 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'docs'
                ? 'bg-brand-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Files className="w-4 h-4 mb-1" />
            <span>Documents</span>
          </button>

          <button
            onClick={() => setActiveTab('metrics')}
            className={`flex flex-col items-center justify-center py-2 px-1 rounded-lg text-xs font-medium transition-all ${
              activeTab === 'metrics'
                ? 'bg-brand-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <BarChart3 className="w-4 h-4 mb-1" />
            <span>Metrics</span>
          </button>
        </div>
      </div>

      {/* Sessions Section Header */}
      <div className="px-4 pt-4 pb-2 flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-400">
        <span>Previous Chats ({sessions.length})</span>
        <button
          onClick={loadSessions}
          disabled={loadingSessions}
          title="Refresh Sessions"
          className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loadingSessions ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Search Input */}
      {sessions.length > 5 && (
        <div className="px-4 pb-2">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search chats..."
              className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-slate-950/40 border border-slate-800 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-brand-500"
            />
          </div>
        </div>
      )}

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5">
        {filteredSessions.length === 0 ? (
          <div className="text-center py-8 px-4 text-xs text-slate-500">
            {searchTerm ? 'No matching chats found.' : 'No saved sessions yet. Start a chat!'}
          </div>
        ) : (
          filteredSessions.map((session) => {
            const isActive = session.session_id === activeSessionId;
            const isEditing = editingId === session.session_id;
            const title = session.title || 'New Chat';
            const created = session.created_at ? session.created_at.slice(0, 10) : '';
            const lastActive = session.last_active ? session.last_active.slice(0, 16).replace('T', ' ') : '';

            return (
              <div
                key={session.session_id}
                onClick={() => {
                  if (!isEditing) {
                    loadSessionMessages(session.session_id);
                    setActiveTab('chat');
                  }
                }}
                className={`group relative p-3 rounded-xl border transition-all cursor-pointer flex flex-col gap-1 ${
                  isActive
                    ? 'bg-slate-800/90 border-brand-500/50 shadow-md ring-1 ring-brand-500/20'
                    : 'bg-slate-950/40 border-slate-800/60 hover:bg-slate-800/50 hover:border-slate-700/60'
                }`}
              >
                {/* Title and Edit Field */}
                <div className="flex items-center justify-between gap-2">
                  {isEditing ? (
                    <div className="flex items-center gap-1 w-full" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="text"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        className="flex-1 px-2 py-0.5 rounded bg-slate-900 border border-brand-500 text-xs text-white focus:outline-none"
                        autoFocus
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveEdit(session.session_id, e);
                          if (e.key === 'Escape') handleCancelEdit(e);
                        }}
                      />
                      <button
                        onClick={(e) => handleSaveEdit(session.session_id, e)}
                        className="p-1 rounded text-emerald-400 hover:bg-emerald-500/10"
                      >
                        <Check className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={handleCancelEdit}
                        className="p-1 rounded text-slate-400 hover:bg-slate-800"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <span className="text-xs font-semibold text-slate-200 truncate flex-1 group-hover:text-white">
                        {title}
                      </span>
                      <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
                        <button
                          onClick={(e) => handleStartEdit(session, e)}
                          title="Rename"
                          className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-700/60"
                        >
                          <Edit2 className="w-3 h-3" />
                        </button>
                        <button
                          onClick={(e) => handleDelete(session.session_id, e)}
                          title="Delete Session"
                          className="p-1 rounded text-rose-400 hover:text-rose-300 hover:bg-rose-500/10"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </>
                  )}
                </div>

                {/* Subtext: ID & Dates */}
                <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                  <span>ID: {session.session_id.slice(0, 8)}</span>
                  <span>{lastActive || created || 'Recent'}</span>
                </div>
              </div>
            );
          })
        )}
      </div>

    </aside>
  );
};
