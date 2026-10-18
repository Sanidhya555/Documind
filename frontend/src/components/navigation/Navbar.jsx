import React, { useEffect, useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import {
  FileText,
  Activity,
  LogOut,
  Copy,
  Check,
  Settings,
  MessageSquare,
  Files,
  BarChart3,
  User,
} from 'lucide-react';

export const Navbar = ({ onOpenSettings }) => {
  const { username, logout } = useAuth();
  const { activeSessionId, activeTab, setActiveTab, addToast } = useApp();
  const [isCopied, setIsCopied] = useState(false);
  const [health, setHealth] = useState({ status: 'checking', docs: 0 });

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const data = await api.checkHealth();
        if (mounted) {
          setHealth({ status: 'healthy', docs: data.documents_indexed || 0 });
        }
      } catch {
        if (mounted) {
          setHealth({ status: 'offline', docs: 0 });
        }
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const copySessionId = () => {
    if (!activeSessionId) return;
    navigator.clipboard.writeText(activeSessionId);
    setIsCopied(true);
    addToast('Session ID copied to clipboard', 'info');
    setTimeout(() => setIsCopied(false), 2000);
  };

  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between z-30 sticky top-0">
      {/* Left: Brand & Session */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-brand-600 to-cyan-500 flex items-center justify-center text-white shadow-lg shadow-brand-500/20">
            <FileText className="w-5 h-5" />
          </div>
          <div className="hidden sm:block">
            <div className="text-sm font-bold text-white tracking-tight flex items-center gap-2">
              <span>Enterprise Doc Intelligence</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-brand-500/10 text-brand-400 border border-brand-500/20">
                v2.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Hybrid Search & Ingestion RAG</p>
          </div>
        </div>

        {/* Active Session badge */}
        <div className="hidden md:flex items-center gap-1.5 pl-3 border-l border-slate-800">
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-slate-800/60 border border-slate-700/60 text-xs">
            <span className="text-slate-400">Session:</span>
            <span className="font-mono text-cyan-300 font-medium">
              {activeSessionId ? `${activeSessionId.slice(0, 12)}...` : 'None'}
            </span>
            <button
              onClick={copySessionId}
              title="Copy Session ID"
              className="text-slate-400 hover:text-white p-0.5 rounded transition-colors"
            >
              {isCopied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Middle: Tab Pills (visible on mobile / tablet) */}
      <div className="flex lg:hidden items-center gap-1 bg-slate-800/80 p-1 rounded-xl border border-slate-700/50">
        <button
          onClick={() => setActiveTab('chat')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'chat'
              ? 'bg-brand-600 text-white shadow'
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          <MessageSquare className="w-3.5 h-3.5" />
          <span>Chat</span>
        </button>
        <button
          onClick={() => setActiveTab('docs')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'docs'
              ? 'bg-brand-600 text-white shadow'
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          <Files className="w-3.5 h-3.5" />
          <span>Docs</span>
        </button>
        <button
          onClick={() => setActiveTab('metrics')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
            activeTab === 'metrics'
              ? 'bg-brand-600 text-white shadow'
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Metrics</span>
        </button>
      </div>

      {/* Right: Health pill, Settings, User profile, Logout */}
      <div className="flex items-center gap-3">
        {/* Backend status indicator */}
        <div
          title={health.status === 'healthy' ? `Backend online (${health.docs} docs)` : 'Backend offline'}
          className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border bg-slate-800/40 border-slate-700/60"
        >
          <span
            className={`w-2 h-2 rounded-full ${
              health.status === 'healthy'
                ? 'bg-emerald-400 animate-pulse'
                : health.status === 'offline'
                ? 'bg-rose-400'
                : 'bg-amber-400'
            }`}
          />
          <span className="text-slate-300 capitalize text-[11px]">{health.status}</span>
        </div>

        {/* API Settings */}
        <button
          onClick={onOpenSettings}
          title="API Configuration"
          className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-800 transition-colors"
        >
          <Settings className="w-4 h-4" />
        </button>

        {/* User Pill & Logout */}
        <div className="flex items-center gap-2 pl-2 border-l border-slate-800">
          <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-xl bg-slate-800/40 border border-slate-700/50 text-xs text-slate-300">
            <User className="w-3.5 h-3.5 text-brand-400" />
            <span className="font-medium">{username || 'admin'}</span>
          </div>
          <button
            onClick={logout}
            title="Log out"
            className="p-2 rounded-xl text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 border border-rose-500/20 transition-all flex items-center gap-1 text-xs font-medium"
          >
            <LogOut className="w-4 h-4" />
            <span className="hidden sm:inline">Logout</span>
          </button>
        </div>
      </div>
    </header>
  );
};
