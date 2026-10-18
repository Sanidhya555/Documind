import React, { useState } from 'react';
import { Modal } from '../common/Modal';
import { getStoredApiUrl, setStoredApiUrl, api } from '../../services/api';
import { Server, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';
import { useApp } from '../../context/AppContext';

export const SettingsModal = ({ isOpen, onClose }) => {
  const { addToast } = useApp();
  const [url, setUrl] = useState(() => getStoredApiUrl());
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch(`${url.replace(/\/+$/, '')}/health`);
      if (res.ok) {
        const data = await res.json();
        setTestResult({ success: true, message: `Connected! Indexed documents: ${data.documents_indexed || 0}` });
      } else {
        setTestResult({ success: false, message: `HTTP Error ${res.status}: ${res.statusText}` });
      }
    } catch (err) {
      setTestResult({ success: false, message: `Connection failed: ${err.message}` });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = () => {
    setStoredApiUrl(url);
    addToast('API URL settings saved', 'success');
    onClose();
  };

  const handleReset = () => {
    const defaultUrl = 'http://localhost:8000/api';
    setUrl(defaultUrl);
    setStoredApiUrl(defaultUrl);
    addToast('Reset to default API URL', 'info');
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="API & Backend Settings">
      <div className="space-y-4 text-xs">
        <div>
          <label className="block text-slate-400 font-semibold uppercase tracking-wider mb-2">
            Backend API URL
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="http://localhost:8000/api"
              className="flex-1 px-3 py-2 rounded-xl glass-input text-white text-xs font-mono focus:outline-none"
            />
            <button
              onClick={handleTest}
              disabled={testing}
              className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium transition-colors flex items-center gap-1.5"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${testing ? 'animate-spin' : ''}`} />
              <span>Test</span>
            </button>
          </div>
          <p className="text-[11px] text-slate-500 mt-1">
            Default endpoint: <code className="text-cyan-400 font-mono">http://localhost:8000/api</code>
          </p>
        </div>

        {testResult && (
          <div
            className={`p-3 rounded-xl border flex items-start gap-2.5 ${
              testResult.success
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                : 'bg-rose-500/10 border-rose-500/30 text-rose-300'
            }`}
          >
            {testResult.success ? (
              <CheckCircle2 className="w-4 h-4 flex-shrink-0 mt-0.5 text-emerald-400" />
            ) : (
              <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5 text-rose-400" />
            )}
            <div className="leading-snug">{testResult.message}</div>
          </div>
        )}

        <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1.5 text-slate-400">
          <div className="font-semibold text-slate-300">Backend Architecture:</div>
          <div className="grid grid-cols-2 gap-1 text-[11px]">
            <div>• Ingestion: <span className="text-slate-200">Docling (PDF, Office, images)</span></div>
            <div>• Embeddings: <span className="text-slate-200">MiniLM-L6-v2 (384d)</span></div>
            <div>• Vector Store: <span className="text-slate-200">FAISS index</span></div>
            <div>• Keyword Search: <span className="text-slate-200">BM25 Retriever</span></div>
            <div>• Reranker: <span className="text-slate-200">Cross-Encoder</span></div>
            <div>• LLM Backend: <span className="text-slate-200">Ollama / OpenAI</span></div>
          </div>
        </div>

        <div className="flex items-center justify-between pt-3 border-t border-slate-800">
          <button
            onClick={handleReset}
            className="text-slate-400 hover:text-white transition-colors"
          >
            Reset Default
          </button>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-3.5 py-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-medium shadow-md shadow-brand-500/20 transition-colors"
            >
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </Modal>
  );
};
