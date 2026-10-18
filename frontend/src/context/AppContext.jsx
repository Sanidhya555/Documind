import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useAuth } from './AuthContext';
import { api } from '../services/api';

const AppContext = createContext(null);

const encodeUsername = (username) =>
  encodeURIComponent(username || 'admin').replace(/[!'()*]/g, (char) =>
    `%${char.charCodeAt(0).toString(16).toUpperCase()}`
  );
const sessionPrefix = (username) => `sess_${encodeUsername(username)}_`;
const generateSessionId = (username) =>
  `${sessionPrefix(username)}${Math.random().toString(36).substring(2, 10)}`;

export const AppProvider = ({ children }) => {
  const { token, username, isAuthenticated } = useAuth();
  
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'docs' | 'metrics'
  const [activeSessionId, setActiveSessionId] = useState(() => {
    return localStorage.getItem('last_session_id') || generateSessionId(username);
  });
  const [sessions, setSessions] = useState([]);
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [systemMetrics, setSystemMetrics] = useState(null);
  const [latencyHistory, setLatencyHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('latency_history');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [toasts, setToasts] = useState([]);

  // Toast notification system
  const addToast = useCallback((message, type = 'info') => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4500);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // Save latency history
  useEffect(() => {
    try {
      localStorage.setItem('latency_history', JSON.stringify(latencyHistory.slice(-50)));
    } catch {
      // ignore
    }
  }, [latencyHistory]);

  // Save active session
  useEffect(() => {
    if (activeSessionId) {
      localStorage.setItem('last_session_id', activeSessionId);
    }
  }, [activeSessionId]);

  // Load sessions
  const loadSessions = useCallback(async () => {
    if (!token) return [];
    setLoadingSessions(true);
    try {
      const data = await api.listSessions(token);
      setSessions(Array.isArray(data) ? data : []);
      return data;
    } catch (err) {
      console.error('Failed to load sessions:', err);
      return [];
    } finally {
      setLoadingSessions(false);
    }
  }, [token]);

  // Load messages for a session
  const loadSessionMessages = useCallback(async (sessionId) => {
    if (!token || !sessionId) return;
    setLoadingMessages(true);
    try {
      const data = await api.getSessionMessages(sessionId, token);
      setMessages(Array.isArray(data) ? data : []);
      setActiveSessionId(sessionId);
    } catch (err) {
      console.error(`Failed to load messages for session ${sessionId}:`, err);
      addToast(`Could not load chat history: ${err.message}`, 'error');
    } finally {
      setLoadingMessages(false);
    }
  }, [token, addToast]);

  // Create new session
  const createNewSession = useCallback(async () => {
    const newId = generateSessionId(username);
    if (token) {
      try {
        await api.createSession(newId, token);
      } catch (err) {
        console.warn('API create session warning:', err);
      }
    }
    setActiveSessionId(newId);
    setMessages([]);
    setActiveTab('chat');
    if (token) {
      loadSessions();
    }
    addToast('Started a new chat session', 'info');
    return newId;
  }, [token, username, loadSessions, addToast]);

  // Delete session
  const deleteSession = useCallback(async (sessionId) => {
    if (!token) return;
    try {
      await api.deleteSession(sessionId, token);
      addToast('Session deleted', 'success');
      
      if (activeSessionId === sessionId) {
        const remaining = sessions.filter((s) => s.session_id !== sessionId);
        if (remaining.length > 0) {
          loadSessionMessages(remaining[0].session_id);
        } else {
          createNewSession();
        }
      }
      loadSessions();
    } catch (err) {
      addToast(`Failed to delete session: ${err.message}`, 'error');
    }
  }, [token, activeSessionId, sessions, loadSessionMessages, createNewSession, loadSessions, addToast]);

  // Rename session
  const renameSession = useCallback(async (sessionId, newTitle) => {
    if (!token || !newTitle.trim()) return;
    try {
      await api.updateSessionTitle(sessionId, newTitle.trim(), token);
      addToast('Session renamed', 'success');
      loadSessions();
    } catch (err) {
      addToast(`Failed to rename session: ${err.message}`, 'error');
    }
  }, [token, loadSessions, addToast]);

  // Load documents
  const loadDocuments = useCallback(async () => {
    if (!token) return [];
    setLoadingDocs(true);
    try {
      const data = await api.listDocuments(token);
      setDocuments(Array.isArray(data) ? data : []);
      return data;
    } catch (err) {
      console.error('Failed to load documents:', err);
      addToast(`Failed to load documents: ${err.message}`, 'error');
      return [];
    } finally {
      setLoadingDocs(false);
    }
  }, [token, addToast]);

  // Delete document
  const deleteDocument = useCallback(async (filename) => {
    if (!token) return;
    try {
      await api.deleteDocument(filename, token);
      addToast(`Document "${filename}" removed from index`, 'success');
      loadDocuments();
    } catch (err) {
      addToast(`Failed to delete document: ${err.message}`, 'error');
    }
  }, [token, loadDocuments, addToast]);

  // Load system metrics
  const loadMetrics = useCallback(async () => {
    if (!token) return null;
    try {
      const data = await api.getMetrics(token);
      setSystemMetrics(data);
      return data;
    } catch (err) {
      console.error('Failed to load metrics:', err);
      return null;
    }
  }, [token]);

  // Add latency record
  const recordLatency = useCallback((item) => {
    setLatencyHistory((prev) => [
      ...prev,
      {
        ...item,
        timestamp: item.timestamp || new Date().toLocaleTimeString(),
      },
    ]);
  }, []);

  // Initial data loading when token is available
  useEffect(() => {
    if (isAuthenticated) {
      loadSessions();
      loadDocuments();
      loadMetrics();
      if (activeSessionId) {
        loadSessionMessages(activeSessionId);
      }
    }
  }, [isAuthenticated]);

  // When accounts change in the same browser, start in a session owned by the new user.
  useEffect(() => {
    if (!isAuthenticated || activeSessionId.startsWith(sessionPrefix(username))) return;
    const newId = generateSessionId(username);
    setActiveSessionId(newId);
    setMessages([]);
    api.createSession(newId, token).then(loadSessions).catch((err) => {
      console.warn('API create session warning:', err);
    });
  }, [activeSessionId, isAuthenticated, loadSessions, token, username]);

  return (
    <AppContext.Provider
      value={{
        activeTab,
        setActiveTab,
        activeSessionId,
        setActiveSessionId,
        sessions,
        messages,
        setMessages,
        documents,
        systemMetrics,
        latencyHistory,
        loadingMessages,
        loadingSessions,
        loadingDocs,
        toasts,
        addToast,
        removeToast,
        loadSessions,
        loadSessionMessages,
        createNewSession,
        deleteSession,
        renameSession,
        loadDocuments,
        deleteDocument,
        loadMetrics,
        recordLatency,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return ctx;
};
