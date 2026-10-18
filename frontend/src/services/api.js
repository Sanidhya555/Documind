/**
 * Enterprise Document Intelligence — API Client v3
 *
 * Backward-compatible with v2 contracts.
 * New in v3:
 *   - query response includes retrieval_decision and evidence_confidence
 *   - /evaluate/security and /evaluate/adaptive endpoints
 *   - metrics response includes adaptive_retrieval_enabled, laya_enabled
 */

const getBaseUrl = () => {
  const saved = localStorage.getItem('api_url');
  if (saved) return saved.replace(/\/+$/, '');
  return import.meta.env.VITE_API_URL || '/api';
};

export const setStoredApiUrl = (url) => {
  if (!url) {
    localStorage.removeItem('api_url');
  } else {
    localStorage.setItem('api_url', url.replace(/\/+$/, ''));
  }
};

export const getStoredApiUrl = getBaseUrl;

const getHeaders = (token, isJson = true) => {
  const headers = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  if (isJson) {
    headers['Content-Type'] = 'application/json';
  }
  return headers;
};

const handleResponse = async (response) => {
  if (!response.ok) {
    let errorDetail = response.statusText;
    try {
      const data = await response.json();
      errorDetail = data.detail || JSON.stringify(data);
    } catch {
      try {
        errorDetail = await response.text();
      } catch {
        // fallback to statusText
      }
    }
    throw new Error(errorDetail || `HTTP Error ${response.status}`);
  }
  return response.json();
};

export const api = {
  // ── Authentication ────────────────────────────────────────────────────────
  login: async (username, password) => {
    const url = `${getBaseUrl()}/login`;
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData,
    });
    return handleResponse(res);
  },

  // ── Health ────────────────────────────────────────────────────────────────
  checkHealth: async () => {
    const res = await fetch(`${getBaseUrl()}/health`);
    return handleResponse(res);
  },

  // ── Document Management ───────────────────────────────────────────────────
  listDocuments: async (token) => {
    const res = await fetch(`${getBaseUrl()}/documents`, {
      headers: getHeaders(token),
    });
    return handleResponse(res);
  },

  uploadDocument: async (file, token) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${getBaseUrl()}/upload`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });
    return handleResponse(res);
  },

  deleteDocument: async (filename, token) => {
    const res = await fetch(
      `${getBaseUrl()}/documents/${encodeURIComponent(filename)}`,
      { method: 'DELETE', headers: getHeaders(token) }
    );
    return handleResponse(res);
  },

  reindexDocument: async (filename, token) => {
    const res = await fetch(
      `${getBaseUrl()}/documents/${encodeURIComponent(filename)}/reindex`,
      { method: 'POST', headers: getHeaders(token) }
    );
    return handleResponse(res);
  },

  // ── Session Management ────────────────────────────────────────────────────
  listSessions: async (token) => {
    const res = await fetch(`${getBaseUrl()}/sessions`, {
      headers: getHeaders(token),
    });
    return handleResponse(res);
  },

  createSession: async (sessionId, token) => {
    const res = await fetch(`${getBaseUrl()}/sessions/new`, {
      method: 'POST',
      headers: getHeaders(token),
      body: JSON.stringify({ session_id: sessionId }),
    });
    return handleResponse(res);
  },

  getSessionMessages: async (sessionId, token) => {
    const res = await fetch(
      `${getBaseUrl()}/sessions/${sessionId}/messages`,
      { headers: getHeaders(token) }
    );
    return handleResponse(res);
  },

  updateSessionTitle: async (sessionId, title, token) => {
    const res = await fetch(`${getBaseUrl()}/sessions/${sessionId}/title`, {
      method: 'PUT',
      headers: getHeaders(token),
      body: JSON.stringify({ session_id: sessionId, title }),
    });
    return handleResponse(res);
  },

  deleteSession: async (sessionId, token) => {
    const res = await fetch(`${getBaseUrl()}/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: getHeaders(token),
    });
    return handleResponse(res);
  },

  // ── Query (synchronous) ───────────────────────────────────────────────────
  query: async ({ question, sessionId, useCache = true }, token) => {
    const res = await fetch(`${getBaseUrl()}/query`, {
      method: 'POST',
      headers: getHeaders(token),
      body: JSON.stringify({
        question,
        session_id: sessionId,
        stream: false,
        use_cache: useCache,
      }),
    });
    return handleResponse(res);
  },

  // ── Query (SSE Streaming) ─────────────────────────────────────────────────
  queryStream: async (
    { question, sessionId, topK = 5, onToken, onError, onDone },
    token
  ) => {
    try {
      const response = await fetch(`${getBaseUrl()}/query`, {
        method: 'POST',
        headers: getHeaders(token),
        body: JSON.stringify({
          question,
          session_id: sessionId,
          top_k: topK,
          stream: true,
          use_cache: false,
        }),
      });

      if (!response.ok) {
        let err = response.statusText;
        try {
          const json = await response.json();
          err = json.detail || err;
        } catch {
          // ignore
        }
        onError && onError(new Error(err));
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('data:')) {
            const dataStr = trimmed.slice(5).trim();
            if (!dataStr) continue;
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.token !== undefined) {
                onToken && onToken(parsed.token);
              } else if (parsed.done) {
                onDone && onDone();
              } else if (parsed.error) {
                onError && onError(new Error(parsed.error));
              }
            } catch {
              onToken && onToken(dataStr);
            }
          }
        }
      }

      onDone && onDone();
    } catch (err) {
      onError && onError(err);
    }
  },

  // ── System Metrics ────────────────────────────────────────────────────────
  getMetrics: async (token) => {
    const res = await fetch(`${getBaseUrl()}/metrics`, {
      headers: getHeaders(token),
    });
    return handleResponse(res);
  },

  // ── Evaluation Suite ──────────────────────────────────────────────────────
  runBatchEvaluation: async (token) => {
    const res = await fetch(`${getBaseUrl()}/evaluate/batch`, {
      method: 'POST',
      headers: getHeaders(token),
    });
    return handleResponse(res);
  },

  runRetrievalEvaluation: async (topK = 5, token) => {
    const res = await fetch(`${getBaseUrl()}/evaluate/retrieval?top_k=${topK}`, {
      method: 'POST',
      headers: getHeaders(token),
    });
    return handleResponse(res);
  },

  /** Legacy security evaluation against evaluation/datasets/eval_set_v2.json. */
  runSecurityEvaluation: async (token) => {
    const res = await fetch(`${getBaseUrl()}/evaluate/security`, {
      method: 'POST',
      headers: getHeaders(token),
    });
    return handleResponse(res);
  },

  /** Legacy adaptive retrieval evaluation against evaluation/datasets/eval_set_v2.json. */
  runAdaptiveEvaluation: async (token) => {
    const res = await fetch(`${getBaseUrl()}/evaluate/adaptive`, {
      method: 'POST',
      headers: getHeaders(token),
    });
    return handleResponse(res);
  },

  runFullEvaluation: async (token) => {
    const res = await fetch(`${getBaseUrl()}/evaluate/all`, {
      method: 'POST',
      headers: getHeaders(token),
    });
    return handleResponse(res);
  },

  runAllTests: async (token) => {
    const res = await fetch(`${getBaseUrl()}/tests/run`, {
      method: 'POST',
      headers: getHeaders(token),
    });
    return handleResponse(res);
  },
};
