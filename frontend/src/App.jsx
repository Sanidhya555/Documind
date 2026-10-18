import React, { useState } from 'react';
import { useAuth } from './context/AuthContext';
import { useApp } from './context/AppContext';
import { LoginPage } from './components/auth/LoginPage';
import { Navbar } from './components/navigation/Navbar';
import { Sidebar } from './components/navigation/Sidebar';
import { ChatView } from './components/chat/ChatView';
import { DocumentsView } from './components/documents/DocumentsView';
import { MetricsView } from './components/metrics/MetricsView';
import { ToastContainer } from './components/common/Toast';
import { SettingsModal } from './components/settings/SettingsModal';

export const App = () => {
  const { isAuthenticated } = useAuth();
  const { activeTab } = useApp();
  const [settingsOpen, setSettingsOpen] = useState(false);

  if (!isAuthenticated) {
    return (
      <>
        <LoginPage onOpenSettings={() => setSettingsOpen(true)} />
        <SettingsModal isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
        <ToastContainer />
      </>
    );
  }

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-950 font-sans text-slate-100 select-none">
      <Navbar onOpenSettings={() => setSettingsOpen(true)} />

      <div className="flex flex-1 overflow-hidden">
        {/* Desktop Sidebar */}
        <div className="hidden lg:block">
          <Sidebar />
        </div>

        {/* Main Content Area */}
        <main className="flex-1 flex flex-col overflow-hidden bg-slate-950 select-text">
          {activeTab === 'chat' && <ChatView />}
          {activeTab === 'docs' && <DocumentsView />}
          {activeTab === 'metrics' && <MetricsView />}
        </main>
      </div>

      <SettingsModal isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <ToastContainer />
    </div>
  );
};

export default App;
