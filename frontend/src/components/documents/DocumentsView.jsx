import React, { useState, useRef } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import {
  UploadCloud,
  FileText,
  FileCheck2,
  Trash2,
  RefreshCw,
  Search,
  AlertCircle,
  CheckCircle2,
  Layers,
  Calendar,
  AlertTriangle,
} from 'lucide-react';
import { StatCard } from '../common/StatCard';

export const DocumentsView = () => {
  const { token } = useAuth();
  const { documents, loadDocuments, deleteDocument, loadingDocs, addToast } = useApp();

  const [dragOver, setDragOver] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState([]); // [{ name, status: 'uploading' | 'success' | 'error', message, chunks }]
  const [searchTerm, setSearchTerm] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(null); // filename to delete
  const [reindexing, setReindexing] = useState(null);
  const fileInputRef = useRef(null);

  const handleFiles = async (fileList) => {
    if (!fileList || fileList.length === 0) return;

    const files = Array.from(fileList);
    const validExtensions = ['.pdf', '.docx', '.xlsx', '.pptx', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'];

    for (const file of files) {
      const ext = '.' + file.name.split('.').pop().toLowerCase();
      if (!validExtensions.includes(ext)) {
        addToast(`Unsupported file: ${file.name}. Use PDF, DOCX, XLSX, PPTX, or a supported image.`, 'error');
        continue;
      }

      if (file.size > 10 * 1024 * 1024) {
        addToast(`File ${file.name} exceeds 10 MB limit.`, 'error');
        continue;
      }

      // Add to uploading queue
      const itemKey = `${file.name}-${Date.now()}`;
      setUploadingFiles((prev) => [
        { key: itemKey, name: file.name, status: 'uploading', message: 'Parsing with Docling & indexing embeddings...' },
        ...prev,
      ]);

      try {
        const result = await api.uploadDocument(file, token);
        setUploadingFiles((prev) =>
          prev.map((item) =>
            item.key === itemKey
              ? {
                  ...item,
                  status: 'success',
                  message: `Indexed successfully (${result.chunks_added || 0} chunks added)`,
                  chunks: result.chunks_added,
                }
              : item
          )
        );
        addToast(`Successfully indexed ${file.name}`, 'success');
        loadDocuments();
      } catch (err) {
        setUploadingFiles((prev) =>
          prev.map((item) =>
            item.key === itemKey
              ? {
                  ...item,
                  status: 'error',
                  message: err.message || 'Indexing failed',
                }
              : item
          )
        );
        addToast(`Failed to index ${file.name}: ${err.message}`, 'error');
      }
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDeleteDoc = async (filename) => {
    await deleteDocument(filename);
    setConfirmDelete(null);
  };

  const handleReindexDoc = async (filename) => {
    setReindexing(filename);
    try {
      const result = await api.reindexDocument(filename, token);
      await loadDocuments();
      addToast(`Reindexed ${filename} (${result.chunks_added} chunks)`, 'success');
    } catch (err) {
      addToast(`Could not reindex ${filename}: ${err.message}`, 'error');
    } finally {
      setReindexing(null);
    }
  };

  const totalChunks = documents.reduce((acc, doc) => acc + (doc.chunks || 0), 0);

  const filteredDocs = documents.filter((doc) =>
    (doc.filename || '').toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 space-y-6 max-w-6xl mx-auto">
      {/* Title & Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Document Library</h1>
          <p className="text-sm text-slate-400 mt-1">
            Upload PDFs or DOCX files for automatic text extraction, recursive chunking, and FAISS indexing.
          </p>
        </div>
        <button
          onClick={loadDocuments}
          disabled={loadingDocs}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 hover:text-white text-xs font-medium transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loadingDocs ? 'animate-spin' : ''}`} />
          <span>Refresh List</span>
        </button>
      </div>

      {/* KPI Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <StatCard
          title="Indexed Documents"
          value={documents.length}
          subtext="Active in corpus"
          icon={FileCheck2}
          color="blue"
        />
        <StatCard
          title="Total Text Chunks"
          value={totalChunks}
          subtext="Dimension: 384 embeddings"
          icon={Layers}
          color="cyan"
        />
        <StatCard
          title="Supported Formats"
          value="PDF, DOCX, XLSX, PPTX, Images"
          subtext="Max file size 10 MB"
          icon={FileText}
          color="emerald"
        />
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-3xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-200 ${
          dragOver
            ? 'border-brand-500 bg-brand-500/10 scale-[1.01]'
            : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/60'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.xlsx,.pptx,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.openxmlformats-officedocument.presentationml.presentation,image/png,image/jpeg,image/tiff,image/bmp,image/webp"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="w-16 h-16 rounded-2xl bg-brand-500/10 border border-brand-500/20 text-brand-400 mx-auto flex items-center justify-center mb-4">
          <UploadCloud className="w-8 h-8" />
        </div>
        <h3 className="text-base font-semibold text-white mb-1">
          Drag & drop your enterprise documents here
        </h3>
        <p className="text-xs text-slate-400 max-w-sm mx-auto mb-4">
          Supports PDF, DOCX, XLSX, PPTX, and PNG, JPEG, TIFF, BMP, or WEBP images up to 10 MB each.
        </p>
        <button
          type="button"
          className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white text-xs font-medium border border-slate-700/80 transition-colors pointer-events-none"
        >
          Select Files from Computer
        </button>
      </div>

      {/* Uploading Queue Feedback */}
      {uploadingFiles.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Upload Progress & Status
          </h4>
          <div className="space-y-2">
            {uploadingFiles.map((file) => (
              <div
                key={file.key}
                className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-between text-xs"
              >
                <div className="flex items-center gap-3 truncate">
                  <FileText className="w-4 h-4 text-brand-400 flex-shrink-0" />
                  <span className="font-medium text-slate-200 truncate">{file.name}</span>
                  <span className="text-slate-400 text-[11px] truncate">{file.message}</span>
                </div>
                <div className="flex-shrink-0">
                  {file.status === 'uploading' && (
                    <div className="w-4 h-4 border-2 border-brand-500/30 border-t-brand-500 rounded-full animate-spin" />
                  )}
                  {file.status === 'success' && (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  )}
                  {file.status === 'error' && (
                    <AlertCircle className="w-4 h-4 text-rose-400" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Document Library Table */}
      <div className="space-y-4 pt-2">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <h3 className="text-lg font-semibold text-white">Indexed Documents ({filteredDocs.length})</h3>

          {documents.length > 3 && (
            <div className="relative w-full sm:w-64">
              <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Filter documents..."
                className="w-full pl-8 pr-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-brand-500"
              />
            </div>
          )}
        </div>

        {documents.length === 0 ? (
          <div className="p-12 text-center border border-slate-800 rounded-2xl bg-slate-900/30">
            <FileText className="w-10 h-10 text-slate-600 mx-auto mb-3" />
            <h4 className="text-sm font-semibold text-slate-300">No documents indexed yet</h4>
            <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1">
              Upload supported documents or images above to start asking questions against your knowledge base.
            </p>
          </div>
        ) : (
          <div className="border border-slate-800 rounded-2xl overflow-hidden bg-slate-900/40">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/80 border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider text-[11px]">
                  <tr>
                    <th className="py-3 px-4">Filename</th>
                    <th className="py-3 px-4">Chunks</th>
                    <th className="py-3 px-4 hidden sm:table-cell">Uploaded At</th>
                    <th className="py-3 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {filteredDocs.map((doc) => {
                    const isDocx = doc.filename.toLowerCase().endsWith('.docx');
                    const uploadDate = doc.uploaded_at
                      ? new Date(doc.uploaded_at).toLocaleString()
                      : '—';

                    return (
                      <tr key={doc.filename} className="hover:bg-slate-800/40 transition-colors">
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2.5">
                            <div
                              className={`p-1.5 rounded-lg ${
                                isDocx
                                  ? 'bg-blue-500/10 text-blue-400'
                                  : 'bg-rose-500/10 text-rose-400'
                              }`}
                            >
                              <FileText className="w-4 h-4" />
                            </div>
                            <span className="font-medium text-slate-200">{doc.filename}</span>
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <span className="px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 font-mono text-[11px]">
                            {doc.chunks ?? 0} chunks
                          </span>
                        </td>
                        <td className="py-3 px-4 text-slate-400 font-mono text-[11px] hidden sm:table-cell">
                          {uploadDate}
                        </td>
                        <td className="py-3 px-4 text-right">
                          <button
                            onClick={() => handleReindexDoc(doc.filename)}
                            disabled={reindexing !== null}
                            title="Re-extract this file and rebuild its chunks and embeddings"
                            className="p-1.5 rounded-lg text-slate-400 hover:text-cyan-300 hover:bg-cyan-500/10 transition-colors disabled:opacity-50"
                          >
                            <RefreshCw className={`w-4 h-4 ${reindexing === doc.filename ? 'animate-spin' : ''}`} />
                          </button>
                          <button
                            onClick={() => setConfirmDelete(doc.filename)}
                            title="Delete document"
                            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {confirmDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 max-w-md w-full shadow-2xl">
            <div className="flex items-center gap-3 text-rose-400 mb-3">
              <AlertTriangle className="w-6 h-6 flex-shrink-0" />
              <h3 className="text-base font-semibold text-white">Delete Document</h3>
            </div>
            <p className="text-xs text-slate-300 mb-6 leading-relaxed">
              Are you sure you want to remove <strong className="text-white font-mono">{confirmDelete}</strong> from the index? This will rebuild the FAISS vector index and BM25 corpus without this file.
            </p>
            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setConfirmDelete(null)}
                className="px-4 py-2 rounded-xl text-xs font-medium text-slate-300 hover:bg-slate-800 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeleteDoc(confirmDelete)}
                className="px-4 py-2 rounded-xl text-xs font-medium bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/20 transition-all"
              >
                Confirm Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
