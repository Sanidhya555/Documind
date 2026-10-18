import React, { useEffect, useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import { Activity, CheckCircle, Clock, Play, RefreshCw, XCircle } from 'lucide-react';

const percent = (value) => (Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—');
const seconds = (value) => (Number.isFinite(value) ? `${value.toFixed(3)}s` : '—');
const display = (value) => (typeof value === 'string' ? value : JSON.stringify(value));

const MetricCard = ({ title, value, detail, accent = 'text-cyan-300' }) => (
  <div className="p-4 rounded-2xl bg-slate-950/80 border border-slate-800 min-w-0">
    <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold block mb-1">{title}</span>
    <div className={`text-2xl font-bold font-mono ${accent}`}>{value}</div>
    {detail && <div className="text-[11px] text-slate-500 mt-1">{detail}</div>}
  </div>
);

export const MetricsView = () => {
  const { token } = useAuth();
  const { systemMetrics, loadMetrics, latencyHistory, addToast } = useApp();
  const [loadingMetrics, setLoadingMetrics] = useState(false);
  const [runningEvaluation, setRunningEvaluation] = useState(false);
  const [evaluation, setEvaluation] = useState(null);

  useEffect(() => {
    loadMetrics();
  }, [loadMetrics]);

  const handleRefreshMetrics = async () => {
    setLoadingMetrics(true);
    await loadMetrics();
    setLoadingMetrics(false);
  };

  const handleRunEvaluation = async () => {
    setRunningEvaluation(true);
    try {
      const result = await api.runFullEvaluation(token);
      setEvaluation(result);
      await loadMetrics();
      const passed = result.suites?.stage_evaluations?.passed ?? 0;
      const failed = result.suites?.stage_evaluations?.failed ?? 0;
      addToast(
        `Stage evaluation finished: ${passed} passed, ${failed} failed`,
        failed === 0 && result.status === 'passed' ? 'success' : 'error'
      );
    } catch (error) {
      setEvaluation({ status: 'error', error: error.message, suites: {} });
      addToast(`Evaluation failed: ${error.message}`, 'error');
    } finally {
      setRunningEvaluation(false);
    }
  };

  const recentHistory = latencyHistory.slice(-20);
  const maxLatency = Math.max(...recentHistory.map((row) => row.total_latency || 0), 0.01);
  const suites = evaluation?.suites || {};
  const stageEvaluation = suites.stage_evaluations || {};
  const stageResults = stageEvaluation.stages || [];
  const integrity = suites.dataset_integrity || {};
  const cacheEntries = systemMetrics?.query_cache_size ?? 0;
  const cacheDetail = `${systemMetrics?.query_cache_hits ?? 0} hits · ${systemMetrics?.query_cache_misses ?? 0} misses`;

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 space-y-7 max-w-6xl mx-auto">
      <header className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Performance Metrics</h1>
          <p className="text-sm text-slate-400 mt-1">System status, per-query performance, and a single evaluation across every pipeline stage.</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRefreshMetrics}
            disabled={loadingMetrics}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 text-xs font-medium transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loadingMetrics ? 'animate-spin' : ''}`} />
            Refresh metrics
          </button>
          <button
            onClick={handleRunEvaluation}
            disabled={runningEvaluation}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-600 hover:bg-brand-500 disabled:opacity-60 text-white text-xs font-semibold shadow transition-colors"
          >
            {runningEvaluation ? (
              <><span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Running all stages…</>
            ) : (
              <><Play className="w-3.5 h-3.5 fill-current" /> Run full evaluation</>
            )}
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <MetricCard title="Indexed documents" value={systemMetrics?.documents_indexed ?? '—'} detail={`${systemMetrics?.total_chunks ?? 0} chunks`} />
        <MetricCard title="Active sessions" value={systemMetrics?.active_sessions ?? '—'} accent="text-indigo-300" />
        <MetricCard title="Cache entries" value={cacheEntries} detail={cacheDetail} accent="text-amber-300" />
      </div>

      {evaluation && (
        <section className="space-y-4">
          <div className={`p-4 rounded-2xl border flex flex-wrap items-center justify-between gap-3 ${evaluation.status === 'passed' ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-rose-500/10 border-rose-500/30'}`}>
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              {evaluation.status === 'passed'
                ? <CheckCircle className="w-4 h-4 text-emerald-400" />
                : <XCircle className="w-4 h-4 text-rose-400" />}
              Full stage evaluation: {evaluation.status}
            </div>
            <div className="text-xs text-slate-300">
              {stageEvaluation.stage_count ?? 0} stages · {stageEvaluation.total_cases ?? 0} cases · {seconds(evaluation.duration_seconds)}
            </div>
          </div>

          {evaluation.error && <p className="text-sm text-rose-300">{evaluation.error}</p>}
          {stageEvaluation.note && <p className="text-xs text-slate-400">{stageEvaluation.note}</p>}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {stageResults.map((stage) => (
              <details key={stage.name} className="group rounded-2xl border border-slate-800 bg-slate-900/50 overflow-hidden">
                <summary className="list-none cursor-pointer p-4 flex items-center justify-between gap-3 hover:bg-slate-800/40">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 text-sm font-semibold text-white capitalize">
                      {stage.status === 'passed'
                        ? <CheckCircle className="w-4 h-4 shrink-0 text-emerald-400" />
                        : <XCircle className="w-4 h-4 shrink-0 text-rose-400" />}
                      {stage.name.replaceAll('_', ' ')}
                    </div>
                    <p className="mt-1 text-[11px] text-slate-500 truncate">{stage.note}</p>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="font-mono text-xs text-slate-200">{stage.passed}/{stage.case_count}</div>
                    <div className={`text-[10px] ${stage.failed ? 'text-rose-300' : 'text-emerald-300'}`}>
                      {stage.failed ? `${stage.failed} failed` : 'all passed'} · {seconds(stage.duration_seconds)}
                    </div>
                  </div>
                </summary>
                {stage.error && <p className="px-4 pb-3 text-xs text-rose-300">{stage.error}</p>}
                <div className="max-h-72 overflow-auto border-t border-slate-800">
                  <table className="w-full text-left text-[10px]">
                    <thead className="sticky top-0 bg-slate-950 text-slate-500 uppercase">
                      <tr><th className="p-2">Case</th><th className="p-2">Result</th><th className="p-2">Expected</th><th className="p-2">Actual</th><th className="p-2">Time</th></tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/80">
                      {(stage.cases || []).map((item) => (
                        <tr key={item.id} className="align-top hover:bg-slate-800/30">
                          <td className="p-2 font-mono text-slate-400">{item.id}</td>
                          <td className={`p-2 ${item.status === 'passed' ? 'text-emerald-300' : 'text-rose-300'}`}>{item.status}</td>
                          <td className="p-2 text-slate-500 max-w-36 break-words">{display(item.expected)}</td>
                          <td className="p-2 text-slate-300 max-w-36 break-words">{item.error || display(item.actual)}</td>
                          <td className="p-2 font-mono text-slate-500">{item.duration_ms?.toFixed(2)}ms</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            ))}
          </div>

          {integrity.datasets?.length > 0 && (
            <details className="rounded-2xl border border-slate-800 bg-slate-900/40">
              <summary className="cursor-pointer px-4 py-3 text-xs font-semibold text-slate-300">Dataset integrity · {integrity.num_cases} cases across {integrity.num_datasets} stages</summary>
              <div className="px-4 pb-4 flex flex-wrap gap-2">
                {integrity.datasets.map((item) => (
                  <span key={item.dataset} className={`rounded-lg border px-2 py-1 text-[10px] ${item.status === 'passed' ? 'border-emerald-500/20 text-emerald-300' : 'border-rose-500/20 text-rose-300'}`}>
                    {item.dataset}: {item.num_cases}
                  </span>
                ))}
              </div>
            </details>
          )}
        </section>
      )}

      <section className="p-5 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2"><Activity className="w-5 h-5 text-cyan-400" /> Performance metrics per query</h2>
          <p className="text-xs text-slate-400 mt-1">Each query includes total, retrieval, and generation latency, faithfulness, and cache status.</p>
        </div>

        {recentHistory.length === 0 ? (
          <div className="p-8 text-center text-xs text-slate-500 border border-slate-800 rounded-2xl bg-slate-950/40">Per-query performance will appear here after you ask a question.</div>
        ) : (
          <>
            <div className="p-4 rounded-2xl bg-slate-950/80 border border-slate-800">
              <div className="flex flex-wrap items-center gap-4 text-[11px] text-slate-400 mb-4">
                <span className="flex items-center gap-1.5"><i className="w-2.5 h-2.5 rounded-sm bg-cyan-400" /> Total</span>
                <span className="flex items-center gap-1.5"><i className="w-2.5 h-2.5 rounded-sm bg-brand-500" /> Retrieval</span>
                <span className="flex items-center gap-1.5"><i className="w-2.5 h-2.5 rounded-sm bg-indigo-500" /> Generation</span>
                <span className="ml-auto">Seconds</span>
              </div>
              <div className="h-48 flex items-end gap-1.5 overflow-x-auto border-b border-slate-800 pb-1">
                {recentHistory.map((row, index) => {
                  const total = row.total_latency || 0;
                  const retrievalTime = row.retrieval_time || 0;
                  const generationTime = row.generation_time || 0;
                  const height = (value) => `${Math.max(value > 0 ? 2 : 0, (value / maxLatency) * 100)}%`;
                  return (
                    <div key={`${row.timestamp}-${index}`} className="group relative h-full min-w-8 flex-1 flex items-end justify-center gap-0.5">
                      <div className="absolute bottom-full mb-2 hidden group-hover:block z-10 max-w-72 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-2 text-[10px] text-white shadow-xl">
                        <div className="max-w-64 truncate font-medium">{row.question || 'Query'}</div>
                        <div><b>Total:</b> {seconds(total)}</div>
                        <div><b>Retrieval:</b> {seconds(retrievalTime)}</div>
                        <div><b>Generation:</b> {seconds(generationTime)}</div>
                        <div><b>Faithfulness:</b> {percent(row.faithfulness_score)}</div>
                        {row.cached && <div className="text-amber-300">Cache hit</div>}
                      </div>
                      <div title={`Total ${seconds(total)}`} className="w-2 rounded-t-sm bg-cyan-400" style={{ height: height(total) }} />
                      <div title={`Retrieval ${seconds(retrievalTime)}`} className="w-2 rounded-t-sm bg-brand-500" style={{ height: height(retrievalTime) }} />
                      <div title={`Generation ${seconds(generationTime)}`} className="w-2 rounded-t-sm bg-indigo-500" style={{ height: height(generationTime) }} />
                      <span className="absolute top-full mt-1 text-[9px] font-mono text-slate-500 whitespace-nowrap">{row.timestamp?.slice(0, 5) || `#${index + 1}`}</span>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="border border-slate-800 rounded-xl overflow-auto max-h-80 bg-slate-950/60">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-slate-900 text-slate-400 uppercase text-[10px]">
                  <tr><th className="p-2.5">Query</th><th className="p-2.5">Time</th><th className="p-2.5">Total</th><th className="p-2.5">Retrieval</th><th className="p-2.5">Generation</th><th className="p-2.5">Faithfulness</th><th className="p-2.5">Cache</th></tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80">
                  {[...recentHistory].reverse().map((row, index) => (
                    <tr key={`${row.timestamp}-${index}`} className="align-top hover:bg-slate-800/30">
                      <td className="p-2.5 text-slate-200 min-w-48 max-w-md">{row.question || '—'}</td>
                      <td className="p-2.5 whitespace-nowrap text-slate-500">{row.timestamp}</td>
                      <td className="p-2.5 font-mono text-cyan-300">{seconds(row.total_latency)}</td>
                      <td className="p-2.5 font-mono text-brand-300">{seconds(row.retrieval_time)}</td>
                      <td className="p-2.5 font-mono text-indigo-300">{seconds(row.generation_time)}</td>
                      <td className="p-2.5 font-mono text-emerald-300">{percent(row.faithfulness_score)}</td>
                      <td className="p-2.5 text-amber-300">{row.cached ? 'hit' : 'miss'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  );
};
