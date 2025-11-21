import type { LoaderFunctionArgs } from "@remix-run/node";
import { json } from "@remix-run/node";
import { useLoaderData } from "@remix-run/react";
import { useEffect, useState } from "react";

async function fetchRun(apiBase: string, id: string) {
  try {
    const res = await fetch(`${apiBase}/runs/${id}`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function fetchProgress(apiBase: string, id: string) {
  try {
    const res = await fetch(`${apiBase}/runs/${id}/progress`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function stopRun(apiBase: string, id: string) {
  try {
    const res = await fetch(`${apiBase}/runs/${id}/stop`, { method: 'POST' });
    return res.ok;
  } catch {
    return false;
  }
}

export async function loader({ params }: LoaderFunctionArgs) {
  const apiBase = process.env.API_BASE || "http://127.0.0.1:5173";
  const id = params.id as string;
  const data = await fetchRun(apiBase, id);
  return json({ run: data, id, apiBase });
}

export default function RunDetail() {
  const { run: initialRun, id, apiBase } = useLoaderData<typeof loader>();
  const [run, setRun] = useState(initialRun);
  const [progress, setProgress] = useState<any>(null);
  const [stopping, setStopping] = useState(false);
  
  useEffect(() => {
    // Poll for progress if run is in progress
    if (!run || run.status === 'completed' || run.status === 'failed' || run.status === 'stopped') {
      return;
    }
    
    const pollProgress = async () => {
      // Fetch both progress and updated run data
      const [prog, updatedRun] = await Promise.all([
        fetchProgress(apiBase, id),
        fetchRun(apiBase, id)
      ]);
      
      if (prog) {
        setProgress(prog);
        // Reload page if run finished to show final results
        if (prog.status !== 'running') {
          window.location.reload();
        }
      }
      
      // Update run to reflect loading_stage removal
      if (updatedRun) {
        setRun(updatedRun);
      }
    };
    
    // Initial poll
    pollProgress();
    
    // Poll every 2 seconds
    const interval = setInterval(pollProgress, 2000);
    return () => clearInterval(interval);
  }, [apiBase, id, run?.status]);
  
  const handleStop = async () => {
    setStopping(true);
    await stopRun(apiBase, id);
    // Progress will update via polling
  };
  
  if (!run) return <div className="card">Run not found: {id}</div>;
  
  const agg = run.aggregate || {};
  const ds = run.dataset || {};
  const isRunning = run.status === 'running';
  const showProgress = isRunning && (run.loading_stage || progress);
  
  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ marginTop: 0 }}>Run {run.run_id}</h2>
          <p className="muted">
            Status: {run.status || 'completed'} • Stage: {run.stage} • Model: {run.model}
          </p>
        </div>
        {isRunning && (
          <button
            onClick={handleStop}
            disabled={stopping}
            style={{
              padding: '8px 16px',
              backgroundColor: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: stopping ? 'not-allowed' : 'pointer',
              fontWeight: 500,
              opacity: stopping ? 0.6 : 1,
            }}
          >
            {stopping ? 'Stopping...' : 'Stop Run'}
          </button>
        )}
      </div>
      
      
      {showProgress && (
        <div style={{ marginTop: 16, padding: 16, backgroundColor: 'rgba(66, 135, 245, 0.1)', borderRadius: 8, border: '1px solid rgba(66, 135, 245, 0.3)' }}>
          {run.loading_stage ? (
            <div style={{ fontSize: 16, fontWeight: 500 }}>Progress: {run.loading_stage}</div>
          ) : progress ? (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontWeight: 500 }}>Progress: {progress.processed_samples} / {progress.total_samples} samples</span>
                <span style={{ fontWeight: 500 }}>{progress.progress_percent}%</span>
              </div>
              <div style={{ width: '100%', height: 8, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ width: `${progress.progress_percent}%`, height: '100%', backgroundColor: '#4287f5', transition: 'width 0.3s ease' }} />
              </div>
              <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
                <div>
                  <div style={{ fontSize: 12, color: 'var(--fg-steel)' }}>Current Cost</div>
                  <div style={{ fontSize: 16, fontWeight: 600, color: progress.current_cost >= progress.cost_limit_per_run * 0.8 ? '#dc3545' : '#4287f5' }}>
                    ${progress.current_cost.toFixed(4)} / ${progress.cost_limit_per_run}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 12, color: 'var(--fg-steel)' }}>Daily Spend</div>
                  <div style={{ fontSize: 16, fontWeight: 600, color: progress.daily_cost >= progress.cost_limit_per_day * 0.8 ? '#dc3545' : '#28a745' }}>
                    ${progress.daily_cost.toFixed(2)} / ${progress.cost_limit_per_day}
                  </div>
                </div>
              </div>
            </>
          ) : null}
        </div>
      )}
      
      {run.stop_reason && (
        <div style={{ marginTop: 16, padding: 12, backgroundColor: 'rgba(255, 193, 7, 0.1)', borderRadius: 6, border: '1px solid rgba(255, 193, 7, 0.3)', color: '#ffc107' }}>
          <strong>⚠️ Stopped:</strong> {run.stop_reason}
        </div>
      )}
      
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(200px,1fr))', gap:16, marginTop:16 }}>
        <Metric label="Dataset" value={ds.dataset_id || '-'} />
        <Metric label="Samples" value={agg.samples ?? '-'} />
        <Metric label="OK Rate" value={agg.ok_rate != null ? (agg.ok_rate*100).toFixed(1)+'%' : '-'} />
        <Metric label="Mean Err" value={fmt(agg.mean_abs_err_mean)} />
        <Metric label="Min Err" value={fmt(agg.min_abs_err_mean)} />
        <Metric label="Max Err" value={fmt(agg.max_abs_err_mean)} />
        <Metric label="Performance" value={fmt(agg.performance)} />
        <Metric label="Started" value={run.started_at?.replace('T',' ').slice(0,19) || '-'} />
        <Metric label="Ended" value={run.ended_at?.replace('T',' ').slice(0,19) || '-'} />
        <Metric label="Total Tokens" value={agg.total_tokens ?? '-'} />
        <Metric label="Cost Total" value={fmt(agg.cost_total)} />
        <Metric label="Cost/Sample" value={fmt(agg.cost_per_sample)} />
      </div>
      <details style={{ marginTop:24 }}>
        <summary style={{ cursor:'pointer', fontWeight:500 }}>Raw Aggregate JSON</summary>
        <pre style={{ whiteSpace: 'pre-wrap', overflowX:'auto', marginTop:12 }}>{JSON.stringify(run.aggregate, null, 2)}</pre>
      </details>
      <details style={{ marginTop:16 }}>
        <summary style={{ cursor:'pointer', fontWeight:500 }}>Per-sample Results</summary>
        <pre style={{ whiteSpace: 'pre-wrap', overflowX:'auto', marginTop:12 }}>{JSON.stringify(run.results, null, 2)}</pre>
      </details>
    </div>
  );
}

function fmt(v: any){
  if (v == null) return '-';
  if (typeof v === 'number') return v.toFixed(4);
  return String(v);
}

function Metric({label, value}:{label:string; value:any}){
  return (
    <div style={{ padding:12, border:'1px solid var(--fg-border)', borderRadius:8 }}>
      <div style={{ fontSize:12, textTransform:'uppercase', letterSpacing:0.5, color:'var(--fg-steel)' }}>{label}</div>
      <div style={{ marginTop:4, fontSize:16, fontWeight:600, color:'var(--fg-platinum)' }}>{value}</div>
    </div>
  );
}
