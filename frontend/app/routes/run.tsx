import type { LoaderFunctionArgs, ActionFunctionArgs } from "@remix-run/node";
import { json } from "@remix-run/node";
import { useLoaderData, useNavigation, Form, useActionData } from "@remix-run/react";
import * as React from "react";

async function fetchMetadata(apiBase: string) {
  const [modelsRes, datasetsRes] = await Promise.all([
    fetch(`${apiBase}/metadata/models`),
    fetch(`${apiBase}/metadata/datasets?stage=telemetry_literacy`)
  ]);
  const models = modelsRes.ok ? await modelsRes.json() : { models: [] };
  const datasets = datasetsRes.ok ? await datasetsRes.json() : { datasets: [] };
  return { models: models.models || [], datasets: datasets.datasets || [] };
}

export async function loader({ request }: LoaderFunctionArgs) {
  const apiBase = process.env.API_BASE || "http://127.0.0.1:5173";
  const metadata = await fetchMetadata(apiBase);
  return json({ ...metadata });
}

export async function action({ request }: ActionFunctionArgs) {
  const apiBase = process.env.API_BASE || "http://127.0.0.1:5173";
  const formData = await request.formData();
  const model = String(formData.get("model") || "").trim();
  const dataset_id = String(formData.get("dataset_id") || "").trim();
  const limitRaw = String(formData.get("limit") || "").trim();
  const limit = limitRaw ? Number(limitRaw) : undefined;
  const errors: Record<string,string> = {};
  if (!model) errors.model = "Model required";
  if (!dataset_id) errors.dataset_id = "Dataset required";
  if (limit != null && (isNaN(limit) || limit <= 0)) errors.limit = "Limit must be positive";
  if (Object.keys(errors).length) return json({ ok:false, errors }, { status: 400 });
  const body = {
    stage: "telemetry_literacy",
    model,
    dataset_source: dataset_id.startsWith("hf_") ? "hf" : "local",
    dataset_id,
    fixture_path: dataset_id.startsWith("local_") ? inferFixturePath(dataset_id) : undefined,
    hf_slug: dataset_id.startsWith("hf_") ? inferHfSlug(dataset_id) : undefined,
    limit, // may be undefined
  };
  try {
    const res = await fetch(`${apiBase}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const contentType = res.headers.get('content-type') || '';
    let data: any = null;
    if (contentType.includes('application/json')) {
      data = await res.json();
    } else {
      const text = await res.text();
      data = { detail: text };
    }
    if (!res.ok) return json({ ok:false, errors:{ server: data.detail || "Run creation failed" } }, { status: res.status });
    return json({ ok:true, run_id: data.run_id });
  } catch (e:any) {
    return json({ ok:false, errors:{ server: e.message || "Network error" } }, { status:500 });
  }
}

function inferFixturePath(dataset_id: string){
  if (dataset_id === 'local_basic') return 'datasets/basic_statistics.json';
  if (dataset_id === 'local_step_functions') return 'datasets/step_functions.json';
  if (dataset_id === 'local_patterns') return 'datasets/pattern_recognition.json';
  return 'datasets/basic_statistics.json';
}

function inferHfSlug(dataset_id: string){
  if (dataset_id === 'hf_factoryset') return 'Forgis/FactorySet';
  return undefined;
}

export default function RunPage(){
  const { models, datasets } = useLoaderData<typeof loader>();
  const actionData = useActionData<typeof action>();
  const nav = useNavigation();
  const creating = nav.state === 'submitting' || nav.state === 'loading';
  const [submittedRun, setSubmittedRun] = React.useState<string|undefined>(undefined);
  React.useEffect(()=>{
    if (actionData?.ok && actionData.run_id){
      setSubmittedRun(actionData.run_id);
    }
  }, [actionData]);

  return (
    <div className="card">
      <h2 style={{ marginTop:0 }}>Create Run</h2>
      <p className="muted">Evaluate a model on a selected telemetry literacy dataset.</p>
      <Form method="post" style={{ display:'grid', gap:16, maxWidth:520, marginTop:16 }}>
        <Field label="Model" name="model" error={actionData?.errors?.model}>
          <select name="model" defaultValue={models[0]?.id || ''} style={selectStyle}>
            {models.map((m:any)=>(<option key={m.id} value={m.id}>{m.name}</option>))}
          </select>
        </Field>
        <Field label="Dataset" name="dataset_id" error={actionData?.errors?.dataset_id}>
          <select name="dataset_id" defaultValue={datasets[0]?.id || ''} style={selectStyle}>
            {datasets.map((d:any)=>(<option key={d.id} value={d.id}>{d.name} ({d.id})</option>))}
          </select>
        </Field>
        <Field label="Limit (optional)" name="limit" error={actionData?.errors?.limit}>
          <input name="limit" type="number" min={1} placeholder="e.g. 25" style={inputStyle} />
        </Field>
        {actionData?.errors?.server && (
          <div style={{ color:'var(--fg-fire)', fontSize:13 }}>{actionData.errors.server}</div>
        )}
        <button className="btn primary" type="submit" disabled={creating}>{creating ? 'Creating...' : 'Run Evaluation'}</button>
      </Form>
      {submittedRun && (
        <div style={{ marginTop:24 }}>
          <div style={{ fontSize:14 }}>Run created: <a href={`/runs/${submittedRun}`}>{submittedRun}</a></div>
        </div>
      )}
    </div>
  );
}

function Field({label, name, children, error}:{label:string; name:string; children:React.ReactNode; error?:string}){
  return (
    <label style={{ display:'flex', flexDirection:'column', gap:6 }}>
      <span style={{ fontSize:13, fontWeight:500 }}>{label}</span>
      {children}
      {error && <span style={{ color:'var(--fg-fire)', fontSize:12 }}>{error}</span>}
    </label>
  );
}

const inputStyle: React.CSSProperties = {
  padding:'8px 12px',
  border:'1px solid var(--fg-steel)',
  borderRadius:6,
  background:'#0e1a22',
  color:'var(--fg-white)',
  fontSize:14,
};
const selectStyle = inputStyle;