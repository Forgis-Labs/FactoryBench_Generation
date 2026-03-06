import type { LoaderFunctionArgs } from "@remix-run/node";
import { json } from "@remix-run/node";
import { useLoaderData, useSearchParams } from "@remix-run/react";
import * as React from "react";

async function fetchRuns(apiBase: string, params?: URLSearchParams) {
  try {
    const url = new URL(`${apiBase}/runs`);
    if (params) {
      params.forEach((value, key) => {
        if (value) url.searchParams.append(key, value);
      });
    }
    const res = await fetch(url.toString());
    if (!res.ok) return { items: [] };
    return await res.json();
  } catch {
    return { items: [] };
  }
}

async function fetchMetadata(apiBase: string) {
  try {
    const [modelsRes, datasetsRes] = await Promise.all([
      fetch(`${apiBase}/metadata/models`),
      fetch(`${apiBase}/metadata/datasets?stage=telemetry_literacy`)
    ]);
    const models = modelsRes.ok ? await modelsRes.json() : { models: [] };
    const datasets = datasetsRes.ok ? await datasetsRes.json() : { datasets: [] };
    return { models: models.models || [], datasets: datasets.datasets || [] };
  } catch {
    return { models: [], datasets: [] };
  }
}

export async function loader({ request }: LoaderFunctionArgs) {
  const apiBase = process.env.API_BASE || "http://127.0.0.1:5173";
  const url = new URL(request.url);
  const params = url.searchParams;
  
  const [runsData, metadata] = await Promise.all([
    fetchRuns(apiBase, params),
    fetchMetadata(apiBase)
  ]);
  
  return json({ 
    items: runsData.items || [],
    models: metadata.models,
    datasets: metadata.datasets
  });
}

export default function Analysis() {
  const { items, models, datasets } = useLoaderData<typeof loader>();
  const [searchParams, setSearchParams] = useSearchParams();
  const [activeTab, setActiveTab] = React.useState<"stage1" | "stage2" | "stage3">("stage1");
  const [modelDropdownOpen, setModelDropdownOpen] = React.useState(false);
  const [datasetDropdownOpen, setDatasetDropdownOpen] = React.useState(false);
  
  const modelFilters = searchParams.getAll("model");
  const datasetFilters = searchParams.getAll("dataset");
  
  // Set default filters on mount
  React.useEffect(() => {
    const params = new URLSearchParams(searchParams);
    let needsUpdate = false;
    
    if (datasetFilters.length === 0) {
      params.append("dataset", "hf_factoryset");
      needsUpdate = true;
    }
    
    if (modelFilters.length === 0) {
      params.append("model", "azure:gpt-4o");
      params.append("model", "azure:gpt-4o-mini");
      needsUpdate = true;
    }
    
    if (needsUpdate) {
      setSearchParams(params, { replace: true });
    }
  }, []); // Only run on mount

  function toggleFilter(key: string, value: string) {
    const params = new URLSearchParams(searchParams);
    const existing = params.getAll(key);
    
    if (existing.includes(value)) {
      // Remove this value
      params.delete(key);
      existing.filter(v => v !== value).forEach(v => params.append(key, v));
    } else {
      // Add this value
      params.append(key, value);
    }
    setSearchParams(params);
  }
  
  // Compute aggregate metrics per stage
  const byStage = items.reduce((acc: any, r: any) => {
    const stage = r.stage || "unknown";
    if (!acc[stage]) acc[stage] = [];
    acc[stage].push(r);
    return acc;
  }, {});

  const tlRuns = byStage.telemetry_literacy || [];
  const rcaRuns = byStage.root_cause_analysis || [];
  const grRuns = byStage.guided_remediation || [];

  const stages = [
    {
      id: "stage1" as const,
      title: "Stage 1: Telemetry Literacy",
      runs: tlRuns,
      charts: [
        { title: "Model Performance Comparison", type: "model_performance" },
        { title: "Cost vs Performance Tradeoff", type: "cost_vs_performance" },
        { title: "Model Metrics Heatmap", type: "metrics_heatmap" }
      ]
    },
    {
      id: "stage2" as const,
      title: "Stage 2: Root Cause Analysis",
      runs: rcaRuns,
      charts: [
        { title: "Retrieval Precision@K", type: "retrieval" },
        { title: "Step Ordering Accuracy (Kendall τ)", type: "ordering" },
        { title: "Fault Classification F1", type: "classification" }
      ]
    },
    {
      id: "stage3" as const,
      title: "Stage 3: Guided Remediation",
      runs: grRuns,
      charts: [
        { title: "Safety Compliance Score", type: "safety" },
        { title: "Instruction Quality (BLEURT)", type: "quality" },
        { title: "Coverage vs Efficiency", type: "coverage_efficiency" }
      ]
    }
  ];

  const currentStage = stages.find(s => s.id === activeTab) || stages[0];

  return (
    <div>
      <div className="card" style={{ marginBottom: 20 }}>
        <h2 style={{ marginTop: 0 }}>Analysis</h2>
        <p className="muted">Track model development progress across all stages.</p>

        {/* Filters */}
        <div style={{ display: "flex", gap: 12, marginTop: 16, flexWrap: "wrap" }}>
          <MultiSelectDropdown
            label="Model"
            options={models.map((m: any) => ({ id: m.id, name: m.name }))}
            selected={modelFilters}
            onToggle={(value) => toggleFilter("model", value)}
            isOpen={modelDropdownOpen}
            setIsOpen={setModelDropdownOpen}
          />

          <MultiSelectDropdown
            label="Dataset"
            options={datasets.map((d: any) => ({ id: d.id, name: d.name }))}
            selected={datasetFilters}
            onToggle={(value) => toggleFilter("dataset", value)}
            isOpen={datasetDropdownOpen}
            setIsOpen={setDatasetDropdownOpen}
          />

          {(modelFilters.length > 0 || datasetFilters.length > 0) && (
            <div style={{ display: "flex", alignItems: "flex-end" }}>
              <button 
                className="btn" 
                onClick={() => setSearchParams(new URLSearchParams())}
                style={{ padding: "8px 12px" }}
              >
                Clear Filters
              </button>
            </div>
          )}
        </div>
        
        {/* WandB Integration */}
        <div style={{ marginTop: 20, padding: 16, border: "1px solid var(--fg-border)", borderRadius: 8 }}>
          <h3 style={{ margin: "0 0 8px 0", fontSize: 16 }}>WandB Project</h3>
          <p className="muted" style={{ margin: "0 0 12px 0" }}>
            View detailed experiment tracking and hyperparameter sweeps in Weights & Biases.
          </p>
          <a 
            href="https://wandb.ai/forgis/factorybench" 
            target="_blank" 
            rel="noopener noreferrer"
            className="btn"
            style={{ display: "inline-block" }}
          >
            Open WandB Dashboard →
          </a>
        </div>
      </div>

      {/* Stage Tabs */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", gap: 8, borderBottom: "2px solid var(--fg-border)", paddingBottom: 8, marginBottom: 24 }}>
          {stages.map(stage => (
            <button
              key={stage.id}
              onClick={() => setActiveTab(stage.id)}
              className="btn"
              style={{
                padding: "8px 16px",
                background: activeTab === stage.id ? "var(--fg-fire)" : "transparent",
                color: activeTab === stage.id ? "var(--fg-white)" : "var(--fg-steel)",
                border: "none",
                borderBottom: activeTab === stage.id ? "2px solid var(--fg-fire)" : "2px solid transparent",
                marginBottom: -2,
                cursor: "pointer",
                fontSize: 14,
                fontWeight: activeTab === stage.id ? 600 : 400
              }}
            >
              {stage.title}
              <span style={{ marginLeft: 8, opacity: 0.7 }}>({stage.runs.length})</span>
            </button>
          ))}
        </div>

        {/* Active Stage Content */}
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 20 }}>
            {currentStage.charts.map(chart => (
              <ChartCard 
                key={chart.type} 
                title={chart.title} 
                type={chart.type} 
                runs={currentStage.runs}
                modelFilters={modelFilters}
                datasetFilters={datasetFilters}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ChartCard({ title, type, runs, modelFilters, datasetFilters }: { 
  title: string; 
  type: string; 
  runs: any[];
  modelFilters?: string[];
  datasetFilters?: string[];
}) {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const [showInfo, setShowInfo] = React.useState(false);

  // Chart explanations
  const chartExplanations: Record<string, string> = {
    model_performance: "Compares models by performance score (average of mean/min/max errors). Lower is better. Shows best run for each model-dataset combination.",
    cost_vs_performance: "Scatter plot of total cost vs performance. Ideal models are in the lower-left (low cost, low error). Point size indicates sample count.",
    metrics_heatmap: "Heatmap showing all error metrics for each model-dataset combo. Darker colors indicate higher values.",
    retrieval: "Measures how accurately the model retrieves relevant information (Precision@K).",
    ordering: "Evaluates step ordering accuracy using Kendall tau correlation.",
    classification: "Shows F1 score for fault classification tasks.",
    safety: "Assesses safety compliance of model outputs.",
    quality: "Rates instruction quality using BLEURT metric.",
    coverage_efficiency: "Compares coverage of solutions versus efficiency (steps taken)."
  };
  const infoText = chartExplanations[type] || "Chart explanation not available.";

  const downloadChart = async () => {
    const apiBase = "http://127.0.0.1:5173";
    const params = new URLSearchParams();
    if (modelFilters && modelFilters.length > 0) {
      modelFilters.forEach(m => params.append("model", m));
    }
    if (datasetFilters && datasetFilters.length > 0) {
      datasetFilters.forEach(d => params.append("dataset", d));
    }
    const url = `${apiBase}/charts/${type}?${params.toString()}`;
    const timestamp = new Date().toISOString().split('T')[0];
    const link = document.createElement("a");
    link.href = url;
    link.download = `factorybench-${type}-${timestamp}.png`;
    link.click();
  };

  // Build chart URL with filters
  const apiBase = "http://127.0.0.1:5173";
  const params = new URLSearchParams();
  if (modelFilters && modelFilters.length > 0) {
    modelFilters.forEach(m => params.append("model", m));
  }
  if (datasetFilters && datasetFilters.length > 0) {
    datasetFilters.forEach(d => params.append("dataset", d));
  }
  params.append("regenerate", "true");
  const chartUrl = `${apiBase}/charts/${type}?${params.toString()}`;

  const chartContent = (
    <img
      src={chartUrl}
      alt={title}
      style={{ width: "100%", height: "auto", borderRadius: 6 }}
    />
  );

  return (
    <>
      <div style={{ border: "1px solid var(--fg-border)", borderRadius: 8, padding: 16 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <h4 style={{ margin: 0, fontSize: 15, fontWeight: 500 }}>{title}</h4>
            <span
              style={{ position: "relative", display: "inline-block" }}
              onMouseEnter={() => setShowInfo(true)}
              onMouseLeave={() => setShowInfo(false)}
            >
              <button
                style={{
                  background: "none",
                  border: "none",
                  padding: 0,
                  marginLeft: 2,
                  cursor: "pointer",
                  color: "var(--fg-steel)",
                  fontSize: 16,
                  lineHeight: 1
                }}
                aria-label="Chart info"
                tabIndex={0}
              >
                <span style={{ fontWeight: 700, fontSize: 16 }}>i</span>
              </button>
              {showInfo && (
                <div
                  style={{
                    position: "absolute",
                    top: "120%",
                    left: "50%",
                    transform: "translateX(-50%)",
                    background: "var(--bg-primary)",
                    color: "var(--fg-white)",
                    border: "1px solid var(--fg-border)",
                    borderRadius: 6,
                    padding: "8px 12px",
                    fontSize: 13,
                    whiteSpace: "pre-line",
                    zIndex: 10,
                    boxShadow: "0 2px 8px rgba(0,0,0,0.12)"
                  }}
                >
                  {infoText}
                </div>
              )}
            </span>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={() => setIsExpanded(true)}
              className="btn"
              style={{ fontSize: 12, padding: "4px 10px" }}
              title="Expand chart"
            >
              ⛶
            </button>
            <button
              onClick={downloadChart}
              className="btn"
              style={{ fontSize: 12, padding: "4px 10px" }}
              title="Download chart"
            >
              ↓
            </button>
          </div>
        </div>
        {chartContent}
      </div>

      {/* Expanded Modal */}
      {isExpanded && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(0, 0, 0, 0.9)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
            padding: 40
          }}
          onClick={() => setIsExpanded(false)}
        >
          <div
            style={{
              background: "var(--bg-primary)",
              borderRadius: 12,
              padding: 32,
              maxWidth: 1200,
              width: "100%",
              maxHeight: "90vh",
              overflow: "auto"
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
              <h3 style={{ margin: 0 }}>{title}</h3>
              <button
                onClick={() => setIsExpanded(false)}
                className="btn"
                style={{ fontSize: 18 }}
              >
                ✕
              </button>
            </div>
            <div style={{ width: "100%", height: "auto", display: "flex", justifyContent: "center" }}>
              <img
                src={chartUrl}
                alt={title}
                style={{ maxWidth: "100%", maxHeight: "70vh", width: "auto", height: "auto", objectFit: "contain" }}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function MultiSelectDropdown({ 
  label, 
  options, 
  selected, 
  onToggle, 
  isOpen, 
  setIsOpen 
}: { 
  label: string;
  options: { id: string; name: string }[];
  selected: string[];
  onToggle: (value: string) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}) {
  const dropdownRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [setIsOpen]);

  const selectedCount = selected.length;
  const displayText = selectedCount === 0 
    ? `All ${label}s` 
    : `${selectedCount} ${label}${selectedCount > 1 ? 's' : ''}`;

  return (
    <div ref={dropdownRef} style={{ position: "relative" }}>
      <label style={{ display: "block", fontSize: 14, marginBottom: 4, color: "var(--fg-platinum)" }}>
        {label}
      </label>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          padding: "8px 12px",
          borderRadius: 6,
          border: `1px solid ${selectedCount > 0 ? "var(--fg-fire)" : "var(--fg-steel)"}`,
          background: "#0e1a22",
          color: selectedCount > 0 ? "var(--fg-fire)" : "var(--fg-white)",
          fontSize: 14,
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 8,
          minWidth: 160,
          justifyContent: "space-between"
        }}
      >
        <span>{displayText}</span>
        <span style={{ opacity: 0.6, fontSize: 12 }}>▼</span>
      </button>
      {isOpen && (
        <div style={{
          position: "absolute",
          top: "100%",
          left: 0,
          marginTop: 4,
          background: "#0e1a22",
          border: "1px solid var(--fg-steel)",
          borderRadius: 6,
          padding: 8,
          minWidth: 200,
          maxHeight: 300,
          overflowY: "auto",
          zIndex: 1000,
          boxShadow: "0 4px 12px rgba(0,0,0,0.4)"
        }}>
          {options.map((option) => (
            <label
              key={option.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "6px 8px",
                cursor: "pointer",
                borderRadius: 4,
                transition: "background 0.15s ease"
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = "rgba(255,77,0,0.1)"}
              onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
            >
              <input
                type="checkbox"
                checked={selected.includes(option.id)}
                onChange={() => onToggle(option.id)}
                style={{
                  width: 16,
                  height: 16,
                  cursor: "pointer",
                  accentColor: "var(--fg-fire)"
                }}
              />
              <span style={{ fontSize: 14, color: "var(--fg-white)" }}>{option.name}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}
