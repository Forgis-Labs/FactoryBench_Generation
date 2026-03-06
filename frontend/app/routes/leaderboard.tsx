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
    if (!res.ok) return { items: [], count: 0 };
    return await res.json();
  } catch {
    return { items: [], count: 0 };
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

export default function Leaderboard() {
  const { items, models, datasets } = useLoaderData<typeof loader>();
  const [searchParams, setSearchParams] = useSearchParams();
  const [modelDropdownOpen, setModelDropdownOpen] = React.useState(false);
  const [datasetDropdownOpen, setDatasetDropdownOpen] = React.useState(false);
  const [sortBy, setSortBy] = React.useState<string>("run_id");
  const [sortDir, setSortDir] = React.useState<"asc" | "desc">("desc");

  const modelFilters = searchParams.getAll("model");
  const datasetFilters = searchParams.getAll("dataset");
  const stageFilter = searchParams.get("stage") || "";
  
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

  function toggleSort(column: string) {
    if (sortBy === column) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortBy(column);
      setSortDir("asc");
    }
  }

  const sortedItems = React.useMemo(() => {
    const sorted = [...items].sort((a: any, b: any) => {
      let aVal: any, bVal: any;
      if (sortBy === "ok_rate") {
        aVal = a.aggregate?.ok_rate ?? 0;
        bVal = b.aggregate?.ok_rate ?? 0;
      } else if (sortBy === "mean_err") {
        aVal = a.aggregate?.mean_abs_err_mean ?? 0;
        bVal = b.aggregate?.mean_abs_err_mean ?? 0;
      } else if (sortBy === "min_err") {
        aVal = a.aggregate?.min_abs_err_mean ?? 0;
        bVal = b.aggregate?.min_abs_err_mean ?? 0;
      } else if (sortBy === "max_err") {
        aVal = a.aggregate?.max_abs_err_mean ?? 0;
        bVal = b.aggregate?.max_abs_err_mean ?? 0;
      } else if (sortBy === "performance") {
        aVal = a.aggregate?.performance ?? 0;
        bVal = b.aggregate?.performance ?? 0;
      } else if (sortBy === "cost_per_sample") {
        aVal = a.aggregate?.cost_per_sample ?? 0;
        bVal = b.aggregate?.cost_per_sample ?? 0;
      } else if (sortBy === "cost_total") {
        aVal = a.aggregate?.cost_total ?? 0;
        bVal = b.aggregate?.cost_total ?? 0;
      } else if (sortBy === "samples") {
        aVal = a.aggregate?.samples ?? 0;
        bVal = b.aggregate?.samples ?? 0;
      } else if (sortBy === "dataset") {
        aVal = a.dataset?.dataset_id || "";
        bVal = b.dataset?.dataset_id || "";
      } else if (sortBy === "model") {
        aVal = a.model || "";
        bVal = b.model || "";
      } else if (sortBy === "stage") {
        aVal = a.stage || "";
        bVal = b.stage || "";
      } else if (sortBy === "status") {
        aVal = a.status || "";
        bVal = b.status || "";
      } else {
        aVal = a.run_id || "";
        bVal = b.run_id || "";
      }
      if (typeof aVal === "string") {
        return sortDir === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortDir === "asc" ? aVal - bVal : bVal - aVal;
    });
    return sorted;
  }, [items, sortBy, sortDir]);

  // Helper components
  const SortTh = ({ column, label, sortBy, sortDir, onToggle }: any) => {
    const isActive = sortBy === column;
    return (
      <th onClick={() => onToggle(column)} style={{ cursor: 'pointer', userSelect: 'none' }}>
        {label} {isActive && (sortDir === 'asc' ? '↑' : '↓')}
      </th>
    );
  };

  const fmtNum = (val: any) => (val != null ? val.toFixed(3) : '-');
  const fmtCost = (val: any) => (val != null ? `$${val.toFixed(3)}` : '-');

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

  function updateStageFilter(value: string) {
    const params = new URLSearchParams(searchParams);
    if (value) {
      params.set("stage", value);
    } else {
      params.delete("stage");
    }
    setSearchParams(params);
  }

  return (
    <div className="card">
      <h2 style={{ marginBottom: 16 }}>Leaderboard</h2>

      {/* Filters */}
      <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
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

        <div>
          <label htmlFor="stage-filter" style={{ display: "block", fontSize: 14, marginBottom: 4, color: "var(--fg-platinum)" }}>Stage</label>
          <select 
            id="stage-filter"
            value={stageFilter} 
            onChange={(e) => updateStageFilter(e.target.value)}
            style={{ 
              padding: "8px 12px", 
              borderRadius: 6, 
              border: "1px solid var(--fg-steel)",
              background: "#0e1a22",
              color: "var(--fg-white)",
              fontSize: 14,
              cursor: "pointer"
            }}
          >
            <option value="">All Stages</option>
            <option value="telemetry_literacy">Telemetry Literacy</option>
          </select>
        </div>

        {(modelFilters.length > 0 || datasetFilters.length > 0 || stageFilter) && (
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

      <table className="table">
        <thead>
          <tr>
            <SortTh column="run_id" label="Run" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="status" label="Status" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="stage" label="Stage" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="model" label="Model" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="dataset" label="Dataset" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="samples" label="Samples" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="ok_rate" label="OK %" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="mean_err" label="Mean" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="min_err" label="Min" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="max_err" label="Max" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="performance" label="Performance" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="cost_per_sample" label="$/Sample" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
            <SortTh column="cost_total" label="Total $" sortBy={sortBy} sortDir={sortDir} onToggle={toggleSort} />
          </tr>
        </thead>
        <tbody>
          {sortedItems.length === 0 && (
            <tr><td colSpan={13} className="muted">No runs match the filters. Try adjusting or clearing them.</td></tr>
          )}
          {sortedItems.map((r: any) => {
            const agg = r.aggregate || {};
            const ds = r.dataset || {};
            const stageShort = r.stage === 'telemetry_literacy' ? 'TL' : r.stage === 'root_cause_analysis' ? 'RCA' : r.stage === 'guided_remediation' ? 'GR' : r.stage || '-';
            return (
              <tr key={r.run_id}>
                <td><a href={`/runs/${r.run_id}`}>{r.run_id}</a></td>
                <td style={{ color: r.status === 'completed' ? 'var(--fg-platinum)' : 'var(--fg-tiger)' }}>{r.status || 'completed'}</td>
                <td>{stageShort}</td>
                <td>{r.model}</td>
                <td>{ds.dataset_id || '-'}</td>
                <td>{agg.samples ?? '-'}</td>
                <td>{((agg.ok_rate ?? 0) * 100).toFixed(1)}%</td>
                <td>{fmtNum(agg.mean_abs_err_mean)}</td>
                <td>{fmtNum(agg.min_abs_err_mean)}</td>
                <td>{fmtNum(agg.max_abs_err_mean)}</td>
                <td>{fmtNum(agg.performance)}</td>
                <td>{fmtCost(agg.cost_per_sample)}</td>
                <td>{fmtCost(agg.cost_total)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
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
