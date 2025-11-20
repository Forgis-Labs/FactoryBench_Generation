import type { LoaderFunctionArgs } from "@remix-run/node";

export async function loader({ request }: LoaderFunctionArgs) {
  return null;
}

export default function Index() {
  return (
    <div className="hero">
      <h1>FactoryBench</h1>
      <p className="muted">Lean benchmark for AI model performance on industrial troubleshooting</p>
      <p style={{ marginTop: 8, fontSize: 14, color: "var(--fg-steel)" }}>
        <strong>Current Stage</strong>: Telemetry Literacy (Stage 1) • <strong>Status</strong>: ✅ MVP Running
      </p>
      
      <div style={{ marginTop: 24, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12, maxWidth: 800 }}>
        <a className="btn primary" href="/leaderboard" style={{ textAlign: "center", padding: "12px 16px" }}>
          📊 Leaderboard
        </a>
        <a className="btn" href="/analysis" style={{ textAlign: "center", padding: "12px 16px" }}>
          📈 Analysis
        </a>
        <a className="btn" href="/run" style={{ textAlign: "center", padding: "12px 16px" }}>
          ▶️ Create Run
        </a>
        <a className="btn" href="/readme" style={{ textAlign: "center", padding: "12px 16px" }}>
          📖 Methodology
        </a>
      </div>

      <div style={{ marginTop: 32, padding: 16, border: "1px solid var(--fg-border)", borderRadius: 8, maxWidth: 800, margin: "32px auto 0" }}>
        <h3 style={{ marginTop: 0, fontSize: 16 }}>Quick Start</h3>
        <ol style={{ fontSize: 14, lineHeight: 1.6, paddingLeft: 20 }}>
          <li>View the <a href="/leaderboard">Leaderboard</a> to see existing benchmark runs</li>
          <li>Explore <a href="/analysis">Analysis</a> for interactive charts & metrics</li>
          <li><a href="/run">Create a new run</a> to evaluate a model</li>
          <li>Read the <a href="/readme">Methodology</a> for detailed metrics & dataset structure</li>
        </ol>
      </div>

      <div style={{ marginTop: 24, textAlign: "center", fontSize: 14, color: "var(--fg-steel)" }}>
        <p>7 models • 4 datasets • Full cost tracking • Sortable leaderboard</p>
      </div>
    </div>
  );
}
