import type { LoaderFunctionArgs } from "@remix-run/node";

export async function loader({ request }: LoaderFunctionArgs) {
  return null;
}

export default function Index() {
  return (
    <div className="hero">
      <h1>FactoryBench</h1>
      <p className="muted">Comprehensive benchmark for AI model performance on industrial troubleshooting tasks</p>
      <p style={{ marginTop: 8, fontSize: 14, color: "var(--fg-steel)" }}>
        Evaluate telemetry literacy • Model comparison analytics • Real-time progress tracking
      </p>

      {/* Feature Highlights */}
      <div style={{ 
        marginTop: 32, 
        padding: "20px 24px", 
        backgroundColor: "var(--bg-secondary)", 
        border: "1px solid var(--fg-border)", 
        borderRadius: 8, 
        maxWidth: 900, 
        margin: "32px auto 0"
      }}>
        <h3 style={{ marginTop: 0, marginBottom: 16, fontSize: 16 }}>Key Features</h3>
        <div style={{ 
          display: "grid", 
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", 
          gap: 16,
          fontSize: 13,
          lineHeight: 1.5
        }}>
          <div>
            <strong style={{ color: "var(--fg-fire)" }}>💰 Cost Safeguards</strong>
            <p style={{ margin: "4px 0 0 0", color: "var(--fg-steel)" }}>$1/run • $20/day limits with pre-flight checks</p>
          </div>
          <div>
            <strong style={{ color: "var(--fg-tiger)" }}>⚡ Real-time Progress</strong>
            <p style={{ margin: "4px 0 0 0", color: "var(--fg-steel)" }}>2s polling • Sample-by-sample updates</p>
          </div>
          <div>
            <strong style={{ color: "var(--fg-flicker)" }}>📊 Model Comparison</strong>
            <p style={{ margin: "4px 0 0 0", color: "var(--fg-steel)" }}>Performance bars • Cost scatter • Traffic-light heatmap</p>
          </div>
          <div>
            <strong style={{ color: "var(--fg-fire)" }}>🎯 Performance Metric</strong>
            <p style={{ margin: "4px 0 0 0", color: "var(--fg-steel)" }}>Composite: avg(mean_err, min_err, max_err)</p>
          </div>
          <div>
            <strong style={{ color: "var(--fg-tiger)" }}>⏹️ Graceful Cancellation</strong>
            <p style={{ margin: "4px 0 0 0", color: "var(--fg-steel)" }}>Stop button saves partial results</p>
          </div>
          <div>
            <strong style={{ color: "var(--fg-flicker)" }}>🔍 Smart Defaults</strong>
            <p style={{ margin: "4px 0 0 0", color: "var(--fg-steel)" }}>Pre-select gpt-4o/mini + FactorySet</p>
          </div>
        </div>
      </div>

      {/* Current Status Cards */}
      <div style={{ 
        marginTop: 32,
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", 
        gap: 12,
        maxWidth: 800,
        margin: "32px auto 0"
      }}>
        <div style={{ padding: "16px 12px", backgroundColor: "var(--bg-secondary)", border: "1px solid var(--fg-border)", borderRadius: 8, textAlign: "center" }}>
          <div style={{ fontSize: 28, fontWeight: "bold", color: "var(--fg-tiger)" }}>5</div>
          <div style={{ fontSize: 12, color: "var(--fg-steel)", marginTop: 4 }}>Models</div>
        </div>
        <div style={{ padding: "16px 12px", backgroundColor: "var(--bg-secondary)", border: "1px solid var(--fg-border)", borderRadius: 8, textAlign: "center" }}>
          <div style={{ fontSize: 28, fontWeight: "bold", color: "var(--fg-flicker)" }}>4</div>
          <div style={{ fontSize: 12, color: "var(--fg-steel)", marginTop: 4 }}>Datasets</div>
        </div>
        <div style={{ padding: "16px 12px", backgroundColor: "var(--bg-secondary)", border: "1px solid var(--fg-border)", borderRadius: 8, textAlign: "center" }}>
          <div style={{ fontSize: 28, fontWeight: "bold", color: "var(--fg-fire)" }}>50k+</div>
          <div style={{ fontSize: 12, color: "var(--fg-steel)", marginTop: 4 }}>Samples</div>
        </div>
        <div style={{ padding: "16px 12px", backgroundColor: "var(--bg-secondary)", border: "1px solid var(--fg-border)", borderRadius: 8, textAlign: "center" }}>
          <div style={{ fontSize: 28, fontWeight: "bold", color: "var(--fg-tiger)" }}>S1</div>
          <div style={{ fontSize: 12, color: "var(--fg-steel)", marginTop: 4 }}>Telemetry</div>
        </div>
      </div>

      {/* Quick Actions */}
      <div style={{ marginTop: 40, maxWidth: 800, margin: "40px auto 0" }}>
        <h3 style={{ fontSize: 18, marginBottom: 16 }}>Quick Actions</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16 }}>
          <div style={{ padding: 20, border: "1px solid var(--fg-border)", borderRadius: 8, backgroundColor: "var(--bg-primary)" }}>
            <h4 style={{ fontSize: 15, marginTop: 0, marginBottom: 8 }}>📊 Leaderboard</h4>
            <p style={{ fontSize: 13, marginBottom: 16, color: "var(--fg-steel)", lineHeight: 1.5 }}>
              View and compare all benchmark runs. Sortable by performance, cost, OK rate. Default filters: gpt-4o/mini.
            </p>
            <a className="btn primary" href="/leaderboard" style={{ textAlign: "center", padding: "10px 20px", display: "block" }}>
              Go to Leaderboard →
            </a>
          </div>

          <div style={{ padding: 20, border: "1px solid var(--fg-border)", borderRadius: 8, backgroundColor: "var(--bg-primary)" }}>
            <h4 style={{ fontSize: 15, marginTop: 0, marginBottom: 8 }}>▶️ Create Run</h4>
            <p style={{ fontSize: 13, marginBottom: 16, color: "var(--fg-steel)", lineHeight: 1.5 }}>
              Start a new benchmark run with real-time progress tracking, cost monitoring, and graceful cancellation.
            </p>
            <a className="btn" href="/run" style={{ textAlign: "center", padding: "10px 20px", display: "block" }}>
              Create Run →
            </a>
          </div>

          <div style={{ padding: 20, border: "1px solid var(--fg-border)", borderRadius: 8, backgroundColor: "var(--bg-primary)" }}>
            <h4 style={{ fontSize: 15, marginTop: 0, marginBottom: 8 }}>📈 Analysis</h4>
            <p style={{ fontSize: 13, marginBottom: 16, color: "var(--fg-steel)", lineHeight: 1.5 }}>
              Visualize model performance with bar charts, cost scatter plots, and traffic-light heatmaps.
            </p>
            <a className="btn" href="/analysis" style={{ textAlign: "center", padding: "10px 20px", display: "block" }}>
              View Charts →
            </a>
          </div>

          <div style={{ padding: 20, border: "1px solid var(--fg-border)", borderRadius: 8, backgroundColor: "var(--bg-primary)" }}>
            <h4 style={{ fontSize: 15, marginTop: 0, marginBottom: 8 }}>📚 Methodology</h4>
            <p style={{ fontSize: 13, marginBottom: 16, color: "var(--fg-steel)", lineHeight: 1.5 }}>
              Learn about the three-stage benchmark progression, evaluation metrics, and industrial datasets.
            </p>
            <a className="btn" href="/readme" style={{ textAlign: "center", padding: "10px 20px", display: "block" }}>
              Read Documentation →
            </a>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{ marginTop: 48, paddingTop: 24, borderTop: "1px solid var(--fg-border)", textAlign: "center", fontSize: 12, color: "var(--fg-steel)" }}>
        <p><strong>Stage 1: Telemetry Literacy</strong> – Statistical analysis of time series data</p>
        <p style={{ marginTop: 8 }}>Maintained by Forgis • Version 0.2.0 • November 2025</p>
      </div>
    </div>
  );
}
