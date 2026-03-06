export default function Readme() {
  return (
    <div className="card" style={{ maxWidth: 900, margin: "0 auto" }}>
      <h1 style={{ marginTop: 0 }}>FactoryBench Methodology</h1>
      <p className="muted">Benchmark design, metrics, and dataset structure</p>

      <hr style={{ margin: "24px 0", border: "none", borderTop: "1px solid var(--fg-border)" }} />

      {/* Three-Stage Progression */}
      <section style={{ marginBottom: 32 }}>
        <h2>Three-Stage Progression</h2>
        <p>
          FactoryBench evaluates AI models through a progressive complexity hierarchy designed to mirror real-world industrial troubleshooting workflows.
        </p>
        
        <div style={{ display: "grid", gap: 16, marginTop: 16 }}>
          {/* Stage 1 */}
          <div style={{ border: "1px solid var(--fg-fire)", borderRadius: 8, padding: 16, background: "rgba(240, 90, 40, 0.05)" }}>
            <h3 style={{ marginTop: 0, color: "var(--fg-fire)" }}>Stage 1: Telemetry Literacy (Current)</h3>
            <p><strong>Goal</strong>: Evaluate basic time series comprehension and statistical analysis capabilities.</p>
            <p><strong>Status</strong>: ✅ Implemented & Running</p>
            
            <h4>Capabilities Tested</h4>
            <ul>
              <li>Statistical measures (mean, min, max) from univariate time series</li>
              <li>Step function detection and characterization</li>
              <li>Pattern recognition in temporal data</li>
              <li>Numerical accuracy and parsing robustness</li>
            </ul>

            <h4>Dataset Structure</h4>
            <p>Each sample contains:</p>
            <pre style={{ background: "#0a1117", padding: 12, borderRadius: 6, fontSize: 13, overflow: "auto" }}>
{`{
  "id": "basic_001",
  "timestamps": [0.0, 1.0, 2.0, 3.0, 4.0],
  "values": [10.0, 20.0, 30.0, 40.0, 50.0],
  "domain": "synthetic",
  "subtype": "linear_increase",
  "statistics": {
    "mean": 30.0,
    "std": 14.142135,
    "min": 10.0,
    "max": 50.0
  }
}`}
            </pre>

            <h4>Prompt Template</h4>
            <p>Models receive structured prompts requesting statistical analysis:</p>
            <pre style={{ background: "#0a1117", padding: 12, borderRadius: 6, fontSize: 13, overflow: "auto" }}>
{`You are an expert in time series analysis.
Given the following time series data, compute the mean, min, and max.

Timestamps: [t1, t2, ..., tn]
Values: [v1, v2, ..., vn]

Respond ONLY with: mean=X min=Y max=Z`}
            </pre>

            <h4>Evaluation Metrics</h4>
            <p><strong>Per-Sample Scoring</strong>:</p>
            <ul style={{ fontFamily: "monospace", fontSize: 14 }}>
              <li>mean_abs_err = |predicted_mean - ground_truth_mean|</li>
              <li>min_abs_err = |predicted_min - ground_truth_min|</li>
              <li>max_abs_err = |predicted_max - ground_truth_max|</li>
              <li>ok = true if all three metrics were successfully extracted, else false</li>
            </ul>

            <p><strong>Aggregate Scoring</strong> (across N samples):</p>
            <ul style={{ fontFamily: "monospace", fontSize: 14 }}>
              <li>mean_abs_err_mean = (Σ mean_abs_err) / N</li>
              <li>min_abs_err_mean = (Σ min_abs_err) / N</li>
              <li>max_abs_err_mean = (Σ max_abs_err) / N</li>
              <li>ok_rate = (count of ok=true) / N</li>
              <li>samples = N</li>
            </ul>

            <p><strong>Cost Metrics</strong>:</p>
            <ul style={{ fontFamily: "monospace", fontSize: 14 }}>
              <li>cost_total = (tokens_input × price_input_per_1k / 1000) + (tokens_output × price_output_per_1k / 1000)</li>
              <li>cost_per_sample = cost_total / samples</li>
            </ul>

            <h4>Available Datasets</h4>
            <table style={{ width: "100%", fontSize: 14, marginTop: 8 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--fg-border)" }}>
                  <th style={{ textAlign: "left", padding: "8px 4px" }}>Dataset ID</th>
                  <th style={{ textAlign: "left", padding: "8px 4px" }}>Samples</th>
                  <th style={{ textAlign: "left", padding: "8px 4px" }}>Focus</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: "8px 4px" }}>local_basic</td>
                  <td style={{ padding: "8px 4px" }}>10</td>
                  <td style={{ padding: "8px 4px" }}>Basic statistical patterns</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 4px" }}>local_step_functions</td>
                  <td style={{ padding: "8px 4px" }}>15</td>
                  <td style={{ padding: "8px 4px" }}>Step changes & plateaus</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 4px" }}>local_patterns</td>
                  <td style={{ padding: "8px 4px" }}>12</td>
                  <td style={{ padding: "8px 4px" }}>Pattern recognition</td>
                </tr>
                <tr>
                  <td style={{ padding: "8px 4px" }}>hf_factoryset</td>
                  <td style={{ padding: "8px 4px" }}>50,000+</td>
                  <td style={{ padding: "8px 4px" }}>Large-scale evaluation</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Stage 2 */}
          <div style={{ border: "1px solid var(--fg-steel)", borderRadius: 8, padding: 16, background: "rgba(110, 121, 131, 0.05)" }}>
            <h3 style={{ marginTop: 0, color: "var(--fg-steel)" }}>Stage 2: Root Cause Analysis (Planned)</h3>
            <p><strong>Goal</strong>: Bridge gap between basic time series understanding and full troubleshooting.</p>
            <p><strong>Status</strong>: 📋 Specification Phase</p>
            
            <h4>Planned Capabilities</h4>
            <ul>
              <li>Correlation between sensor readings and fault types</li>
              <li>Basic diagnostic reasoning from multi-sensor data</li>
              <li>Understanding of machine state transitions</li>
              <li>Context integration (manual snippets + sensor data)</li>
              <li>Fault code interpretation</li>
            </ul>

            <h4>Dataset Structure (TBD)</h4>
            <p>Each sample will likely contain:</p>
            <pre style={{ background: "#0a1117", padding: 12, borderRadius: 6, fontSize: 13, overflow: "auto" }}>
{`{
  "id": "rca_001",
  "sensors": {
    "temperature": [...],
    "pressure": [...],
    "vibration": [...]
  },
  "fault_codes": ["E1402", "W0234"],
  "manual_excerpt": "...",
  "ground_truth": {
    "root_cause": "bearing_wear",
    "evidence_sensors": ["vibration", "temperature"],
    "confidence": 0.95
  }
}`}
            </pre>

            <h4>Evaluation Metrics (TBD)</h4>
            <p>Planned metrics:</p>
            <ul>
              <li><strong>Retrieval Precision@K</strong>: Accuracy of identifying relevant sensors</li>
              <li><strong>Fault Classification F1</strong>: Precision & recall for fault type identification</li>
              <li><strong>Reasoning Quality</strong>: Human-evaluated logical coherence (1-5 scale)</li>
              <li><strong>Context Utilization</strong>: Measure of how well manual excerpts are integrated</li>
            </ul>
          </div>

          {/* Stage 3 */}
          <div style={{ border: "1px solid var(--fg-steel)", borderRadius: 8, padding: 16, background: "rgba(110, 121, 131, 0.05)" }}>
            <h3 style={{ marginTop: 0, color: "var(--fg-steel)" }}>Stage 3: Guided Remediation (Future)</h3>
            <p><strong>Goal</strong>: Evaluate complete diagnostic and repair instruction generation.</p>
            <p><strong>Status</strong>: 🔮 Research & Curation Phase</p>
            
            <h4>Planned Capabilities</h4>
            <ul>
              <li>Multi-modal input integration (time series + PDFs + metadata)</li>
              <li>Step-by-step repair instruction generation</li>
              <li>Safety protocol compliance</li>
              <li>Understanding of factory process dependencies</li>
              <li>Complete troubleshooting workflow orchestration</li>
            </ul>

            <h4>Dataset Structure (FactorySet)</h4>
            <p>Real industrial scenarios with expert-validated solutions:</p>
            <pre style={{ background: "#0a1117", padding: 12, borderRadius: 6, fontSize: 13, overflow: "auto" }}>
{`{
  "id": "scenario_001",
  "machine_metadata": {
    "model": "CNC-3000",
    "manufacturer": "ACME",
    "install_date": "2018-03-15"
  },
  "timeseries": {
    "parquet_url": "s3://..."
  },
  "manual_pdf": "s3://...",
  "fault_codes": ["E1402"],
  "process_context": {
    "upstream_machines": [...],
    "production_impact": "high"
  },
  "ground_truth_instructions": [
    {
      "step": 1,
      "action": "Isolate machine power",
      "safety_critical": true,
      "tools_required": ["lockout_kit"]
    },
    ...
  ]
}`}
            </pre>

            <h4>Evaluation Metrics (TBD)</h4>
            <p>Comprehensive multi-dimensional scoring:</p>
            <ul>
              <li><strong>Instruction Correctness</strong>: % of steps matching ground truth</li>
              <li><strong>Safety Compliance</strong>: Binary pass/fail on safety-critical steps</li>
              <li><strong>Completeness</strong>: % of necessary steps included</li>
              <li><strong>Efficiency</strong>: Ratio of necessary to total steps (penalize redundancy)</li>
              <li><strong>Ordering Accuracy</strong>: Kendall τ correlation with expert sequence</li>
              <li><strong>Expert Rating</strong>: Human evaluation (1-5 scale) by domain experts</li>
              <li><strong>BLEURT Quality</strong>: Automated instruction quality assessment</li>
            </ul>
          </div>
        </div>
      </section>

      <hr style={{ margin: "24px 0", border: "none", borderTop: "1px solid var(--fg-border)" }} />

      {/* Mathematical Foundations */}
      <section style={{ marginBottom: 32 }}>
        <h2>Mathematical Foundations</h2>
        
        <h3>Stage 1: Statistical Error Metrics</h3>
        <p>For a time series <code>X = [x₁, x₂, ..., xₙ]</code>:</p>
        
        <div style={{ background: "#0a1117", padding: 16, borderRadius: 6, marginTop: 12, marginBottom: 12 }}>
          <p style={{ margin: "4px 0", fontFamily: "monospace", fontSize: 14 }}>
            <strong>Mean</strong>: μ = (1/n) Σxᵢ
          </p>
          <p style={{ margin: "4px 0", fontFamily: "monospace", fontSize: 14 }}>
            <strong>Min</strong>: min(X) = min(x₁, x₂, ..., xₙ)
          </p>
          <p style={{ margin: "4px 0", fontFamily: "monospace", fontSize: 14 }}>
            <strong>Max</strong>: max(X) = max(x₁, x₂, ..., xₙ)
          </p>
          <p style={{ margin: "16px 0 4px", fontFamily: "monospace", fontSize: 14 }}>
            <strong>Absolute Error</strong>: |ŷ - y| where ŷ is prediction, y is ground truth
          </p>
          <p style={{ margin: "4px 0", fontFamily: "monospace", fontSize: 14 }}>
            <strong>Mean Absolute Error (MAE)</strong>: (1/N) Σ|ŷᵢ - yᵢ| across N samples
          </p>
        </div>

        <h3>Cost Calculation</h3>
        <p>Token-based pricing for Azure OpenAI models:</p>
        <div style={{ background: "#0a1117", padding: 16, borderRadius: 6, marginTop: 12, marginBottom: 12 }}>
          <p style={{ margin: "4px 0", fontFamily: "monospace", fontSize: 14 }}>
            <strong>Input Cost</strong>: C_input = (tokens_input / 1000) × price_input_per_1k
          </p>
          <p style={{ margin: "4px 0", fontFamily: "monospace", fontSize: 14 }}>
            <strong>Output Cost</strong>: C_output = (tokens_output / 1000) × price_output_per_1k
          </p>
          <p style={{ margin: "4px 0", fontFamily: "monospace", fontSize: 14 }}>
            <strong>Total Cost</strong>: C_total = C_input + C_output
          </p>
          <p style={{ margin: "4px 0", fontFamily: "monospace", fontSize: 14 }}>
            <strong>Cost per Sample</strong>: C_avg = C_total / N
          </p>
        </div>

        <p><strong>Current Pricing (Nov 2024)</strong>:</p>
        <table style={{ width: "100%", fontSize: 14, marginTop: 8 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--fg-border)" }}>
              <th style={{ textAlign: "left", padding: "8px 4px" }}>Model</th>
              <th style={{ textAlign: "right", padding: "8px 4px" }}>Input ($/1K)</th>
              <th style={{ textAlign: "right", padding: "8px 4px" }}>Output ($/1K)</th>
            </tr>
          </thead>
          <tbody>
            <tr><td style={{ padding: "8px 4px" }}>gpt-4o</td><td style={{ textAlign: "right", padding: "8px 4px" }}>$0.0025</td><td style={{ textAlign: "right", padding: "8px 4px" }}>$0.01</td></tr>
            <tr><td style={{ padding: "8px 4px" }}>gpt-4o-mini</td><td style={{ textAlign: "right", padding: "8px 4px" }}>$0.00015</td><td style={{ textAlign: "right", padding: "8px 4px" }}>$0.0006</td></tr>
            <tr><td style={{ padding: "8px 4px" }}>gpt-5 (planned)</td><td style={{ textAlign: "right", padding: "8px 4px" }}>$0.00125</td><td style={{ textAlign: "right", padding: "8px 4px" }}>$0.01</td></tr>
            <tr><td style={{ padding: "8px 4px" }}>o3-2025 (planned)</td><td style={{ textAlign: "right", padding: "8px 4px" }}>$0.002</td><td style={{ textAlign: "right", padding: "8px 4px" }}>$0.008</td></tr>
          </tbody>
        </table>
      </section>

      <hr style={{ margin: "24px 0", border: "none", borderTop: "1px solid var(--fg-border)" }} />

      {/* Benchmark Philosophy */}
      <section style={{ marginBottom: 32 }}>
        <h2>Benchmark Philosophy</h2>
        
        <h3>Progressive Complexity</h3>
        <p>
          FactoryBench is designed with a "crawl-walk-run" philosophy. Each stage builds on the previous:
        </p>
        <ol>
          <li><strong>Crawl (Stage 1)</strong>: Establish baseline numerical reasoning & parsing</li>
          <li><strong>Walk (Stage 2)</strong>: Add diagnostic reasoning & context integration</li>
          <li><strong>Run (Stage 3)</strong>: Full multi-modal troubleshooting workflows</li>
        </ol>

        <h3>Reproducibility</h3>
        <p>
          All runs generate immutable JSON artifacts containing:
        </p>
        <ul>
          <li>Complete run configuration (model, dataset, parameters)</li>
          <li>Per-sample predictions & ground truth</li>
          <li>Token usage & costs</li>
          <li>Aggregate metrics</li>
          <li>Timestamps & versioning</li>
        </ul>

        <h3>Cost Transparency</h3>
        <p>
          Every run tracks token usage and API costs. This enables:
        </p>
        <ul>
          <li>Cost-quality tradeoff analysis</li>
          <li>Budget planning for large-scale evaluations</li>
          <li>Model efficiency comparisons</li>
        </ul>
      </section>

      <hr style={{ margin: "24px 0", border: "none", borderTop: "1px solid var(--fg-border)" }} />

      <footer style={{ textAlign: "center", color: "var(--fg-steel)", fontSize: 14, marginTop: 32 }}>
        <p>For implementation details, see the <a href="https://github.com/Xelerit-Robotics/FactoryBench" target="_blank" rel="noopener noreferrer">GitHub repository</a>.</p>
        <p>Maintained by <strong>Forgis</strong> • Last Updated: November 2025</p>
      </footer>
    </div>
  );
}
