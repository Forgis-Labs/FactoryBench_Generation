#!/usr/bin/env python
"""Sample Level-4 answers + their judge scores for a human-baseline review.

Draws a stratified sample across (question_type x median judge score) so the
review covers the full score range -- including the rare 0.5/1 verdicts where
the judges are most likely to be wrong -- rather than being ~90% trivial
"no-anomaly / score 0" items. Pools answers across the evaluee panel (we are
validating the *judges*, so any model's answer is fair game).

Outputs, under output/human_baseline/:
  sample.json  -- full records (question, answer, ground truth, per-judge votes)
  sample.csv   -- flat table with blank human_score / human_notes columns
  review.html  -- self-contained blind-review tool (data inlined)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# evaluee reply dirs on disk (the GPT folder is gpt_5_1-1)
PANEL = ["claude-sonnet-4_6", "deepseek-v3_2", "gpt_5_1-1",
         "mistral-large-3", "qwen-3-235b", "qwen-3-4b"]
JUDGES = ["gpt-5.1-1", "claude-sonnet-4.6", "deepseek-v3.2"]
SHORT = {"gpt-5.1-1": "GPT-5.1", "claude-sonnet-4.6": "Claude 4.6",
         "deepseek-v3.2": "DeepSeek"}


def load(repo: Path):
    root = repo / "output" / "test_eval" / "replies" / "level4"
    rows = []
    for model in PANEL:
        mdir = root / model
        if not mdir.is_dir():
            continue
        for f in mdir.glob("*_answer.json"):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            votes = d.get("llm_judge_votes") or {}
            if not all((votes.get(j) or {}).get("score") is not None for j in JUDGES):
                continue
            med = d.get("llm_judge_score")
            if med is None:
                continue
            q = d.get("prompt", "").split("Question:")[-1].strip()
            rows.append({
                "id": d.get("custom_id"),
                "model": model,
                "question_type": d.get("question_type", ""),
                "question": q,
                "ground_truth": d.get("ground_truth", ""),
                "answer": d.get("answer", ""),
                "median": float(med),
                "judges": {SHORT[j]: {"score": float(votes[j]["score"]),
                                      "reason": votes[j].get("reason", "")}
                           for j in JUDGES},
            })
    return rows


def stratified(rows, n, seed=0):
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for r in rows:
        strata[(r["question_type"], r["median"])].append(r)
    keys = sorted(strata)
    pools = {}
    for k in keys:
        idx = rng.permutation(len(strata[k]))
        pools[k] = [strata[k][i] for i in idx]
    # round-robin draw -> balanced coverage, rare strata taken in full first
    out = []
    while len(out) < n and any(pools.values()):
        for k in keys:
            if pools[k]:
                out.append(pools[k].pop())
                if len(out) >= n:
                    break
    rng.shuffle(out)  # present in random order (blind review)
    for i, r in enumerate(out):
        r["review_id"] = i + 1
    return out


def main():
    repo = Path(".")
    out_dir = repo / "output" / "human_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load(repo)
    print(f"loaded {len(rows)} judged L4 answers")
    sample = stratified(rows, 100, seed=0)

    from collections import Counter
    comp = Counter((r["question_type"], r["median"]) for r in sample)
    print("sample composition (question_type, median -> count):")
    for k in sorted(comp):
        print(f"  {k}: {comp[k]}")

    json.dump(sample, open(out_dir / "sample.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    with open(out_dir / "sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "id", "model", "question_type",
                    "GPT-5.1", "Claude 4.6", "DeepSeek", "median",
                    "human_score", "human_notes", "ground_truth", "answer"])
        for r in sample:
            j = r["judges"]
            w.writerow([r["review_id"], r["id"], r["model"], r["question_type"],
                        j["GPT-5.1"]["score"], j["Claude 4.6"]["score"],
                        j["DeepSeek"]["score"], r["median"], "", "",
                        r["ground_truth"], r["answer"]])

    build_html(sample, out_dir / "review.html")
    print(f"\nwrote sample.json, sample.csv, review.html to {out_dir}")


def build_html(sample, path: Path):
    tmpl = _HTML_TEMPLATE.replace(
        "__DATA__", json.dumps(sample, ensure_ascii=False).replace("</", "<\\/"))
    path.write_text(tmpl, encoding="utf-8")


_HTML_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FactoryBench L4 - human baseline review</title>
<style>
:root{--ink:#122128;--orange:#FF5A00;--steel:#878F92;--amber:#FFF3CD;--panel:#F0F4F8;--line:#dfe6ea}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:#fafbfc}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
header h1{font-size:15px;margin:0;font-weight:700}
.prog{font-size:13px;color:var(--steel)}
.bar{height:6px;background:var(--panel);border-radius:4px;flex:1;min-width:120px;overflow:hidden}
.bar > i{display:block;height:100%;background:var(--orange);width:0}
button{font:inherit;border:1px solid var(--line);background:#fff;border-radius:8px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--steel)}
.wrap{max-width:900px;margin:18px auto;padding:0 16px}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 1px 3px rgba(18,33,40,.05)}
.meta{font-size:12px;color:var(--steel);letter-spacing:.02em;text-transform:uppercase;margin-bottom:10px}
.lab{font-size:12px;font-weight:700;color:var(--steel);text-transform:uppercase;letter-spacing:.04em;margin:16px 0 6px}
.q{background:var(--panel);border-radius:10px;padding:12px 14px}
.gt{background:var(--amber);border-radius:10px;padding:12px 14px;border:1px solid #f0e2b0}
.ans{white-space:pre-wrap;background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px;max-height:420px;overflow:auto}
.score{display:flex;gap:10px;margin-top:8px}
.score button{font-size:16px;font-weight:700;padding:10px 20px;border-radius:10px}
.score button.sel{background:var(--orange);color:#fff;border-color:var(--orange)}
.notes{width:100%;margin-top:10px;border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit;min-height:44px}
.nav{display:flex;justify-content:space-between;align-items:center;margin-top:16px;gap:10px}
.judges{margin-top:14px;border-top:1px dashed var(--line);padding-top:12px}
.judges.hidden{display:none}
.jrow{display:flex;gap:8px;margin:6px 0;font-size:13px}
.pill{flex:0 0 76px;font-weight:700}
.s0{color:#b23b1e}.s5{color:#c98a2b}.s1{color:#2e8b57}
.reveal{color:var(--orange);cursor:pointer;font-size:13px;font-weight:600}
.done{color:#2e8b57;font-weight:700}
small{color:var(--steel)}
</style></head><body>
<header>
  <h1>FactoryBench L4 &mdash; human baseline</h1>
  <div class="bar"><i id="pfill"></i></div>
  <div class="prog"><span id="pcount">0</span>/<span id="ptot">0</span> scored</div>
  <button onclick="jump(-1)">&larr; Prev</button>
  <button onclick="jump(1)">Next &rarr;</button>
  <button onclick="exportCSV()">Export CSV</button>
  <button onclick="agree()">Agreement</button>
</header>
<div class="wrap"><div class="card" id="card"></div>
  <div class="nav">
    <button onclick="jump(-1)">&larr; Prev</button>
    <small>Score blind, then optionally reveal the judges. Progress autosaves in this browser.</small>
    <button onclick="jump(1)">Next &rarr;</button>
  </div>
</div>
<script>
const DATA = __DATA__;
const KEY = "fb_l4_human_v1";
let human = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;
const $ = s => document.querySelector(s);
document.getElementById("ptot").textContent = DATA.length;

function setScore(id, v){ human[id] = {...(human[id]||{}), score:v}; save(); render(); }
function setNotes(id, v){ human[id] = {...(human[id]||{}), notes:v}; save(); }
function save(){ localStorage.setItem(KEY, JSON.stringify(human)); prog(); }
function prog(){
  const n = Object.values(human).filter(h=>h && h.score!==undefined).length;
  $("#pcount").textContent = n;
  $("#pfill").style.width = (100*n/DATA.length)+"%";
}
function jump(d){ i=Math.max(0,Math.min(DATA.length-1,i+d)); render(); }

function render(){
  const r = DATA[i]; const h = human[r.id]||{};
  const sc = h.score;
  const jb = Object.entries(r.judges).map(([n,o])=>{
    const cls = o.score===0?"s0":o.score===0.5?"s5":"s1";
    return `<div class="jrow"><span class="pill ${cls}">${n}: ${o.score}</span><span>${esc(o.reason)}</span></div>`;
  }).join("");
  const mcls = r.median===0?"s0":r.median===0.5?"s5":"s1";
  $("#card").innerHTML = `
    <div class="meta">Item ${r.review_id} of ${DATA.length} &middot; ${r.question_type} &middot; <small>${r.model}</small></div>
    <div class="lab">Question</div><div class="q">${esc(r.question)}</div>
    <div class="lab">Ground-truth reference</div><div class="gt">${esc(r.ground_truth)}</div>
    <div class="lab">Model answer</div><div class="ans">${esc(r.answer)}</div>
    <div class="lab">Your verdict ${sc!==undefined?'<span class="done">&#10003; scored</span>':''}</div>
    <div class="score">
      ${[0,0.5,1].map(v=>`<button class="${sc===v?'sel':''}" onclick="setScore('${r.id}',${v})">${v}</button>`).join("")}
    </div>
    <textarea class="notes" placeholder="notes (optional)" onchange="setNotes('${r.id}',this.value)">${esc(h.notes||"")}</textarea>
    <div class="reveal" onclick="this.nextElementSibling.classList.toggle('hidden')">&#9662; reveal judges (after you score)</div>
    <div class="judges ${sc===undefined?'hidden':'hidden'}">
      <div class="jrow"><span class="pill ${mcls}">median: ${r.median}</span></div>${jb}
    </div>`;
}
function esc(s){ return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function exportCSV(){
  const rows = [["review_id","id","model","question_type","human_score","human_notes","median","GPT-5.1","Claude 4.6","DeepSeek"]];
  DATA.forEach(r=>{const h=human[r.id]||{};rows.push([r.review_id,r.id,r.model,r.question_type,
    h.score===undefined?"":h.score, (h.notes||"").replace(/\n/g," "), r.median,
    r.judges["GPT-5.1"].score, r.judges["Claude 4.6"].score, r.judges["DeepSeek"].score]);});
  const csv = rows.map(r=>r.map(x=>`"${String(x).replace(/"/g,'""')}"`).join(",")).join("\n");
  const ov=document.createElement("div");
  ov.style.cssText="position:fixed;inset:0;background:rgba(18,33,40,.45);display:flex;align-items:center;justify-content:center;z-index:99";
  ov.innerHTML=`<div style="background:#fff;padding:18px;border-radius:14px;max-width:760px;width:92%">
    <div style="font-weight:700;margin-bottom:8px">Your scores &mdash; copy this, or Download</div>
    <textarea style="width:100%;height:320px;font:12px/1.4 monospace"></textarea>
    <div style="display:flex;gap:8px;margin-top:10px;justify-content:flex-end">
      <button id="dl">Download CSV</button><button id="cl">Close</button></div></div>`;
  document.body.appendChild(ov);
  const ta=ov.querySelector("textarea"); ta.value=csv; ta.focus(); ta.select();
  ov.querySelector("#cl").onclick=()=>ov.remove();
  ov.querySelector("#dl").onclick=()=>{try{const a=document.createElement("a");
    a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download="human_scores.csv";a.click();}catch(e){alert("Download blocked here; copy the text instead.");}};
}
function agree(){
  const scored = DATA.filter(r=>(human[r.id]||{}).score!==undefined);
  if(!scored.length){alert("Score some items first.");return;}
  let exact=0; const conf={};
  scored.forEach(r=>{const hs=human[r.id].score; if(hs===r.median)exact++;
    const k=hs+"|"+r.median; conf[k]=(conf[k]||0)+1;});
  alert(`Human vs. judge-median on ${scored.length} scored items\n`+
        `exact agreement: ${(100*exact/scored.length).toFixed(1)}%\n\n`+
        Object.entries(conf).sort().map(([k,v])=>`human ${k.split("|")[0]} / median ${k.split("|")[1]}: ${v}`).join("\n"));
}
render(); prog();
</script></body></html>"""


if __name__ == "__main__":
    main()
