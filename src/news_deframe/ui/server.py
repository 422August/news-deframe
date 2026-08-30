"""Lightweight built-in HTTP server providing a browser UI for news-deframe.

Start with:
    news-deframe --ui [--port PORT] [--host HOST]

Architecture
------------
  POST /api/run       → reads files, enqueues a background job, returns {"job_id": "..."}
  GET  /api/status    → ?id=<job_id> — returns job status / result (quick poll)

This async-job design avoids HTTP connection timeouts caused by port-forwarding
proxies (e.g. VS Code / GitHub Codespaces) when model loading takes > ~30 s.
"""
from __future__ import annotations

import json
import socketserver
import threading
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# ── Job registry ─────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}   # job_id → {"status": str, "result": dict|None}
_jobs_lock = threading.Lock()


def _new_job(jid: str | None = None) -> str:
    if not jid:
        jid = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[jid] = {"status": "pending", "result": None}
    return jid


def _set_job(jid: str, status: str, result: dict | None = None) -> None:
    with _jobs_lock:
        _jobs[jid] = {"status": status, "result": result}


def _get_job(jid: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(jid)


# ── HTML page ─────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>news-deframe UI</title>
<style>
  :root {
    --bg:#0f1117; --surface:#1a1d27; --border:#2d3147;
    --accent:#6c8ef5; --accent2:#5dd9c1;
    --text:#e0e4f6; --muted:#8891b4;
    --error:#f87171; --success:#4ade80; --warn:#fbbf24;
    --radius:10px; --mono:'JetBrains Mono','Fira Code',monospace;
  }
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;min-height:100vh;display:flex;flex-direction:column}
  header{padding:18px 28px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;background:var(--surface)}
  header h1{font-size:1.15rem;font-weight:600;letter-spacing:.3px}
  .badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:.7rem;font-weight:600;letter-spacing:.5px;text-transform:uppercase;background:color-mix(in srgb,var(--accent) 18%,transparent);color:var(--accent);border:1px solid color-mix(in srgb,var(--accent) 35%,transparent)}
  main{flex:1;display:grid;grid-template-columns:360px 1fr;height:calc(100vh - 57px)}
  .sidebar{background:var(--surface);border-right:1px solid var(--border);padding:24px 20px;display:flex;flex-direction:column;gap:20px;overflow-y:auto}
  .result-panel{padding:24px 28px;overflow-y:auto;display:flex;flex-direction:column;gap:16px}
  label{font-size:.82rem;color:var(--muted);display:block;margin-bottom:6px}
  .field{display:flex;flex-direction:column;gap:4px}
  select,input[type=number]{background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);padding:8px 12px;font-size:.88rem;width:100%;outline:none;transition:border-color .15s}
  select:focus,input[type=number]:focus{border-color:var(--accent)}
  input[type=range]{padding:4px 0;cursor:pointer;accent-color:var(--accent);width:100%}
  .upload-area{border:1.5px dashed var(--border);border-radius:var(--radius);padding:14px;text-align:center;cursor:pointer;transition:border-color .2s,background .2s;position:relative}
  .upload-area:hover{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 6%,transparent)}
  .upload-area input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .upload-area .icon{font-size:1.4rem;margin-bottom:6px}
  .upload-area p{font-size:.8rem;color:var(--muted)}
  .fname{font-size:.82rem;color:var(--accent2);margin-top:4px;word-break:break-all}
  .thr-row{display:flex;justify-content:space-between;align-items:center;font-size:.8rem;color:var(--muted)}
  .thr-row strong{color:var(--text)}
  button.run-btn{background:var(--accent);color:#fff;border:none;border-radius:var(--radius);padding:10px 0;font-size:.9rem;font-weight:600;cursor:pointer;width:100%;transition:opacity .15s,transform .1s}
  button.run-btn:hover{opacity:.88}
  button.run-btn:active{transform:scale(.98)}
  button.run-btn:disabled{opacity:.4;cursor:not-allowed}
  .section-title{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--muted);padding-bottom:8px;border-bottom:1px solid var(--border)}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px}
  .card h3{font-size:.9rem;margin-bottom:10px;color:var(--accent)}
  .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px}
  .stat{background:color-mix(in srgb,var(--accent) 7%,transparent);border:1px solid color-mix(in srgb,var(--accent) 18%,transparent);border-radius:8px;padding:10px 12px;text-align:center}
  .stat .val{font-size:1.4rem;font-weight:700;color:var(--accent)}
  .stat .lbl{font-size:.72rem;color:var(--muted);margin-top:2px}
  .sentence-list{display:flex;flex-direction:column;gap:6px}
  .sent{font-size:.82rem;padding:8px 12px;border-radius:6px;background:color-mix(in srgb,var(--bg) 60%,transparent);border-left:3px solid var(--border);line-height:1.6}
  .sent.ua{border-left-color:var(--error)}
  .sent.ub{border-left-color:var(--accent2)}
  .sent.ok{border-left-color:var(--success)}
  .align-row{display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:start;font-size:.82rem}
  .sim{text-align:center;padding-top:8px;font-size:.75rem;color:var(--muted);white-space:nowrap}
  .sim span{display:block;font-size:.9rem;font-weight:700;color:var(--accent2)}
  .cluster-badge{display:inline-block;padding:1px 7px;border-radius:20px;font-size:.7rem;font-weight:600;background:color-mix(in srgb,var(--accent2) 15%,transparent);color:var(--accent2);border:1px solid color-mix(in srgb,var(--accent2) 30%,transparent);margin-right:4px}
  pre{font-family:var(--mono);font-size:.78rem;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:14px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;color:var(--text);max-height:400px;overflow-y:auto}
  .status-bar{padding:8px 12px;border-radius:8px;font-size:.82rem;display:flex;align-items:center;gap:8px}
  .status-bar.loading{background:color-mix(in srgb,var(--warn) 12%,transparent);color:var(--warn)}
  .status-bar.error{background:color-mix(in srgb,var(--error) 12%,transparent);color:var(--error)}
  .status-bar.ok{background:color-mix(in srgb,var(--success) 12%,transparent);color:var(--success)}
  .spinner{display:inline-block;animation:spin .7s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .progress-row{display:flex;align-items:center;gap:10px;margin-bottom:8px}
  .progress-row label{min-width:80px;font-size:.8rem;color:var(--muted);margin:0}
  .progress-bg{flex:1;background:var(--border);border-radius:4px;height:8px;overflow:hidden}
  .passive-bar{height:8px;border-radius:4px;background:linear-gradient(90deg,var(--error),var(--warn));transition:width .4s}
  .tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:.7rem;background:color-mix(in srgb,var(--muted) 14%,transparent);color:var(--muted);margin:1px}
  .multi-file-list{display:flex;flex-direction:column;gap:6px;margin-top:8px}
  .file-item{display:flex;align-items:center;justify-content:space-between;background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:6px 10px;font-size:.82rem}
  .file-item button{background:none;border:none;color:var(--error);cursor:pointer;font-size:1rem;line-height:1;padding:0 2px}
  @media(max-width:768px){main{grid-template-columns:1fr}.sidebar{border-right:none;border-bottom:1px solid var(--border)}}
</style>
</head>
<body>
<header>
  <span>📰</span>
  <h1>news-deframe</h1>
  <span class="badge">UI</span>
  <span style="margin-left:auto;font-size:.78rem;color:var(--muted)">Structural framing analysis</span>
</header>
<main>
  <aside class="sidebar">
    <div class="field">
      <span class="section-title">Mode</span>
      <select id="mode" onchange="onModeChange()">
        <option value="diff">Diff — compare two articles</option>
        <option value="analyze">Analyze — multiple articles</option>
      </select>
    </div>

    <div id="diff-inputs">
      <div class="field" style="margin-bottom:12px">
        <label>Article A</label>
        <div class="upload-area">
          <div class="icon">📄</div>
          <p>Click or drag a .txt file</p>
          <div class="fname" id="fname-a">No file selected</div>
          <input type="file" accept=".txt,text/plain" id="file-a" onchange="onFileChange('a')">
        </div>
      </div>
      <div class="field">
        <label>Article B</label>
        <div class="upload-area">
          <div class="icon">📄</div>
          <p>Click or drag a .txt file</p>
          <div class="fname" id="fname-b">No file selected</div>
          <input type="file" accept=".txt,text/plain" id="file-b" onchange="onFileChange('b')">
        </div>
      </div>
    </div>

    <div id="analyze-inputs" style="display:none">
      <div class="field">
        <label>Articles (2 or more)</label>
        <div class="upload-area" style="padding:12px">
          <div class="icon">📂</div>
          <p>Click to select multiple .txt files</p>
          <input type="file" accept=".txt,text/plain" id="file-multi" multiple onchange="onMultiChange()">
        </div>
        <div class="multi-file-list" id="multi-list"></div>
      </div>
      <div class="field">
        <label>Clusters (optional)</label>
        <input type="number" id="n-clusters" min="1" max="20" placeholder="Auto (min 3, n_articles)">
      </div>
      <div class="field" style="flex-direction:row;align-items:center;gap:10px">
        <input type="checkbox" id="chk-details" style="width:auto;accent-color:var(--accent)">
        <label for="chk-details" style="margin:0;color:var(--text)">Show claim-level details</label>
      </div>
    </div>

    <div class="field">
      <div class="thr-row">
        <label style="margin:0">Similarity threshold</label>
        <strong id="thr-val">0.60</strong>
      </div>
      <input type="range" id="threshold" min="0" max="1" step="0.01" value="0.60"
             oninput="document.getElementById('thr-val').textContent=parseFloat(this.value).toFixed(2)">
    </div>

    <button class="run-btn" id="run-btn" onclick="runAnalysis()">▶ Run Analysis</button>
  </aside>

  <section class="result-panel" id="result-panel">
    <div style="margin:auto;text-align:center;color:var(--muted)">
      <div style="font-size:3rem;margin-bottom:12px">📊</div>
      <div>Select articles and click "Run Analysis"</div>
    </div>
  </section>
</main>

<script>
let fileA = null, fileB = null, multiFiles = [];
let _pollTimer = null;

function onModeChange() {
  const m = document.getElementById('mode').value;
  document.getElementById('diff-inputs').style.display    = m === 'diff'    ? '' : 'none';
  document.getElementById('analyze-inputs').style.display = m === 'analyze' ? '' : 'none';
}

function onFileChange(w) {
  const f = document.getElementById('file-' + w).files[0];
  if (w === 'a') { fileA = f; document.getElementById('fname-a').textContent = f ? f.name : 'No file selected'; }
  else           { fileB = f; document.getElementById('fname-b').textContent = f ? f.name : 'No file selected'; }
}

function onMultiChange() {
  const inp = document.getElementById('file-multi');
  Array.from(inp.files).forEach(f => { if (!multiFiles.find(x => x.name === f.name)) multiFiles.push(f); });
  renderMultiList();
  inp.value = '';
}
function removeFile(n) { multiFiles = multiFiles.filter(f => f.name !== n); renderMultiList(); }
function renderMultiList() {
  document.getElementById('multi-list').innerHTML = multiFiles.map(f =>
    `<div class="file-item"><span>📄 ${esc(f.name)}</span><button onclick="removeFile('${esc(f.name)}')" title="Remove">✕</button></div>`
  ).join('');
}
function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

// ── Run ───────────────────────────────────────────────────────────────────────
function genId() {
  // Use crypto.randomUUID when available (all modern browsers), fallback otherwise.
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

async function runAnalysis() {
  const mode = document.getElementById('mode').value;
  const threshold = parseFloat(document.getElementById('threshold').value);
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  clearTimeout(_pollTimer);

  if (mode === 'diff' && (!fileA || !fileB)) {
    setResult('<div class="status-bar error">⚠ Please select both Article A and Article B.</div>');
    btn.disabled = false; return;
  }
  if (mode === 'analyze' && multiFiles.length < 2) {
    setResult('<div class="status-bar error">⚠ Please select at least 2 articles.</div>');
    btn.disabled = false; return;
  }

  // Generate the job ID here so we can start polling even if the POST
  // response is never received (the proxy may drop the connection mid-write).
  const jobId = genId();

  const fd = new FormData();
  fd.append('job_id',   jobId);   // server registers this exact ID
  fd.append('mode',     mode);
  fd.append('threshold', threshold);

  if (mode === 'diff') {
    fd.append('file_a', fileA);
    fd.append('file_b', fileB);
  } else {
    multiFiles.forEach(f => fd.append('files', f));
    const nc = document.getElementById('n-clusters').value;
    if (nc) fd.append('n_clusters', nc);
    if (document.getElementById('chk-details').checked) fd.append('details', '1');
  }

  setResult(`<div class="status-bar loading"><span class="spinner">⟳</span> Submitting job…</div>`);

  // Fire-and-forget POST. If the proxy drops the response (BrokenPipe), the
  // job is still running server-side because it was registered before the write.
  fetch('/api/run', { method: 'POST', body: fd }).catch(() => { /* expected on proxy timeout */ });

  // Start polling immediately — works whether or not the POST response arrived.
  setResult(`<div class="status-bar loading"><span class="spinner">⟳</span> Analysing — please wait. NLP model loading may take up to a minute on first run.</div>`);
  poll(jobId, mode, btn, 0);
}

function poll(jobId, mode, btn, elapsed) {
  _pollTimer = setTimeout(async () => {
    // Outer try/catch: any uncaught error reschedules the next poll instead of
    // silently breaking the chain (which is what caused the frozen counter).
    try {
      let data;
      try {
        // Hard 10 s timeout on each status request — prevents the fetch from
        // hanging indefinitely when the VS Code proxy stalls while NLP runs.
        const ctrl = new AbortController();
        const abort = setTimeout(() => ctrl.abort(), 10000);
        const resp  = await fetch(`/api/status?id=${encodeURIComponent(jobId)}`, { signal: ctrl.signal });
        clearTimeout(abort);
        data = await resp.json();
      } catch (_e) {
        // Network error or AbortError — reschedule without incrementing so
        // the user can tell the counter froze vs. a real timeout.
        poll(jobId, mode, btn, elapsed);
        return;
      }

      if (data.status === 'done') {
        try {
          renderResult(mode, data.result);
        } catch (renderErr) {
          setResult(`<div class="status-bar error">⚠ Render error: ${esc(String(renderErr))}</div><pre>${esc(JSON.stringify(data.result, null, 2).slice(0, 3000))}</pre>`);
        }
        btn.disabled = false;
      } else if (data.status === 'error') {
        setResult(`<div class="status-bar error">⚠ ${esc(data.result.error)}</div><pre>${esc(data.result.traceback || '')}</pre>`);
        btn.disabled = false;
      } else {
        // pending / running / not_found — keep polling
        const secs = elapsed + 2;
        setResult(`<div class="status-bar loading"><span class="spinner">⟳</span> Analysing… ${secs}s elapsed. NLP model loading may take up to a minute on first run.</div>`);
        poll(jobId, mode, btn, secs);
      }
    } catch (outerErr) {
      // Should never reach here, but ensures the chain is never silently broken.
      poll(jobId, mode, btn, elapsed + 2);
    }
  }, 2000);
}


function setResult(html) { document.getElementById('result-panel').innerHTML = html; }
function pct(v) { return (v * 100).toFixed(1) + '%'; }
function renderResult(mode, data) { mode === 'diff' ? renderDiff(data) : renderAnalyze(data); }

function renderDiff(d) {
  const r = d.report;
  const matched = r.alignments.filter(a => a.sent_b !== null);
  let html = `
  <div class="status-bar ok" style="margin-bottom:4px">✓ Analysis complete</div>
  <div class="card"><h3>📊 Overview</h3>
    <div class="stat-grid">
      <div class="stat"><div class="val">${r.alignments.length}</div><div class="lbl">Sentences in A</div></div>
      <div class="stat"><div class="val">${matched.length}</div><div class="lbl">Matched pairs</div></div>
      <div class="stat"><div class="val">${r.unshared_claims_a.length}</div><div class="lbl">A-only claims</div></div>
      <div class="stat"><div class="val">${r.unshared_claims_b.length}</div><div class="lbl">B-only claims</div></div>
    </div>
    <div style="margin-top:14px">
      <div class="progress-row">
        <label>${esc(r.article_a_id)} passive</label>
        <div class="progress-bg"><div class="passive-bar" style="width:${pct(r.passive_ratio_a)}"></div></div>
        <span style="font-size:.8rem;min-width:40px">${pct(r.passive_ratio_a)}</span>
      </div>
      <div class="progress-row">
        <label>${esc(r.article_b_id)} passive</label>
        <div class="progress-bg"><div class="passive-bar" style="width:${pct(r.passive_ratio_b)}"></div></div>
        <span style="font-size:.8rem;min-width:40px">${pct(r.passive_ratio_b)}</span>
      </div>
    </div>
  </div>`;

  if (r.unshared_claims_a.length)
    html += `<div class="card"><h3 style="color:var(--error)">🔴 ${esc(r.article_a_id)} — unique claims</h3>
    <div class="sentence-list">${r.unshared_claims_a.map(s=>`<div class="sent ua">${esc(s)}</div>`).join('')}</div></div>`;

  if (r.unshared_claims_b.length)
    html += `<div class="card"><h3 style="color:var(--accent2)">🟢 ${esc(r.article_b_id)} — unique claims</h3>
    <div class="sentence-list">${r.unshared_claims_b.map(s=>`<div class="sent ub">${esc(s)}</div>`).join('')}</div></div>`;

  if (matched.length)
    html += `<div class="card"><h3>🔗 Aligned pairs (${matched.length})</h3>
    <div class="sentence-list">${matched.slice(0,30).map(a=>`
      <div class="align-row">
        <div class="sent ok">${esc(a.sent_a)}</div>
        <div class="sim"><span>${a.similarity_score.toFixed(2)}</span>sim</div>
        <div class="sent ok">${esc(a.sent_b)}</div>
      </div>`).join('')}
    ${matched.length>30?`<div style="color:var(--muted);font-size:.8rem;text-align:center">…${matched.length-30} more</div>`:''}
    </div></div>`;

  html += rawJson(d.report);
  setResult(html);
}

function renderAnalyze(d) {
  const a = d.analysis;

  // ── consensus_view.claims (not consensus_claims)
  const claims   = (a.consensus_view && a.consensus_view.claims) || [];
  // ── entity_outlet_matrix (not entity_matrix), profiles (not entities)
  const eom      = a.entity_outlet_matrix || {};
  const profiles = eom.profiles || [];
  const artIds   = eom.article_ids || [];
  // ── framing_clusters
  const fcs = a.framing_clusters || [];

  let html = `<div class="status-bar ok" style="margin-bottom:4px">✓ Analysis complete — event: ${esc(a.event_id)}</div>
  <div class="card"><h3>📊 Overview</h3><div class="stat-grid">
    <div class="stat"><div class="val">${(a.articles||[]).length}</div><div class="lbl">Articles</div></div>
    <div class="stat"><div class="val">${fcs.length || '—'}</div><div class="lbl">Framing clusters</div></div>
    <div class="stat"><div class="val">${claims.length || '—'}</div><div class="lbl">Claim clusters</div></div>
  </div></div>`;

  // ── Framing clusters
  if (fcs.length) {
    html += `<div class="card"><h3>🗂 Framing Clusters</h3>`;
    fcs.forEach((cl, i) => {
      html += `<div style="margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <span class="cluster-badge">${esc(cl.label || 'Cluster ' + (i+1))}</span>
          <span style="font-size:.82rem;color:var(--muted)">${(cl.article_ids||[]).map(esc).join(', ')}</span>
        </div>
      </div>`;
    });
    html += `</div>`;
  }

  // ── Claim clusters (consensus_view.claims — representative + coverage)
  if (claims.length) {
    // Group by category
    const widely  = claims.filter(c => c.coverage_category === 'Widely shared');
    const common  = claims.filter(c => c.coverage_category === 'Commonly reported');
    const single  = claims.filter(c => c.coverage_count === 1);

    html += `<div class="card"><h3>📋 Claim Coverage</h3>
    <div class="stat-grid" style="margin-bottom:12px">
      <div class="stat"><div class="val">${widely.length}</div><div class="lbl">Widely shared</div></div>
      <div class="stat"><div class="val">${common.length}</div><div class="lbl">Commonly reported</div></div>
      <div class="stat"><div class="val">${single.length}</div><div class="lbl">Single-outlet</div></div>
    </div>
    <div class="sentence-list">
      ${claims.slice(0, 12).map(c => {
        const ratio = c.total_articles ? `${c.coverage_count}/${c.total_articles}` : c.coverage_count;
        const absent = (c.outlets_absent||[]).length ? ` <span style="color:var(--error);font-size:.75rem">missing: ${c.outlets_absent.map(esc).join(', ')}</span>` : '';
        return `<div class="sent ok"><strong>[${ratio}]</strong> <span style="color:var(--muted);font-size:.75rem">${esc(c.coverage_category||'')}</span>${absent}<br>${esc(c.representative)}</div>`;
      }).join('')}
      ${claims.length > 12 ? `<div style="color:var(--muted);font-size:.8rem;text-align:center">…${claims.length-12} more</div>` : ''}
    </div></div>`;
  }

  // ── Entity × outlet framing table
  // EntityOutletProfile: entity_name, article_id, agent_ratio, patient_ratio, modifiers
  // We pivot: rows = distinct entity names, cols = article_ids
  if (profiles.length && artIds.length) {
    // Build pivot map: entity_name → { article_id → profile }
    const pivot = {};
    const entityNames = [...new Set(profiles.map(p => p.entity_name))].slice(0, 8);
    profiles.forEach(p => {
      if (!pivot[p.entity_name]) pivot[p.entity_name] = {};
      pivot[p.entity_name][p.article_id] = p;
    });

    html += `<div class="card"><h3>👤 Entity Framing by Outlet</h3>
    <div style="overflow-x:auto"><table style="width:100%;font-size:.76rem;border-collapse:collapse">
      <thead><tr>
        <th style="text-align:left;padding:4px 8px;color:var(--muted)">Entity</th>
        ${artIds.map(id => `<th style="padding:4px 8px;color:var(--muted);text-align:center">${esc(id)}</th>`).join('')}
      </tr></thead>
      <tbody>
        ${entityNames.map(ename => `<tr>
          <td style="padding:4px 8px;font-weight:600">${esc(ename)}</td>
          ${artIds.map(aid => {
            const p = pivot[ename] && pivot[ename][aid];
            if (!p) return `<td style="padding:4px 8px;text-align:center;color:var(--border)">—</td>`;
            const mods = (p.modifiers||[]).slice(0,3).map(m=>`<span class="tag">${esc(m)}</span>`).join('');
            return `<td style="padding:4px 8px;text-align:center">
              <div style="font-size:.7rem;color:var(--muted)">Ag ${(p.agent_ratio*100).toFixed(0)}% Pt ${(p.patient_ratio*100).toFixed(0)}%</div>
              ${mods || ''}
            </td>`;
          }).join('')}
        </tr>`).join('')}
      </tbody>
    </table></div></div>`;
  }

  html += rawJson(d.analysis);
  setResult(html);
}


function rawJson(obj) {
  return `<div class="card"><h3>📋 Raw JSON</h3><pre>${esc(JSON.stringify(obj,null,2))}</pre></div>`;
}
</script>
</body>
</html>
"""


# ── Threading HTTP server ─────────────────────────────────────────────────────

class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ── Request handler ───────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_json_or_html(_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._handle_status()
        else:
            self.send_error(404, "Not found")

    def _handle_status(self):
        qs = parse_qs(urlparse(self.path).query)
        jid = (qs.get("id") or [""])[0]
        job = _get_job(jid)
        if job is None:
            body = json.dumps({"status": "not_found"}).encode("utf-8")
        else:
            body = json.dumps(job, ensure_ascii=False).encode("utf-8")
        self._send_json_or_html(body, "application/json; charset=utf-8")

    def _send_json_or_html(self, body: bytes, ctype: str, status: int = 200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Proxy dropped the connection — the job is already queued,
            # and the client will recover by polling /api/status.
            pass

    # ── POST ─────────────────────────────────────────────────────────────────
    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            self.send_error(404); return

        # Read body immediately in this thread (socket is not thread-safe).
        # If multipart parsing fails we still try to send an error response,
        # but a BrokenPipeError there is silenced by _send_json_or_html.
        try:
            fields = _parse_multipart(self)
        except Exception as exc:
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send_json_or_html(err, "application/json; charset=utf-8", 400)
            return

        # Use a client-supplied job_id so polling works even when the POST
        # response is dropped by the proxy before reaching the browser.
        client_jid = _field(fields, "job_id")
        jid = _new_job(client_jid)

        t = threading.Thread(target=_run_job, args=(jid, fields), daemon=True)
        t.start()

        # Best-effort response — BrokenPipeError is silenced inside _send_json_or_html.
        body = json.dumps({"job_id": jid}).encode("utf-8")
        self._send_json_or_html(body, "application/json; charset=utf-8")


# ── Multipart parser (stdlib only) ───────────────────────────────────────────

def _parse_multipart(handler: BaseHTTPRequestHandler) -> dict:
    from email.parser import BytesParser

    ctype = handler.headers.get("Content-Type", "")
    if "boundary=" not in ctype:
        raise ValueError("Expected multipart/form-data Content-Type")

    length = int(handler.headers.get("Content-Length", 0))
    raw = handler.rfile.read(length)

    msg = BytesParser().parsebytes(
        b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw
    )
    payload = msg.get_payload()
    if not isinstance(payload, list):
        raise ValueError("Failed to parse multipart body")

    fields: dict[str, list] = {}
    for part in payload:
        disp = part.get("Content-Disposition", "")
        params: dict[str, str] = {}
        for seg in disp.split(";"):
            seg = seg.strip()
            if "=" in seg:
                k, v = seg.split("=", 1)
                params[k.strip()] = v.strip().strip('"')
        name = params.get("name", "")
        filename = params.get("filename")
        data = part.get_payload(decode=True) or b""
        fields.setdefault(name, []).append({"filename": filename, "data": data})

    return fields


def _field(fields: dict, name: str, default: str | None = None) -> str | None:
    parts = fields.get(name)
    if not parts:
        return default
    return parts[0]["data"].decode("utf-8", errors="replace")


# ── Background job runner ─────────────────────────────────────────────────────

def _run_job(jid: str, fields: dict) -> None:
    _set_job(jid, "running")
    try:
        result = _handle_api(fields)
        _set_job(jid, "done", result)
    except Exception as exc:
        _set_job(jid, "error", {"error": str(exc), "traceback": traceback.format_exc()})


def _handle_api(fields: dict) -> dict:
    mode = _field(fields, "mode", "diff")
    threshold = float(_field(fields, "threshold") or "0.60")

    from news_deframe.cli import _parse_article
    from news_deframe.diff.coverage import compute_coverage
    from news_deframe.formatters.json_export import report_to_json

    if mode == "diff":
        parts_a = fields.get("file_a", [])
        parts_b = fields.get("file_b", [])
        if not parts_a or not parts_b:
            raise ValueError("Both file_a and file_b are required for diff mode.")
        text_a = parts_a[0]["data"].decode("utf-8", errors="replace")
        text_b = parts_b[0]["data"].decode("utf-8", errors="replace")
        id_a = (parts_a[0]["filename"] or "article_a").rsplit(".", 1)[0]
        id_b = (parts_b[0]["filename"] or "article_b").rsplit(".", 1)[0]

        art_a = _parse_article(text_a, id_a)
        art_b = _parse_article(text_b, id_b)
        report = compute_coverage(art_a, art_b, threshold=threshold)
        return {"report": json.loads(report_to_json(report))}

    else:  # analyze
        file_parts = fields.get("files", [])
        if len(file_parts) < 2:
            raise ValueError("At least 2 files are required for analyze mode.")

        n_clusters_raw = _field(fields, "n_clusters")
        n_clusters = int(n_clusters_raw) if n_clusters_raw and n_clusters_raw.strip() else None

        from news_deframe.analysis.event import run_event_analysis

        articles = []
        for fp in file_parts:
            text = fp["data"].decode("utf-8", errors="replace")
            art_id = (fp["filename"] or "article").rsplit(".", 1)[0]
            articles.append(_parse_article(text, art_id))

        analysis = run_event_analysis(
            event_id="event",
            articles=articles,
            threshold=threshold,
            n_framing_clusters=n_clusters,
        )
        return {"analysis": json.loads(analysis.model_dump_json())}


# ── Public entry point ────────────────────────────────────────────────────────

def launch(host: str = "127.0.0.1", port: int = 8080, open_browser: bool = True) -> None:
    """Start the threaded web server (blocking)."""
    server = _ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print(f"\n  🌐  news-deframe UI  →  {url}")
    print("  Press Ctrl+C to stop\n")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()
