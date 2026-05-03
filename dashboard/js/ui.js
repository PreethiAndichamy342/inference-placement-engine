'use strict';

// ── Escape helpers ────────────────────────────────────────────────────────

function esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function fmtTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], {
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch { return iso; }
}

function envIcon(env) {
  return { aws: '☁', gcp: '🌐', azure: '⬡', on_prem: '🖥', neocloud: '✦' }[env] || '•';
}

// ── Render dispatcher ─────────────────────────────────────────────────────

function renderAll(newIds = [], total = null, returned = null) {
  renderConnection();
  renderStats();
  renderServerCards();
  renderRoutingLog(newIds, total, returned);
}

// ── Connection indicator ──────────────────────────────────────────────────

function renderConnection() {
  const el  = document.getElementById('live-indicator');
  const lbl = document.getElementById('live-label');
  if (state.connected) {
    el.className    = 'live-indicator connected';
    lbl.textContent = 'Live · 5s';
  } else {
    el.className    = 'live-indicator error';
    lbl.textContent = 'Disconnected';
  }
}

// ── Stat cards ────────────────────────────────────────────────────────────

function renderStats() {
  if (state.health) {
    document.getElementById('stat-healthy').textContent =
      state.health.healthy_server_count;
    document.getElementById('stat-total-servers').textContent =
      `of ${state.health.total_server_count} registered`;
  }

  if (state.logStats) {
    const s      = state.logStats;
    const phi    = (s.by_sensitivity.phi || 0) + (s.by_sensitivity.phi_strict || 0);
    const routed = s.total - s.rejected_count;

    document.getElementById('stat-routed').textContent = routed;
    document.getElementById('stat-rejected-sub').textContent =
      `${s.rejected_count} rejected`;
    document.getElementById('stat-phi').textContent = phi;
    document.getElementById('stat-phi-sub').textContent =
      `phi: ${s.by_sensitivity.phi || 0}  strict: ${s.by_sensitivity.phi_strict || 0}`;
  }

  // Average routing latency from the 20 most recent log entries
  const recent = state.logEntries.slice(0, 20);
  if (recent.length > 0) {
    const avg = recent.reduce((s, e) => s + e.routing_latency_ms, 0) / recent.length;
    document.getElementById('stat-latency').textContent =
      avg < 1 ? avg.toFixed(3) + ' ms' : avg.toFixed(2) + ' ms';
  }

  const totalPhi = state.logEntries.reduce(
    (sum, e) => sum + (e.phi_entities_detected || 0), 0,
  );
  if (totalPhi > 0 || state.lastPhiEntities !== null) {
    updatePhiEntityStat();
  }
}

function updatePhiEntityStat() {
  const lastVal = state.lastPhiEntities;
  const el      = document.getElementById('stat-phi-entities');
  const sub     = document.getElementById('stat-phi-entities-sub');
  if (lastVal === null) return;
  el.textContent  = lastVal;
  sub.textContent = lastVal === 0 ? 'none in last request' : 'detected in last request';
}

// ── Server health cards ───────────────────────────────────────────────────

function renderServerCards() {
  if (!state.metrics) return;
  const servers = state.metrics.servers;
  const grid    = document.getElementById('server-grid');

  if (!servers.length) {
    grid.innerHTML = '<div class="server-placeholder">No servers registered.</div>';
    return;
  }

  // Build circuit breaker lookup: server_id → circuit info
  const cbMap = {};
  (state.circuitStatus?.servers || []).forEach(c => { cbMap[c.server_id] = c; });

  grid.innerHTML = servers.map(s => {
    const loadPct  = Math.round((s.current_load || 0) * 100);
    const latPct   = Math.min(100, Math.round(((s.avg_latency_ms || 0) / 1000) * 100));
    const loadClass =
      loadPct < 50 ? 'load-low' : loadPct < 80 ? 'load-medium' : 'load-high';
    const latClass =
      (s.avg_latency_ms || 0) < 200 ? 'latency-low' :
      (s.avg_latency_ms || 0) < 600 ? 'latency-medium' : 'latency-high';
    const gpuInfo  = s.gpu_count > 0 ? `${s.gpu_count}× ${s.gpu_type || 'GPU'}` : 'CPU only';

    const cb      = cbMap[s.server_id];
    const cbState = cb?.state || 'closed';
    const cbDot   = cbState !== 'closed'
      ? `<span class="cb-dot ${cbState}" title="Circuit ${cbState.toUpperCase()}"></span>`
      : '';

    return `
    <div class="server-card ${s.status}" onclick="openModal('${esc(s.server_id)}')">
      ${cbDot}
      <div class="server-header">
        <span class="server-status-dot"></span>
        <div class="server-info">
          <div class="server-id">${esc(s.server_id)}</div>
          <div class="server-meta">${esc(s.region)} · ${gpuInfo}</div>
        </div>
        <span class="server-status-badge status-badge-${s.status}">${s.status}</span>
      </div>
      <div class="server-bars">
        <div class="bar-row">
          <div class="bar-label-row">
            <span class="bar-label">Load</span>
            <span class="bar-value">${loadPct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill ${loadClass}" style="width:${loadPct}%"></div>
          </div>
        </div>
        <div class="bar-row">
          <div class="bar-label-row">
            <span class="bar-label">Avg Latency</span>
            <span class="bar-value">${(s.avg_latency_ms || 0).toFixed(1)} ms</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill ${latClass}" style="width:${latPct}%"></div>
          </div>
        </div>
      </div>
      <div class="server-chips">
        <span class="chip ${s.cloud_env}">${envIcon(s.cloud_env)} ${s.cloud_env}</span>
        <span class="chip">$${(s.cost_per_token || 0).toFixed(4)}/1K tok</span>
        ${s.gpu_count > 0 ? `<span class="chip">GPU ×${s.gpu_count}</span>` : ''}
      </div>
    </div>`;
  }).join('');
}

// ── Routing log ───────────────────────────────────────────────────────────

function renderRoutingLog(newIds = [], total = null, returned = null) {
  const entries  = state.logEntries;
  const tbody    = document.getElementById('log-tbody');
  const countEl  = document.getElementById('log-count');
  const resultEl = document.getElementById('log-result-count');

  countEl.textContent = state.logStats ? state.logStats.total : entries.length;

  if (total !== null && returned !== null && total !== returned) {
    resultEl.textContent = `${returned} of ${total} shown`;
  } else if (returned !== null) {
    resultEl.textContent = `${returned} shown`;
  } else {
    resultEl.textContent = '';
  }

  if (!entries.length) {
    tbody.innerHTML = `<tr><td colspan="12">
      <div class="log-empty">
        <div class="empty-icon">📭</div>
        Send a request via the test panel or<br>
        call <code>POST /route</code> to see routing decisions.
      </div>
    </td></tr>`;
    return;
  }

  const newSet = new Set(newIds);

  tbody.innerHTML = entries.map(e => {
    const isNew      = newSet.has(e.request_id);
    const isExpanded = state.expandedIds.has(e.request_id);
    const time       = fmtTime(e.timestamp);
    const env        = e.cloud_env
      ? `<span class="env env-${e.cloud_env}">${envIcon(e.cloud_env)} ${e.cloud_env}</span>`
      : '<span class="text-muted">—</span>';
    const server     = e.selected_server_id
      ? `<code>${esc(e.selected_server_id)}</code>`
      : '<span class="text-muted">—</span>';
    const latMs      = e.routing_latency_ms;
    const latCls     = latMs < 1 ? 'status-ok' : latMs < 10 ? '' : 'status-rej';
    const status     = e.rejected
      ? `<span class="status-rej" title="${esc(e.rejection_reason || '')}">✗ Rejected</span>`
      : `<span class="status-ok">✓ Routed</span>`;
    const phiN       = e.phi_entities_detected || 0;
    const phiBadge   = phiN > 0
      ? `<span class="phi-entity-badge" title="${phiN} PHI entity${phiN !== 1 ? 'ies' : ''} de-identified">🛡 ${phiN}</span>`
      : '<span class="text-muted">—</span>';
    const reasonCell = e.rejected && e.rejection_reason
      ? `<span class="reason-cell" title="${esc(e.rejection_reason)}">${esc(e.rejection_reason)}</span>`
      : '<span class="text-muted">—</span>';
    const arrow      = isExpanded ? '▼' : '▶';
    const rid        = esc(e.request_id);

    return `<tr class="log-row${isNew ? ' new-entry' : ''}" onclick="toggleExpand('${rid}', event)">
      <td style="width:28px;padding:9px 6px 9px 12px"><span class="expand-arrow" id="arr-${rid}">${arrow}</span></td>
      <td class="mono-sm">${time}</td>
      <td>${esc(e.tenant_id)}</td>
      <td><span class="badge badge-${e.data_sensitivity}">${e.data_sensitivity}</span></td>
      <td>${phiBadge}</td>
      <td>${server}</td>
      <td>${env}</td>
      <td class="mono-sm ${latCls}">${latMs.toFixed(3)} ms</td>
      <td><span class="strategy-pill">${esc(e.strategy_used)}</span></td>
      <td class="mono-sm text-muted">${e.candidate_count ?? '—'}</td>
      <td>${status}</td>
      <td>${reasonCell}</td>
    </tr>
    <tr class="detail-row" id="exp-${rid}"${isExpanded ? '' : ' style="display:none"'}>
      <td colspan="12" style="padding:0;border-bottom:none">
        <div class="detail-panel${isExpanded ? ' open' : ''}">${buildDetailPanel(e)}</div>
      </td>
    </tr>`;
  }).join('');
}

// Keep the public name that external callers (api.js) use
const renderLog = renderRoutingLog;

// ── Expand / collapse detail row ──────────────────────────────────────────

function toggleExpand(reqId, event) {
  if (event && event.target.closest && event.target.closest('button, a, code')) return;

  const detailTr = document.getElementById('exp-' + reqId);
  const arrow    = document.getElementById('arr-' + reqId);
  const panel    = detailTr?.querySelector('.detail-panel');
  if (!panel) return;

  if (state.expandedIds.has(reqId)) {
    state.expandedIds.delete(reqId);
    panel.classList.remove('open');
    if (arrow) arrow.textContent = '▶';
    panel.addEventListener('transitionend', () => {
      if (!state.expandedIds.has(reqId)) detailTr.style.display = 'none';
    }, { once: true });
  } else {
    state.expandedIds.add(reqId);
    detailTr.style.display = '';
    // Double rAF: ensures display:'' is painted before max-height transition starts
    requestAnimationFrame(() => requestAnimationFrame(() => panel.classList.add('open')));
    if (arrow) arrow.textContent = '▼';
  }
}

// ── Detail panel content builder ──────────────────────────────────────────

function buildDetailPanel(e) {
  const decisionHtml = `
  <div class="dp-section">
    <div class="dp-section-title">Decision</div>
    <div class="dp-kv">
      <div class="dp-kv-row">
        <span class="dp-kv-label">Request ID</span>
        <span class="dp-kv-value">${esc(e.request_id)}</span>
      </div>
      <div class="dp-kv-row">
        <span class="dp-kv-label">Strategy</span>
        <span class="dp-kv-value">${esc(STRATEGY_DESC[e.strategy_used] || e.strategy_used)}</span>
      </div>
      <div class="dp-kv-row">
        <span class="dp-kv-label">Candidates</span>
        <span class="dp-kv-value">${e.candidate_count} server${e.candidate_count !== 1 ? 's' : ''} passed policy filter</span>
      </div>
      <div class="dp-kv-row">
        <span class="dp-kv-label">Routing time</span>
        <span class="dp-kv-value">${e.routing_latency_ms.toFixed(3)} ms</span>
      </div>
      ${e.selected_server_id ? `
      <div class="dp-kv-row">
        <span class="dp-kv-label">Selected</span>
        <span class="dp-kv-value">${esc(e.selected_server_id)}</span>
      </div>` : ''}
    </div>
  </div>`;

  // Score breakdown table
  let scoreHtml = '';
  const scores    = e.score_breakdown || {};
  const scoreKeys = Object.keys(scores);
  if (scoreKeys.length) {
    const rows = scoreKeys.map(sid => {
      const s          = scores[sid];
      const isSelected = sid === e.selected_server_id;
      return `<tr class="${isSelected ? 'score-selected' : ''}">
        <td>${esc(sid)}${isSelected ? ' ★' : ''}</td>
        <td>${((s.current_load || 0) * 100).toFixed(1)}%</td>
        <td>${(s.avg_latency_ms || 0).toFixed(1)} ms</td>
        <td>$${(s.cost_per_token || 0).toFixed(5)}</td>
      </tr>`;
    }).join('');
    scoreHtml = `
    <div class="dp-section">
      <div class="dp-section-title">Candidate Scores</div>
      <table class="score-table">
        <thead><tr><th>Server</th><th>Load</th><th>Latency</th><th>Cost/1K</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  }

  // PHI entity breakdown
  let phiHtml = '';
  const breakdown = e.phi_entity_breakdown || {};
  const phiKeys   = Object.keys(breakdown);
  if (phiKeys.length) {
    const pills = phiKeys.map(type =>
      `<span class="phi-type-pill">${breakdown[type]} ${esc(type)}</span>`,
    ).join('');
    phiHtml = `
    <div class="dp-section">
      <div class="dp-section-title">PHI Detected (${e.phi_entities_detected} total)</div>
      <div class="phi-breakdown">${pills}</div>
    </div>`;
  }

  // Rejection reason
  let rejHtml = '';
  if (e.rejected && e.rejection_reason) {
    rejHtml = `
    <div class="dp-section">
      <div class="dp-section-title">Rejection Reason</div>
      <div class="rejection-box">${esc(e.rejection_reason)}</div>
    </div>`;
  }

  return `<div class="detail-panel-inner">${decisionHtml}${scoreHtml}${phiHtml}${rejHtml}</div>`;
}

// ── Test panel — preset buttons ───────────────────────────────────────────

async function sendPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return;

  const tier      = preset.data_sensitivity;
  const prompt    = randomPrompt(tier);
  const tenant    = document.getElementById('cb-tenant')?.value.trim();
  const strategy  = document.getElementById('cb-strategy')?.value;
  const cloudPref = document.getElementById('cb-cloud-pref')?.value;

  const body = { ...preset, payload: { prompt } };
  if (tenant)    body.tenant_id   = tenant;
  if (strategy)  body.strategy    = strategy;
  if (cloudPref) body.region_hint = cloudPref;
  else           delete body.region_hint;

  await sendRoute(body, name);
}

// ── Test panel — JSON builder ─────────────────────────────────────────────

function generateSampleJSON() {
  const tenant      = document.getElementById('cb-tenant').value.trim() || 'hospital_A';
  const sensitivity = document.getElementById('cb-sensitivity').value;
  const strategy    = document.getElementById('cb-strategy').value;
  const cloudPref   = document.getElementById('cb-cloud-pref').value;

  const body = {
    model_id:         'tinyllama:latest',
    payload:          { prompt: randomPrompt(sensitivity) },
    tenant_id:        tenant,
    data_sensitivity: sensitivity,
    strategy,
  };
  if (cloudPref) body.region_hint = cloudPref;

  document.getElementById('custom-json').value = JSON.stringify(body, null, 2);
}

async function sendCustom() {
  const btn = document.getElementById('send-custom-btn');
  const raw = document.getElementById('custom-json').value.trim();
  let body;
  try {
    body = JSON.parse(raw);
  } catch {
    showResponse('error', 'JSON parse error — check the textarea', '');
    toast('Invalid JSON', 'error');
    return;
  }
  btn.disabled = true;
  await sendRoute(body, 'custom');
  btn.disabled = false;
}

// ── Test panel — burst generator ──────────────────────────────────────────

async function sendBurst() {
  const btn      = document.getElementById('burst-btn');
  const progress = document.getElementById('burst-progress');
  const label    = document.getElementById('burst-label');
  const fill     = document.getElementById('burst-fill');
  const TOTAL    = 20;

  btn.disabled = true;
  progress.classList.add('visible');

  let sent = 0, ok = 0, failed = 0;

  for (let i = 0; i < TOTAL; i++) {
    const sensitivity = weightedPick(BURST_SENSITIVITY_WEIGHTS);
    const body = {
      model_id:         'tinyllama:latest',
      payload:          { prompt: randomPrompt(sensitivity) },
      tenant_id:        randPick(BURST_TENANTS),
      data_sensitivity: sensitivity,
      strategy:         randPick(BURST_STRATEGIES),
    };

    try {
      const succeeded = await postRoute(body); // postRoute is in api.js
      succeeded ? ok++ : failed++;
    } catch {
      failed++;
    }

    sent++;
    label.textContent = `Sent ${sent}/${TOTAL}  ✓ ${ok}  ✗ ${failed}`;
    fill.style.width  = `${(sent / TOTAL) * 100}%`;

    if (i < TOTAL - 1) await sleep(200);
  }

  label.textContent = `Done — ${ok} routed, ${failed} failed`;
  toast(`Burst complete: ${ok}/${TOTAL} routed`, ok === TOTAL ? 'ok' : 'info');

  await fetchFilteredLog();
  poll(); // poll is in main.js

  btn.disabled = false;
  setTimeout(() => {
    progress.classList.remove('visible');
    fill.style.width = '0%';
  }, 4000);
}

// ── Response preview (test panel result area) ─────────────────────────────

function showResponse(type, header, body) {
  const el  = document.getElementById('response-preview');
  const hEl = document.getElementById('response-header');
  const bEl = document.getElementById('response-body');
  el.style.display = '';
  hEl.className    = `response-preview-header ${type}`;
  hEl.textContent  = header;
  bEl.textContent  = body;
}

// ── Server detail modal ───────────────────────────────────────────────────

function openModal(serverId) {
  state.currentModalServerId = serverId;
  const server = (state.metrics?.servers || []).find(s => s.server_id === serverId);
  const cb     = (state.circuitStatus?.servers || []).find(s => s.server_id === serverId);

  const dot = document.getElementById('modal-server-dot');
  dot.className = `server-status-dot ${server?.status || ''}`;
  document.getElementById('modal-title').textContent = serverId;

  document.getElementById('modal-id').textContent      = serverId;
  document.getElementById('modal-env').textContent     =
    server ? `${envIcon(server.cloud_env)} ${server.cloud_env}` : '—';
  document.getElementById('modal-region').textContent  = server?.region || '—';
  document.getElementById('modal-health').textContent  = server?.status || '—';
  document.getElementById('modal-load').textContent    =
    server != null ? `${(server.current_load * 100).toFixed(1)}%` : '—';
  document.getElementById('modal-latency').textContent =
    server != null ? `${server.avg_latency_ms.toFixed(1)} ms` : '—';

  const cbState  = cb?.state || 'closed';
  const cbDot    = document.getElementById('modal-cb-dot');
  cbDot.className = `modal-cb-dot ${cbState}`;

  const cbLabels = { closed: 'Closed', open: 'Open', half_open: 'Half-Open' };
  const cbSubs   = {
    closed:    'Normal operation — requests pass through',
    open:      'Fast-failing — requests rejected until cooldown',
    half_open: 'Probing — one trial request allowed',
  };
  const labelEl = document.getElementById('modal-cb-label');
  labelEl.textContent = cbLabels[cbState] || cbState;
  labelEl.className   = `modal-cb-label ${cbState}`;
  document.getElementById('modal-cb-sub').textContent       = cbSubs[cbState] || '';
  document.getElementById('modal-cb-failures').textContent  =
    cb != null ? `${cb.consecutive_failures}` : '—';
  document.getElementById('modal-cb-threshold').textContent =
    cb != null ? `${cb.failure_threshold}` : '—';

  const lastFail = cb?.last_failure_time;
  document.getElementById('modal-cb-last-fail').textContent =
    lastFail ? new Date(lastFail * 1000).toLocaleTimeString() : 'None';

  // Clear previous action result
  const resultEl = document.getElementById('modal-action-result');
  if (resultEl) resultEl.innerHTML = '';

  document.getElementById('modal-overlay').classList.add('visible');
}

function closeModal(event) {
  if (event && event.target !== document.getElementById('modal-overlay')) return;
  document.getElementById('modal-overlay').classList.remove('visible');
  state.currentModalServerId = null;
}

// ── Server Action: Test Health Check ─────────────────────────────────────

async function testHealthCheck() {
  const serverId = state.currentModalServerId;
  if (!serverId) return;
  const resultEl = document.getElementById('modal-action-result');
  resultEl.innerHTML = '<div class="action-result pending">Probing…</div>';
  try {
    const r = await fetchHealthCheck(serverId);
    const cls = r.status === 'healthy' ? 'ok' : 'error';
    const latency = r.latency_ms != null ? ` · ${r.latency_ms.toFixed(1)} ms` : '';
    const errNote = r.error ? `<div class="action-result-detail">${esc(r.error)}</div>` : '';
    resultEl.innerHTML = `<div class="action-result ${cls}">Health probe: ${r.status}${latency}${errNote}</div>`;
  } catch (err) {
    resultEl.innerHTML = `<div class="action-result error">Probe failed: ${esc(err.message)}</div>`;
  }
}

// ── Server Action: View Recent Logs ──────────────────────────────────────

async function viewServerLogs() {
  const serverId = state.currentModalServerId;
  if (!serverId) return;
  const resultEl = document.getElementById('modal-action-result');
  resultEl.innerHTML = '<div class="action-result pending">Fetching logs…</div>';
  try {
    const r = await fetchServerLogs(serverId, 10);
    if (!r.entries.length) {
      resultEl.innerHTML = '<div class="action-result info">No routing log entries for this server yet.</div>';
      return;
    }
    const rows = r.entries.map(e => {
      const t   = fmtTime(e.timestamp);
      const cls = e.rejected ? 'status-rej' : 'status-ok';
      return `<div class="log-mini-row">
        <span class="mono-sm">${t}</span>
        <span class="badge badge-${e.data_sensitivity}">${e.data_sensitivity}</span>
        <span class="${cls}">${e.rejected ? '✗' : '✓'}</span>
        <span class="mono-sm text-muted">${e.routing_latency_ms.toFixed(2)} ms</span>
      </div>`;
    }).join('');
    resultEl.innerHTML = `<div class="action-result info">
      <div class="action-result-title">Last ${r.entries.length} of ${r.total} routes to ${esc(serverId)}</div>
      <div class="log-mini">${rows}</div>
    </div>`;
  } catch (err) {
    resultEl.innerHTML = `<div class="action-result error">Failed: ${esc(err.message)}</div>`;
  }
}

// ── Server Action: Force Health Poll ─────────────────────────────────────

async function runForceHealthPoll() {
  const serverId = state.currentModalServerId;
  if (!serverId) return;
  const resultEl = document.getElementById('modal-action-result');
  resultEl.innerHTML = '<div class="action-result pending">Forcing health poll…</div>';
  try {
    const r   = await forceHealthPoll(serverId);
    const cls = r.new_status === 'healthy' ? 'ok' : 'error';
    const changed = r.previous_status !== r.new_status;
    const latency = r.latency_ms != null ? ` · ${r.latency_ms.toFixed(1)} ms` : '';
    resultEl.innerHTML = `<div class="action-result ${cls}">
      ${changed ? `Status changed: ${r.previous_status} → <strong>${r.new_status}</strong>` : `Status confirmed: ${r.new_status}`}${latency}
    </div>`;
    flashServerCard(serverId, r.new_status === 'healthy' ? 'green' : 'red');
    // Refresh modal's health field immediately
    document.getElementById('modal-health').textContent = r.new_status;
    const dot = document.getElementById('modal-server-dot');
    if (dot) dot.className = `server-status-dot ${r.new_status}`;
    toast(
      changed ? `${serverId}: ${r.previous_status} → ${r.new_status}` : `${serverId}: ${r.new_status} (confirmed)`,
      cls,
    );
    poll(); // refresh all panels
  } catch (err) {
    resultEl.innerHTML = `<div class="action-result error">Poll failed: ${esc(err.message)}</div>`;
  }
}

// ── Flash server card ─────────────────────────────────────────────────────

function flashServerCard(serverId, color) {
  const card = document.querySelector(`.server-card[onclick*="${CSS.escape(serverId)}"]`);
  if (!card) return;
  const cls = `flash-${color}`;
  card.classList.add(cls);
  setTimeout(() => card.classList.remove(cls), 1200);
}

// ── Live Diagnostics panel ────────────────────────────────────────────────

function toggleDiagnostics() {
  state.diagExpanded = !state.diagExpanded;
  const panel  = document.getElementById('diag-panel');
  const toggle = document.getElementById('diag-toggle');
  if (!panel) return;
  if (state.diagExpanded) {
    panel.classList.add('open');
    if (toggle) toggle.textContent = '▼';
    renderDiagnostics();
  } else {
    panel.classList.remove('open');
    if (toggle) toggle.textContent = '▶';
  }
}

function renderDiagnostics() {
  if (!state.diagExpanded) return;
  const panel = document.getElementById('diag-panel');
  if (!panel) return;

  const servers = state.metrics?.servers || [];
  if (!servers.length) {
    panel.innerHTML = '<div class="diag-empty">No server metrics available.</div>';
    return;
  }

  panel.innerHTML = servers.map(s => {
    const history = state.latencyHistory[s.server_id] || [];
    const sparkSvg = renderSparkline(history);
    const statusCls = s.status === 'healthy' ? 'ok' : s.status === 'degraded' ? 'warn' : 'error';
    return `<div class="diag-server-row">
      <div class="diag-server-info">
        <span class="diag-status-dot ${s.status}"></span>
        <span class="diag-server-id">${esc(s.server_id)}</span>
        <span class="diag-status-badge ${statusCls}">${s.status}</span>
      </div>
      <div class="diag-sparkline-area" title="Latency history (ms)">
        ${sparkSvg}
        <span class="diag-latency-label">${s.avg_latency_ms.toFixed(0)} ms avg</span>
      </div>
      <div class="diag-load-label">${Math.round(s.current_load * 100)}% load</div>
    </div>`;
  }).join('');
}

function renderSparkline(history) {
  const W = 80, H = 24;
  if (!history.length) {
    return `<svg class="sparkline" width="${W}" height="${H}"><text x="0" y="${H-2}" font-size="9" fill="var(--text-muted)">no data</text></svg>`;
  }
  const values = history.map(h => h.latency_ms);
  const min    = Math.min(...values);
  const max    = Math.max(...values) || 1;
  const range  = max - min || 1;

  const pts = values.map((v, i) => {
    const x = Math.round((i / Math.max(values.length - 1, 1)) * W);
    const y = Math.round(H - ((v - min) / range) * (H - 4) - 2);
    return `${x},${y}`;
  }).join(' ');

  // Color based on latest status
  const last   = history[history.length - 1];
  const stroke = last?.status === 'healthy' ? '#22c55e' : last?.status === 'degraded' ? '#f59e0b' : '#ef4444';

  return `<svg class="sparkline" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
    <polyline points="${pts}" fill="none" stroke="${stroke}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>
  </svg>`;
}

// ── Toast notifications ───────────────────────────────────────────────────

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className  = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// ── updateCircuitIndicators ───────────────────────────────────────────────
// Called after a circuit-status update to refresh cb-dot badges on server cards
// without a full re-render. Full re-render via renderServerCards() is preferred;
// this exists as a lighter alternative for future use.

function updateCircuitIndicators() {
  if (!state.circuitStatus) return;
  state.circuitStatus.servers.forEach(cb => {
    const card = document.querySelector(
      `.server-card[onclick*="${CSS.escape(cb.server_id)}"]`,
    );
    if (!card) return;
    let dot = card.querySelector('.cb-dot');
    if (cb.state === 'closed') {
      if (dot) dot.remove();
    } else {
      if (!dot) {
        dot = document.createElement('span');
        dot.className = `cb-dot ${cb.state}`;
        dot.title = `Circuit ${cb.state.toUpperCase()}`;
        card.prepend(dot);
      } else {
        dot.className = `cb-dot ${cb.state}`;
      }
    }
  });
}
