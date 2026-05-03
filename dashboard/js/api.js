'use strict';

// ── Low-level fetch helper ────────────────────────────────────────────────

async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} from ${url}`);
  return resp.json();
}

// ── Individual endpoint fetchers ──────────────────────────────────────────

async function fetchHealth() {
  return fetchJSON('/health');
}

async function fetchMetrics() {
  return fetchJSON('/metrics');
}

async function fetchCircuitStatus() {
  return fetchJSON('/circuit-status');
}

async function fetchLogStats() {
  return fetchJSON('/logs/stats');
}

async function fetchLogs() {
  return fetchJSON(buildLogsUrl());
}

async function fetchTestPrompts() {
  try {
    const data = await fetchJSON('/test-prompts');
    state.testPrompts = data.prompts || {};
  } catch (err) {
    console.warn('Could not load /test-prompts:', err.message);
  }
}

// ── Filter URL builder (reads live filter controls from DOM) ──────────────

function buildLogsUrl() {
  const params      = new URLSearchParams();
  const search      = document.getElementById('log-search')?.value.trim();
  const sensitivity = document.getElementById('log-sensitivity')?.value;
  const cloudEnv    = document.getElementById('log-cloud-env')?.value;
  const rejected    = document.getElementById('log-rejected')?.value;
  const limit       = document.getElementById('log-limit')?.value || '50';

  if (search)      params.set('search',      search);
  if (sensitivity) params.set('sensitivity', sensitivity);
  if (cloudEnv)    params.set('cloud_env',   cloudEnv);
  if (rejected)    params.set('rejected',    rejected);
  params.set('limit', limit);

  return '/logs?' + params.toString();
}

// ── Filtered log fetch — updates state and re-renders log table ───────────

async function fetchFilteredLog() {
  try {
    const logs    = await fetchLogs();
    const entries = logs.entries || [];

    const newEntries = entries.filter(e => !state.seenIds.has(e.request_id));
    newEntries.forEach(e => state.seenIds.add(e.request_id));
    state.logEntries = entries;

    // renderLog is defined in ui.js — safe to call at runtime
    renderLog(newEntries.map(e => e.request_id), logs.total, logs.returned);
  } catch (_err) {
    // Silent: live-tail errors must not change connection state
  }
}

// ── POST /route with response preview and toast feedback ──────────────────

async function sendRoute(body, label) {
  // showResponse / toast / updatePhiEntityStat / poll are defined in ui.js
  // and main.js respectively — all scripts are loaded before any user action.
  showResponse('pending', `Sending ${label}…`, '');
  try {
    const resp = await fetch('/route', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const data = await resp.json();

    if (resp.ok) {
      const server   = data.selected_server?.server_id || '—';
      const env      = data.selected_server?.cloud_env  || '—';
      const phiCount = data.phi_entities_detected ?? 0;
      state.lastPhiEntities = phiCount;
      updatePhiEntityStat();
      const phiNote = phiCount > 0
        ? `  🛡 ${phiCount} entity${phiCount !== 1 ? 'ies' : ''} de-id'd`
        : '';
      showResponse(
        'ok',
        `✓ Routed → ${server} (${env})  ${data.routing_latency_ms?.toFixed(3)} ms${phiNote}`,
        JSON.stringify(data, null, 2),
      );
      toast(
        phiCount > 0
          ? `Routed to ${server} — ${phiCount} PHI entities de-identified`
          : `Routed to ${server}`,
        'ok',
      );
    } else {
      showResponse(
        'error',
        `✗ ${resp.status} — ${data.detail || 'Rejected'}`,
        JSON.stringify(data, null, 2),
      );
      toast(`Rejected: ${data.detail || resp.status}`, 'error');
    }

    poll(); // refresh all panels immediately
  } catch (err) {
    showResponse('error', `Network error: ${err.message}`, '');
    toast('Network error', 'error');
  }
}

// ── Raw POST /route without UI side-effects (used by burst generator) ─────

async function postRoute(body) {
  const resp = await fetch('/route', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  });
  return resp.ok;
}

// ── Troubleshooting endpoints ─────────────────────────────────────────────

async function fetchHealthCheck(serverId) {
  return fetchJSON(`/health-check/${encodeURIComponent(serverId)}`);
}

async function fetchServerLogs(serverId, limit = 20) {
  return fetchJSON(`/server-logs/${encodeURIComponent(serverId)}?limit=${limit}`);
}

async function forceHealthPoll(serverId) {
  const resp = await fetch(`/force-health-poll/${encodeURIComponent(serverId)}`, {
    method: 'POST',
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}
