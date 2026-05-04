'use strict';

// ── Polling — fetches all data, updates state, triggers full re-render ─────

async function poll() {
  try {
    const [health, metrics, circuit, logs, stats] = await Promise.all([
      fetchHealth(),
      fetchMetrics(),
      fetchCircuitStatus(),
      fetchLogs(),
      fetchLogStats(),
    ]);

    state.health        = health;
    state.circuitStatus = circuit;
    state.logStats      = stats;

    const entries    = logs.entries || [];
    const newEntries = entries.filter(e => !state.seenIds.has(e.request_id));
    newEntries.forEach(e => state.seenIds.add(e.request_id));
    state.logEntries = entries;

    state.connected     = true;
    state.pollErrors    = 0;
    state.lastConnectedAt = new Date();

    // ── Record latency history + detect status changes ────────────────
    const now     = new Date();
    let   hasAlert = false;

    (metrics.servers || []).forEach(s => {
      // Latency history — keep last 20 samples
      if (!state.latencyHistory[s.server_id]) state.latencyHistory[s.server_id] = [];
      const hist = state.latencyHistory[s.server_id];
      hist.push({ ts: now, latency_ms: s.p99_latency_ms, status: s.status });
      if (hist.length > 20) hist.shift();

      // Status change detection
      const prev = state.prevServerStatuses[s.server_id];
      if (prev && prev !== s.status) {
        const isRecovery = s.status === 'healthy';
        const type = isRecovery ? 'ok' : 'error';
        toast(
          `${s.server_id}: ${prev} → ${s.status}`,
          type,
        );
        flashServerCard(s.server_id, isRecovery ? 'green' : 'red');
        if (!isRecovery) hasAlert = true;
      }
      state.prevServerStatuses[s.server_id] = s.status;
    });

    state.metrics = metrics;

    // Auto-expand diagnostics on first unhealthy server detection
    if (hasAlert && !state.diagExpanded) toggleDiagnostics();

    renderAll(newEntries.map(e => e.request_id), logs.total, logs.returned);
    renderDiagnostics();
  } catch (_err) {
    state.pollErrors++;
    if (state.pollErrors >= 2) {
      state.connected = false;
      renderConnection();
    }
  }
}

// ── Live tail toggle ──────────────────────────────────────────────────────

function toggleLiveTail() {
  const btn = document.getElementById('btn-live-tail');
  state.liveTailActive = !state.liveTailActive;

  if (state.liveTailActive) {
    btn.classList.add('active');
    btn.querySelector('.tail-dot').style.opacity = '1';
    btn.childNodes[1].textContent = ' Live Tail: ON';
    fetchFilteredLog();
    state.liveTailTimer = setInterval(fetchFilteredLog, 3000);
  } else {
    btn.classList.remove('active');
    btn.childNodes[1].textContent = ' Live Tail';
    clearInterval(state.liveTailTimer);
    state.liveTailTimer = null;
  }
}

// ── Clock ─────────────────────────────────────────────────────────────────

function updateClock() {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString([], {
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────

// Filter controls — any change triggers an immediate re-fetch
['log-sensitivity', 'log-cloud-env', 'log-rejected', 'log-limit'].forEach(id => {
  document.getElementById(id)?.addEventListener('change', fetchFilteredLog);
});

// Search — fetch on Enter or after 400 ms debounce
let searchDebounce = null;
document.getElementById('log-search')?.addEventListener('input', () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(fetchFilteredLog, 400);
});
document.getElementById('log-search')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') { clearTimeout(searchDebounce); fetchFilteredLog(); }
});

fetchTestPrompts();        // load test prompts — no medical text in JS
poll();                    // first data load
setInterval(poll, 5000);   // auto-refresh every 5 s
setInterval(updateClock, 1000);
updateClock();
