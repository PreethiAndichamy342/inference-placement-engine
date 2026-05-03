'use strict';

// ── Central application state ─────────────────────────────────────────────

const state = {
  connected:      false,
  health:         null,
  metrics:        null,
  circuitStatus:  null,
  logEntries:     [],
  logStats:       null,
  seenIds:        new Set(),
  pollErrors:     0,
  liveTailActive: false,
  liveTailTimer:  null,
  lastPhiEntities: null,  // phi_entities_detected from most recent /route response
  testPrompts:    {},     // tier → string[] loaded from GET /test-prompts
  expandedIds:    new Set(), // request_ids of currently expanded detail rows
  // ── Diagnostics & alerts ─────────────────────────────────────────────
  latencyHistory:       {},   // server_id → [{ts: Date, latency_ms, status}] (last 20)
  prevServerStatuses:   {},   // server_id → status string from last poll
  lastConnectedAt:      null, // Date when we last had a successful poll
  diagExpanded:         false,
  currentModalServerId: null, // server_id currently shown in the detail modal
};

// ── Preset routing configurations ─────────────────────────────────────────
// Prompts are fetched from GET /test-prompts at runtime so no medical text
// is hardcoded here.

const PRESETS = {
  public: {
    model_id:         'tinyllama:latest',
    tenant_id:        'demo_tenant',
    data_sensitivity: 'public',
    strategy:         'least_loaded',
  },
  internal: {
    model_id:         'tinyllama:latest',
    tenant_id:        'demo_tenant',
    data_sensitivity: 'internal',
    strategy:         'round_robin',
  },
  sensitive: {
    model_id:         'tinyllama:latest',
    tenant_id:        'demo_tenant',
    data_sensitivity: 'sensitive',
    strategy:         'compliance_first',
  },
  phi: {
    model_id:         'tinyllama:latest',
    tenant_id:        'hospital_A',
    data_sensitivity: 'phi',
    strategy:         'compliance_first',
  },
  phi_strict: {
    model_id:         'tinyllama:latest',
    tenant_id:        'hospital_A',
    data_sensitivity: 'phi_strict',
    strategy:         'compliance_first',
  },
};

// ── Burst generator constants ─────────────────────────────────────────────

const BURST_TENANTS = ['hospital_A', 'hospital_B', 'clinic_C'];

const BURST_SENSITIVITY_WEIGHTS = [
  { value: 'public',     weight: 60 },
  { value: 'internal',   weight: 20 },
  { value: 'sensitive',  weight: 10 },
  { value: 'phi',        weight:  6 },
  { value: 'phi_strict', weight:  4 },
];

const BURST_STRATEGIES = [
  'compliance_first', 'least_loaded', 'latency_optimized',
  'cost_optimized', 'round_robin',
];

// ── Strategy descriptions (used in detail panel) ──────────────────────────

const STRATEGY_DESC = {
  compliance_first:  'Policy filtering first, then load / latency',
  least_loaded:      'Selects server with lowest current load',
  latency_optimized: 'Minimizes expected response latency',
  cost_optimized:    'Selects least expensive eligible server',
  round_robin:       'Distributes evenly across eligible servers',
};

// ── Pure utility functions ────────────────────────────────────────────────

function weightedPick(items) {
  const total = items.reduce((s, i) => s + i.weight, 0);
  let r = Math.random() * total;
  for (const item of items) {
    r -= item.weight;
    if (r <= 0) return item.value;
  }
  return items[items.length - 1].value;
}

function randPick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function randomPrompt(tier) {
  const pool = state.testPrompts[tier];
  if (!pool || !pool.length) return `Sample ${tier} request.`;
  return pool[Math.floor(Math.random() * pool.length)];
}
