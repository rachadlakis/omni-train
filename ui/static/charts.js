/* Charts, memory bars, log viewer and failure hints for the console. Loaded before app.js.
   Series colours (--c1..--c3) are validated for colour-blind separation: at most 3 series. */
'use strict';

const VIZ = {
  esc: (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])),

  /** Round axis ticks covering [lo, hi]. */
  ticks(lo, hi, n = 5, integer = false) {
    const span = hi - lo || 1;
    const pow = 10 ** Math.floor(Math.log10(span / n));
    let step = [1, 2, 2.5, 5, 10].map((m) => m * pow).find((s) => span / s <= n) || 10 * pow;
    if (integer) step = Math.max(1, Math.round(step));
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10));
    return out;
  },

  num(v) {
    if (v == null || !isFinite(v)) return '–';
    const a = Math.abs(v);
    if (a >= 10000) return Math.round(v).toLocaleString();
    if (a >= 100) return v.toFixed(1);
    if (a > 0 && a < 0.001) return v.toExponential(1);
    return a >= 1 ? v.toFixed(3) : v.toFixed(4);
  },
  axis(v) {
    const a = Math.abs(v);
    if (a >= 10000) return `${+(v / 1000).toFixed(1)}K`;
    if (a > 0 && a < 0.001) return v.toExponential(0);
    return String(+v.toFixed(4));
  },

  /** Exponential moving average, debiased (TensorBoard-style). */
  smooth(points, w) {
    let last = 0, norm = 0;
    return points.map(([x, y]) => { last = last * w + (1 - w) * y; norm = norm * w + (1 - w); return [x, last / norm]; });
  },
};

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

const METRIC_INFO = {
  loss: ['Loss', 'lower is better'],
  reward: ['Reward', 'higher is better'],
  reward_std: ['Reward spread', 'within a group; 0 means every answer scored the same'],
  kl: ['KL', 'distance from the starting model'],
  len: ['Answer length', 'tokens per answer'],
  acc: ['Preference accuracy', 'share of pairs where chosen beats rejected'],
  margin: ['Reward margin', 'chosen minus rejected'],
  grad_norm: ['Gradient norm', 'before clipping'],
  tokens_per_s: ['Tokens per second', ''],
};
function metricLabel(key) {
  if (METRIC_INFO[key]) return METRIC_INFO[key][0];
  return key.startsWith('r_') ? `Reward: ${key.slice(2)}` : key.replace(/_/g, ' ');
}
/** Plottable metrics, the most telling one first. */
function metricKeys(points) {
  const skip = new Set(['epoch', 'epochs', 'step', 'steps', 'global_step', 'avg']);
  const seen = new Set();
  (points || []).forEach((p) => Object.entries(p).forEach(([k, v]) => { if (!skip.has(k) && typeof v === 'number') seen.add(k); }));
  const order = ['reward', 'loss', 'acc', 'margin', 'reward_std', 'len', 'kl', 'grad_norm', 'tokens_per_s'];
  if (!seen.has('reward')) order.splice(0, 2, 'loss');
  return [...order.filter((k) => seen.has(k)), ...[...seen].filter((k) => !order.includes(k)).sort()];
}
const metricSeries = (points, key) => (points || []).filter((p) => typeof p[key] === 'number').map((p) => [p.global_step, p[key]]);

// ---------------------------------------------------------------------------
// Line chart: crosshair + tooltip on hover, arrow keys when focused
// ---------------------------------------------------------------------------

/** series: [{ name, color, points: [[step, value]] }]; opts: { label, smoothing, height, empty } */
function drawLineChart(el, series, opts = {}) {
  const live = series.filter((s) => s.points.length);
  if (!live.length || Math.max(...live.map((s) => s.points.length)) < 2) {
    el.innerHTML = `<p class="chart-empty">${VIZ.esc(opts.empty || 'The curve appears after the first logged steps.')}</p>`;
    return;
  }
  const w = opts.smoothing || 0;
  const lines = live.map((s) => ({ ...s, shown: w ? VIZ.smooth(s.points, w) : s.points }));
  const W = Math.max(280, el.clientWidth || 600), H = opts.height || 280, L = 52, R = 14, T = 12, B = 30;
  const all = lines.flatMap((s) => s.points);
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = (y1 - y0) * 0.06 || Math.abs(y1) * 0.1 || 0.5;
  const floor = y0 >= 0 ? 0 : -Infinity;            // never draw a negative axis for data that cannot be negative
  y0 = Math.max(floor, y0 - pad); y1 += pad;
  const sx = (x) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const d = (pts) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join('');
  const few = all.length <= 40;
  const grid = VIZ.ticks(y0, y1, 6).map((v) => `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}"/><text class="axis-text" x="${L - 10}" y="${sy(v) + 4}" text-anchor="end">${VIZ.axis(v)}</text>`).join('')
    + VIZ.ticks(x0, x1, Math.min(8, Math.floor(W / 90)), true).map((v) => `<text class="axis-text" x="${sx(v)}" y="${H - 8}" text-anchor="middle">${VIZ.axis(v)}</text>`).join('');
  const marks = lines.map((s) => {
    const area = lines.length === 1 && y0 === 0 ? `<path class="area" style="fill:${s.color}" d="${d(s.shown)}L${sx(s.shown[s.shown.length - 1][0])},${H - B}L${sx(s.shown[0][0])},${H - B}Z"/>` : '';
    const raw = w ? `<path class="line raw" style="stroke:${s.color}" d="${d(s.points)}"/>` : '';
    const dots = few && !w ? s.shown.map(([x, y]) => `<circle class="pt" cx="${sx(x)}" cy="${sy(y)}" r="4" style="fill:${s.color}"/>`).join('') : '';
    return `${area}${raw}<path class="line" style="stroke:${s.color}" d="${d(s.shown)}"/>${dots}`;
  }).join('');
  const last = lines[0].shown[lines[0].shown.length - 1];
  el.innerHTML = `<div class="chart-box">
    <svg class="chart" viewBox="0 0 ${W} ${H}" height="${H}" tabindex="0" role="img"
      aria-label="${VIZ.esc(`${opts.label || 'Value'} by training step, steps ${x0} to ${x1}, last value ${VIZ.num(last[1])}. Use the arrow keys to read values.`)}">
      ${grid}${marks}<line class="cross" y1="${T}" y2="${H - B}" hidden></line>${lines.map((s) => `<circle class="focus" r="5" style="fill:${s.color}" hidden></circle>`).join('')}
    </svg><div class="chart-tip" hidden></div></div>
    ${lines.length > 1 || w ? `<div class="legend">${lines.map((s) => `<span><i class="key" style="background:${s.color}"></i>${VIZ.esc(s.name)}</span>`).join('')}${w ? '<span class="faint">faint line: unsmoothed</span>' : ''}</div>` : ''}`;

  const svg = el.querySelector('svg'), tip = el.querySelector('.chart-tip'), cross = svg.querySelector('.cross'), focus = [...svg.querySelectorAll('.focus')];
  const steps = [...new Set(xs)].sort((a, b) => a - b);
  let at = -1;
  const show = (i) => {
    at = Math.max(0, Math.min(steps.length - 1, i));
    const x = steps[at];
    cross.setAttribute('x1', sx(x)); cross.setAttribute('x2', sx(x)); cross.removeAttribute('hidden');
    tip.textContent = '';
    const head = document.createElement('div'); head.className = 'tip-head'; head.textContent = `Step ${x.toLocaleString()}`; tip.appendChild(head);
    lines.forEach((s, k) => {
      const j = s.points.findIndex((p) => p[0] === x);
      if (j < 0) { focus[k].setAttribute('hidden', ''); return; }
      focus[k].setAttribute('cx', sx(x)); focus[k].setAttribute('cy', sy(s.shown[j][1])); focus[k].removeAttribute('hidden');
      const row = document.createElement('div'); row.className = 'tip-row';
      const key = document.createElement('i'); key.className = 'key'; key.style.background = s.color;
      const val = document.createElement('b'); val.textContent = VIZ.num(s.points[j][1]);
      const name = document.createElement('span'); name.textContent = lines.length > 1 ? s.name : (w ? `smoothed ${VIZ.num(s.shown[j][1])}` : '');
      row.append(key, val, name); tip.appendChild(row);
    });
    tip.hidden = false;
    const ratio = svg.clientWidth / W, left = sx(x) * ratio;
    tip.style.left = `${left > svg.clientWidth / 2 ? left - tip.offsetWidth - 12 : left + 12}px`;
  };
  const hide = () => { tip.hidden = true; cross.setAttribute('hidden', ''); focus.forEach((f) => f.setAttribute('hidden', '')); at = -1; };
  svg.onpointermove = (e) => {
    const r = svg.getBoundingClientRect();
    const x = x0 + (((e.clientX - r.left) / r.width) * W - L) / (W - L - R) * (x1 - x0);
    let best = 0; steps.forEach((s, i) => { if (Math.abs(s - x) < Math.abs(steps[best] - x)) best = i; });
    show(best);
  };
  svg.onpointerleave = hide; svg.onblur = hide;
  svg.onfocus = () => { if (svg.matches(':focus-visible')) show(steps.length - 1); };
  svg.onkeydown = (e) => {
    const move = { ArrowLeft: -1, ArrowRight: 1, PageUp: -10, PageDown: 10 }[e.key];
    if (move) { e.preventDefault(); show((at < 0 ? steps.length - 1 : at) + move); }
    if (e.key === 'Escape') hide();
  };
}

// ---------------------------------------------------------------------------
// Memory bars: one row per option, shared scale, GPU limit marked
// ---------------------------------------------------------------------------

const SEGMENTS = [
  ['weights', 'Weights', 'var(--m-weights)'],
  ['gradients', 'Gradients', 'var(--m-grads)'],
  ['optimizer', 'AdamW states', 'var(--m-optim)'],
  ['activations', 'Activations', 'var(--m-acts)'],
  ['logits', 'Logits + loss', 'var(--m-logits)'],
  ['overhead', 'CUDA + NCCL', 'var(--m-over)'],
];

function memoryBars(options, capGb, usableGb, chosenLabel) {
  const biggest = Math.max(...options.map((o) => o.memory_gb.total));
  const scale = Math.min(Math.max(biggest, capGb * 1.08), capGb * 2.4);
  const rows = options.map((o) => {
    const m = o.memory_gb, cut = m.total > scale;
    const segs = SEGMENTS.filter(([k]) => (m[k] || 0) / scale > 0.002).map(([k, name, color]) =>
      `<span class="mseg" style="flex-grow:${m[k]};background:${color}" title="${name}: ${m[k].toFixed(2)} GB"></span>`).join('');
    const state = !o.fits ? ['over', 'Too big'] : m.total > usableGb * 0.9 ? ['tight', 'Tight'] : ['fit', 'Fits'];
    return `<div class="mrow${o.label === chosenLabel ? ' chosen' : ''}">
      <div class="mlabel"><b>${VIZ.esc(o.label)}</b>${o.label === chosenLabel ? '<span class="pill">selected</span>' : ''}<span class="mval num">${m.total.toFixed(m.total < 100 ? 1 : 0)} GB</span><span class="pill ${state[0]}">${state[1]}</span></div>
      <div class="mtrack" role="img" aria-label="${VIZ.esc(`${o.label}: ${m.total.toFixed(1)} GB per GPU of ${capGb} GB. ${SEGMENTS.map(([k, n]) => `${n} ${(m[k] || 0).toFixed(1)}`).join(', ')}`)}">
        <div class="mfill${cut ? ' cut' : ''}" style="width:${(Math.min(m.total, scale) / scale) * 100}%">${segs}</div>
        <i class="mcap" style="left:${(capGb / scale) * 100}%"></i></div></div>`;
  }).join('');
  return `<div class="mbars">${rows}</div>
    <div class="legend">${SEGMENTS.map(([, n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join('')}<span><i class="capkey"></i>GPU limit, ${capGb} GB</span></div>`;
}

// ---------------------------------------------------------------------------
// Log viewer: search, level filter, wrap, copy, download
// ---------------------------------------------------------------------------

function classifyLine(line) {
  if (/^Traceback \(|^\[rank \d+\].*❌|❌|\bexited with code [1-9]|^\s*[\w.]*(?:Error|Exception)\b:|\bError:\s|\bFAILED\b/.test(line)) return 'e';
  if (/\bWarning\b|\bWARN(ING)?\b|⚠/.test(line)) return 'w';
  if (/✅|\bEpoch \d+ complete\b|^\s*(?:Checkpoint|Model) saved\b/.test(line)) return 's';
  if (/Epoch \d+\/\d+ \[/.test(line)) return 'p';
  return '';
}

function errorSummary(logs) {
  for (let i = logs.length - 1; i >= 0; i--) {
    if (/^[A-Za-z_.]*(Error|Exception)\b.*:/.test(logs[i].trim())) return logs[i].trim();
  }
  return '';
}

class LogView {
  constructor(root, { name, url }) {
    this.lines = []; this.query = ''; this.level = 'all'; this.stick = true; this.sig = '';
    root.innerHTML = `<div class="log-bar">
        <input type="search" placeholder="Search the log" aria-label="Search the log">
        <select aria-label="Which lines to show"><option value="all">All lines</option><option value="quiet">Hide progress lines</option><option value="bad">Errors and warnings</option></select>
        <label class="check"><input type="checkbox" data-wrap checked><span>Wrap</span></label>
        <span class="log-count faint num" aria-live="polite"></span>
        <button class="btn small" data-copy-log>Copy</button><a class="btn small" download="${VIZ.esc(name)}.log" href="${VIZ.esc(url)}">Download</a>
      </div><div class="console" tabindex="0" aria-label="Run log"></div><button class="btn small log-jump" hidden>Jump to latest</button>`;
    this.box = root.querySelector('.console'); this.count = root.querySelector('.log-count'); this.jump = root.querySelector('.log-jump');
    root.querySelector('input[type="search"]').oninput = (e) => { this.query = e.target.value.trim().toLowerCase(); this.render(true); };
    root.querySelector('select').onchange = (e) => { this.level = e.target.value; this.render(true); };
    root.querySelector('[data-wrap]').onchange = (e) => this.box.classList.toggle('nowrap', !e.target.checked);
    root.querySelector('[data-copy-log]').onclick = () => copyText(this.visible().map((x) => x[1]).join('\n'));
    this.jump.onclick = () => { this.stick = true; this.box.scrollTop = this.box.scrollHeight; };
    this.box.onscroll = () => {
      this.stick = this.box.scrollTop + this.box.clientHeight >= this.box.scrollHeight - 30;
      this.jump.hidden = this.stick;
    };
  }

  visible() {
    return this.lines.map((l, i) => [i, l, classifyLine(l)]).filter(([, l, c]) =>
      (this.level === 'all' || (this.level === 'quiet' ? c !== 'p' : c === 'e' || c === 'w')) && (!this.query || l.toLowerCase().includes(this.query)));
  }

  set(lines) { this.lines = lines || []; this.render(false); }

  render(force) {
    const sig = `${this.lines.length}|${this.lines[this.lines.length - 1] || ''}|${this.level}|${this.query}`;
    if (sig === this.sig && !force) return;                  // unchanged: keep the reader's text selection
    this.sig = sig;
    const rows = this.visible();
    const mark = (t) => {
      if (!this.query) return VIZ.esc(t);
      const out = []; const low = t.toLowerCase(); let i = 0, j;
      while ((j = low.indexOf(this.query, i)) >= 0) { out.push(VIZ.esc(t.slice(i, j)), `<mark>${VIZ.esc(t.slice(j, j + this.query.length))}</mark>`); i = j + this.query.length; }
      return out.join('') + VIZ.esc(t.slice(i));
    };
    this.box.innerHTML = rows.length
      ? rows.map(([i, l, c]) => `<div class="ln ${c}" data-i="${i}">${mark(l) || ' '}</div>`).join('')
      : `<div class="ln w">${this.lines.length ? 'No line matches.' : 'Waiting for output…'}</div>`;
    this.count.textContent = rows.length === this.lines.length ? `${rows.length.toLocaleString()} lines` : `${rows.length.toLocaleString()} of ${this.lines.length.toLocaleString()}`;
    if (this.stick) this.box.scrollTop = this.box.scrollHeight;
  }

  /** Show the last traceback. */
  showError() {
    let i = this.lines.length - 1;
    while (i >= 0 && !/^Traceback \(/.test(this.lines[i])) i--;
    if (i < 0) { i = this.lines.length - 1; while (i >= 0 && classifyLine(this.lines[i]) !== 'e') i--; }
    if (i < 0) return;
    this.level = 'all'; this.query = ''; this.stick = false;
    this.box.parentNode.querySelector('select').value = 'all'; this.box.parentNode.querySelector('input[type="search"]').value = '';
    this.render(true);
    const ln = this.box.querySelector(`[data-i="${i}"]`);
    this.box.scrollTop = ln.offsetTop - this.box.offsetTop - 8;
    this.box.scrollIntoView({ block: 'center' });
  }
}

// ---------------------------------------------------------------------------
// What a failed run most likely needs
// ---------------------------------------------------------------------------

const FAILURE_HINTS = [
  [/out of memory|OutOfMemoryError/i, 'The GPU ran out of memory.', 'Lower the batch size or the sequence length, turn on gradient checkpointing, or use LoRA or QLoRA. The plan panel shows what fits.'],
  [/\b(401|403)\b|gated repo|GatedRepo|awaiting a review|not authorized/i, 'Hugging Face refused access.', 'Accept the model licence on its Hub page and set HF_TOKEN in .env (see System).'],
  [/ConnectionError|MaxRetryError|offline mode|couldn.t connect|Name or service not known|LocalEntryNotFound/i, 'A download failed.', 'Check the network, or download the model and dataset first. HF_HUB_OFFLINE=1 only works with files already cached.'],
  [/fewer than one global batch/i, 'The dataset is smaller than one batch.', 'Use a larger split, or lower the batch size.'],
  [/reference answer column/i, 'A reward needs reference answers the dataset lacks.', 'Use a dataset with an answer column, or pick rewards that need none (format, length).'],
  [/has no prompt|SFT row has none of|preference strings without/i, 'The dataset columns do not match the goal.', 'Compare the dataset with the expected columns on the Data step.'],
  [/chat template/i, 'The tokenizer has no usable chat template.', 'Use an instruct model, or a dataset with plain-text prompts.'],
  [/Configuration error|Unsupported|must be (divisible|>=|<=|>)/i, 'The config was rejected.', 'The message above names the option to change.'],
  [/address already in use|EADDRINUSE/i, 'Another process holds the network port.', 'Start the run again: a new port is picked each time.'],
  [/\bnan\b/i, 'The numbers diverged (NaN).', 'Lower the learning rate and keep gradient clipping at 1.0.'],
  [/CheckpointError/i, 'Gradient checkpointing saw different tensors on its second pass.', 'Turn gradient checkpointing off for this run, and report it: this is a trainer bug rather than a setting.'],
  [/Server was restarted/i, 'The console restarted while this run was active.', 'Clone the run and launch it again, or resume from its last checkpoint.'],
];
function failureHint(text) {
  const hit = FAILURE_HINTS.find(([re]) => re.test(text || ''));
  return hit ? { what: hit[1], fix: hit[2] } : null;
}
