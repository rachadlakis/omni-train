/* OMNI-Train console: runs, new-run wizard, capacity planner, checkpoints, system, API. */
'use strict';

// ===========================================================================
// Utilities
// ===========================================================================

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clone = (o) => JSON.parse(JSON.stringify(o));

function getPath(obj, path) {
  return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
}
function setPath(obj, path, value) {
  const keys = path.split('.');
  let o = obj;
  keys.slice(0, -1).forEach((k) => { if (typeof o[k] !== 'object' || o[k] === null) o[k] = {}; o = o[k]; });
  o[keys[keys.length - 1]] = value;
}
function deepMerge(base, over) {
  for (const [k, v] of Object.entries(over || {})) {
    if (v && typeof v === 'object' && !Array.isArray(v) && base[k] && typeof base[k] === 'object' && !Array.isArray(base[k])) deepMerge(base[k], v);
    else base[k] = v;
  }
  return base;
}

async function api(path, opts = {}) {
  const init = { headers: {}, ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || 'POST';
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, init);
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) {
    let msg = data && data.detail ? data.detail : `${res.status} ${res.statusText}`;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg || JSON.stringify(d)).join('; ');
    throw new Error(msg);
  }
  return data;
}

const fmt = {
  params(n) {
    if (!n && n !== 0) return '–';
    if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
    return `${(n / 1e3).toFixed(0)}K`;
  },
  gb(n, digits = 1) { return n == null ? '–' : `${Number(n).toFixed(n < 10 ? digits : 0)} GB`; },
  int(n) { return n == null ? '–' : Math.round(n).toLocaleString(); },
  bytes(n) {
    if (n == null) return '–';
    const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
  },
  dur(s) {
    if (s == null || !isFinite(s)) return '–';
    s = Math.max(0, Math.round(s));
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
    return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
  },
  ago(iso) {
    if (!iso) return '–';
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  },
  lr(v) { return typeof v === 'number' ? (v !== 0 && Math.abs(v) < 0.001 ? String(+v.toPrecision(6)).includes('e') ? String(v) : v.toExponential().replace('+', '') : String(v)) : String(v ?? ''); },
};

function toast(message, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  $('#toasts').appendChild(el);
  setTimeout(() => el.remove(), type === 'error' ? 7000 : 3500);
}

function dialog({ title, body, actions }) {
  const root = $('#dialog');
  $('#dialog-title').textContent = title;
  $('#dialog-body').innerHTML = body;
  const bar = $('#dialog-actions');
  bar.innerHTML = '';
  const close = () => { root.hidden = true; document.removeEventListener('keydown', onKey); };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  (actions || [{ label: 'Close' }]).forEach((a) => {
    const b = document.createElement('button');
    b.className = `btn ${a.kind || ''}`;
    b.textContent = a.label;
    b.onclick = async () => { if (a.onClick) { const keep = await a.onClick(); if (keep === true) return; } close(); };
    bar.appendChild(b);
  });
  root.hidden = false;
  root.onclick = (e) => { if (e.target === root) close(); };
  document.addEventListener('keydown', onKey);
  bar.lastChild && bar.lastChild.focus();
  return close;
}

function download(filename, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/yaml' }));
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied to clipboard'); }
  catch (e) { toast('Copy failed: select the text and copy it manually', 'error'); }
}

const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
  del(k) { try { localStorage.removeItem(k); } catch (e) { /* ignore */ } },
};

// ===========================================================================
// Shared data
// ===========================================================================

const S = { catalog: null, queue: null };

async function loadCatalog() {
  if (!S.catalog) S.catalog = await api('/api/catalog');
  return S.catalog;
}

const ALGO_LABEL = { pretrain: 'Pretrain', sft: 'SFT', dpo: 'DPO', simpo: 'SimPO', orpo: 'ORPO', grpo: 'GRPO', ppo: 'PPO' };
const MODEL_TYPES = [
  { id: 'llm', label: 'Causal language model', hint: 'Llama, Qwen, Mistral, OPT, GPT-2 …' },
  { id: 'custom_transformer', label: 'Transformer from scratch', hint: 'Random init, your architecture; optional MoE' },
  { id: 'seq2seq', label: 'Encoder-decoder', hint: 'T5, BART, mT5' },
  { id: 'vlm', label: 'Vision-language model', hint: 'SmolVLM, Idefics3, LLaVA' },
  { id: 'vision', label: 'Image classification', hint: 'ViT, Swin, DeiT, ResNet' },
  { id: 'yolo', label: 'Object detection', hint: 'DETR, RT-DETR, YOLOS' },
  { id: 'encoder', label: 'Text encoder / embeddings', hint: 'BERT, RoBERTa, E5' },
];
const PEFT_TYPES = new Set(['llm', 'seq2seq', 'encoder', 'vlm', 'vision']);
const QLORA_TYPES = new Set(['llm', 'seq2seq', 'vlm']);
const DATA_HINT = {
  llm: 'A text column. Local .json/.jsonl/.csv/.parquet files work as the dataset name.',
  seq2seq: 'A text column.',
  encoder: 'A text column.',
  vision: 'An image column (image or img) and an integer label column.',
  yolo: 'image, objects (bbox in COCO [x, y, w, h], category) and image_id.',
  vlm: 'An image column and a caption column (text, caption, captions or sentence).',
  custom_transformer: 'Synthetic random tokens. Under strategy 5d you can also pack an HF text dataset.',
  sft: 'messages, prompt + completion, instruction (+ input) + output, or text.',
  preference: 'prompt + chosen + rejected, chosen/rejected conversations (last turn is the answer), or hh-rlhf strings.',
  rl: 'a prompt (prompt, question, problem or messages) and, for the accuracy reward, a reference answer (answer, solution, ground_truth, target or final_answer). GSM8K works as is.',
};
const DEFAULT_MODELS = {
  llm: 'Qwen/Qwen2.5-0.5B-Instruct', seq2seq: 'google/flan-t5-small', vlm: 'HuggingFaceTB/SmolVLM-256M-Instruct',
  vision: 'google/vit-base-patch16-224-in21k', yolo: 'hustvl/yolos-tiny', encoder: 'bert-base-uncased',
};
const DEFAULT_DATA = {
  llm: { name: 'wikitext', subset: 'wikitext-2-raw-v1', split: 'train[:2%]' },
  seq2seq: { name: 'wikitext', subset: 'wikitext-2-raw-v1', split: 'train[:2%]' },
  encoder: { name: 'wikitext', subset: 'wikitext-2-raw-v1', split: 'train[:2%]' },
  vision: { name: 'cifar10', subset: null, split: 'train[:5%]' },
  yolo: { name: 'cppe-5', subset: null, split: 'train' },
  vlm: { name: 'nlphuji/flickr30k', subset: null, split: 'test[:1%]' },
  custom_transformer: { name: 'synthetic', subset: null, split: 'train' },
};

const isRL = (a) => a === 'grpo' || a === 'ppo';

function defaultConfig() {
  return {
    model_name: 'Qwen/Qwen2.5-0.5B-Instruct',
    model_type: 'llm',
    dataset: { name: 'wikitext', subset: 'wikitext-2-raw-v1', split: 'train[:2%]' },
    algorithm: { type: 'pretrain', beta: 0.1, dpo_loss: 'sigmoid', label_smoothing: 0, simpo_gamma: 0.5, max_prompt_length: 256, completion_only: true,
      num_generations: 4, max_completion_length: 320, temperature: 1.0, top_p: 1.0, epsilon: 0.2, epsilon_high: 0.28,
      loss_type: 'dapo', scale_rewards: true, rewards: { accuracy: 1.0 }, custom_reward: '',
      ppo_epochs: 4, num_minibatches: 2, gamma: 1.0, gae_lambda: 0.95, vf_coef: 0.1, value_clip: 0.2, value_lr: 1e-4, eos_penalty: 0,
      reward_model: '', reward_model_weight: 1,
      system_prompt: 'Please reason step by step, and put your final answer within \\boxed{}.' },
    training: { epochs: 1, batch_size: 4, max_length: 512, learning_rate: 2e-4, gradient_checkpointing: true,
      warmup_steps: 20, lr_scheduler: 'cosine', weight_decay: 0.0, grad_clip: 1.0 },
    strategy: 'auto', num_gpus: 'auto', checkpoint_dir: 'checkpoints', save: true,
    dist_parameters: { mixed_precision: true, param_dtype: 'bfloat16', reduce_dtype: 'float32', output_dtype: 'bfloat16',
      cast_forward_inputs: false, distribute_api: 'dcp_api' },
    save_load: { resume: false, resume_path: '', load_model_from_hf: true },
    prefetch: { explicit: true, forward: 2, backward: 2 },
    peft: { enabled: true, type: 'lora', r: 16, alpha: 32, dropout: 0.05, target_modules: 'all-linear', bias: 'none' },
    quantization: { enabled: false, bits: 4, quant_type: 'nf4', compute_dtype: 'bfloat16', double_quant: true },
    parallelism: { dp: 'auto', tp: 1, sp: false, cp: 1, pp: 1, pp_schedule: '1f1b', micro_batches: 1, ep: 1, zero1: false },
    custom_transformer_args: { n_layers: 6, vocab_size: 8192, max_seq_len: 512, dim: 512, n_heads: 8, n_kv_heads: 0,
      dropout_p: 0.0, num_experts: 0, top_k: 2 },
    wandb: { wandb_log_with_train: false, wandb_entity: '', wandb_project: 'omni-train', wandb_run_name: '' },
  };
}

/** A full draft from a saved config (template, cloned run, pasted YAML, stored draft). Defaults fill
 *  gaps, except where a default would change what the config trains. */
function loadConfig(raw) {
  const src = clone(raw || {});
  const cfg = deepMerge(defaultConfig(), src);
  if (!src.peft) cfg.peft.enabled = false;
  const algo = src.algorithm || {};
  if (!src.algorithm) cfg.algorithm.type = 'pretrain';
  if (isRL(algo.type)) {
    const rm = algo.reward_model;                       // a name, or {name, weight, max_length, batch_size}
    cfg.algorithm.reward_model = (rm && typeof rm === 'object' ? rm.name : rm) || '';
    cfg.algorithm.reward_model_weight = rm && typeof rm === 'object' ? (rm.weight ?? 1) : 1;
    cfg.algorithm.reward_model_extra = rm && typeof rm === 'object'
      ? Object.fromEntries(Object.entries(rm).filter(([k]) => k !== 'name' && k !== 'weight')) : {};
    const spec = algo.rewards ?? (rm ? {} : ['accuracy']);
    cfg.algorithm.rewards = Array.isArray(spec) || typeof spec === 'string'
      ? Object.fromEntries([].concat(spec).map((n) => [String(n), 1])) : { ...spec };
    if (algo.system_prompt === undefined || algo.system_prompt === null) cfg.algorithm.system_prompt = '';
    if (algo.beta === undefined) cfg.algorithm.beta = 0;
    if (algo.custom_reward === undefined) cfg.algorithm.custom_reward = '';
    const len = (src.training || {}).max_length;
    if (len && algo.max_prompt_length === undefined) cfg.algorithm.max_prompt_length = Math.floor(len / 2);
    if (len && algo.max_completion_length === undefined) cfg.algorithm.max_completion_length = len - cfg.algorithm.max_prompt_length;
    if (algo.type === 'ppo') {                          // the trainer's defaults, not the wizard's
      if (algo.num_generations === undefined) cfg.algorithm.num_generations = 1;
      if (algo.num_minibatches === undefined) cfg.algorithm.num_minibatches = 1;
    }
    ['epsilon', 'loss_type'].forEach((k) => { if (algo[k] === undefined) cfg.algorithm[k] = { epsilon: 0.2, loss_type: 'grpo' }[k]; });
    if (algo.epsilon_high === undefined) cfg.algorithm.epsilon_high = cfg.algorithm.epsilon;
  }
  return cfg;
}

const methodOf = (cfg) => (!cfg.peft.enabled ? 'full' : (cfg.peft.type === 'qlora' || cfg.quantization.enabled ? 'qlora' : 'lora'));
const goalOf = (cfg) => {
  const a = cfg.algorithm.type;
  if (a === 'sft') return 'sft';
  if (['dpo', 'simpo', 'orpo'].includes(a)) return 'preference';
  if (isRL(a)) return 'rl';
  return 'pretrain';
};

/** The config sent to the server: only the keys the chosen options use. */
function exportConfig(cfg) {
  const c = clone(cfg);
  const a = c.algorithm.type;
  const algo = { type: a };
  if (a === 'sft') algo.completion_only = !!c.algorithm.completion_only;
  if (['dpo', 'simpo', 'orpo'].includes(a)) {
    algo.beta = Number(c.algorithm.beta);
    algo.max_prompt_length = Number(c.algorithm.max_prompt_length);
    if (a === 'dpo') { algo.dpo_loss = c.algorithm.dpo_loss; if (c.algorithm.dpo_loss === 'sigmoid' && Number(c.algorithm.label_smoothing)) algo.label_smoothing = Number(c.algorithm.label_smoothing); }
    if (a === 'simpo') algo.simpo_gamma = Number(c.algorithm.simpo_gamma);
  }
  if (isRL(a)) {
    const g = c.algorithm;
    Object.assign(algo, {
      num_generations: Number(g.num_generations), max_prompt_length: Number(g.max_prompt_length),
      max_completion_length: Number(g.max_completion_length), temperature: Number(g.temperature), top_p: Number(g.top_p),
      beta: c.peft.enabled ? Number(g.beta) : 0, epsilon: Number(g.epsilon),
    });
    if (Number(g.top_k)) algo.top_k = Number(g.top_k);
    if (a === 'grpo') Object.assign(algo, { epsilon_high: Number(g.epsilon_high), loss_type: g.loss_type, scale_rewards: !!g.scale_rewards });
    else ['ppo_epochs', 'num_minibatches', 'gamma', 'gae_lambda', 'vf_coef', 'value_clip', 'value_lr', 'eos_penalty'].forEach((k) => { algo[k] = Number(g[k]); });
    const rmName = String(g.reward_model || '').trim();
    const rmWeight = Number(g.reward_model_weight);
    if (rmName) {
      const extra = g.reward_model_extra || {};
      algo.reward_model = rmWeight === 1 && !Object.keys(extra).length ? rmName : { name: rmName, weight: rmWeight, ...extra };
    }
    const rewards = {};
    for (const [name, w] of Object.entries(g.rewards || {})) if (w !== null && w !== '' && Number(w)) rewards[name] = Number(w);
    if (g.custom_reward && g.custom_reward.includes(':')) rewards[g.custom_reward.trim()] = 1.0;
    if (Object.keys(rewards).length) algo.rewards = rewards;
    else if (!rmName) algo.rewards = { accuracy: 1.0 };
    if (g.system_prompt && String(g.system_prompt).trim()) algo.system_prompt = String(g.system_prompt).trim();
    c.training.max_length = Math.max(Number(c.training.max_length), algo.max_prompt_length + algo.max_completion_length);
  }
  c.algorithm = algo;
  if (c.model_type === 'custom_transformer') {
    const ct = c.custom_transformer_args;
    c.training.max_length = Number(ct.max_seq_len);
    if (!Number(ct.n_kv_heads)) delete ct.n_kv_heads;
    if (!Number(ct.num_experts)) { delete ct.num_experts; delete ct.top_k; }
    if (!['5d', 'auto'].includes(c.strategy)) { delete ct.n_kv_heads; delete ct.num_experts; delete ct.top_k; }
  } else {
    delete c.custom_transformer_args;
  }
  c.quantization.bits = Number(c.quantization.bits);
  if (c.parallelism.dp !== 'auto') c.parallelism.dp = Number(c.parallelism.dp);
  if (c.strategy === 'solo') c.num_gpus = 1;
  if (c.strategy !== 'auto' && c.num_gpus === 'auto') c.num_gpus = 1;
  if (!c.wandb.wandb_run_name) delete c.wandb.wandb_run_name;
  if (!c.dataset.subset) c.dataset.subset = null;
  return c;
}

// ===========================================================================
// Router
// ===========================================================================

const timers = new Set();
function every(ms, fn) { const id = setInterval(fn, ms); timers.add(id); return id; }
let navId = 0;
/** Call at page start; the returned function says whether that page is still showing. */
function stillHere() { const mine = navId; return () => mine === navId; }
function clearTimers() { timers.forEach(clearInterval); timers.clear(); }

const routes = {
  runs: renderRuns,
  run: renderRun,
  compare: renderCompare,
  new: renderWizard,
  planner: renderPlanner,
  checkpoints: renderCheckpoints,
  system: renderSystem,
  api: renderApi,
};

const VIEW_TITLE = { runs: 'Runs', run: 'Runs', compare: 'Runs', new: 'New run', planner: 'Planner', checkpoints: 'Checkpoints', system: 'System', api: 'API reference' };
const VIEW_GROUP = { system: 'Infrastructure', api: 'Infrastructure' };

function setCrumbs(view, arg) {
  const el = $('#crumbs');
  if (!el) return;
  const group = `<span>${VIEW_GROUP[view] || 'Workspace'}</span><span aria-hidden="true">/</span>`;
  if (view === 'compare') arg = 'Compare';
  el.innerHTML = (view === 'run' || view === 'compare') && arg
    ? `${group}<a href="#/runs">Runs</a><span aria-hidden="true">/</span><span aria-current="page" id="crumb-leaf">${esc(arg)}</span>`
    : `${group}<span aria-current="page">${esc(VIEW_TITLE[view] || 'Runs')}</span>`;
}

function route() {
  clearTimers();
  navId += 1;
  const [name, arg] = (location.hash.replace(/^#\/?/, '') || 'runs').split('/');
  const view = routes[name] ? name : 'runs';
  $$('.rail nav a').forEach((a) => a.toggleAttribute('aria-current', a.dataset.nav === (view === 'run' || view === 'compare' ? 'runs' : view)));
  $$('.rail nav a').forEach((a) => { if (a.hasAttribute('aria-current')) a.setAttribute('aria-current', 'page'); });
  setCrumbs(view, arg ? decodeURIComponent(arg) : undefined);
  const main = $('#main');
  main.innerHTML = '';
  window.scrollTo(0, 0);
  const here = stillHere();
  Promise.resolve(routes[view](main, arg ? decodeURIComponent(arg) : undefined)).catch((e) => {
    if (here()) main.innerHTML = `<div class="page"><div class="notice error"><div><strong>This page could not load.</strong> ${esc(e.message)}</div></div></div>`;
  });
}

// ===========================================================================
// Runs
// ===========================================================================

function statusPill(status) {
  const label = { pending: 'Queued', running: 'Running', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled' }[status] || status;
  return `<span class="pill ${esc(status)}">${esc(label)}</span>`;
}

function strategyLabel(cfg) {
  const s = String(cfg.strategy || 'solo');
  if (s === '5d') {
    const p = cfg.parallelism || {};
    const parts = ['tp', 'pp', 'cp', 'ep'].filter((k) => Number(p[k]) > 1).map((k) => `${k.toUpperCase()}${p[k]}`);
    return `5D ${parts.join('×') || 'DP'}`;
  }
  return s === 'auto' ? 'Automatic' : s.toUpperCase();
}

function runTitle(job) {
  if (job.name) return job.name;
  const c = job.config || {};
  const model = c.model_type === 'custom_transformer' ? 'custom transformer' : String(c.model_name || '').split('/').pop();
  return `${ALGO_LABEL[(c.algorithm || {}).type || 'pretrain'] || 'Run'} ${model}`;
}

const RUN_FILTERS = [['all', 'All'], ['running', 'Running'], ['pending', 'Queued'], ['completed', 'Completed'], ['failed', 'Failed'], ['cancelled', 'Cancelled']];
const MAX_COMPARE = 3;
const seconds = (j) => (j.started_at ? ((j.finished_at ? new Date(j.finished_at) : new Date()) - new Date(j.started_at)) / 1000 : null);
const modelOf = (c) => (c.model_type === 'custom_transformer' ? 'custom transformer' : String(c.model_name || ''));
function methodLabel(c) {
  if (!c.peft || !c.peft.enabled) return '';
  return c.quantization && c.quantization.enabled ? 'QLoRA' : 'LoRA';
}

async function renderRuns(main) {
  const here = stillHere();
  let filter = 'all', query = '', jobs = [];
  const picked = new Set();
  main.innerHTML = `
    <div class="page">
      <div class="page-head">
        <div><h1>Runs</h1><p>Every training job on this machine. Queued jobs start as soon as enough GPUs are free.</p></div>
        <a class="btn primary" href="#/new">New run</a>
      </div>
      <div class="stats" id="run-stats"></div>
      <div class="list-bar" id="list-bar" hidden>
        <div class="segmented" role="group" aria-label="Filter by status" id="run-filter"></div>
        <input type="search" id="run-search" placeholder="Search name, model or dataset" aria-label="Search runs">
        <button class="btn small" id="run-compare" disabled>Compare</button>
      </div>
      <div class="panel" id="runs-table"><div class="panel-pad"><div class="skeleton"></div></div></div>
    </div>`;

  const matches = (j) => {
    const c = j.config || {};
    const hay = [runTitle(j), c.model_name, (c.dataset || {}).name, (c.algorithm || {}).type, c.strategy].join(' ').toLowerCase();
    return (filter === 'all' || j.status === filter) && (!query || hay.includes(query));
  };

  const drawCompare = () => {
    const b = $('#run-compare');
    b.disabled = picked.size < 2;
    b.textContent = picked.size ? `Compare ${picked.size} of ${MAX_COMPARE}` : 'Compare';
    b.title = picked.size < 2 ? 'Tick two or three runs to compare them' : '';
  };

  const drawTable = () => {
    const el = $('#runs-table');
    if (!jobs.length) {
      el.innerHTML = `<div class="empty"><h2>No runs yet</h2>
        <p>Pick a goal, a model and a dataset. The planner tells you whether it fits on one GPU or needs FSDP or 5D parallelism.</p>
        <a class="btn primary" href="#/new">Start a run</a></div>`;
      return;
    }
    const shown = jobs.filter(matches);
    if (!shown.length) { el.innerHTML = '<div class="empty"><h2>No run matches</h2><p>Change the filter or clear the search.</p></div>'; return; }
    el.innerHTML = `<div class="table-wrap"><table class="runs">
      <thead><tr><th><span class="sr">Select to compare</span></th><th>Run</th><th>Status</th><th>Goal</th><th>Strategy</th><th>Result</th><th class="right">Duration</th><th class="right">Created</th></tr></thead>
      <tbody>${shown.map((j) => {
        const c = j.config || {}, p = j.progress, m = j.summary;
        const frac = p ? p.fraction : 0;
        const result = j.status === 'running'
          ? `<div class="bar" role="progressbar" aria-valuenow="${Math.round(frac * 100)}" aria-valuemin="0" aria-valuemax="100"><span style="width:${(frac * 100).toFixed(1)}%"></span></div>
             <div class="run-sub num">${p ? `${Math.round(frac * 100)}%, step ${fmt.int(p.global_step)} of ${fmt.int(p.total_steps)}` : 'starting'}</div>`
          : m ? `<span class="num"><b>${VIZ.num(m.value)}</b> <span class="faint">${esc(m.key)}</span></span>`
          : j.status === 'failed' ? '<span class="faint">stopped before the first step</span>' : '<span class="faint">–</span>';
        const can = j.status !== 'pending';
        return `<tr class="clickable" data-id="${esc(j.id)}">
          <td class="pick"><input type="checkbox" data-pick="${esc(j.id)}" aria-label="Compare ${esc(runTitle(j))}" ${picked.has(j.id) ? 'checked' : ''} ${can ? '' : 'disabled'}></td>
          <td data-l="Run"><a class="run-name" href="#/run/${esc(j.id)}">${esc(runTitle(j))}</a><div class="run-sub">${esc(modelOf(c))}</div></td>
          <td data-l="Status">${statusPill(j.status)}</td>
          <td data-l="Goal">${esc(ALGO_LABEL[(c.algorithm || {}).type || 'pretrain'] || '')} <span class="faint">${methodLabel(c)}</span></td>
          <td data-l="Strategy">${esc(strategyLabel(c))} <span class="faint num">${j.gpu_count} GPU${j.gpu_count > 1 ? 's' : ''}</span></td>
          <td data-l="Result" class="result">${result}</td>
          <td data-l="Duration" class="right num">${j.status === 'pending' ? '–' : esc(fmt.dur(seconds(j)))}</td>
          <td data-l="Created" class="right run-sub">${esc(fmt.ago(j.created_at))}</td></tr>`;
      }).join('')}</tbody></table></div>`;
  };

  $('#runs-table').addEventListener('click', (e) => {
    const box = e.target.closest('[data-pick]');
    if (box) {
      if (box.checked && picked.size >= MAX_COMPARE) { box.checked = false; toast(`Compare up to ${MAX_COMPARE} runs at a time`); } else if (box.checked) picked.add(box.dataset.pick); else picked.delete(box.dataset.pick);
      drawCompare();
      return;
    }
    if (e.target.closest('a, .pick')) return;
    const tr = e.target.closest('tr[data-id]');
    if (tr) location.hash = `#/run/${tr.dataset.id}`;
  });
  $('#run-search').oninput = (e) => { query = e.target.value.trim().toLowerCase(); drawTable(); };
  $('#run-filter').onclick = (e) => { const b = e.target.closest('button'); if (b) { filter = b.dataset.val; draw(); } };
  $('#run-compare').onclick = () => { location.hash = `#/compare/${[...picked].join(',')}`; };

  const draw = async () => {
    let q;
    try { const [a, b] = await Promise.all([api('/api/queue/jobs?limit=500'), api('/api/queue/status')]); if (!here()) return; jobs = a.jobs; q = b; } catch (e) {
      if (here()) $('#runs-table').innerHTML = `<div class="panel-pad"><div class="notice error"><div>${esc(e.message)}</div></div></div>`;
      return;
    }
    const count = (st) => jobs.filter((j) => j.status === st).length;
    const busy = q.total_gpus - q.available_gpus;
    const finished = count('completed') + count('failed');
    const tile = (k, v, sub, dot = '', extra = '') => `<div class="stat"><div class="k"><span>${k}</span>${dot ? `<i class="${dot}"></i>` : ''}</div><div class="v">${v}</div>${extra}<div class="s">${sub}</div></div>`;
    $('#run-stats').innerHTML = [
      tile('Running', count('running'), busy ? `${busy} GPU${busy > 1 ? 's' : ''} busy` : 'No GPU reserved', 'running'),
      tile('Queued', count('pending'), count('pending') ? 'Starts when GPUs free up' : 'Queue is empty', 'pending'),
      tile('GPUs free', `${q.available_gpus}<small> of ${q.total_gpus}</small>`, busy ? `${busy} in use` : 'All idle'),
      tile('Completed', count('completed'), finished ? `${Math.round((count('completed') / finished) * 100)}% of finished runs` : 'None yet', 'completed'),
    ].join('');
    $('#list-bar').hidden = !jobs.length;
    $('#run-filter').innerHTML = RUN_FILTERS.map(([id, label]) => [id, label, id === 'all' ? jobs.length : count(id)]).filter(([id, , n]) => n || id === 'all' || id === filter)
      .map(([id, label, n]) => `<button type="button" data-val="${id}" aria-pressed="${id === filter}">${label} <span class="num faint">${n}</span></button>`).join('');
    [...picked].forEach((id) => { if (!jobs.some((j) => j.id === id)) picked.delete(id); });
    const focused = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.pick : null;
    drawTable(); drawCompare();
    if (focused) { const f = $(`[data-pick="${focused}"]`); if (f) f.focus(); }
  };
  await draw();
  if (here()) every(3000, draw);
}

// ---------------------------------------------------------------------------
// Run detail
// ---------------------------------------------------------------------------

async function renderRun(main, id) {
  const here = stillHere();
  main.innerHTML = '<div class="page"><div class="skeleton" style="width:40%;height:28px"></div></div>';
  let job = await api(`/api/queue/jobs/${encodeURIComponent(id)}?tail=5000`);
  if (!here()) return;
  let tab = 'logs', metric = null, smoothing = store.get('omni-smooth', 0), sig = '', log = null;

  main.innerHTML = `
    <div class="page">
      <div class="run-head">
        <div><h1><span>${esc(runTitle(job))}</span><span id="run-status"></span></h1><div class="meta" id="run-meta"></div></div>
        <div class="btn-row" id="run-actions"></div>
      </div>
      <div id="run-error"></div>
      <div class="kpis panel" id="run-kpis"></div>
      <div class="panel" id="chart-panel">
        <div class="panel-head wrap">
          <div class="segmented" role="group" aria-label="Metric" id="metric-pick"></div>
          <div class="btn-row"><label class="inline">Smoothing <select id="smooth"><option value="0">Off</option><option value="0.6">Light</option><option value="0.9">Strong</option></select></label>
          <button class="btn small" id="csv">Download CSV</button></div>
        </div>
        <div class="panel-pad"><p class="chart-note faint" id="chart-note"></p><div id="chart"></div></div>
      </div>
      <div class="panel">
        <div class="tabs" role="tablist">
          <button role="tab" data-tab="logs" aria-selected="true">Log</button>
          <button role="tab" data-tab="config" aria-selected="false">Config</button>
        </div>
        <div id="tab-logs"></div><div id="tab-config" hidden></div>
      </div>
    </div>`;
  const leaf = $('#crumb-leaf'); if (leaf) leaf.textContent = runTitle(job);
  log = new LogView($('#tab-logs'), { name: (job.name || id).replace(/[^\w.-]+/g, '_'), url: `/api/queue/jobs/${encodeURIComponent(id)}/log` });
  $('#smooth').value = String(smoothing);
  $('#smooth').onchange = (e) => { smoothing = Number(e.target.value); store.set('omni-smooth', smoothing); drawChart(true); };
  $('#metric-pick').onclick = (e) => { const b = e.target.closest('button'); if (b) { metric = b.dataset.val; drawChart(true); } };
  $('#csv').onclick = () => {
    const pts = (job.metrics || {}).points || [], keys = ['global_step', 'epoch', ...metricKeys(pts)];
    download(`${(job.name || 'run').replace(/[^\w.-]+/g, '_')}-metrics.csv`, [keys.join(','), ...pts.map((p) => keys.map((k) => p[k] ?? '').join(','))].join('\n'));
  };
  $$('.tabs button', main).forEach((b) => { b.onclick = async () => {
    tab = b.dataset.tab;
    $$('.tabs button', main).forEach((x) => x.setAttribute('aria-selected', String(x === b)));
    $('#tab-logs').hidden = tab !== 'logs'; $('#tab-config').hidden = tab !== 'config';
    if (tab === 'config' && !$('#tab-config').innerHTML) {
      const r = await api('/api/config/render', { json: { config: job.config } });
      if (here()) $('#tab-config').innerHTML = `<div class="panel-pad"><pre class="code">${esc(r.yaml)}</pre></div>`;
    }
  }; });

  const cloneRun = () => {
    store.set('omni-draft', { cfg: loadConfig(job.config), ui: { name: `${runTitle(job)}-copy` }, step: 4 });
    location.hash = '#/new';
  };
  const act = {
    stop: () => dialog({ title: job.status === 'pending' ? 'Cancel this queued run?' : 'Stop this run?',
      body: '<p class="muted">The process is terminated. Checkpoints already saved stay on disk.</p>',
      actions: [{ label: 'Keep it' }, { label: job.status === 'pending' ? 'Cancel run' : 'Stop run', kind: 'primary', onClick: async () => {
        try { await api(`/api/queue/jobs/${id}/cancel`, { method: 'POST' }); toast(job.status === 'pending' ? 'Run cancelled' : 'Run stopped'); refresh(); } catch (e) { toast(e.message, 'error'); }
      } }] }),
    clone: cloneRun,
    yaml: async () => { const r = await api('/api/config/render', { json: { config: job.config } }); download(`${(job.name || 'run').replace(/[^\w.-]+/g, '_')}.yaml`, r.yaml); },
    delete: () => dialog({ title: 'Delete this run from history?', body: '<p class="muted">Its log is deleted. Checkpoints on disk are kept.</p>',
      actions: [{ label: 'Keep' }, { label: 'Delete run', kind: 'primary', onClick: async () => {
        try { await api(`/api/queue/jobs/${id}`, { method: 'DELETE' }); toast('Run deleted'); location.hash = '#/runs'; } catch (e) { toast(e.message, 'error'); }
      } }] }),
    error: () => { $('[data-tab="logs"]').click(); log.showError(); },
  };
  main.addEventListener('click', (e) => { const b = e.target.closest('[data-act]'); if (b && act[b.dataset.act]) act[b.dataset.act](); });

  const drawChart = (force) => {
    const pts = (job.metrics || {}).points || [];
    const keys = metricKeys(pts);
    const now = `${pts.length}|${metric}|${smoothing}|${$('#chart').clientWidth}`;
    if (!force && now === sig) return;                       // keep the hover readout while nothing changed
    sig = now;
    const none = pts.length < 2;
    $('#chart-panel').hidden = none && !['running', 'pending'].includes(job.status);
    if (!metric || !keys.includes(metric)) metric = keys[0] || 'loss';
    $('#metric-pick').innerHTML = (keys.length ? keys : ['loss']).map((k) => `<button type="button" data-val="${esc(k)}" aria-pressed="${k === metric}">${esc(metricLabel(k))}</button>`).join('');
    $('#csv').disabled = none;
    const series = metricSeries(pts, metric);
    const mean = series.length ? series.reduce((a, [, y]) => a + y, 0) / series.length : null;
    $('#chart-note').textContent = none ? '' : [(METRIC_INFO[metric] || [])[1], `mean ${VIZ.num(mean)}`, `last ${VIZ.num(series.length ? series[series.length - 1][1] : null)}`].filter(Boolean).join(', ');
    drawLineChart($('#chart'), [{ name: metricLabel(metric), color: 'var(--c1)', points: series }], { label: metricLabel(metric), smoothing });
  };

  const draw = () => {
    const c = job.config || {}, m = job.metrics || {}, p = m.progress;
    const active = job.status === 'running' || job.status === 'pending';
    $('#run-status').innerHTML = statusPill(job.status);
    $('#run-meta').innerHTML = [modelOf(c), [ALGO_LABEL[(c.algorithm || {}).type || 'pretrain'], methodLabel(c)].filter(Boolean).join(' + '),
      `${strategyLabel(c)} on ${job.gpu_count} GPU${job.gpu_count > 1 ? 's' : ''}`, (c.dataset || {}).name].filter(Boolean).map((t) => `<span>${esc(t)}</span>`).join('');
    $('#run-actions').innerHTML = `${active ? `<button class="btn danger" data-act="stop">${job.status === 'pending' ? 'Cancel' : 'Stop run'}</button>` : ''}
      <button class="btn" data-act="clone">Clone and edit</button><button class="btn" data-act="yaml">Download YAML</button>
      ${active ? '' : '<button class="btn ghost danger" data-act="delete">Delete</button>'}`;

    const frac = job.status === 'completed' ? 1 : p ? p.fraction : 0;
    const elapsed = seconds(job);
    const eta = job.status === 'running' && p && frac > 0 && elapsed ? (elapsed * (1 - frac)) / frac : null;
    const sum = p && p.reward != null ? (() => { const r = metricSeries(m.points, 'reward').slice(-10).map((x) => x[1]); return ['Reward, last 10 steps', VIZ.num(r.reduce((a, b) => a + b, 0) / (r.length || 1))]; })() : p ? ['Loss', VIZ.num(p.loss)] : null;
    const kpis = [
      p ? ['Progress', `${Math.round(frac * 100)}%`, frac] : job.status === 'pending' ? ['Queue', job.queue_position ? `Position ${job.queue_position}` : 'Waiting'] : null,
      p ? ['Step', `${fmt.int(p.global_step)}<small> of ${fmt.int(p.total_steps)}</small>`] : null,
      p && p.epochs > 1 ? ['Epoch', `${p.epoch}<small> of ${p.epochs}</small>`] : null,
      sum,
      p && p.reward == null && p.avg != null ? ['Average loss', VIZ.num(p.avg)] : null,
      p && p.tokens_per_s ? ['Tokens per second', fmt.int(p.tokens_per_s)] : null,
      elapsed != null ? [job.status === 'running' ? 'Elapsed' : 'Duration', fmt.dur(elapsed)] : null,
      eta != null ? ['Time left', fmt.dur(eta)] : job.finished_at ? ['Finished', fmt.ago(job.finished_at)] : null,
    ].filter(Boolean);
    $('#run-kpis').innerHTML = kpis.map(([k, v, f]) => `<div class="kpi"><div class="k">${k}</div><div class="v num">${v}</div>${f != null ? `<div class="bar"><span style="width:${(f * 100).toFixed(1)}%"></span></div>` : ''}</div>`).join('');
    $('#run-kpis').hidden = !kpis.length;

    const err = job.status === 'failed' ? (errorSummary(job.logs || []) || job.error_message || 'The process stopped without an error message.') : '';
    const hint = err ? failureHint(`${err}\n${(job.logs || []).slice(-60).join('\n')}`) : null;
    $('#run-error').innerHTML = err ? `<div class="fail panel"><div class="fail-body"><h2>${esc(hint ? hint.what : 'The run failed.')}</h2>
      ${hint ? `<p>${esc(hint.fix)}</p>` : ''}<pre>${esc(err.length > 400 ? `${err.slice(0, 400)}…` : err)}</pre>
      <div class="btn-row"><button class="btn primary small" data-act="clone">Clone and fix</button><button class="btn small" data-act="error">Show in log</button></div></div></div>` : '';
    log.set(job.logs || []);
    drawChart(false);
  };

  const refresh = async () => {
    try { const j = await api(`/api/queue/jobs/${encodeURIComponent(id)}?tail=5000`); if (here()) { job = j; draw(); } } catch (e) { /* transient: the next poll retries */ }
  };
  draw();
  every(2000, () => { if (job.status === 'running' || job.status === 'pending') refresh(); });
  let rt = null;
  const onResize = () => { if (!here()) { window.removeEventListener('resize', onResize); return; } clearTimeout(rt); rt = setTimeout(() => here() && drawChart(true), 120); };
  window.addEventListener('resize', onResize);
}

// ---------------------------------------------------------------------------
// Compare runs
// ---------------------------------------------------------------------------

function flatConfig(o, prefix = '', out = {}) {
  Object.entries(o || {}).forEach(([k, v]) => {
    if (v && typeof v === 'object' && !Array.isArray(v)) flatConfig(v, `${prefix}${k}.`, out); else out[prefix + k] = v == null ? '' : Array.isArray(v) ? v.join(', ') : String(v);
  });
  return out;
}

async function renderCompare(main, arg) {
  const here = stillHere();
  const ids = String(arg || '').split(',').filter(Boolean).slice(0, MAX_COMPARE);
  const got = await Promise.allSettled(ids.map((id) => api(`/api/queue/jobs/${encodeURIComponent(id)}?tail=1`)));
  if (!here()) return;
  const jobs = got.filter((g) => g.status === 'fulfilled').map((g) => g.value);
  if (jobs.length < 2) {
    main.innerHTML = '<div class="page"><div class="panel empty"><h2>Pick at least two runs</h2><p>Tick two or three runs in the list, then press Compare.</p><a class="btn primary" href="#/runs">Go to runs</a></div></div>';
    return;
  }
  const color = (i) => `var(--c${i + 1})`;
  const names = jobs.map((j, i) => (jobs.some((k, n) => n !== i && runTitle(k) === runTitle(j)) ? `${runTitle(j)} (${j.id.slice(0, 4)})` : runTitle(j)));
  const keys = [...new Set(jobs.flatMap((j) => metricKeys(j.metrics.points)))];
  const shared = keys.filter((k) => jobs.every((j) => metricSeries(j.metrics.points, k).length > 1));
  let metric = shared[0] || keys[0], smoothing = store.get('omni-smooth', 0);
  const flat = jobs.map((j) => flatConfig(j.config));
  const diff = [...new Set(flat.flatMap(Object.keys))].filter((k) => new Set(flat.map((f) => f[k] ?? '')).size > 1).sort();
  const head = names.map((n, i) => `<th><i class="key" style="background:${color(i)}"></i><a href="#/run/${esc(jobs[i].id)}">${esc(n)}</a></th>`).join('');
  const last = (j, k) => { const se = metricSeries(j.metrics.points, k).slice(k === 'loss' ? -1 : -10); return se.length ? VIZ.num(se.reduce((a, x) => a + x[1], 0) / se.length) : '–'; };
  main.innerHTML = `<div class="page">
    <div class="page-head"><div><h1>Compare runs</h1><p>${names.map(esc).join(', ')}</p></div><a class="btn" href="#/runs">Back to runs</a></div>
    <div class="panel"><div class="panel-head wrap"><div class="segmented" role="group" aria-label="Metric" id="metric-pick"></div>
      <label class="inline">Smoothing <select id="smooth"><option value="0">Off</option><option value="0.6">Light</option><option value="0.9">Strong</option></select></label></div>
      <div class="panel-pad"><div id="chart"></div></div></div>
    <div class="panel"><div class="panel-head"><h2>Results</h2><span class="faint">last loss; mean of the last 10 steps for everything else</span></div><div class="table-wrap"><table class="cmp"><thead><tr><th>Metric</th>${head}</tr></thead><tbody>
      <tr><td>Status</td>${jobs.map((j) => `<td>${statusPill(j.status)}</td>`).join('')}</tr>
      <tr><td>Duration</td>${jobs.map((j) => `<td class="num">${esc(fmt.dur(seconds(j)))}</td>`).join('')}</tr>
      ${keys.map((k) => `<tr><td>${esc(metricLabel(k))}</td>${jobs.map((j) => `<td class="num">${last(j, k)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></div>
    <div class="panel"><div class="panel-head"><h2>Config differences</h2><span class="faint">${diff.length} of ${new Set(flat.flatMap(Object.keys)).size} options</span></div>
      ${diff.length ? `<div class="table-wrap"><table class="cmp"><thead><tr><th>Option</th>${head}</tr></thead><tbody>${diff.map((k) => `<tr><td><code>${esc(k)}</code></td>${flat.map((f) => `<td>${esc(f[k] ?? '–') || '–'}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : '<div class="panel-pad muted">These runs use the same config.</div>'}</div></div>`;
  const draw = () => {
    $('#metric-pick').innerHTML = keys.map((k) => `<button type="button" data-val="${esc(k)}" aria-pressed="${k === metric}">${esc(metricLabel(k))}</button>`).join('');
    drawLineChart($('#chart'), jobs.map((j, i) => ({ name: names[i], color: color(i), points: metricSeries(j.metrics.points, metric) })), { label: metricLabel(metric), smoothing, height: 340, empty: 'No run logged this metric twice.' });
  };
  $('#smooth').value = String(smoothing);
  $('#smooth').onchange = (e) => { smoothing = Number(e.target.value); store.set('omni-smooth', smoothing); draw(); };
  $('#metric-pick').onclick = (e) => { const b = e.target.closest('button'); if (b) { metric = b.dataset.val; draw(); } };
  draw();
}

// ===========================================================================
// Plan rendering (shared by the wizard and the planner page)
// ===========================================================================

const VERDICT_TAG = {
  single_gpu: ['fit', 'One GPU is enough'],
  distributed_optional: ['fit', 'One GPU is enough'],
  distributed_recommended: ['running', 'More GPUs recommended'],
  distributed_required: ['running', 'Needs several GPUs'],
  infeasible: ['over', 'Does not fit'],
};

/**
 * Render a plan. opts.onApply(option) applies a recommendation, opts.onSuggest(workload) a what-if.
 */
function renderPlan(plan, opts = {}) {
  const cap = plan.cluster.gpu_mem_gb;
  const rec = plan.recommended;
  const sel = plan.selected;
  const [tagCls, tagText] = VERDICT_TAG[plan.verdict] || ['', plan.verdict];
  const options = plan.options.slice();
  const sameAsRec = sel && sel.label === rec.label;
  if (sel && !options.some((o) => o.label === sel.label)) options.push(sel);
  const mg = plan.min_gpus;
  const perf = (sel && sel.fits ? sel : rec).perf;
  const step = perf.step_seconds;

  const adv = plan.advice, auto = sel && sel.auto;
  const changed = Object.keys(rec.changes || {}).length > 0;
  const canApply = opts.onApply && adv.fits && !auto && (!sameAsRec || changed);
  let html = `
    <div class="verdict ${esc(plan.verdict)}">
      <div class="tag"><span class="pill ${tagCls}">${tagText}</span></div>
      <p class="need">${esc(adv.need)}${adv.data ? ` ${esc(adv.data)}` : ''}</p>
      <h3>${adv.fits ? '<small>Recommended</small>' : ''}${esc(adv.title)}${adv.tech ? ` <span class="tech">${esc(adv.tech)}</span>` : ''}</h3>
      <p>${esc(adv.detail)}</p>
      ${[adv.why, ...adv.changes].filter(Boolean).map((t) => `<p>${esc(t)}</p>`).join('')}
      ${auto && adv.fits ? `<p class="auto-note">${isRemotePlan() ? 'Automatic decides again on the machine that runs it.' : 'Automatic uses this when the run starts.'}</p>` : ''}
      ${canApply ? `<div class="btn-row" style="margin-top:12px"><button class="btn primary small" data-apply="rec">Use recommended</button></div>` : ''}
    </div>
    ${adv.others.length ? `<div class="plan-sec"><h4>${esc(adv.others_title)}</h4><ul class="plan-list">${adv.others.map((o, i) => `
      <li class="sugg"><div><strong>${esc(o.title)}</strong><br><span class="muted">${esc(o.detail)}.${o.why ? ` ${esc(o.why)}` : ''}</span></div>
      ${opts.onSuggest && Object.keys(o.workload || {}).length ? `<button class="btn small" data-sugg="${i}">${{ lora: 'Use LoRA', qlora: 'Use QLoRA' }[o.workload.method] || 'Apply'}</button>` : ''}</li>`).join('')}</ul></div>` : ''}
    ${adv.tips.length ? `<div class="plan-sec">${adv.tips.map((t) => `<div class="notice"><div>${esc(t)}</div></div>`).join('')}</div>` : ''}
    <div class="plan-sec">
      <h4>Estimated peak memory per GPU</h4>
      ${memoryBars(options, cap, plan.cluster.usable_gb, sel && sel.label)}
    </div>`;

  const detail = sel || rec;
  const fold = (id, title, right, body) => `<details class="fold" data-fold="${id}" ${store.get('omni-fold-' + id, id === 'speed') ? 'open' : ''}><summary><span>${title}</span><span class="num">${right}</span></summary>${body}</details>`;
  html += fold('speed', 'Speed', `${fmt.int(perf.tokens_per_second)} tok/s`, `<ul class="plan-list">
      <li><span class="muted">Step time</span><span class="num">${step < 1 ? `${Math.round(step * 1000)} ms` : `${step.toFixed(2)} s`}</span></li>
      ${perf.generation_seconds ? `<li><span class="muted">of which writing answers</span><span class="num">${perf.generation_seconds.toFixed(2)} s</span></li>` : ''}
      ${perf.total_seconds ? `<li><span class="muted">Whole run, ${fmt.int(perf.total_steps)} steps</span><span class="num">${fmt.dur(perf.total_seconds)}</span></li>` : ''}
      <li><span class="muted">Assumed GPU utilisation</span><span class="num">${Math.round(perf.mfu * 100)}%</span></li></ul>`);
  html += fold('mem', `Memory breakdown, ${esc(detail.label)}`, fmt.gb(detail.memory_gb.total), `<table class="mem-table"><tbody>
      ${SEGMENTS.map(([k, label, c]) => `<tr><td><i style="background:${c}"></i>${label}</td><td>${fmt.gb(detail.memory_gb[k], 2)}</td></tr>`).join('')}
      <tr class="total"><td>Total, ${fmt.gb(plan.cluster.usable_gb)} usable</td><td>${fmt.gb(detail.memory_gb.total, 2)}</td></tr></tbody></table>`);
  html += fold('min', 'Fewest GPUs that fit', mg.single ? '1' : esc(String([mg.fsdp, mg['5d']].filter(Boolean).sort((a, b) => a - b)[0] || 'none')), `<ul class="plan-list">
      <li><span class="muted">One GPU</span><span>${mg.single ? 'yes' : 'no'}</span></li>
      <li><span class="muted">FSDP</span><span class="num">${mg.fsdp ? `${mg.fsdp} GPU${mg.fsdp > 1 ? 's' : ''}` : 'not possible'}</span></li>
      <li><span class="muted">5D</span><span class="num">${mg['5d'] ? `${mg['5d']} GPUs, ${esc(mg['5d_layout'])}` : 'not available'}</span></li></ul>`);

  const notes = [plan.summary, ...plan.notes];
  const model = plan.model;
  notes.unshift(`${model.name}: ${fmt.params(model.params)} parameters${model.num_experts ? ` (${fmt.params(model.active_params)} active)` : ''}, ${model.layers} layers, vocab ${fmt.int(model.vocab)} (${model.source === 'hf-config' ? 'from config.json' : model.source}).`);
  html += fold('notes', 'Notes', String(notes.length), `<ul class="plan-list">${notes.map((n) => `<li class="muted" style="display:block">${esc(n)}</li>`).join('')}</ul>`);

  const root = document.createElement('div');
  root.innerHTML = html;
  $$('[data-fold]', root).forEach((d) => { d.ontoggle = () => store.set('omni-fold-' + d.dataset.fold, d.open); });
  const applyBtn = $('[data-apply]', root);
  if (applyBtn) applyBtn.onclick = () => opts.onApply(rec);
  $$('[data-sugg]', root).forEach((b) => { b.onclick = () => opts.onSuggest(adv.others[Number(b.dataset.sugg)].workload, adv.others[Number(b.dataset.sugg)]); });
  return root;
}

/** Is the plan on screen for a machine other than this one? (Only the wizard knows where a run goes.) */
const isRemotePlan = () => !!W && location.hash.startsWith('#/new') && !isLocal();

/** Use a concrete option of a plan: its strategy, GPUs, layout and the changes it needs. */
function applyOption(cfg, o) {
  cfg.strategy = o.strategy;
  cfg.num_gpus = o.gpus;
  if (o.parallelism) cfg.parallelism = { ...cfg.parallelism, ...o.parallelism };
  if (o.changes) applyWorkload(cfg, o.changes);
}

/** Map planner what-if changes back onto a config. */
function applyWorkload(cfg, w) {
  if ('grad_ckpt' in w) cfg.training.gradient_checkpointing = w.grad_ckpt;
  if ('batch_size' in w) cfg.training.batch_size = w.batch_size;
  if ('seq_len' in w) { cfg.training.max_length = w.seq_len; if (cfg.model_type === 'custom_transformer') cfg.custom_transformer_args.max_seq_len = w.seq_len; }
  if ('method' in w) setMethod(cfg, w.method);
  if ('lora_r' in w) { cfg.peft.r = w.lora_r; cfg.peft.alpha = 2 * w.lora_r; }
  if ('quant_bits' in w) cfg.quantization.bits = w.quant_bits;
  if (cfg.quantization.enabled && cfg.strategy === 'fsdp') cfg.strategy = cfg.num_gpus > 1 ? 'ddp' : 'solo';
  if (cfg.peft.enabled && cfg.strategy === '5d') cfg.strategy = cfg.num_gpus > 1 ? 'fsdp' : 'solo';
}

function setMethod(cfg, method) {
  if (method === 'full') { cfg.peft.enabled = false; cfg.quantization.enabled = false; }
  if (method === 'lora') { cfg.peft.enabled = true; cfg.peft.type = 'lora'; cfg.quantization.enabled = false; }
  if (method === 'qlora') {
    cfg.peft.enabled = true; cfg.peft.type = 'qlora'; cfg.quantization.enabled = true;
    if (![4, 8].includes(Number(cfg.quantization.bits))) cfg.quantization.bits = 4;
  }
}

// ---------------------------------------------------------------------------
// 5D rank grid
// ---------------------------------------------------------------------------

function rankGrid(par, numGpus, perNode) {
  const tp = +par.tp || 1, pp = +par.pp || 1, cp = +par.cp || 1;
  const dp = par.dp === 'auto' || !par.dp ? Math.floor(numGpus / (tp * pp * cp)) : +par.dp;
  if (dp * tp * pp * cp !== numGpus || numGpus < 1) return '';
  if (numGpus > 128) return `<p class="muted">${numGpus} ranks: DP ${dp} × PP ${pp} × CP ${cp} × TP ${tp}.</p>`;
  const nodes = Math.ceil(numGpus / perNode);
  let html = '<div class="rankgrid">';
  for (let n = 0; n < nodes; n++) {
    html += `<div class="node"><div class="node-label">Node ${n}</div><div class="node-gpus" style="grid-template-columns:repeat(${Math.min(perNode, 8) > 4 ? 4 : Math.min(perNode, 4)},34px)">`;
    for (let r = n * perNode; r < Math.min(numGpus, (n + 1) * perNode); r++) {
      // rank = ((d * pp + p) * cp + c) * tp + t   (TP fastest)
      const t = r % tp, c = Math.floor(r / tp) % cp, p = Math.floor(r / (tp * cp)) % pp, d = Math.floor(r / (tp * cp * pp));
      html += `<div class="gpu" style="background:var(--s${p % 8})" title="rank ${r}: dp ${d}, pp stage ${p}, cp ${c}, tp ${t}">r${r}<small>${tp > 1 ? `t${t}` : ''}${cp > 1 ? `c${c}` : ''}${!(tp > 1 || cp > 1) ? `d${d}` : ''}</small></div>`;
    }
    html += '</div></div>';
  }
  html += '</div>';
  html += `<div class="axis-legend">
    <span><b>Colour</b>: pipeline stage (${pp})</span>
    <span><b>t</b>: tensor-parallel rank (${tp}), kept adjacent</span>
    ${cp > 1 ? `<span><b>c</b>: context-parallel rank (${cp})</span>` : ''}
    <span><b>DP</b>: ${dp} replica${dp > 1 ? 's' : ''}${par.zero1 ? ', ZeRO-1 sharded' : ''}</span>
  </div>`;
  return html;
}

function fiveDErrors(cfg) {
  const p = cfg.parallelism, n = +cfg.num_gpus;
  const tp = +p.tp, pp = +p.pp, cp = +p.cp, ep = +p.ep;
  const errs = [];
  const mp = tp * pp * cp;
  if ([tp, pp, cp, ep, +p.micro_batches].some((v) => !(v >= 1))) errs.push('Every size must be at least 1.');
  if (n % mp) errs.push(`GPU count (${n}) must be divisible by TP × PP × CP (${mp}).`);
  const dp = p.dp === 'auto' ? Math.floor(n / mp) : +p.dp;
  if (dp * mp !== n) errs.push(`DP × TP × PP × CP = ${dp * mp}, but the run has ${n} GPUs.`);
  if (p.sp && tp === 1) errs.push('Sequence parallelism needs TP > 1.');
  if (dp && dp % ep) errs.push(`EP (${ep}) must divide DP (${dp}).`);
  const ct = cfg.custom_transformer_args;
  if (ep > 1 && !(cfg.model_type === 'custom_transformer' && +ct.num_experts > 0)) errs.push('EP > 1 needs a mixture-of-experts model (custom transformer with experts).');
  const seq = cfg.model_type === 'custom_transformer' ? +ct.max_seq_len : +cfg.training.max_length;
  if (seq % cp) errs.push(`Sequence length ${seq} must be divisible by CP (${cp}).`);
  if (p.sp && (seq / cp) % tp) errs.push(`With SP, sequence length ${seq} must be divisible by CP × TP (${cp * tp}).`);
  if (cfg.model_type === 'custom_transformer' && (+ct.n_heads % tp)) errs.push(`Attention heads (${ct.n_heads}) must be divisible by TP (${tp}).`);
  if (cfg.model_type === 'custom_transformer' && pp > +ct.n_layers) errs.push(`PP (${pp}) cannot exceed the layer count (${ct.n_layers}).`);
  const warns = [];
  if (pp > 1 && +p.micro_batches < pp) warns.push(`Use at least ${pp} micro-batches: the pipeline bubble is (PP − 1) / micro-batches.`);
  if (tp > 8) warns.push('TP above 8 crosses node boundaries; keep TP within NVLink.');
  return { errs, warns, dp };
}

// ===========================================================================
// New-run wizard
// ===========================================================================

const STEPS = ['Goal', 'Model', 'Data', 'Compute', 'Training', 'Review'];
let W = null;

function wizardState() {
  const draft = store.get('omni-draft', null);
  const cfg = draft && draft.cfg ? loadConfig(draft.cfg) : defaultConfig();
  return {
    step: (draft && draft.step) || 0,
    cfg,
    ui: deepMerge({ name: '', target: 'local', hw: defaultHw(), algo: 'dpo' }, draftUi(draft)),
    plan: null, planSeq: 0, data: null, dataFor: null, spec: null, specFor: null, validation: null, yaml: null, editingYaml: false,
  };
}

function defaultHw() {
  return { gpu: 'h100', nodes: 1, gpus_per_node: 1, host: '', port: 29500, partition: 'gpu', time: '24:00:00', cpus: 8, mem: '64G' };
}
/** Stored drafts from before "Other machines" kept their hardware under `slurm`. */
function draftUi(draft) {
  const ui = clone((draft && draft.ui) || {});
  if (ui.slurm) { ui.hw = { ...ui.slurm, ...(ui.hw || {}) }; delete ui.slurm; }
  return ui;
}
/** Machines and GPUs per machine for a GPU count, as a starting point the user can change. */
function hwFor(gpus) { return gpus <= 8 ? { nodes: 1, gpus_per_node: gpus } : { nodes: Math.ceil(gpus / 8), gpus_per_node: 8 }; }
const isLocal = () => W.ui.target === 'local';
const gpusHere = () => (W.cfg.strategy === 'solo' ? 1 : +W.ui.hw.nodes * +W.ui.hw.gpus_per_node || 1);

function saveDraft() { store.set('omni-draft', { cfg: W.cfg, ui: W.ui, step: W.step }); }

async function renderWizard(main) {
  W = wizardState();
  const catalog = await loadCatalog();
  W.local = catalog.local_cluster;
  main.innerHTML = `
    <div class="page">
      <div class="page-head">
        <div><h1>New run</h1><p>Choose what to train and on what. The plan on the right updates as you go and tells you whether one GPU is enough, and what to do if it is not.</p></div>
        <div class="btn-row"><button class="btn ghost" id="wiz-reset">Start over</button></div>
      </div>
      <nav class="steps" id="steps" aria-label="Steps"></nav>
      <div class="split">
        <div class="panel panel-pad" id="step-panel"></div>
        <aside class="panel sticky plan-side" id="plan-panel" aria-live="polite" aria-label="Capacity plan"></aside>
      </div>
    </div>`;
  $('#wiz-reset').onclick = () => {
    dialog({ title: 'Start over?', body: '<p class="muted">This clears every choice in this draft.</p>', actions: [
      { label: 'Keep editing' }, { label: 'Start over', kind: 'primary', onClick: () => { store.del('omni-draft'); route(); } }] });
  };
  bindStep($('#step-panel'));
  drawSteps();
  drawStep();
  schedulePlan(0);
}

function drawSteps() {
  $('#steps').innerHTML = STEPS.map((s, i) => `<button type="button" data-step="${i}" class="${i < W.step ? 'done' : ''}" ${i === W.step ? 'aria-current="step"' : ''}>${s}</button>`).join('');
  $$('#steps button').forEach((b) => { b.onclick = () => goStep(Number(b.dataset.step)); });
}

function goStep(i) {
  W.step = Math.max(0, Math.min(STEPS.length - 1, i));
  W.editingYaml = false;
  saveDraft(); drawSteps(); drawStep();
  window.scrollTo({ top: 0 });
}

function field(label, path, { type = 'text', hint = '', step, min, placeholder = '', rerender = false, cls = '' } = {}) {
  const v = getPath(W.cfg, path);
  const kind = type === 'int' || type === 'float' ? 'number' : 'text';
  const val = type === 'float' && typeof v === 'number' ? fmt.lr(v) : (v ?? '');
  const id = `f-${path.replace(/\./g, '-')}`;
  return `<div class="field ${cls}"><label for="${id}">${label}</label>
    <input id="${id}" type="${type === 'float' ? 'text' : kind}" inputmode="${type === 'float' ? 'decimal' : type === 'int' ? 'numeric' : 'text'}" data-bind="${path}" data-type="${type}" value="${esc(val)}" ${step ? `step="${step}"` : ''} ${min != null ? `min="${min}"` : ''} placeholder="${esc(placeholder)}" ${rerender ? 'data-rerender' : ''}>
    ${hint ? `<span class="hint">${hint}</span>` : ''}</div>`;
}
function select(label, path, options, { hint = '', rerender = false } = {}) {
  const v = String(getPath(W.cfg, path));
  const id = `f-${path.replace(/\./g, '-')}`;
  return `<div class="field"><label for="${id}">${label}</label><select id="${id}" data-bind="${path}" data-type="str" ${rerender ? 'data-rerender' : ''}>
    ${options.map(([val, text]) => `<option value="${esc(val)}" ${String(val) === v ? 'selected' : ''}>${esc(text)}</option>`).join('')}</select>
    ${hint ? `<span class="hint">${hint}</span>` : ''}</div>`;
}
function check(label, path, small = '', { rerender = false } = {}) {
  return `<label class="check"><input type="checkbox" data-bind="${path}" data-type="bool" ${getPath(W.cfg, path) ? 'checked' : ''} ${rerender ? 'data-rerender' : ''}><span>${label}${small ? `<small>${small}</small>` : ''}</span></label>`;
}
function segmented(name, options, value) {
  return `<div class="segmented" role="group" data-seg="${name}">${options.map(([v, label, disabled, why]) => `<button type="button" data-val="${esc(v)}" aria-pressed="${String(v) === String(value)}" ${disabled ? `disabled title="${esc(why || '')}"` : ''}>${esc(label)}</button>`).join('')}</div>`;
}

function strategyAvailability() {
  const c = W.cfg, m = methodOf(c), a = c.algorithm.type;
  const why5d = !['llm', 'custom_transformer'].includes(c.model_type) ? '5D supports causal LMs and the custom transformer'
    : m !== 'full' ? '5D does not support LoRA or quantization yet'
    : a !== 'pretrain' ? 'Post-training does not run under 5D yet'
    : (c.model_type === 'llm' && W.spec && W.spec.name === c.model_name && !W.spec.five_d_ready) ? `5D loads Llama, Mistral and Qwen2 checkpoints (this is ${W.spec.arch || 'another architecture'})`
    : '';
  return {
    solo: '', ddp: '',
    fsdp: c.quantization.enabled ? 'FSDP cannot shard bitsandbytes-quantized layers' : '',
    '5d': why5d,
  };
}

function normalise() {
  const c = W.cfg;
  const avail = strategyAvailability();
  if (avail[c.strategy]) c.strategy = c.num_gpus > 1 && !avail.fsdp ? 'fsdp' : (c.num_gpus > 1 ? 'ddp' : 'solo');
  if (!isLocal()) c.num_gpus = gpusHere();
  else if (c.strategy === 'auto') { if (c.num_gpus !== 'auto') c.num_gpus = Math.max(1, Math.min(+c.num_gpus || 1, W.local.gpus_per_node)); }
  else if (c.num_gpus === 'auto') c.num_gpus = c.strategy === 'solo' ? 1 : W.local.gpus_per_node;
  if (c.model_type === 'custom_transformer') { setMethod(c, 'full'); c.training.max_length = +c.custom_transformer_args.max_seq_len; }
  if (!PEFT_TYPES.has(c.model_type)) setMethod(c, 'full');
  if (!QLORA_TYPES.has(c.model_type) && c.quantization.enabled) setMethod(c, 'lora');
}

function drawStep() {
  normalise();
  const panel = $('#step-panel');
  const body = [stepGoal, stepModel, stepData, stepCompute, stepTraining, stepReview][W.step]();
  panel.innerHTML = `<div class="step-body">${body}
    <div class="step-nav">
      <button class="btn" type="button" id="step-back" ${W.step === 0 ? 'disabled' : ''}>Back</button>
      ${W.step < STEPS.length - 1 ? `<button class="btn primary" type="button" id="step-next">Continue to ${STEPS[W.step + 1].toLowerCase()}</button>` : `<button class="btn primary" type="button" id="launch">${launchLabel()}</button>`}
    </div></div>`;
  $('#step-back').onclick = () => goStep(W.step - 1);
  const next = $('#step-next');
  if (next) next.onclick = () => goStep(W.step + 1);
  if (W.step === 1) inspectModelSoon();
  if (W.step >= 2) inspectDataSoon(0);
  if (W.step === 5) loadReview();
}

let nameTimer = null;
function bindStep(panel) {
  const onInput = (e) => {
    const el = e.target;
    if (!el.dataset || !el.dataset.bind) return;
    const t = el.dataset.type;
    let v = el.value;
    if (t === 'bool') v = el.checked;
    else if (t === 'int') { v = parseInt(v, 10); if (!Number.isFinite(v)) { el.classList.add('invalid'); return; } }
    else if (t === 'float') { v = Number(String(v).trim()); if (!Number.isFinite(v) || String(el.value).trim() === '') { el.classList.add('invalid'); return; } }
    el.classList.remove('invalid');
    if (el.dataset.bind.startsWith('ui.')) setPath(W, el.dataset.bind, v); else setPath(W.cfg, el.dataset.bind, v);
    if (e.type === 'input' && el.tagName === 'SELECT') return;
    if (el.dataset.bind === 'ui.name') {                 // only names the file: the config and its YAML are unchanged
      saveDraft(); clearTimeout(nameTimer); nameTimer = setTimeout(() => { const f = $('#run-file'); if (f) f.textContent = runFile(); loadCommands(); }, 300);
      return;
    }
    onCfgChange(el.hasAttribute('data-rerender') && e.type === 'change');
    if (el.dataset.bind === 'model_name') inspectModelSoon();
  };
  panel.addEventListener('input', onInput);
  panel.addEventListener('change', onInput);
  panel.addEventListener('click', (e) => {
    const b = e.target.closest('[data-model]');
    if (!b) return;
    const input = $('#f-model_name', panel);
    input.value = b.dataset.model;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    $$('[data-model]', panel).forEach((x) => x.classList.toggle('on', x === b));
  });
}

function onCfgChange(rerender) {
  normalise();
  saveDraft();
  W.validation = null; W.yaml = null;
  schedulePlan();
  if (W.step >= 2) inspectDataSoon();
  if (rerender) drawStep();
}

// ---------------------------------------------------------------------------
// Step 1: goal
// ---------------------------------------------------------------------------

function stepGoal() {
  const goal = goalOf(W.cfg);
  const goals = [
    ['pretrain', 'Pretrain', 'Next-token training on raw text, or the task loss of a vision, detection, VLM or embedding model. Every strategy, including 5D.', 'Solo, DDP, FSDP, 5D'],
    ['sft', 'Supervised fine-tuning', 'Teach a model to follow instructions or chat. Loss on the assistant turns only.', 'Solo, DDP, FSDP'],
    ['preference', 'Preference alignment', 'DPO, SimPO or ORPO on chosen / rejected answer pairs. No reward model needed.', 'DPO, SimPO, ORPO'],
    ['rl', 'Reinforcement learning', 'The model writes answers, reward functions or a reward model score them, and it learns from the scores.', 'GRPO, PPO'],
  ];
  const algo = W.cfg.algorithm.type;
  const cat = S.catalog.algorithms;
  const prefAlgo = ['dpo', 'simpo', 'orpo'].includes(algo) ? algo : W.ui.algo;
  setTimeout(() => {
    $$('.choice[data-goal]').forEach((b) => { b.onclick = () => setGoal(b.dataset.goal); });
    const rlSeg = $('[data-seg="rlalgo"]');
    if (rlSeg) rlSeg.onclick = (e) => { const b = e.target.closest('button'); if (b && !b.disabled) setGoal('rl', b.dataset.val); };
    const seg = $('[data-seg="algo"]');
    if (seg) seg.onclick = (e) => { const b = e.target.closest('button'); if (b && !b.disabled) { W.ui.algo = b.dataset.val; setGoal('preference', b.dataset.val); } };
    $('#tpl-list') && loadTemplates();
  });
  return `
    <section class="step-section">
      <h3>What do you want to do?</h3>
      <p>The goal decides the training objective, the dataset format and which strategies are available.</p>
      <div class="choices">${goals.map(([id, title, text, meta]) => `
        <button type="button" class="choice" data-goal="${id}" aria-pressed="${goal === id}">
          <span class="choice-title">${title}</span>
          <span class="choice-text">${text}</span>
          <span class="choice-meta">${meta}</span>
        </button>`).join('')}</div>
    </section>
    ${goal === 'preference' ? `<section class="step-section">
      <h3>Preference method</h3>
      <p>${esc((cat.find((a) => a.id === prefAlgo) || {}).summary || '')}</p>
      ${segmented('algo', [['dpo', 'DPO'], ['simpo', 'SimPO'], ['orpo', 'ORPO']], prefAlgo)}
    </section>` : ''}
    ${goal === 'rl' ? `<section class="step-section">
      <h3>RL method</h3>
      <p>${esc((cat.find((a) => a.id === W.cfg.algorithm.type) || {}).summary || '')}</p>
      ${segmented('rlalgo', [['grpo', 'GRPO'], ['ppo', 'PPO']], W.cfg.algorithm.type)}
    </section>` : ''}
    <section class="step-section">
      <h3>Or start from a template</h3>
      <p>Ready-made configs from <code>configs/</code>. Loading one replaces this draft.</p>
      <div class="tpl-groups" id="tpl-list"><div class="skeleton"></div></div>
    </section>`;
}

function setGoal(goal, algo) {
  const c = W.cfg;
  const prev = c.algorithm.type;
  const type = goal === 'pretrain' ? 'pretrain' : goal === 'sft' ? 'sft' : goal === 'rl' ? (algo || (isRL(prev) ? prev : 'grpo')) : (algo || W.ui.algo || 'dpo');
  c.algorithm.type = type;
  const info = S.catalog.algorithms.find((a) => a.id === type);
  if (goal !== 'pretrain') {
    if (c.model_type !== 'llm') { c.model_type = 'llm'; c.model_name = DEFAULT_MODELS.llm; }
    if (info && info.example_dataset && (prev === 'pretrain' || goalOf({ algorithm: { type: prev } }) !== goal)) {
      c.dataset = { name: info.example_dataset.name, subset: info.example_dataset.subset || null, split: info.example_dataset.split };
    }
    if (info && info.beta) c.algorithm.beta = info.beta;
    if (isRL(type) && methodOf(c) === 'full') setMethod(c, 'lora');
    if (isRL(type) && !isRL(prev)) c.algorithm.max_prompt_length = 192;
    if (!isRL(type) && isRL(prev)) c.algorithm.max_prompt_length = 256;
    if (type === 'grpo' && prev !== 'grpo') { c.training.batch_size = Math.min(c.training.batch_size, 2); c.algorithm.num_generations = 4; }  // x G answers each
    if (type === 'ppo' && prev !== 'ppo') { c.training.batch_size = 8; c.algorithm.num_generations = 1; c.algorithm.num_minibatches = 2; }
    if (c.strategy === '5d') c.strategy = c.num_gpus > 1 ? 'fsdp' : 'solo';
  } else if (prev !== 'pretrain' && c.model_type === 'llm') {
    c.dataset = clone(DEFAULT_DATA.llm);
  }
  if (info && info.lr) c.training.learning_rate = info.lr[methodOf(c) === 'full' ? 'full' : 'lora'];
  if (type !== prev) c.training.max_length = Math.max(c.training.max_length, goal === 'preference' || goal === 'rl' ? 512 : c.training.max_length);
  onCfgChange(true);
}

/** Template groups, decided from what each config actually contains. */
const TPL_GROUPS = [
  { id: 'post', title: 'Post-training', sub: 'SFT, preference alignment, reinforcement learning', test: (t) => t.algorithm !== 'pretrain' },
  { id: '5d', title: '5D parallelism', sub: 'Tensor, pipeline, context, expert and data parallel', test: (t) => t.strategy === '5d' },
  { id: 'llm', title: 'LLM fine-tuning', sub: 'Full, LoRA and QLoRA', test: (t) => t.model_type === 'llm' },
  { id: 'vision', title: 'Vision and multimodal', sub: 'Classification, detection, VLM', experimental: true, test: (t) => ['vision', 'yolo', 'vlm'].includes(t.model_type) },
  { id: 'other', title: 'Embeddings and other', sub: 'Encoders, seq2seq, from scratch', experimental: true, test: () => true },
];

function describeTemplate(t) {
  const short = t.model ? String(t.model).split('/').pop().replace(/-(Instruct|Chat)$/i, '') : 'Custom transformer';
  const title = t.algorithm !== 'pretrain' ? `${ALGO_LABEL[t.algorithm] || t.algorithm}, ${short}` : t.moe ? 'Mixture of experts' : short;
  const tags = [{ lora: 'LoRA', qlora: t.bits === 8 ? 'QLoRA 8-bit' : 'QLoRA', full: t.bits ? `${t.bits}-bit` : 'Full' }[t.method]];
  tags.push(t.strategy === 'solo' ? '1 GPU' : `${t.strategy === '5d' ? '5D' : t.strategy.toUpperCase()}, ${t.num_gpus} GPU${t.num_gpus > 1 ? 's' : ''}`);
  return { title, tags: tags.filter(Boolean) };
}

async function loadTemplates() {
  const el = $('#tpl-list');
  const here = stillHere();
  try {
    const { templates } = await api('/api/configs');
    if (!here() || !el.isConnected) return;
    const ok = templates.filter((t) => !t.error);
    if (!ok.length) { el.innerHTML = '<p class="faint" style="padding:16px">No templates in configs/.</p>'; return; }
    const want = { sft: 'post', preference: 'post', rl: 'post' }[goalOf(W.cfg)];
    el.innerHTML = TPL_GROUPS.map((g) => [g, ok.filter((t) => TPL_GROUPS.find((x) => x.test(t)) === g)]).filter(([, list]) => list.length).map(([g, list]) => `
      <details class="tpl-group"${g.id === want ? ' open' : ''}>
        <summary><svg class="chev" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>
          <span class="tpl-group-title">${esc(g.title)}</span>${g.experimental ? '<span class="pill">Experimental</span>' : ''}
          <span class="tpl-group-sub">${esc(g.sub)}</span><span class="tpl-count">${list.length}</span></summary>
        <div class="tpl-grid">${list.map((t) => {
          const d = describeTemplate(t);
          return `<button type="button" class="tpl-item tall" data-tpl="${esc(t.name)}" title="Load configs/${esc(t.name)}.yaml">
            <span class="tpl-name">${esc(d.title)}</span><span class="tpl-tags">${d.tags.map((x) => `<span>${esc(x)}</span>`).join('')}</span>
            <span class="tpl-file">${esc(t.name)}.yaml</span></button>`;
        }).join('')}</div>
      </details>`).join('');
    $$('[data-tpl]', el).forEach((b) => { b.onclick = () => applyTemplate(b.dataset.tpl); });
  } catch (e) { el.innerHTML = `<p class="faint" style="padding:16px">${esc(e.message)}</p>`; }
}

async function applyTemplate(name) {
  try {
    const t = await api(`/api/configs/${encodeURIComponent(name)}`);
    W.cfg = loadConfig(t.config);
    W.ui.name = name.replace(/_/g, '-');
    const need = W.cfg.strategy === 'solo' ? 1 : +W.cfg.num_gpus || 1;
    W.ui.target = need <= W.local.gpus_per_node ? 'local' : 'remote';
    if (!isLocal()) Object.assign(W.ui.hw, hwFor(need));
    toast(`Loaded ${name}.yaml`);
    onCfgChange(true);
  } catch (e) { toast(e.message, 'error'); }
}

// ---------------------------------------------------------------------------
// Step 2: model
// ---------------------------------------------------------------------------

function stepModel() {
  const c = W.cfg;
  const goal = goalOf(c);
  const types = goal === 'pretrain' ? MODEL_TYPES : MODEL_TYPES.filter((t) => t.id === 'llm');
  const method = methodOf(c);
  const ct = c.custom_transformer_args;
  setTimeout(() => {
    $$('.choice[data-mtype]').forEach((b) => { b.onclick = () => {
      const t = b.dataset.mtype;
      if (t === c.model_type) return;
      c.model_type = t;
      if (DEFAULT_MODELS[t]) c.model_name = DEFAULT_MODELS[t];
      c.dataset = clone(DEFAULT_DATA[t] || DEFAULT_DATA.llm);
      if (t === 'custom_transformer') { c.training.learning_rate = 3e-4; }
      onCfgChange(true);
    }; });
    const seg = $('[data-seg="method"]');
    if (seg) seg.onclick = (e) => { const b = e.target.closest('button'); if (b && !b.disabled) {
      setMethod(c, b.dataset.val);
      const info = S.catalog.algorithms.find((a) => a.id === c.algorithm.type);
      if (info && info.lr) c.training.learning_rate = info.lr[b.dataset.val === 'full' ? 'full' : 'lora'];
      onCfgChange(true);
    } };
  });
  const presets = S.catalog.models;
  return `
    <section class="step-section">
      <h3>Model type</h3>
      <p>${goal === 'pretrain' ? 'Each type has its own data format and training loss.' : 'Post-training works on causal language models.'}</p>
      <div class="choices">${types.map((t) => `<button type="button" class="choice" data-mtype="${t.id}" aria-pressed="${c.model_type === t.id}">
        <span class="choice-title">${t.label}</span><span class="choice-text">${t.hint}</span></button>`).join('')}</div>
    </section>
    ${c.model_type === 'custom_transformer' ? `
    <section class="step-section">
      <h3>Architecture</h3>
      <p>Built from random weights. Grouped-query attention and experts apply under strategy 5D.</p>
      <div class="grid">
        ${field('Layers', 'custom_transformer_args.n_layers', { type: 'int', min: 1 })}
        ${field('Hidden size', 'custom_transformer_args.dim', { type: 'int', min: 8 })}
        ${field('Attention heads', 'custom_transformer_args.n_heads', { type: 'int', min: 1, hint: 'Must divide the hidden size' })}
        ${field('Key/value heads', 'custom_transformer_args.n_kv_heads', { type: 'int', min: 0, hint: '0 = same as heads (5D only)' })}
        ${field('Vocabulary', 'custom_transformer_args.vocab_size', { type: 'int', min: 2 })}
        ${field('Sequence length', 'custom_transformer_args.max_seq_len', { type: 'int', min: 8 })}
        ${field('Experts', 'custom_transformer_args.num_experts', { type: 'int', min: 0, hint: '0 = dense; MoE runs under 5D' })}
        ${field('Experts per token', 'custom_transformer_args.top_k', { type: 'int', min: 1 })}
        ${field('Dropout', 'custom_transformer_args.dropout_p', { type: 'float' })}
      </div>
      <div class="model-card" style="margin-top:14px" id="model-card"></div>
    </section>` : `
    <section class="step-section">
      <h3>Checkpoint</h3>
      <p>Any Hugging Face model id or a local path. Gated models (Llama, Gemma) need <code>HF_TOKEN</code> in <code>.env</code>.</p>
      <div class="grid two">
        <div class="field span-all"><label for="f-model_name">Model</label>
          <input id="f-model_name" type="text" list="model-presets" data-bind="model_name" data-type="str" value="${esc(c.model_name)}" placeholder="org/model-name" autocomplete="off" spellcheck="false">
          <datalist id="model-presets">${presets.map((m) => `<option value="${esc(m.name)}">${fmt.params(m.params)}${m.five_d_ready ? ', 5D ready' : ''}</option>`).join('')}</datalist>
          <div class="model-presets" role="group" aria-label="Popular models">${presets.map((m) => `<button type="button" class="tpl-item${m.name === c.model_name ? ' on' : ''}" data-model="${esc(m.name)}" title="${esc(m.name)}"><span class="tpl-name">${esc(m.name.split('/').pop())}</span><span class="tpl-tags"><span>${fmt.params(m.params)}</span>${m.five_d_ready ? '<span>5D</span>' : ''}</span></button>`).join('')}</div>
        </div>
      </div>
      <div class="model-card" style="margin-top:14px" id="model-card"><div class="skeleton"></div></div>
    </section>`}
    <section class="step-section">
      <h3>What to train</h3>
      <p>Full fine-tuning updates every weight. LoRA trains small adapters on a frozen base; QLoRA also stores the base in 4 or 8 bits.</p>
      ${segmented('method', [
        ['full', 'Full fine-tune'],
        ['lora', 'LoRA', !PEFT_TYPES.has(c.model_type), 'Not available for this model type'],
        ['qlora', 'QLoRA', !QLORA_TYPES.has(c.model_type), 'Available for language and vision-language models'],
      ], method)}
      ${method !== 'full' ? `<div class="grid" style="margin-top:16px">
        ${field('Rank (r)', 'peft.r', { type: 'int', min: 1, hint: '8–64 typical' })}
        ${field('Alpha', 'peft.alpha', { type: 'int', min: 1, hint: 'Usually 2 × r' })}
        ${field('Dropout', 'peft.dropout', { type: 'float' })}
        ${field('Target modules', 'peft.target_modules', { hint: 'all-linear, or e.g. q_proj,v_proj' })}
        ${method === 'qlora' ? select('Base precision', 'quantization.bits', [['4', '4-bit NF4'], ['8', '8-bit LLM.int8()']]) : ''}
      </div>` : ''}
    </section>`;
}

let inspectTimer = null;
function inspectModelSoon() {
  clearTimeout(inspectTimer);
  inspectTimer = setTimeout(inspectModel, 450);
}
async function inspectModel() {
  const card = $('#model-card');
  const c = W.cfg;
  if (c.model_type === 'custom_transformer') {
    if (card) card.innerHTML = W.plan ? modelCard(W.plan.model) : '';
    return;
  }
  const name = c.model_name;
  if (!name) { if (card) card.innerHTML = '<span class="faint">Enter a model id.</span>'; return; }
  if (W.specFor !== name) {
    try { W.spec = await api(`/api/models/inspect?name=${encodeURIComponent(name)}&model_type=${encodeURIComponent(c.model_type)}`); W.specFor = name; }
    catch (e) { if (card) card.innerHTML = `<span class="faint">${esc(e.message)}</span>`; return; }
  }
  if (card && W.cfg.model_name === name) card.innerHTML = modelCard(W.spec);
}
function modelCard(m) {
  const kv = [
    ['Parameters', fmt.params(m.params)], ['Layers', m.layers], ['Hidden', fmt.int(m.hidden)],
    ['Heads / KV', `${m.heads} / ${m.kv_heads}`], ['Vocabulary', fmt.int(m.vocab)],
    ['Architecture', m.arch || 'unknown'], ['5D support', m.five_d_ready ? 'yes' : 'no'],
    ['Source', m.source === 'hf-config' ? 'config.json' : m.source === 'estimate' ? 'guessed from name' : m.source],
  ];
  if (m.num_experts) kv.splice(1, 0, ['Active / token', fmt.params(m.active_params)], ['Experts', `${m.num_experts}, top ${m.top_k}`]);
  return kv.map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${esc(v)}</div></div>`).join('');
}

// ---------------------------------------------------------------------------
// Step 3: data
// ---------------------------------------------------------------------------

function stepData() {
  const c = W.cfg;
  const goal = goalOf(c);
  const hint = goal === 'sft' ? DATA_HINT.sft : goal === 'preference' ? DATA_HINT.preference : goal === 'rl' ? DATA_HINT.rl : DATA_HINT[c.model_type];
  const info = S.catalog.algorithms.find((a) => a.id === c.algorithm.type);
  setTimeout(() => {
    const ex = $('#use-example');
    if (ex) ex.onclick = () => { c.dataset = { name: info.example_dataset.name, subset: info.example_dataset.subset || null, split: info.example_dataset.split }; onCfgChange(true); };
  });
  return `
    <section class="step-section">
      <h3>Dataset</h3>
      <p>Expected columns: ${esc(hint)}</p>
      <div class="grid">
        ${field('Dataset', 'dataset.name', { placeholder: 'org/dataset or ./data.jsonl', cls: 'span-all' })}
        ${field('Subset', 'dataset.subset', { placeholder: 'optional, e.g. wikitext-2-raw-v1' })}
        ${field('Split', 'dataset.split', { placeholder: 'train', hint: 'Slice for quick runs: train[:2000], train[:5%]' })}
      </div>
      <div id="data-check" class="data-check" aria-live="polite"></div>
      ${info && info.example_dataset && c.dataset.name !== info.example_dataset.name ? `<div class="btn-row" style="margin-top:12px"><button class="btn small" id="use-example">Use ${esc(info.example_dataset.name)}</button></div>` : ''}
    </section>
    ${goal === 'sft' ? `<section class="step-section"><h3>Loss mask</h3><p>Usually the model should learn to answer, not to repeat the prompt.</p>
      ${check('Train on the answer tokens only', 'algorithm.completion_only', 'Off = also learn the user and system turns')}</section>` : ''}
    ${goal === 'preference' ? `<section class="step-section"><h3>Lengths</h3><p>Prompts longer than the limit are cut from the left; chosen and rejected answers share the rest of the sequence.</p>
      <div class="grid">${field('Max prompt tokens', 'algorithm.max_prompt_length', { type: 'int', min: 1 })}${field('Max sequence tokens', 'training.max_length', { type: 'int', min: 16 })}</div></section>` : ''}
    ${goal === 'rl' ? `<section class="step-section"><h3>Prompts</h3><p>Prompts longer than the limit are cut from the left. The system prompt is added to every prompt; the default is Qwen's math prompt, whose <code>\\boxed{}</code> answer the accuracy reward reads. Asking a small model for &ldquo;only the number&rdquo; makes it skip its reasoning.</p>
      <div class="grid">${field('Max prompt tokens', 'algorithm.max_prompt_length', { type: 'int', min: 1 })}${field('Max answer tokens', 'algorithm.max_completion_length', { type: 'int', min: 1, hint: 'Generated per completion' })}</div>
      <div class="field" style="margin-top:14px"><label for="f-algorithm-system_prompt">System prompt</label>
        <textarea id="f-algorithm-system_prompt" rows="3" data-bind="algorithm.system_prompt" data-type="str" placeholder="optional" style="font-family:inherit">${esc(c.algorithm.system_prompt || '')}</textarea></div></section>` : ''}`;
}

/** Everything the measured lengths depend on (the server keeps its own copy of this rule). */
const dataKey = (c) => JSON.stringify([c.dataset, c.model_name, c.model_type, c.algorithm.type, c.algorithm.max_prompt_length, c.algorithm.system_prompt, c.training.max_length, c.training.batch_size]);
let dataTimer = null;
/** Measure the dataset once the user has reached the Data step: it may need the network. */
function inspectDataSoon(delay = 1000) {
  drawDataCheck();
  clearTimeout(dataTimer);
  dataTimer = setTimeout(inspectData, delay);
}
async function inspectData() {
  if (!W || W.dataFor === dataKey(W.cfg)) return;
  const wizard = W, key = dataKey(W.cfg);
  W.dataFor = key; W.data = null;
  drawDataCheck();
  let found;
  try { found = await api('/api/data/inspect', { json: { config: exportConfig(W.cfg) } }); }
  catch (e) { found = { data: null, error: e.message }; }
  if (W !== wizard || W.dataFor !== key) return;
  W.data = found;
  drawDataCheck();
  if (found.data) schedulePlan(0);
}
function drawDataCheck() {
  const el = $('#data-check');
  if (!el) return;
  if (!W.dataFor || W.dataFor !== dataKey(W.cfg) || !W.data) {
    el.innerHTML = '<p class="faint">Checking your data…</p>';
    return;
  }
  const d = W.data.data;
  if (W.data.error) { el.innerHTML = `<div class="notice warn"><div><strong>Could not check this dataset.</strong> ${esc(W.data.error)}</div></div>`; return; }
  if (!d) { el.innerHTML = ''; return; }
  const n = (v) => fmt.int(v);
  const kv = [['Examples', d.rows ? `${d.rows_exact ? '' : '~'}${n(d.rows)}` : 'unknown'], ['Average', `${n(Math.round(d.mean))} tokens`],
    ['95% are under', `${n(d.p95)} tokens`], [d.longest_exact ? 'Longest' : 'Longest seen', `${n(d.longest_seen)} tokens`],
    ['Longer than the limit', `${d.over_limit_pct}%`]];
  el.innerHTML = `<div class="model-card">${kv.map(([k, v]) => `<div><div class="k">${k}</div><div class="v">${esc(v)}</div></div>`).join('')}</div>
    <p class="faint">Measured with this model's tokenizer on ${n(d.sampled)} examples. The limit is ${n(d.limit)} tokens.</p>`;
}

// ---------------------------------------------------------------------------
// Step 4: compute
// ---------------------------------------------------------------------------

function uiField(label, path, value, { type = 'str', hint = '', placeholder = '', rerender = false } = {}) {
  const id = `f-${path.replace(/\./g, '-')}`;
  return `<div class="field"><label for="${id}">${label}</label><input id="${id}" type="${type === 'int' ? 'number' : 'text'}" ${type === 'int' ? 'min="1"' : ''} data-bind="${path}" data-type="${type}" value="${esc(value)}" placeholder="${esc(placeholder)}" ${rerender ? 'data-rerender' : ''}>${hint ? `<span class="hint">${hint}</span>` : ''}</div>`;
}
function uiSelect(label, path, options, value) {
  const id = `f-${path.replace(/\./g, '-')}`;
  return `<div class="field"><label for="${id}">${label}</label><select id="${id}" data-bind="${path}" data-type="str" data-rerender>${options.map(([v, t]) => `<option value="${esc(v)}" ${v === value ? 'selected' : ''}>${esc(t)}</option>`).join('')}</select></div>`;
}
async function loadSlurm() {
  const here = stillHere();
  try { S.slurm = await api('/api/slurm/status'); } catch (e) { S.slurm = { available: false, reason: e.message, partitions: [] }; }
  if (!here() || !W) return;
  if (S.slurm.partitions.length && !S.slurm.partitions.includes(W.ui.hw.partition)) { W.ui.hw.partition = S.slurm.partitions[0]; saveDraft(); }
  if (W.ui.target === 'slurm' && W.step >= 3) drawStep();
}
function launchLabel() {
  if (isLocal()) return 'Launch run';
  return W.ui.target === 'slurm' && S.slurm && S.slurm.available ? 'Submit to SLURM' : 'Copy launch command';
}

function stepCompute() {
  const c = W.cfg;
  const avail = strategyAvailability();
  const local = W.local;
  const target = W.ui.target, hw = W.ui.hw, slurm = S.slurm || { available: false, reason: 'Checking SLURM…', partitions: [] };
  const names = { auto: 'Automatic', solo: 'Single GPU', ddp: 'DDP', fsdp: 'FSDP', '5d': '5D parallel' };
  const desc = {
    auto: 'Looks at the model, the data and the GPUs, and uses the simplest strategy that fits and is fast enough.',
    solo: 'One process, one GPU.',
    ddp: 'A full copy on every GPU; gradients are averaged. Scales throughput.',
    fsdp: 'Weights, gradients and optimizer states sharded across GPUs. For models that do not fit on one.',
    '5d': 'Combine data, tensor, sequence, context, pipeline and expert parallelism for very large models and long sequences.',
  };
  setTimeout(() => {
    $$('.choice[data-strategy]').forEach((b) => { b.onclick = () => {
      c.strategy = b.dataset.strategy;
      if (c.strategy === 'auto' && isLocal()) c.num_gpus = 'auto';
      if (!['solo', 'auto'].includes(c.strategy) && c.num_gpus < 2 && isLocal()) c.num_gpus = Math.max(1, Math.min(2, local.gpus_per_node));
      onCfgChange(true);
    }; });
    $('[data-seg="target"]').onclick = (e) => {
      const b = e.target.closest('button');
      if (!b || b.dataset.val === W.ui.target) return;
      if (isLocal() && c.num_gpus > 1) Object.assign(W.ui.hw, hwFor(+c.num_gpus));
      W.ui.target = b.dataset.val;
      if (isLocal() && c.strategy === 'auto') c.num_gpus = 'auto';
      else if (isLocal()) c.num_gpus = Math.max(1, Math.min(c.num_gpus, local.gpus_per_node));
      onCfgChange(true);
    };
    if (target === 'slurm' && !S.slurm) loadSlurm();
  });
  const p = c.parallelism;
  const five = c.strategy === '5d' ? fiveDErrors(c) : null;
  const gpuOpts = S.catalog.gpus.map((g) => [g.id, `${g.name}`]);
  return `
    <section class="step-section">
      <h3>Where it runs</h3>
      ${segmented('target', [['local', 'This machine'], ['remote', 'Other machines'], ['slurm', 'SLURM cluster']], target)}
      <p style="margin-top:12px">${{
        local: `${local.detected ? `${local.gpus_per_node} × ${esc(local.gpu_name)}, ${fmt.gb(local.gpu_mem_gb)} each.` : 'No CUDA GPU was detected.'} The run joins this machine's queue and you follow it here.`,
        remote: 'Any GPU server, rented machine or group of machines, with no scheduler. You get the YAML and the exact commands; plain torchrun connects the machines.',
        slurm: 'For clusters that schedule jobs with SLURM. If yours does not, use Other machines.' }[target]}</p>
      ${target === 'local' && c.strategy !== 'solo' && +c.num_gpus > local.gpus_per_node ? `<div class="notice warn" style="margin-top:12px"><div>This run wants ${c.num_gpus} GPUs and this machine has ${local.gpus_per_node}. Lower the count, or pick Other machines.</div></div>` : ''}
      ${target !== 'local' ? `<div class="grid" style="margin-top:4px">
        ${uiSelect('GPU type', 'ui.hw.gpu', gpuOpts, hw.gpu)}
        ${c.strategy === 'solo' ? '<div class="field"><span class="label">GPUs</span><div class="fixed">1, single-GPU run</div></div>' : `
        ${uiField('Machines', 'ui.hw.nodes', hw.nodes, { type: 'int', rerender: true })}
        ${uiField('GPUs per machine', 'ui.hw.gpus_per_node', hw.gpus_per_node, { type: 'int', rerender: true })}`}
        ${target === 'remote' && c.strategy !== 'solo' && +hw.nodes > 1 ? `${uiField('First machine', 'ui.hw.host', hw.host, { placeholder: 'host name or IP', hint: 'All machines connect to it' })}${uiField('Port', 'ui.hw.port', hw.port, { type: 'int' })}` : ''}
        ${target === 'slurm' ? `${slurm.partitions.length ? uiSelect('Partition', 'ui.hw.partition', slurm.partitions.map((x) => [x, x]), hw.partition) : uiField('Partition', 'ui.hw.partition', hw.partition)}
        ${uiField('Time limit', 'ui.hw.time', hw.time, { placeholder: 'HH:MM:SS' })}${uiField('CPUs per machine', 'ui.hw.cpus', hw.cpus, { type: 'int' })}${uiField('Memory per machine', 'ui.hw.mem', hw.mem, { placeholder: '64G' })}` : ''}
      </div>` : ''}
      ${target === 'slurm' && !slurm.available ? `<div class="notice warn" style="margin-top:14px"><div><strong>Cannot submit from here.</strong> ${esc(slurm.reason)} You still get the command to run on a login node.</div></div>` : ''}
      ${target === 'remote' && ['fsdp', 'auto'].includes(c.strategy) && +hw.nodes > 1 ? '<div class="notice" style="margin-top:14px"><div>FSDP across machines needs the checkpoint folder on storage all machines share.</div></div>' : ''}
    </section>
    <section class="step-section">
      <h3>Strategy</h3>
      <p>How the work is spread over the GPUs. Leave it on Automatic unless you want a specific one; the plan on the right shows what it picks and why.</p>
      <div class="choices">${['auto', 'solo', 'ddp', 'fsdp', '5d'].map((s) => `
        <button type="button" class="choice" data-strategy="${s}" aria-pressed="${c.strategy === s}" ${avail[s] ? `disabled title="${esc(avail[s])}"` : ''}>
          <span class="choice-title">${names[s]}${s === 'auto' ? '<span class="pill fit">Recommended</span>' : ''}</span><span class="choice-text">${desc[s]}</span>
          ${avail[s] ? `<span class="choice-meta">${esc(avail[s])}</span>` : ''}
        </button>`).join('')}</div>
      <div class="grid" style="margin-top:16px">
        ${c.strategy !== 'solo' ? (target !== 'local'
          ? `<div class="field"><span class="label">GPUs</span><div class="fixed">${c.num_gpus}, ${hw.nodes} × ${hw.gpus_per_node}</div></div>`
          : c.strategy === 'auto' ? select('GPUs to use at most', 'num_gpus', [['auto', `all (${local.gpus_per_node})`], ...Array.from({ length: local.gpus_per_node }, (_, i) => [String(i + 1), String(i + 1)])], { rerender: true })
          : field('GPUs', 'num_gpus', { type: 'int', min: 1, hint: `Up to ${local.gpus_per_node} on this machine`, rerender: true })) : ''}
        ${check('Mixed precision (bf16)', 'dist_parameters.mixed_precision', 'fp32 weights, bf16 compute')}
      </div>
    </section>
    ${c.strategy === '5d' ? `
    <section class="step-section">
      <h3>5D layout</h3>
      <p>DP × TP × PP × CP must equal the GPU count. TP splits each layer (keep it inside a node), PP splits the layer stack, CP splits the sequence, EP spreads MoE experts over DP ranks.</p>
      <div class="grid">
        ${select('Data parallel (DP)', 'parallelism.dp', [['auto', `auto (${five.dp || '?'})`], ...[1, 2, 4, 8, 16, 32, 64].map((v) => [String(v), String(v)])], { rerender: true })}
        ${field('Tensor parallel (TP)', 'parallelism.tp', { type: 'int', min: 1, rerender: true })}
        ${field('Pipeline stages (PP)', 'parallelism.pp', { type: 'int', min: 1, rerender: true })}
        ${field('Context parallel (CP)', 'parallelism.cp', { type: 'int', min: 1, rerender: true })}
        ${field('Expert parallel (EP)', 'parallelism.ep', { type: 'int', min: 1, rerender: true })}
        ${field('Micro-batches', 'parallelism.micro_batches', { type: 'int', min: 1, hint: 'Global batch = batch × micro-batches × DP', rerender: true })}
        ${select('Pipeline schedule', 'parallelism.pp_schedule', [['1f1b', '1F1B (less memory)'], ['afab', 'All-forward-all-backward']])}
      </div>
      <div class="grid" style="margin-top:14px">
        ${check('Sequence parallel (SP)', 'parallelism.sp', 'Also split norms and dropout over TP ranks', { rerender: true })}
        ${check('ZeRO-1', 'parallelism.zero1', 'Shard AdamW states over DP ranks', { rerender: true })}
      </div>
      ${five.errs.length ? `<div class="notice error" style="margin-top:14px"><div>${five.errs.map(esc).join('<br>')}</div></div>` : ''}
      ${five.warns.length ? `<div class="notice warn" style="margin-top:14px"><div>${five.warns.map(esc).join('<br>')}</div></div>` : ''}
      ${!five.errs.length ? `<div style="margin-top:16px">${rankGrid({ ...p, dp: p.dp === 'auto' ? 'auto' : +p.dp }, +c.num_gpus, target !== 'local' ? +hw.gpus_per_node : Math.max(1, local.gpus_per_node))}</div>` : ''}
    </section>` : ''}`;
}

// ---------------------------------------------------------------------------
// Step 5: training
// ---------------------------------------------------------------------------

function stepTraining() {
  const c = W.cfg;
  const a = c.algorithm.type;
  const pref = ['dpo', 'simpo', 'orpo'].includes(a);
  const rl = isRL(a);
  const betaHint = { dpo: 'KL strength against the reference; 0.1 typical', simpo: 'Reward scale; 2.0 typical', orpo: 'Weight λ of the odds-ratio term; 0.1 typical' }[a];
  return `
    <section class="step-section">
      <h3>Optimisation</h3>
      <p>${pref ? `${ALGO_LABEL[a]} wants a lower learning rate than SFT: about 1e-5 to 1e-4 with LoRA, 5e-7 to 5e-6 for a full fine-tune.` : rl ? `${ALGO_LABEL[a]} likes small steps: about 1e-5 to 5e-5 with LoRA, around 1e-6 for a full fine-tune.` : 'AdamW with linear warmup, then the chosen schedule.'}</p>
      <div class="grid">
        ${field('Epochs', 'training.epochs', { type: 'int', min: 1 })}
        ${field(pref ? 'Pairs per GPU' : rl ? 'Prompts per GPU' : 'Batch per GPU', 'training.batch_size', { type: 'int', min: 1, hint: pref ? 'Each pair runs chosen + rejected' : rl ? (a === 'ppo' ? 'Prompts answered per rollout' : `Each prompt is answered ${c.algorithm.num_generations} times`) : '' })}
        ${c.model_type === 'custom_transformer' || pref || rl ? '' : field('Max sequence tokens', 'training.max_length', { type: 'int', min: 8 })}
        ${field('Learning rate', 'training.learning_rate', { type: 'float' })}
        ${select('Schedule', 'training.lr_scheduler', [['cosine', 'Cosine'], ['linear', 'Linear'], ['constant', 'Constant']])}
        ${field('Warmup steps', 'training.warmup_steps', { type: 'int', min: 0 })}
        ${field('Weight decay', 'training.weight_decay', { type: 'float' })}
        ${field('Gradient clip', 'training.grad_clip', { type: 'float' })}
      </div>
      <div style="margin-top:14px">${check('Gradient checkpointing', 'training.gradient_checkpointing', 'Recompute activations in the backward pass: much less memory, ~30% slower')}</div>
    </section>
    ${pref ? `<section class="step-section">
      <h3>${ALGO_LABEL[a]} settings</h3>
      <div class="grid">
        ${field('Beta', 'algorithm.beta', { type: 'float', hint: betaHint })}
        ${a === 'dpo' ? select('Loss', 'algorithm.dpo_loss', [['sigmoid', 'Sigmoid (standard DPO)'], ['ipo', 'IPO'], ['hinge', 'Hinge']], { rerender: true }) : ''}
        ${a === 'dpo' && c.algorithm.dpo_loss === 'sigmoid' ? field('Label smoothing', 'algorithm.label_smoothing', { type: 'float', hint: 'Assumed share of flipped labels, < 0.5' }) : ''}
        ${a === 'simpo' ? field('Target margin (γ)', 'algorithm.simpo_gamma', { type: 'float' }) : ''}
      </div></section>` : ''}
    ${rl ? rlSettings(c) : ''}
    <section class="step-section">
      <h3>Saving and tracking</h3>
      <div class="grid">
        ${field('Checkpoint folder', 'checkpoint_dir')}
        ${field('W&B project', 'wandb.wandb_project')}
        ${field('W&B entity', 'wandb.wandb_entity', { placeholder: 'your default entity' })}
      </div>
      <div class="grid" style="margin-top:14px">
        ${check('Save checkpoints', 'save')}
        ${check('Log to Weights & Biases', 'wandb.wandb_log_with_train', 'Needs WANDB_API_KEY in .env')}
        ${check('Resume from the latest checkpoint', 'save_load.resume')}
      </div>
    </section>`;
}

// ---------------------------------------------------------------------------
// Step 6: review & launch
// ---------------------------------------------------------------------------

function stepReview() {
  const c = W.cfg;
  const file = runFile();
  if (W.ui.target === 'slurm' && !S.slurm) loadSlurm();
  setTimeout(() => {
    $('#launch').onclick = launch;
    $('#dl-yaml').onclick = () => W.yaml && download(runFile().split('/').pop(), W.yaml);
    $('#edit-yaml').onclick = () => { W.editingYaml = !W.editingYaml; drawYaml(); };
  });
  return `
    <section class="step-section">
      <h3>Name</h3>
      <div class="grid two"><div class="field"><label for="f-ui-name">Run name</label>
        <input id="f-ui-name" type="text" data-bind="ui.name" data-type="str" value="${esc(W.ui.name)}" placeholder="${esc(defaultRunName())}" maxlength="120"></div></div>
    </section>
    <section class="step-section">
      <h3>Summary</h3>
      <div class="model-card">${[
        ['Goal', ALGO_LABEL[c.algorithm.type]], ['Model', c.model_type === 'custom_transformer' ? 'custom transformer' : c.model_name.split('/').pop()],
        ['Method', { full: 'Full fine-tune', lora: 'LoRA', qlora: `QLoRA ${c.quantization.bits}-bit` }[methodOf(c)]],
        ['Data', `${c.dataset.name} ${c.dataset.split}`], ['Strategy', c.strategy === 'auto' && W.plan && W.plan.advice.fits ? `Automatic: ${W.plan.recommended.label}` : strategyLabel(c)], ['Runs on', { local: 'This machine', remote: +W.ui.hw.nodes > 1 && c.strategy !== 'solo' ? `${W.ui.hw.nodes} machines` : 'Another machine', slurm: 'SLURM' }[W.ui.target]], ['GPUs', c.strategy === 'solo' ? 1 : c.strategy === 'auto' ? (W.plan && W.plan.advice.fits ? W.plan.recommended.gpus : 'automatic') : c.num_gpus],
        ['Batch', `${c.training.batch_size} × ${c.model_type === 'custom_transformer' ? c.custom_transformer_args.max_seq_len : c.training.max_length} tokens`],
        ['Learning rate', fmt.lr(c.training.learning_rate)],
      ].map(([k, v]) => `<div><div class="k">${k}</div><div class="v" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(v)}">${esc(v)}</div></div>`).join('')}</div>
    </section>
    <section class="step-section">
      <div id="validation"><div class="notice">Checking the config…</div></div>
    </section>
    <section class="step-section">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
        <h3>Training YAML</h3>
        <div class="btn-row"><button class="btn small" id="edit-yaml">Edit YAML</button><button class="btn small" id="dl-yaml">Download</button></div>
      </div>
      <div id="yaml-box"><div class="skeleton"></div></div>
    </section>
    <section class="step-section">
      <h3>${isLocal() ? 'Or run it from a terminal' : 'How to start it'}</h3>
      <p>Save the YAML above as <code id="run-file">${esc(file)}</code>, then run:</p>
      <div id="cmd-box"><div class="skeleton"></div></div>
    </section>`;
}

const REWARD_INFO = {
  accuracy: 'Final answer matches the reference (\\boxed{}, answer tags, #### or last number)',
  format: 'Exactly <think>…</think> <answer>…</answer> (for models trained on that format)',
  exact_match: 'Whole answer equals the reference',
  contains: 'Reference appears in the answer',
  length: 'Penalty for running into the token limit',
};

function rlSettings(c) {
  const g = c.algorithm;
  const ppo = g.type === 'ppo';
  const lora = c.peft.enabled;
  setTimeout(() => {
    $$('[data-reward]').forEach((el) => {
      el.oninput = el.onchange = () => {
        const name = el.dataset.reward;
        const on = $(`[data-reward-on="${name}"]`).checked;
        const w = Number($(`[data-reward="${name}"][type="text"]`).value);
        $(`[data-reward="${name}"][type="text"]`).disabled = !on;
        g.rewards = { ...(g.rewards || {}) };
        if (on && Number.isFinite(w) && w) g.rewards[name] = w; else delete g.rewards[name];
        onCfgChange(false);
      };
    });
  });
  const extra = Object.keys(g.rewards || {}).filter((n) => !REWARD_INFO[n]).map((n) => [n, 'Custom function']);
  const rows = [...Object.entries(REWARD_INFO), ...extra].map(([name, text]) => {
    const w = (g.rewards || {})[name];
    return `<div style="display:grid;grid-template-columns:minmax(0,1fr) 90px;gap:10px;align-items:center">
      <label class="check"><input type="checkbox" data-reward="${esc(name)}" data-reward-on="${esc(name)}" ${w ? 'checked' : ''}><span><code>${esc(name)}</code><small>${esc(text)}</small></span></label>
      <input type="text" inputmode="decimal" data-reward="${esc(name)}" value="${w || 1}" aria-label="${esc(name)} weight" ${w ? '' : 'disabled'}></div>`;
  }).join('');
  return `<section class="step-section">
      <h3>${ALGO_LABEL[g.type]} settings</h3>
      ${ppo ? `<p>Each batch of prompts is answered once (the rollout), then several gradient steps learn from the answers. The progress line shows <code>reward</code> and <code>v_err</code>, the value head's error.</p>
      <div class="grid">
        ${field('Answers per prompt', 'algorithm.num_generations', { type: 'int', min: 1, rerender: true })}
        ${field('Passes per rollout', 'algorithm.ppo_epochs', { type: 'int', min: 1, hint: '4 typical' })}
        ${field('Minibatches per pass', 'algorithm.num_minibatches', { type: 'int', min: 1, hint: `Must divide prompts × answers (${Number(c.training.batch_size) * Number(g.num_generations)})` })}
        ${field('Temperature', 'algorithm.temperature', { type: 'float', hint: '1.0 keeps exploration' })}
        ${field('KL beta', 'algorithm.beta', { type: 'float', hint: lora ? 'Penalty for leaving the start model; 0.05 typical, 0 = off' : 'Needs LoRA (the reference is the model without adapters); 0 for a full fine-tune' })}
        ${field('Clip ε', 'algorithm.epsilon', { type: 'float' })}
        ${field('GAE λ', 'algorithm.gae_lambda', { type: 'float', hint: '0.95 typical' })}
        ${field('Discount γ', 'algorithm.gamma', { type: 'float' })}
        ${field('Value loss weight', 'algorithm.vf_coef', { type: 'float' })}
        ${field('Value learning rate', 'algorithm.value_lr', { type: 'float', hint: "The value head's own optimizer" })}
        ${field('Value clip', 'algorithm.value_clip', { type: 'float', hint: '0 = off' })}
        ${field('Cut-answer penalty', 'algorithm.eos_penalty', { type: 'float', hint: 'Subtracted when an answer hits the token limit' })}
      </div>` : `<p>Each step samples a group of answers per prompt; answers that score above their group's average are reinforced. The progress line shows <code>reward</code>; the loss itself stays near 0.</p>
      <div class="grid">
        ${field('Answers per prompt (G)', 'algorithm.num_generations', { type: 'int', min: 2, hint: '4–16 typical; more = better signal, slower', rerender: true })}
        ${field('Temperature', 'algorithm.temperature', { type: 'float', hint: '1.0 keeps exploration' })}
        ${field('KL beta', 'algorithm.beta', { type: 'float', hint: lora ? 'Pull towards the start model; 0.04 typical, 0 = off' : 'Needs LoRA (the reference is the model without adapters); 0 for a full fine-tune' })}
        ${field('Clip ε (low)', 'algorithm.epsilon', { type: 'float' })}
        ${field('Clip ε (high)', 'algorithm.epsilon_high', { type: 'float', hint: '0.28 = DAPO clip-higher' })}
        ${select('Loss normalisation', 'algorithm.loss_type', [['dapo', 'DAPO (per token)'], ['grpo', 'GRPO (per answer)'], ['dr_grpo', 'Dr. GRPO (fixed length)']])}
      </div>
      <div style="margin-top:14px">${check('Scale advantages by the group std', 'algorithm.scale_rewards', 'Off = Dr. GRPO (no bias towards very easy or very hard prompts)')}</div>`}
      <h3 style="margin-top:20px">Rewards</h3>
      <p>Weighted sum of the selected functions and the reward model, if any. Accuracy needs a reference-answer column.</p>
      <div class="stack" style="gap:10px">${rows}</div>
      <div class="grid two" style="margin-top:14px">${field('Custom reward function', 'algorithm.custom_reward', { placeholder: 'my_pkg.rewards:is_polite', hint: 'Importable module:function(prompts, completions, answers, **info) -> list of floats' })}
        ${field('Reward model', 'algorithm.reward_model', { placeholder: 'Skywork/Skywork-Reward-V2-Qwen3-0.6B', hint: 'Optional Hugging Face reward model (one output). Kept frozen on every GPU; alone if no function is selected.' })}
        ${field('Reward model weight', 'algorithm.reward_model_weight', { type: 'float' })}</div>
    </section>`;
}

function defaultRunName() {
  const c = W.cfg;
  const model = c.model_type === 'custom_transformer' ? 'custom' : String(c.model_name).split('/').pop();
  return `${c.algorithm.type}-${model}`.toLowerCase().replace(/[^\w.-]+/g, '-');
}

const runFile = () => `configs/${(W.ui.name || defaultRunName()).replace(/[^\w.-]+/g, '_')}.yaml`;

async function loadCommands() {
  const box = $('#cmd-box');
  if (!box) return;
  const here = stillHere();
  try {
    const r = await api('/api/launch/commands', { json: { config: exportConfig(W.cfg), target: W.ui.target, hardware: W.ui.hw, file: runFile() } });
    if (!here() || !box.isConnected) return;
    W.commands = r.commands;
    box.innerHTML = r.commands.map((c, i) => `<div class="cmd"><div class="cmd-head"><span>${esc(c.title)}</span><button class="btn small" data-cmd="${i}">Copy</button></div><pre class="code">${esc(c.command)}</pre></div>`).join('')
      + (r.notes.length ? `<ul class="cmd-notes">${r.notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul>` : '');
    $$('[data-cmd]', box).forEach((b) => { b.onclick = () => copyText(r.commands[+b.dataset.cmd].command); });
  } catch (e) {
    if (!here() || !box.isConnected) return;
    W.commands = null;
    box.innerHTML = `<div class="notice error"><div>${esc(e.message)}</div></div>`;
  }
}

async function loadReview() {
  loadCommands();
  const cfg = exportConfig(W.cfg);
  const [val, ren] = await Promise.allSettled([
    api('/api/config/validate', { json: { config: cfg } }),
    api('/api/config/render', { json: { config: cfg } }),
  ]);
  if (W.step !== 5) return;
  W.validation = val.status === 'fulfilled' ? val.value : { valid: false, error: val.reason.message };
  W.yaml = ren.status === 'fulfilled' ? ren.value.yaml : '';
  const v = W.validation;
  const plan = W.plan;
  const fitWarn = plan && plan.selected && plan.selected.auto && !plan.advice.fits
    ? `<div class="notice error" style="margin-top:10px"><div><strong>Nothing fits on ${isLocal() ? 'this machine' : 'these GPUs'} as configured.</strong> Automatic will not start this run. The plan on the right lists what would work; to try anyway, choose a strategy yourself.</div></div>`
    : plan && plan.selected && !plan.selected.fits
    ? `<div class="notice warn" style="margin-top:10px"><div><strong>The planner estimates this runs out of memory</strong> (${fmt.gb(plan.selected.per_gpu_gb)} per GPU). You can still launch it.</div></div>` : '';
  $('#validation').innerHTML = (v.valid
    ? '<div class="notice ok"><div><strong>The config is valid.</strong> The trainer accepts every option.</div></div>'
    : `<div class="notice error"><div><strong>The trainer rejects this config.</strong> ${esc(v.error)}</div></div>`) + fitWarn;
  $('#launch').disabled = !v.valid;
  drawYaml();
}

function drawYaml() {
  const box = $('#yaml-box');
  if (!box) return;
  $('#edit-yaml').textContent = W.editingYaml ? 'Cancel editing' : 'Edit YAML';
  if (!W.editingYaml) { box.innerHTML = `<pre class="code">${esc(W.yaml || '')}</pre>`; return; }
  box.innerHTML = `<textarea id="yaml-edit" rows="24" spellcheck="false" aria-label="Training YAML">${esc(W.yaml || '')}</textarea>
    <div class="btn-row" style="margin-top:10px"><button class="btn primary small" id="yaml-apply">Apply YAML</button><span class="faint" style="font-size:.8rem">Replaces every choice in this draft.</span></div>`;
  $('#yaml-apply').onclick = async () => {
    try {
      const { config } = await api('/api/config/parse', { json: { yaml: $('#yaml-edit').value } });
      W.cfg = loadConfig(config);
      W.editingYaml = false;
      toast('YAML applied');
      onCfgChange(true);
    } catch (e) { toast(e.message, 'error'); }
  };
}

async function launch() {
  const btn = $('#launch');
  btn.disabled = true;
  const cfg = exportConfig(W.cfg);
  const name = W.ui.name || defaultRunName();
  try {
    if (!isLocal() && launchLabel() === 'Copy launch command') {
      if (!W.commands) throw new Error('Fix the hardware options first: see the message under "How to start it".');
      await copyText(W.commands[0].command);
    } else if (W.ui.target === 'slurm') {
      const r = await api('/api/slurm/submit', { json: { config: cfg, hardware: W.ui.hw, name } });
      dialog({ title: r.job_id ? `Submitted as SLURM job ${r.job_id}` : 'Submitted to SLURM',
        body: `<p class="muted">Follow it with <code>squeue</code> and the files in <code>logs/</code>.</p><pre class="code">${esc(r.output)}</pre>` });
    } else {
      const r = await api('/api/queue/submit', { json: { config: cfg, name } });
      toast((r.status === 'running' ? 'Run started' : `Run queued at position ${r.queue_position}`) + (r.chosen ? `: ${strategyLabel({ strategy: r.strategy })} on ${r.gpu_count} GPU${r.gpu_count > 1 ? 's' : ''}` : ''));
      store.del('omni-draft');
      location.hash = `#/run/${r.job_id}`;
    }
  } catch (e) {
    toast(e.message, 'error');
  } finally { if (btn.isConnected) btn.disabled = false; }
}

// ---------------------------------------------------------------------------
// Live plan panel
// ---------------------------------------------------------------------------

let planTimer = null;
function schedulePlan(delay = 450) {
  clearTimeout(planTimer);
  planTimer = setTimeout(refreshPlan, delay);
}

function wizardCluster() {
  if (!isLocal()) {
    const hw = W.ui.hw, g = S.catalog.gpus.find((x) => x.id === hw.gpu) || S.catalog.gpus[0];
    return { gpu_mem_gb: g.memory_gb, gpu_name: g.id, tflops: g.tflops, gpus_per_node: Math.max(1, +hw.gpus_per_node || 1), nodes: Math.max(1, +hw.nodes || 1) };
  }
  const cap = W.cfg.strategy === 'auto' && W.cfg.num_gpus !== 'auto' ? +W.cfg.num_gpus : W.local.gpus_per_node;
  return { gpu_mem_gb: W.local.gpu_mem_gb, gpu_name: W.local.gpu_name, gpus_per_node: Math.min(cap, W.local.gpus_per_node), nodes: 1 };
}

async function refreshPlan() {
  const panel = $('#plan-panel');
  if (!panel || !W) return;
  const seq = ++W.planSeq;
  if (!W.plan) panel.innerHTML = '<div class="panel-pad stack"><div class="skeleton"></div><div class="skeleton" style="width:70%"></div><div class="skeleton" style="height:170px"></div></div>';
  else panel.style.opacity = '.6';
  try {
    const plan = await api('/api/plan', { json: { config: exportConfig(W.cfg), cluster: wizardCluster() } });
    if (seq !== W.planSeq) return;
    W.plan = plan;
    panel.style.opacity = '';
    panel.innerHTML = '';
    panel.appendChild(renderPlan(plan, {
      onApply: (rec) => {
        applyOption(W.cfg, rec);
        toast(`Using ${rec.label}`);
        onCfgChange(true);
      },
      onSuggest: (w, o) => {
        applyWorkload(W.cfg, w);
        if (W.cfg.strategy !== 'auto' && o.strategy) applyOption(W.cfg, o);
        toast('Applied');
        onCfgChange(true);
      },
    }));
    if (W.step === 1 && W.cfg.model_type === 'custom_transformer') inspectModel();
  } catch (e) {
    if (seq !== W.planSeq) return;
    panel.style.opacity = '';
    panel.innerHTML = `<div class="panel-pad"><div class="notice error"><div><strong>No estimate yet.</strong> ${esc(e.message)}</div></div></div>`;
  }
}

// ===========================================================================
// Planner page
// ===========================================================================

const TIER = {
  solo: { cls: 'one', glyph: '●', name: '1 GPU' }, ddp: { cls: 'one', glyph: '●', name: 'DDP' },
  fsdp: { cls: 'fsdp', glyph: '◆', name: 'FSDP' }, '5d': { cls: 'multi', glyph: '▲', name: '3D/5D' },
};

/** An estimate, said like one: "~48 min", "~2.3 h", "~3 days". */
function roughTime(s) {
  if (s == null) return '';
  if (s < 90) return '~1 min';
  if (s < 3600) return `~${Math.round(s / 60)} min`;
  if (s < 48 * 3600) return `~${(s / 3600).toFixed(s < 36000 ? 1 : 0)} h`;
  return `~${Math.round(s / 86400)} days`;
}

/** Short name of a cell's layout: "1 GPU", "DDP", "FSDP", "TP2·CP2". */
function cellName(c) {
  if (!c.fits) return 'Too big';
  if (c.strategy !== '5d') return TIER[c.strategy].name;
  const p = c.parallelism || {};
  return ['tp', 'pp', 'cp', 'ep'].filter((k) => +p[k] > 1).map((k) => `${k.toUpperCase()}${p[k]}`).join('·') || `DP${p.dp}`;
}

async function renderPlanner(main) {
  const catalog = await loadCatalog();
  const here = stillHere();
  const local = catalog.local_cluster;
  const saved = store.get('omni-planner', null);
  const P = deepMerge({
    model: 'meta-llama/Llama-3.1-8B', method: 'full', algorithm: 'pretrain', batch: 1, seq: 4096,
    gpu: local.detected ? 'local' : 'h100', perMachine: 8, samples: 0, axis: 'method', pick: null,
  }, saved || {});
  if (P.gpu === 'local' && !local.detected) P.gpu = 'h100';
  const gpuOpts = [...(local.detected ? [['local', `this machine's ${local.gpu_name} (${fmt.gb(local.gpu_mem_gb)})`]] : []),
    ...catalog.gpus.map((g) => [g.id, g.name])];
  const opt = (list, v) => list.map(([k, t]) => `<option value="${esc(k)}" ${String(k) === String(v) ? 'selected' : ''}>${esc(t)}</option>`).join('');
  main.innerHTML = `
    <div class="page planner-page">
      <div class="page-head"><div><h1>Capacity planner</h1>
        <p>What does a model need before you set up a run? Each square is one setup: how many GPUs, and which training method or sequence length. Click one to see why.</p></div></div>
      <form class="panel query" id="planner-form" onsubmit="return false">
        <div class="query-line">
          <span>Train</span>
          <input id="p-model" class="q-wide" type="text" list="p-presets" value="${esc(P.model)}" spellcheck="false" autocomplete="off" aria-label="Model">
          <datalist id="p-presets">${catalog.models.map((m) => `<option value="${esc(m.name)}">${fmt.params(m.params)}</option>`).join('')}</datalist>
          <span>for</span>
          <select id="p-algorithm" aria-label="Goal">${opt(['pretrain', 'sft', 'dpo', 'simpo', 'orpo', 'grpo', 'ppo'].map((a) => [a, ALGO_LABEL[a]]), P.algorithm)}</select>
          <span>on</span>
          <select id="p-gpu" class="q-wide" aria-label="GPU type">${opt(gpuOpts, P.gpu)}</select>
          <span>GPUs.</span>
        </div>
        <div class="query-line">
          <span>Machines hold</span>
          <select id="p-per" aria-label="GPUs per machine">${opt([1, 2, 4, 8, 16].map((n) => [n, String(n)]), P.perMachine)}</select>
          <span>GPUs each. Batch of</span><input id="p-batch" class="q-num" type="number" min="1" value="${P.batch}" aria-label="Batch per GPU">
          <span class="q-seq">×</span><input id="p-seq" class="q-num q-seq" type="number" min="16" step="256" value="${P.seq}" aria-label="Sequence length"><span class="q-seq">tokens</span>
          <span class="q-method">, trained with</span><select id="p-method" class="q-method" aria-label="Training method">${opt([['full', 'full fine-tuning'], ['lora', 'LoRA'], ['qlora', 'QLoRA']], P.method)}</select>
          <span>, over</span><input id="p-samples" class="q-num" type="number" min="0" value="${P.samples || ''}" placeholder="?" aria-label="Examples in the dataset"><span>examples.</span>
        </div>
        <div class="query-foot">
          <span class="faint">Compare</span>
          ${segmented('axis', [['method', 'Training methods'], ['seq', 'Sequence lengths']], P.axis)}
          <span class="faint q-hint">Without a number of examples there are no times. More examples take longer, never more memory.</span>
        </div>
      </form>
      <div class="split map-split">
        <section class="panel map-panel" aria-live="polite"><div id="map" class="panel-pad"><div class="skeleton" style="height:320px"></div></div></section>
        <aside class="panel sticky" id="cell-detail" aria-live="polite"></aside>
      </div>
    </div>`;

  const axisSeg = $('[data-seg="axis"]');
  const showAxis = () => {
    $$('.q-seq').forEach((el) => { el.hidden = P.axis === 'seq'; });
    $$('.q-method').forEach((el) => { el.hidden = P.axis === 'method'; });
    $$('button', axisSeg).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.val === P.axis)));
  };
  showAxis();
  const read = () => {
    P.model = $('#p-model').value.trim(); P.algorithm = $('#p-algorithm').value; P.gpu = $('#p-gpu').value;
    P.perMachine = +$('#p-per').value; P.batch = Math.max(1, +$('#p-batch').value || 1);
    P.seq = Math.max(16, +$('#p-seq').value || 512); P.method = $('#p-method').value;
    P.samples = Math.max(0, Math.floor(+$('#p-samples').value || 0));
    store.set('omni-planner', P);
  };
  const toCfg = (change = {}) => {
    const c = defaultConfig();
    c.model_name = P.model; c.algorithm.type = P.algorithm;
    setMethod(c, change.method || P.method);
    c.training.batch_size = P.batch; c.training.max_length = change.seq_len || P.seq; c.training.gradient_checkpointing = true;
    if (isRL(P.algorithm)) {
      c.algorithm.max_prompt_length = Math.floor(c.training.max_length / 2);
      c.algorithm.max_completion_length = c.training.max_length - c.algorithm.max_prompt_length;
    }
    return c;
  };
  const hardware = () => {
    if (P.gpu === 'local') return { gpu_mem_gb: local.gpu_mem_gb, gpu_name: local.gpu_name, gpus_per_node: P.perMachine };
    const g = catalog.gpus.find((x) => x.id === P.gpu) || catalog.gpus[0];
    return { gpu_mem_gb: g.memory_gb, gpu_name: g.id, tflops: g.tflops, gpus_per_node: P.perMachine };
  };

  let grid = null, seq = 0;
  const drawMap = () => {
    const cols = grid.columns;
    // the fewest GPUs that work, per column
    const fewest = cols.map((_, j) => grid.cells.findIndex((row) => row[j] && row[j].fits));
    const head = `<tr><th scope="col"><span class="sr-only">GPUs</span></th>${cols.map((c) => `<th scope="col" title="${esc(c.title)}">${esc(c.label)}</th>`).join('')}</tr>`;
    const body = grid.rows.map((n, i) => `<tr><th scope="row">${n} GPU${n > 1 ? 's' : ''}${n > grid.per_machine ? `<small>${n / grid.per_machine} machines</small>` : ''}</th>${cols.map((col, j) => {
      const c = grid.cells[i][j];
      if (!c) return `<td><div class="cell na" title="${esc(col.why)}">–</div></td>`;
      const t = c.fits ? TIER[c.strategy] : { cls: 'no', glyph: '✕' };
      const on = P.pick && P.pick[0] === n && P.pick[1] === col.key;
      const changed = c.fits && Object.keys(c.changes || {}).length;
      const label = `${n} GPUs, ${col.title}: ${c.fits ? cellName(c) : 'does not fit'}, ${fmt.gb(c.per_gpu_gb)} per GPU${c.total_seconds ? `, ${roughTime(c.total_seconds)}` : ''}${fewest[j] === i ? ', the fewest GPUs that work' : ''}`;
      return `<td><button type="button" class="cell ${t.cls}${on ? ' on' : ''}${fewest[j] === i ? ' first' : ''}" data-cell="${i},${j}" aria-pressed="${on}" aria-label="${esc(label)}">
        <span class="cell-name"><i aria-hidden="true">${t.glyph}</i>${esc(cellName(c))}${changed ? '<sup title="With a smaller batch or gradient checkpointing">*</sup>' : ''}</span>
        <span class="cell-gb">${fmt.gb(c.per_gpu_gb)}</span>
        ${c.total_seconds && c.fits ? `<span class="cell-time">${roughTime(c.total_seconds)}</span>` : ''}</button></td>`;
    }).join('')}</tr>`).join('');
    const m = grid.model;
    $('#map').innerHTML = `
      <div class="map-head"><h2>${esc(m.name)}</h2><span class="faint">${fmt.params(m.params)} parameters${m.source === 'estimate' ? ', size guessed from the name' : ''} · ${fmt.gb(grid.gpu_mem_gb)} per GPU</span></div>
      <div class="map-scroll"><table class="cap-map">${head}${body}</table></div>
      <div class="legend map-legend">
        <span><b class="k one">●</b>1 GPU or DDP: the model fits on one GPU</span>
        <span><b class="k fsdp">◆</b>FSDP: the model is split across the GPUs</span>
        <span><b class="k multi">▲</b>3D/5D: layers or sequences are split too</span>
        <span><b class="k no">✕</b>Does not fit</span><span><i class="first-key"></i>The fewest GPUs that work</span><span><b>*</b>with a smaller batch or checkpointing</span>
      </div>`;
    $$('[data-cell]').forEach((b) => { b.onclick = () => {
      const [i, j] = b.dataset.cell.split(',').map(Number);
      P.pick = [grid.rows[i], grid.columns[j].key]; store.set('omni-planner', P);
      $$('[data-cell]').forEach((x) => { const on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-pressed', String(on)); });
      drawDetail();
    }; });
  };

  const drawDetail = () => {
    const box = $('#cell-detail');
    let i = P.pick ? grid.rows.indexOf(P.pick[0]) : -1;
    let j = P.pick ? grid.columns.findIndex((c) => c.key === P.pick[1]) : -1;
    if (i < 0 || j < 0 || !grid.cells[i][j]) {                // default: the fewest GPUs in the first column that has an answer
      j = grid.columns.findIndex((_, k) => grid.cells.some((row) => row[k] && row[k].fits));
      i = j < 0 ? 0 : grid.cells.findIndex((row) => row[j] && row[j].fits);
      j = Math.max(0, j);
      const btn = $(`[data-cell="${i},${j}"]`);
      if (btn) { btn.classList.add('on'); btn.setAttribute('aria-pressed', 'true'); }
    }
    const c = grid.cells[i][j], col = grid.columns[j], n = grid.rows[i];
    if (!c) { box.innerHTML = ''; return; }
    const a = c.advice;
    const others = a.others.map((o) => `<li><strong>${esc(o.title)}</strong><br><span class="muted">${esc(o.detail)}.</span></li>`).join('');
    box.innerHTML = `
      <div class="panel-pad detail">
        <p class="detail-kicker">${n} GPU${n > 1 ? 's' : ''} · ${esc(col.title)}${c.total_seconds && c.fits ? ` · ${roughTime(c.total_seconds)}` : ''}</p>
        <h3>${esc(a.title)}${a.tech ? ` <span class="tech">${esc(a.tech)}</span>` : ''}</h3>
        <p>${esc(a.detail)}</p>
        ${[a.why, ...a.changes].filter(Boolean).map((t) => `<p class="muted">${esc(t)}</p>`).join('')}
        <p class="muted small">${esc(a.need)}</p>
        ${others ? `<h4>${esc(a.others_title)}</h4><ul class="plan-list">${others}</ul>` : ''}
        ${a.fits ? `<h4>Memory on each GPU</h4>${memoryBars([{ label: cellName(c), memory_gb: c.memory_gb, fits: c.fits }], grid.gpu_mem_gb, grid.usable_gb, '')}
        <div class="btn-row" style="margin-top:18px"><button class="btn primary" id="p-start">Start a run with this</button></div>` : ''}
      </div>`;
    const start = $('#p-start');
    if (start) start.onclick = () => {
      const cfg = toCfg(col.change);
      if (P.algorithm !== 'pretrain') {
        const info = catalog.algorithms.find((x) => x.id === P.algorithm);
        if (info && info.example_dataset) cfg.dataset = { name: info.example_dataset.name, subset: info.example_dataset.subset || null, split: info.example_dataset.split };
      }
      cfg.strategy = 'auto';                    // decides again with the real data and GPUs, the same way
      cfg.num_gpus = n;
      if (c.changes) applyWorkload(cfg, c.changes);
      const onHere = P.gpu === 'local' && n <= local.gpus_per_node;
      const ui = onHere ? { target: 'local' } : { target: 'remote', hw: { nodes: Math.max(1, n / P.perMachine), gpus_per_node: Math.min(n, P.perMachine), ...(P.gpu === 'local' ? {} : { gpu: P.gpu }) } };
      store.set('omni-draft', { cfg, ui, step: 1 });
      location.hash = '#/new';
    };
  };

  const run = async () => {
    read();
    const my = ++seq;
    $('#map').style.opacity = '.6';
    try {
      const g = await api('/api/plan/grid', { json: { config: exportConfig(toCfg()), cluster: hardware(), axis: P.axis, num_samples: P.samples } });
      if (!here() || my !== seq) return;
      grid = g;
      $('#map').style.opacity = '';
      drawMap();
      drawDetail();
    } catch (e) {
      if (!here() || my !== seq) return;
      $('#map').style.opacity = '';
      $('#map').innerHTML = `<div class="notice error"><div>${esc(e.message)}</div></div>`;
      $('#cell-detail').innerHTML = '';
    }
  };
  axisSeg.onclick = (e) => { const b = e.target.closest('button'); if (!b || b.dataset.val === P.axis) return; P.axis = b.dataset.val; P.pick = null; showAxis(); run(); };
  let t = null;
  $('#planner-form').addEventListener('input', () => { clearTimeout(t); t = setTimeout(run, 450); });
  $('#planner-form').addEventListener('change', () => { clearTimeout(t); t = setTimeout(run, 50); });
  run();
}

// ===========================================================================
// Checkpoints
// ===========================================================================

async function renderCheckpoints(main) {
  let dir = store.get('omni-ckpt-dir', 'checkpoints');
  main.innerHTML = `
    <div class="page">
      <div class="page-head"><div><h1>Checkpoints</h1><p>Saved weights under a checkpoint folder, newest first. Resume a run by turning on “Resume from the latest checkpoint”.</p></div></div>
      <div class="panel">
        <div class="panel-head"><form id="ck-form" class="btn-row" style="flex:1">
          <label for="ck-dir" class="faint" style="font-size:.85rem">Folder</label>
          <input id="ck-dir" type="text" value="${esc(dir)}" style="max-width:340px">
          <button class="btn small" type="submit">Show</button></form><span class="faint" id="ck-meta"></span></div>
        <div id="ck-body"><div class="panel-pad"><div class="skeleton"></div></div></div>
      </div>
    </div>`;
  const draw = async () => {
    const body = $('#ck-body');
    try {
      const r = await api(`/api/checkpoints?dir=${encodeURIComponent(dir)}`);
      $('#ck-meta').textContent = `${r.count} checkpoint${r.count === 1 ? '' : 's'}`;
      if (!r.count) {
        body.innerHTML = `<div class="empty"><h2>No checkpoints in ${esc(r.base_dir_relative)}</h2><p>Runs with “Save checkpoints” on write here when an epoch ends.</p></div>`;
        return;
      }
      body.innerHTML = `<div class="table-wrap"><table><thead><tr><th>Path</th><th>Strategy</th><th>Tag</th><th class="right">Size</th><th>Saved</th><th></th></tr></thead><tbody>
        ${r.checkpoints.map((c) => `<tr><td><code>${esc(c.path)}</code></td><td>${esc(c.strategy)}</td><td>${esc(c.tag.replace(/^__/, '') || '–')}</td>
          <td class="right num">${fmt.bytes(c.size_bytes)}</td><td class="run-sub">${esc(new Date(c.mtime * 1000).toLocaleString())}</td>
          <td class="right"><button class="btn small ghost" data-copy="${esc(c.path)}">Copy path</button></td></tr>`).join('')}
        </tbody></table></div>`;
      $$('[data-copy]', body).forEach((b) => { b.onclick = () => copyText(b.dataset.copy); });
    } catch (e) { body.innerHTML = `<div class="panel-pad"><div class="notice error"><div>${esc(e.message)}</div></div></div>`; }
  };
  $('#ck-form').onsubmit = (e) => { e.preventDefault(); dir = $('#ck-dir').value.trim() || 'checkpoints'; store.set('omni-ckpt-dir', dir); draw(); };
  draw();
}

// ===========================================================================
// System
// ===========================================================================

async function renderSystem(main) {
  main.innerHTML = `
    <div class="page">
      <div class="page-head"><div><h1>System</h1><p>The GPUs this server schedules runs on, and the credentials training needs.</p></div></div>
      <div class="stack">
        <div class="panel"><div class="panel-head"><h2>GPUs</h2><span class="faint" id="sys-ver"></span></div><div id="sys-gpus"><div class="panel-pad"><div class="skeleton"></div></div></div></div>
        <div class="panel"><div class="panel-head"><h2>Credentials</h2></div><div class="panel-pad" id="sys-env"></div></div>
      </div>
    </div>`;
  const [gpus, q, env] = await Promise.allSettled([api('/api/system/gpus'), api('/api/queue/status'), api('/api/env/status')]);
  const g = gpus.status === 'fulfilled' ? gpus.value : { available: false, error: gpus.reason.message, gpus: [] };
  const alloc = q.status === 'fulfilled' ? q.value.gpu_details : [];
  $('#sys-ver').textContent = g.pytorch_version ? `PyTorch ${g.pytorch_version}${g.cuda_version ? `, CUDA ${g.cuda_version}` : ''}` : '';
  $('#sys-gpus').innerHTML = !g.available
    ? `<div class="panel-pad"><div class="notice warn"><div><strong>${esc(g.error || 'No CUDA GPU')}.</strong> Runs still start on CPU for small models and tests, one job at a time.</div></div></div>`
    : `<div class="table-wrap"><table><thead><tr><th>#</th><th>Device</th><th class="right">Memory in use</th><th>Compute</th><th>Queue</th></tr></thead><tbody>
      ${g.gpus.map((d) => { const a = alloc.find((x) => x.index === d.index) || {}; const frac = d.used_memory_gb / d.total_memory_gb; return `<tr>
        <td class="num">${d.index}</td><td><div class="run-name">${esc(d.name)}</div><div class="run-sub">${d.multi_processor_count} SMs</div></td>
        <td class="right" style="min-width:180px"><div class="bar"><span style="width:${(frac * 100).toFixed(1)}%;background:var(--${frac > 0.85 ? 'over' : frac > 0.5 ? 'tight' : 'fit'})"></span></div><div class="run-sub num">${d.used_memory_gb.toFixed(1)} / ${d.total_memory_gb.toFixed(1)} GB</div></td>
        <td class="num">sm_${esc(d.compute_capability.replace('.', ''))}</td>
        <td>${a.job_id ? `<a href="#/run/${esc(a.job_id)}">${statusPill('running')}</a>` : '<span class="pill fit">Free</span>'}</td></tr>`; }).join('')}
      </tbody></table></div>`;
  const e = env.status === 'fulfilled' ? env.value : { found: false, keys: {} };
  const row = (k, why) => `<li><span><code>${k}</code> <span class="muted">${why}</span></span>${e.keys[k] ? '<span class="pill fit">Set</span>' : '<span class="pill">Not set</span>'}</li>`;
  $('#sys-env').innerHTML = `<ul class="plan-list">${row('HF_TOKEN', 'gated models (Llama, Gemma) and rate-limited datasets')}${row('WANDB_API_KEY', 'Weights & Biases logging')}</ul>
    <p class="faint" style="font-size:.82rem;margin-top:12px">${e.found ? 'Read from .env in the project root.' : 'No .env file in the project root.'} Values are never sent to the browser.</p>`;
}

// ===========================================================================
// API
// ===========================================================================

function renderApi(main) {
  const o = location.origin;
  const snippet = (title, text, code) => `<div class="panel"><div class="panel-head"><h3>${title}</h3><button class="btn small" data-copy="${esc(code)}">Copy</button></div>
    <div class="panel-pad stack"><p class="muted">${text}</p><pre class="code">${esc(code)}</pre></div></div>`;
  main.innerHTML = `
    <div class="page">
      <div class="page-head"><div><h1>API</h1><p>Everything in this console is a REST call, so scripts, CI jobs and other services can plan and launch runs too. Requests must come through a host name the server allows (localhost, an IP, this machine's name or <code>OMNI_UI_ALLOWED_HOSTS</code>).</p></div>
        <a class="btn" href="/docs" target="_blank" rel="noopener">Open interactive docs</a></div>
      <div class="stack">
        ${snippet('Is distributed training needed?', 'POST a training config (config.yaml schema) and optionally the hardware. The response has the verdict, a recommended strategy or 5D layout, per-GPU memory for every option and the minimum GPU counts.',
          `curl -s ${o}/api/plan -H 'Content-Type: application/json' -d '{
  "config": {"model_name": "meta-llama/Llama-3.1-8B", "model_type": "llm", "strategy": "auto",
             "training": {"batch_size": 1, "max_length": 4096, "gradient_checkpointing": true}},
  "cluster": {"gpu_name": "h100", "gpu_mem_gb": 80, "gpus_per_node": 8, "nodes": 1}
}' | jq '{verdict, headline, recommended: .recommended.label, min_gpus}'`)}
        ${snippet('Launch a run', 'Submits to the queue. It starts when enough GPUs are free; the response has the job id.',
          `python - <<'EOF'
import requests, yaml
cfg = yaml.safe_load(open("configs/llm_sft_lora_single_gpu.yaml"))
r = requests.post("${o}/api/queue/submit", json={"config": cfg, "name": "sft-qwen-demo"})
print(r.json())
EOF`)}
        ${snippet('Follow a run', 'Status, the last log lines, the parsed loss curve and progress.',
          `curl -s ${o}/api/queue/jobs/<job_id>?tail=50 | jq '{status, progress: .metrics.progress, last: .logs[-1]}'`)}
        ${snippet('From the command line', 'The same planner runs without the server.',
          'python -m omni_train.planner --model Qwen/Qwen2.5-7B-Instruct --method lora --algorithm dpo --gpus 4 --gpu a100-80 --seq 2048')}
      </div>
    </div>`;
  $$('[data-copy]', main).forEach((b) => { b.onclick = () => copyText(b.dataset.copy); });
}

// ===========================================================================
// Boot
// ===========================================================================

async function refreshChip() {
  const dot = $('.cluster-state .dot');
  try {
    const [q, sys] = await Promise.all([api('/api/queue/status'), S.gpuName !== undefined ? null : api('/api/system/gpus').catch(() => null)]);
    S.queue = q;
    if (sys) {
      const list = sys.gpus || sys.devices || (Array.isArray(sys) ? sys : []);
      const first = list[0] || {};
      const name = first.name || first.gpu_name;
      const mem = first.memory_total_gb || first.total_memory_gb || first.memory_gb;
      S.gpuName = name ? `${q.total_gpus} × ${String(name).replace(/^NVIDIA\s+|GeForce\s+|\s+GPU$/g, '')}${mem ? ` ${Math.round(mem)} GB` : ''}` : '';
    }
    const busy = q.total_gpus - q.available_gpus;
    dot.className = `dot ${q.available_gpus ? 'ok' : 'busy'}`;
    $('#cluster-state-text').textContent = q.total_gpus ? (q.available_gpus ? 'Online' : 'Fully booked') : 'No GPUs';
    $('#cluster-name').textContent = S.gpuName || (q.total_gpus ? `${q.total_gpus} GPU${q.total_gpus === 1 ? '' : 's'}` : 'CPU only');
    $('#cluster-meter').innerHTML = Array.from({ length: Math.min(q.total_gpus, 16) }, (_, i) => `<span class="${i < busy ? 'busy' : ''}"></span>`).join('');
    $('#cluster-chip-text').textContent = q.total_gpus
      ? `${busy} in use · ${q.available_gpus} free${q.pending_jobs ? ` · ${q.pending_jobs} queued` : ''}`
      : 'Jobs run on CPU';
    const badge = $('#nav-running');
    badge.textContent = q.running_jobs || '';
    badge.hidden = !q.running_jobs;
  } catch (e) {
    dot.className = 'dot down';
    $('#cluster-state-text').textContent = 'Offline';
    $('#cluster-chip-text').textContent = 'Server unreachable';
  }
}

function initTheme() {
  $('#theme-btn').onclick = () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('omni-theme', root.dataset.theme); } catch (e) { /* ignore */ }
  };
}

window.addEventListener('hashchange', route);
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  refreshChip();
  setInterval(refreshChip, 5000);
  route();
});
