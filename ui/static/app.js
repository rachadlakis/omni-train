// =========================================================================
// OMNI-Train Application JavaScript
// =========================================================================

// =========================================================================
// State
// =========================================================================
let selectedModelType = null;
let currentConfig = null;
let pollInterval = null;
let startedFromYaml = false;
let allTemplates = [];
let sideTemplatesFilter = 'all';
let yamlFromForm = false;
let pendingTemplateSelection = null;
let yamlCodeMirror = null;

// Training time estimation state
let trainingStartTime = null;
let estimatedTotalSeconds = null;
let timerInterval = null;
let lastEtaFromLogs = null;

// Queue state
let currentJobId = null;
let queuePollInterval = null;
let isJobQueued = false;

// =========================================================================
// Training Time Estimation
// =========================================================================

/**
 * Fetch training time estimate from backend API
 * Returns estimated time in seconds
 */
async function fetchTimeEstimate(config) {
  try {
    const res = await fetch('/api/estimate-time', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config }),
    });
    if (res.ok) {
      const data = await res.json();
      return data;
    }
  } catch (e) {
    console.error('Failed to fetch time estimate:', e);
  }
  // Fallback to local estimate
  return { est_total_seconds: estimateTrainingTimeLocal(config), readable: null };
}

/**
 * Local fallback estimate if API fails
 */
function estimateTrainingTimeLocal(config) {
  const modelType = config.model?.type || 'llm';
  const epochs = config.training?.epochs || 3;
  const batchSize = config.training?.batch_size || 8;

  const baseTime = { cnn: 0.05, llm: 2.5, vlm: 3.0, detection: 0.3, embedding: 0.8 };
  const datasetSize = { cnn: 50000, llm: 10000, vlm: 5000, detection: 8000, embedding: 20000 };

  const timePerStep = baseTime[modelType] || 1.0;
  const samples = datasetSize[modelType] || 10000;
  const totalSteps = Math.ceil(samples / batchSize) * epochs;

  return Math.round(totalSteps * timePerStep + 30);
}

/**
 * Format seconds into HH:MM:SS string
 */
function formatTime(seconds) {
  if (seconds <= 0 || isNaN(seconds)) return '--:--:--';

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Parse ETA from training logs
 * Looks for patterns like "ETA: 01:23:45" or "ETA: 12:34"
 */
function parseEtaFromLogs(logs) {
  if (!logs || logs.length === 0) return null;

  // Search from the end of logs for the most recent ETA
  for (let i = logs.length - 1; i >= Math.max(0, logs.length - 50); i--) {
    const line = logs[i];

    // Match ETA patterns: "ETA: HH:MM:SS" or "ETA: MM:SS"
    const etaMatch = line.match(/ETA:\s*(\d{1,2}):(\d{2})(?::(\d{2}))?/i);
    if (etaMatch) {
      let hours = 0, minutes = 0, seconds = 0;

      if (etaMatch[3]) {
        // HH:MM:SS format
        hours = parseInt(etaMatch[1]);
        minutes = parseInt(etaMatch[2]);
        seconds = parseInt(etaMatch[3]);
      } else {
        // MM:SS format
        minutes = parseInt(etaMatch[1]);
        seconds = parseInt(etaMatch[2]);
      }

      return hours * 3600 + minutes * 60 + seconds;
    }
  }

  return null;
}

/**
 * Start the countdown timer
 */
async function startTrainingTimer(config) {
  trainingStartTime = Date.now();
  lastEtaFromLogs = null;

  // Show initial loading state
  updateTimerDisplay(0, 'Calculating estimate...');

  // Fetch estimate from backend
  const estimate = await fetchTimeEstimate(config);
  estimatedTotalSeconds = estimate.est_total_seconds || 0;
  const epochs = Math.max(1, parseInt(estimate.epochs || config?.training?.epochs || 1, 10) || 1);
  const perEpochSeconds = estimatedTotalSeconds > 0 ? (estimatedTotalSeconds / epochs) : 0;
  const totalText = estimatedTotalSeconds > 0 ? formatTime(estimatedTotalSeconds) : (estimate.readable || '--:--:--');
  const perEpochText = perEpochSeconds > 0 ? formatTime(perEpochSeconds) : '--:--:--';

  const vramText = estimate.vram?.total_gb
    ? ` | VRAM~${estimate.vram.total_gb} GB/GPU`
    : '';

  const infoText = estimatedTotalSeconds > 0
    ? `Estimated total: ${totalText} | per epoch: ${perEpochText} (${epochs} epochs, ${estimate.total_steps || '?'} steps)${vramText}`
    : 'Initial estimate based on configuration';

  updateTimerDisplay(estimatedTotalSeconds, infoText);

  // Clear any existing interval
  if (timerInterval) {
    clearInterval(timerInterval);
  }

  // Update timer every second
  timerInterval = setInterval(updateTimerCountdown, 1000);
}

/**
 * Update the timer countdown
 */
function updateTimerCountdown() {
  if (!trainingStartTime || !estimatedTotalSeconds) return;

  const elapsed = Math.floor((Date.now() - trainingStartTime) / 1000);
  let remaining;

  if (lastEtaFromLogs !== null) {
    // Use ETA from logs if available (more accurate)
    remaining = lastEtaFromLogs;
  } else {
    // Use initial estimate minus elapsed time
    remaining = Math.max(0, estimatedTotalSeconds - elapsed);
  }

  updateTimerDisplay(remaining);
}

/**
 * Update the timer display elements
 */
function updateTimerDisplay(seconds, info = null) {
  const clockEl = document.getElementById('timer-clock');
  const infoEl = document.getElementById('timer-info');

  if (clockEl) {
    clockEl.textContent = formatTime(seconds);
  }

  if (infoEl && info) {
    infoEl.textContent = info;
    infoEl.classList.remove('updated');
  }
}

/**
 * Update timer based on log parsing
 */
function updateTimerFromLogs(logs) {
  const etaSeconds = parseEtaFromLogs(logs);

  if (etaSeconds !== null && etaSeconds !== lastEtaFromLogs) {
    lastEtaFromLogs = etaSeconds;

    const infoEl = document.getElementById('timer-info');
    if (infoEl) {
      infoEl.textContent = 'Updated from training progress';
      infoEl.classList.add('updated');
    }

    updateTimerDisplay(etaSeconds);
  }
}

/**
 * Stop the training timer
 */
function stopTrainingTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  trainingStartTime = null;
  estimatedTotalSeconds = null;
  lastEtaFromLogs = null;

  const clockEl = document.getElementById('timer-clock');
  const infoEl = document.getElementById('timer-info');

  if (clockEl) clockEl.textContent = '--:--:--';
  if (infoEl) {
    infoEl.textContent = '';
    infoEl.classList.remove('updated');
  }
}

// =========================================================================
// State Persistence (localStorage)
// =========================================================================
function saveFormState() {
  const formData = {};
  const inputs = document.querySelectorAll('input, select');
  inputs.forEach(el => {
    if (el.id) {
      if (el.type === 'checkbox') {
        formData[el.id] = el.checked;
      } else {
        formData[el.id] = el.value;
      }
    }
  });
  localStorage.setItem('omni_form_state', JSON.stringify(formData));
}

function loadFormState() {
  const data = localStorage.getItem('omni_form_state');
  if (!data) return;
  try {
    const formData = JSON.parse(data);
    Object.keys(formData).forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        if (el.type === 'checkbox') {
          el.checked = formData[id];
        } else {
          el.value = formData[id];
        }
      }
    });
  } catch (e) {
    console.error('Failed to load form state:', e);
  }
}

function clearFormState() {
  localStorage.removeItem('omni_form_state');
}

// =========================================================================
// Theme Toggle
// =========================================================================
function toggleTheme() {
  document.body.classList.toggle('light');
  const isLight = document.body.classList.contains('light');
  localStorage.setItem('theme', isLight ? 'light' : 'dark');

  // YAML editor always follows the global toggle — no separate user preference.
  _syncEditorTheme();
}

function loadTheme() {
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'light') {
    document.body.classList.add('light');
  }
}

function _syncEditorTheme() {
  const theme = document.body.classList.contains('light') ? 'eclipse' : 'material';
  if (yamlCodeMirror) {
    yamlCodeMirror.setOption('theme', theme);
    yamlCodeMirror.refresh();
  }
}

function getYamlEditorValue() {
  if (yamlCodeMirror) {
    return yamlCodeMirror.getValue();
  }

  const editor = document.getElementById('yaml-editor');
  return editor ? editor.value : '';
}

function setYamlEditorValue(value) {
  const content = value || '';
  const editor = document.getElementById('yaml-editor');
  if (editor) {
    editor.value = content;
  }

  if (yamlCodeMirror) {
    yamlCodeMirror.setValue(content);
  }
}

function setupYamlCodeEditor() {
  const editor = document.getElementById('yaml-editor');
  if (!editor || typeof window.CodeMirror === 'undefined' || yamlCodeMirror) return;

  // Theme always derived from the global dark/light mode — no separate preference stored.
  const initialTheme = document.body.classList.contains('light') ? 'eclipse' : 'material';

  yamlCodeMirror = window.CodeMirror.fromTextArea(editor, {
    mode: 'yaml',
    lineNumbers: true,
    lineWrapping: true,
    tabSize: 2,
    indentUnit: 2,
    theme: initialTheme,
    extraKeys: {
      Tab: function(cm) {
        cm.replaceSelection('  ', 'end');
      }
    }
  });

  let _yamlLintTimer = null;
  yamlCodeMirror.on('change', function(cm) {
    editor.value = cm.getValue();

    // Debounced real-time syntax check
    clearTimeout(_yamlLintTimer);
    _yamlLintTimer = setTimeout(() => {
      const val = cm.getValue().trim();
      const statusEl = document.getElementById('yaml-lint-status');
      if (!statusEl) return;
      if (!val) { statusEl.textContent = ''; return; }
      try {
        if (typeof window.jsyaml !== 'undefined') {
          window.jsyaml.load(val);
        }
        statusEl.textContent = '✓ Valid YAML';
        statusEl.style.color = 'var(--green)';
      } catch (e) {
        const loc = e.mark ? ` (line ${e.mark.line + 1})` : '';
        statusEl.textContent = `✗ ${e.reason || 'Syntax error'}${loc}`;
        statusEl.style.color = 'var(--red)';
      }
    }, 400);
  });
}

// =========================================================================
// Message Modal
// =========================================================================
function showMessage(title, text, type = 'info') {
  const icons = {
    info: 'info',
    success: 'check',
    error: 'error',
    warning: 'warning'
  };
  const iconEl = document.getElementById('message-modal-icon');
  const titleEl = document.getElementById('message-modal-title');
  const textEl = document.getElementById('message-modal-text');
  const modal = document.getElementById('message-modal');

  if (iconEl) iconEl.textContent = icons[type] || icons.info;
  if (titleEl) titleEl.textContent = title;
  if (textEl) {
    textEl.textContent = text;
    textEl.classList.remove('validation-summary');
  }
  if (modal) modal.classList.add('active');
}

function closeMessageModal() {
  const modal = document.getElementById('message-modal');
  if (modal) modal.classList.remove('active');
}

// =========================================================================
// Navigation
// =========================================================================
function navigateTo(page) {
  saveFormState();
  // Pages are served from /static/ directory
  window.location.href = '/static/' + page;
}

function goBack() {
  // Go back to root which serves index.html
  window.location.href = '/';
}

function goBackFromYaml() {
  const fromForm = localStorage.getItem('yaml_from_form') === 'true';
  if (fromForm) {
    navigateTo('config.html');
  } else {
    window.location.href = '/';
  }
}

function goBackFromEnv() {
  navigateTo('config.html');
}

function proceedToYaml() {
  localStorage.setItem('yaml_from_form', 'true');
  saveFormState();
  navigateTo('yaml.html');
}

// =========================================================================
// Template Modal
// =========================================================================
function openTemplateModal() {
  loadTemplatesForModal();
  const modal = document.getElementById('template-modal');
  if (modal) modal.classList.add('active');
}

function closeTemplateModal(event) {
  if (event && event.target !== event.currentTarget) return;
  const modal = document.getElementById('template-modal');
  if (modal) modal.classList.remove('active');
}

// Template name mappings
const templateDisplayNames = {
  // LLM
  'llm_full_finetune_ddp': 'LLM Full Fine Tuning',
  'llm_full_finetune_fsdp': 'LLM Full Fine Tuning',
  'llm_full_quantized_single_gpu': 'LLM Fine Tune with Quantization',
  'llm_lora_ddp': 'LLM Fine Tune with LoRA',
  'llm_lora_local_file': 'LLM Fine Tune with LoRA',
  'llm_lora_quantized_single_gpu': 'LLM Fine Tune with QLoRA',
  'llm_hybrid_2d_dp_tp': 'LLM Hybrid 2D (DP+TP)',
  'llm_fsdp_mini_project_style': 'LLM FSDP Example Small Model',
  // VLM
  'vlm_llava_lora_single_gpu': 'VLM Fine Tune with LoRA',
  // Vision (CNN)
  'cnn_resnet_single_gpu': 'Vision Full Training (Single GPU)',
  'cnn_vit_ddp': 'Vision Full Training (DDP)',
  // Detection
  'detection_coco_format': 'Detection Full Training (COCO)',
  'detection_yolo_single_gpu': 'Detection Full Training (YOLO)',
  // Embedding
  'embedding_text_infonce': 'Embedding Text Training',
  'embedding_bert_lora_triplet': 'Embedding BERT with LoRA',
  'embedding_vision_resnet': 'Embedding Vision Training',
  'embedding_clip_finetune': 'Embedding CLIP Fine-Tuning',
};

// Additional templates
const extraTemplates = [
  // Top Open Source LLMs (2026)
  { name: 'llm_glm5', display: 'GLM-5 (THUDM)', type: 'llm', desc: 'Reasoning & document analysis' },
  { name: 'llm_kimi_k25', display: 'Kimi K2.5 (Moonshot)', type: 'llm', desc: 'Top-tier reasoning & coding' },
  { name: 'llm_deepseek_v3', display: 'DeepSeek-V3', type: 'llm', desc: '671B MoE, exceptional coding' },
  { name: 'llm_deepseek_r1', display: 'DeepSeek-R1', type: 'llm', desc: 'Specialized reasoning' },
  { name: 'llm_qwen3_235b', display: 'Qwen3-235B (Alibaba)', type: 'llm', desc: '1M+ token context' },
  { name: 'llm_llama33_70b', display: 'Llama 3.3 70B (Meta)', type: 'llm', desc: 'Versatile, 128k context' },
  { name: 'llm_mimo_v2_flash', display: 'MiMo-V2-Flash (Xiaomi)', type: 'llm', desc: 'Ultra-fast, 256K context' },
  // Other LLMs
  { name: 'llm_mistral_7b_lora', display: 'Mistral 7B', type: 'llm' },
  { name: 'llm_phi3_mini_lora', display: 'Phi-3 Mini', type: 'llm' },
  // Top Open Source VLMs (2026)
  { name: 'vlm_full_finetune', display: 'VLM Full Fine Tuning', type: 'vlm', desc: 'Full parameter fine-tuning' },
  { name: 'vlm_qlora_general', display: 'VLM Fine Tune with QLoRA', type: 'vlm', desc: 'Quantized LoRA fine-tuning' },
  { name: 'vlm_qwen2_vl_72b', display: 'Qwen2-VL 72B (Alibaba)', type: 'vlm', desc: 'State-of-the-art vision-language' },
  { name: 'vlm_internvl2_26b', display: 'InternVL2 26B', type: 'vlm', desc: 'Strong multimodal reasoning' },
  { name: 'vlm_llava_next_34b', display: 'LLaVA-NeXT 34B', type: 'vlm', desc: 'Enhanced visual instruction' },
  { name: 'vlm_cogvlm2', display: 'CogVLM2 (THUDM)', type: 'vlm', desc: 'Visual understanding & grounding' },
  { name: 'vlm_phi3_vision', display: 'Phi-3 Vision (Microsoft)', type: 'vlm', desc: 'Efficient multimodal, 128k ctx' },
  { name: 'vlm_idefics2_8b', display: 'Idefics2 8B (HuggingFace)', type: 'vlm', desc: 'Open multimodal assistant' },
  { name: 'vlm_paligemma_3b', display: 'PaliGemma 3B (Google)', type: 'vlm', desc: 'Lightweight vision-language' },
  // Other VLMs
  { name: 'vlm_llava_7b_lora', display: 'LLaVA 1.5 7B', type: 'vlm' },
  { name: 'vlm_blip2_flan_lora', display: 'BLIP-2 Flan-T5', type: 'vlm' },
  // Small/test LLMs
  { name: 'llm_opt_125m', display: 'OPT-125M (Facebook)', type: 'llm', desc: 'Tiny model, fast for testing' },
  { name: 'llm_opt_350m', display: 'OPT-350M (Facebook)', type: 'llm', desc: 'Small model for quick runs' },
  { name: 'llm_gpt2', display: 'GPT-2 (OpenAI)', type: 'llm', desc: 'Classic small language model' },
  { name: 'llm_gpt2_medium', display: 'GPT-2 Medium (OpenAI)', type: 'llm', desc: '345M params' },
  // Vision (CNN)
  // ResNet
  { name: 'cnn_resnet18', display: 'ResNet-18', type: 'cnn', desc: 'Lightweight, fast baseline' },
  { name: 'cnn_resnet50', display: 'ResNet-50', type: 'cnn', desc: 'Classic strong baseline' },
  { name: 'cnn_resnet101', display: 'ResNet-101', type: 'cnn', desc: 'Deeper ResNet variant' },
  // ViT
  { name: 'cnn_vit_base', display: 'ViT Base (16×16)', type: 'vision', desc: 'Vision Transformer base' },
  { name: 'cnn_vit_large', display: 'ViT Large (16×16)', type: 'vision', desc: 'Vision Transformer large' },
  { name: 'cnn_vit_huge', display: 'ViT Huge (14×14)', type: 'vision', desc: 'Vision Transformer huge, ImageNet-21k' },
  // Swin Transformer
  { name: 'cnn_swin_tiny', display: 'Swin Transformer Tiny', type: 'vision', desc: 'Efficient hierarchical ViT' },
  { name: 'cnn_swin_base', display: 'Swin Transformer Base', type: 'vision', desc: 'Balanced accuracy/speed' },
  { name: 'cnn_swin_large', display: 'Swin Transformer Large', type: 'vision', desc: 'High-accuracy hierarchical ViT' },
  // DeiT
  { name: 'cnn_deit_tiny', display: 'DeiT Tiny', type: 'vision', desc: 'Data-efficient ViT, tiny' },
  { name: 'cnn_deit_small', display: 'DeiT Small', type: 'vision', desc: 'Data-efficient ViT, small' },
  { name: 'cnn_deit_base', display: 'DeiT Base', type: 'vision', desc: 'Data-efficient ViT, base' },
  // EfficientNet
  { name: 'cnn_efficientnet_b0', display: 'EfficientNet B0', type: 'cnn', desc: 'Compact & efficient' },
  { name: 'cnn_efficientnet_b4', display: 'EfficientNet B4', type: 'vision', desc: 'Balanced accuracy/size' },
  { name: 'cnn_efficientnet_b7', display: 'EfficientNet B7', type: 'vision', desc: 'Highest accuracy in family' },
  // ConvNeXT
  { name: 'cnn_convnext_tiny', display: 'ConvNeXT Tiny', type: 'vision', desc: 'Modern CNN with ViT-like design' },
  { name: 'cnn_convnext_base', display: 'ConvNeXT Base', type: 'vision', desc: 'Strong modern CNN baseline' },
  { name: 'cnn_convnext_large', display: 'ConvNeXT Large', type: 'vision', desc: 'High-capacity modern CNN' },
  // BEiT
  { name: 'cnn_beit_base', display: 'BEiT Base', type: 'vision', desc: 'BERT-style pre-trained ViT' },
  { name: 'cnn_beit_large', display: 'BEiT Large', type: 'vision', desc: 'Large BERT-style ViT' },
  // Detection — YOLOS (MIT/Apache 2.0, HuggingFace-native)
  { name: 'detection_yolos_tiny',        display: 'YOLOS Tiny',         type: 'detection', desc: 'Fastest YOLO via ViT (MIT)' },
  { name: 'detection_yolos_small',       display: 'YOLOS Small',        type: 'detection', desc: 'Balanced YOLO via ViT (MIT)' },
  { name: 'detection_yolos_base',        display: 'YOLOS Base',         type: 'detection', desc: 'Strongest YOLO via ViT (MIT)' },
  // Detection — RT-DETR (Apache 2.0)
  { name: 'detection_rtdetr_r18',        display: 'RT-DETR R18',        type: 'detection', desc: 'Real-time DETR, lightest (Apache 2.0)' },
  { name: 'detection_rtdetr_r50',        display: 'RT-DETR R50',        type: 'detection', desc: 'Real-time DETR, balanced (Apache 2.0)' },
  { name: 'detection_rtdetr_r101',       display: 'RT-DETR R101',       type: 'detection', desc: 'Real-time DETR, strongest (Apache 2.0)' },
  // Detection — DETR family (Apache 2.0)
  { name: 'detection_detr_r50',          display: 'DETR R50',           type: 'detection', desc: 'Original DETR (Apache 2.0)' },
  { name: 'detection_detr_r101',         display: 'DETR R101',          type: 'detection', desc: 'DETR with ResNet-101 backbone (Apache 2.0)' },
  { name: 'detection_conditional_detr',  display: 'Conditional DETR',   type: 'detection', desc: '10× faster convergence than DETR (Apache 2.0)' },
  { name: 'detection_deformable_detr',   display: 'Deformable DETR',    type: 'detection', desc: 'Sparse attention, strong real-world perf (Apache 2.0)' },
  { name: 'detection_dab_detr',          display: 'DAB-DETR R50',       type: 'detection', desc: 'Dynamic anchor boxes DETR (Apache 2.0)' },
  // Detection — OWL-ViT zero-shot (Apache 2.0)
  { name: 'detection_owlvit_base',       display: 'OWL-ViT Base',       type: 'detection', desc: 'Zero-shot open-vocabulary detection (Apache 2.0)' },
  { name: 'detection_owlvit_large',      display: 'OWL-ViT Large',      type: 'detection', desc: 'Larger zero-shot detection model (Apache 2.0)' },
  // Embedding — text
  { name: 'embedding_e5_large_v2', display: 'E5-Large-v2 (Microsoft)', type: 'embedding', desc: 'State-of-the-art text embeddings' },
  { name: 'embedding_e5_mistral_7b', display: 'E5-Mistral-7B (Microsoft)', type: 'embedding', desc: 'LLM-based text embeddings' },
  { name: 'embedding_gte_qwen2_7b', display: 'GTE-Qwen2-7B (Alibaba)', type: 'embedding', desc: 'Top multilingual embeddings' },
  { name: 'embedding_bge_large_en', display: 'BGE-Large-EN (BAAI)', type: 'embedding', desc: 'Best English text embeddings' },
  { name: 'embedding_bge_m3', display: 'BGE-M3 (BAAI)', type: 'embedding', desc: 'Multi-lingual, multi-granularity' },
  { name: 'embedding_minilm_l6', display: 'all-MiniLM-L6-v2', type: 'embedding', desc: 'Fast, lightweight sentence model' },
  { name: 'embedding_mpnet_base', display: 'all-mpnet-base-v2', type: 'embedding', desc: 'High quality sentence embeddings' },
  { name: 'embedding_gte_large', display: 'GTE-Large (Alibaba)', type: 'embedding', desc: 'General text embeddings' },
  { name: 'embedding_nomic_text', display: 'Nomic Embed Text v1.5', type: 'embedding', desc: '8192 token context, open source' },
  // Embedding — vision
  { name: 'embedding_dinov2_large', display: 'DINOv2 Large (Meta)', type: 'embedding', desc: 'Self-supervised vision features' },
  { name: 'embedding_clip_vit_l14', display: 'CLIP ViT-L/14 (OpenAI)', type: 'embedding', desc: 'Strong visual + text alignment' },
  { name: 'embedding_siglip_so400m', display: 'SigLIP SO400M (Google)', type: 'embedding', desc: 'Sigmoid image-text contrastive' },
  { name: 'embedding_imagebind', display: 'ImageBind (Meta)', type: 'embedding', desc: 'Multi-modal joint embedding' },
];

const hiddenTemplates = ['llm_lora_s3', 'config', 'llm_hybrid_2d_dp_tp'];

async function loadTemplatesForModal() {
  try {
    const res = await fetch('/api/configs');
    const data = await res.json();
    allTemplates = data.configs;
    renderTemplateGrid(allTemplates);
  } catch (e) {
    console.error('Failed to load templates:', e);
  }
}

function renderTemplateGrid(templates, filter = 'all') {
  const grid = document.getElementById('template-grid');
  if (!grid) return;
  grid.innerHTML = '';

  const seen = new Set();
  const filtered = templates.filter(name => {
    if (hiddenTemplates.includes(name)) return false;
    const displayName = templateDisplayNames[name] || name;
    if (seen.has(displayName)) return false;
    seen.add(displayName);
    return true;
  });

  const items = filtered.map(name => {
    let type = 'llm';
    if (name.startsWith('cnn')) type = 'cnn';
    else if (name.startsWith('vlm')) type = 'vlm';
    else if (name.startsWith('detection')) type = 'detection';
    else if (name.startsWith('embedding')) type = 'embedding';
    return {
      name,
      display: templateDisplayNames[name] || name.replace(/_/g, ' '),
      type,
      isExtra: false
    };
  });

  extraTemplates.forEach(t => {
    if (!seen.has(t.display)) {
      seen.add(t.display);
      items.push({ ...t, isExtra: true });
    }
  });

  const finalItems = filter === 'all' ? items : items.filter(t => {
    if (filter === 'cnn') return t.type === 'cnn' || t.type === 'vision';
    return t.type === filter;
  });
  const typeOrder = { llm: 0, vlm: 1, cnn: 2, vision: 2, detection: 3, embedding: 4 };
  finalItems.sort((a, b) => typeOrder[a.type] - typeOrder[b.type]);

  finalItems.forEach(item => {
    const tag = item.type === 'cnn' ? 'VISION CNN' :
                item.type === 'vision' ? 'VIS TRANSF' :
                item.type === 'vlm' ? 'VLM' :
                item.type === 'detection' ? 'OBJ DET' :
                item.type === 'embedding' ? 'EMBED' : 'LLM';

    const div = document.createElement('div');
    div.className = 'template-card';
    div.setAttribute('data-type', item.type);
    const descHtml = item.desc ? `<p class="template-desc">${item.desc}</p>` : '';
    div.innerHTML = `
      <span class="template-tag ${item.type}">${tag}</span>
      <div class="template-card-info">
        <h4>${item.display}</h4>
        ${descHtml}
      </div>
      <span class="template-card-arrow">-></span>
    `;
    div.onclick = () => selectTemplateFromModal(item.name, item.type, item.isExtra ? item : null);
    grid.appendChild(div);
  });
}

function filterTemplates(filter, btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderTemplateGrid(allTemplates, filter);
}

async function selectTemplateFromModal(name, type, extraTemplate = null) {
  closeTemplateModal();
  localStorage.setItem('selected_template', JSON.stringify({ name, type, isExtra: !!extraTemplate, extra: extraTemplate }));
  navigateTo('config.html');
}

// =========================================================================
// Custom Setup
// =========================================================================
function openCustomSetup() {
  clearFormState();
  localStorage.removeItem('selected_template');
  navigateTo('config.html');
}

function openYamlEditor() {
  localStorage.setItem('yaml_from_form', 'false');
  navigateTo('yaml.html');
}

function openGuidePage() {
  navigateTo('guide.html');
}

function closeGuidePage() {
  navigateTo('index.html');
}

function openCheckpointsPage() {
  // Pass the currently-configured checkpoint dir so the page lists the right folder
  const dir = (typeof getVal === 'function' ? getVal('f-checkpoint-dir') : '') || 'checkpoints';
  const q = '?dir=' + encodeURIComponent(dir);
  window.location.href = '/static/checkpoints.html' + q;
}

// =========================================================================
// Form Logic (Config Page)
// =========================================================================

function onModelTypeChange() {
  resetFsdpCheck();
  const t = getVal('f-model-type');

  document.getElementById('f-model-source').selectedIndex = 0;
  setVal('f-model-name', '');
  setVal('f-model-url', '');
  setVal('f-model-path', '');
  clearModelUpload();
  toggle('fg-model-name', false);
  toggle('fg-model-url', false);
  toggle('fg-model-path', false);
  toggle('fg-model-upload', false);

  document.getElementById('f-finetune-mode').selectedIndex = 0;
  setVal('f-lora-r', '');
  setVal('f-lora-alpha', '');
  setVal('f-lora-dropout', '');
  const quantizeEl = document.getElementById('f-quantize');
  if (quantizeEl) quantizeEl.checked = false;
  document.getElementById('f-quant-bits').selectedIndex = 0;
  setVal('f-max-seq-len', '');

  setVal('f-num-classes', '');
  const freezeEl = document.getElementById('f-freeze-backbone');
  if (freezeEl) freezeEl.checked = false;

  document.getElementById('f-yolo-model').selectedIndex = 0;
  setVal('f-image-size', '');

  const isCustom = t === 'custom_transformer';
  toggle('fg-model-source', !isCustom);
  toggle('fg-num-classes', t === 'cnn');
  toggle('fg-finetune-mode', !isCustom && (t === 'llm' || t === 'vlm' || t === 'embedding' || t === 'vision'));
  toggle('fg-yolo-model', t === 'detection');
  toggle('tr-quantize', t === 'llm' || t === 'vlm');
  toggle('tr-freeze', t === 'cnn');
  toggle('fg-max-seq-len', !isCustom && (t === 'llm' || t === 'vlm' || t === 'embedding'));
  toggle('fg-image-size', t === 'cnn' || t === 'vlm' || t === 'detection' || t === 'vision');
  toggle('fg-custom-arch', isCustom);
  if (isCustom) updateCustomParamCount();

  // Hide QLoRA option for vision transformers (quantization not applicable)
  const qloraOpt = document.querySelector('#f-finetune-mode option[value="qlora"]');
  if (qloraOpt) qloraOpt.style.display = (t === 'vision' || t === 'embedding') ? 'none' : '';

  if (t === 'llm' || t === 'vlm' || t === 'embedding' || t === 'vision') {
    onFinetuneModeChange();
  } else {
    toggle('lora-options', false);
    toggle('tr-quant-bits', false);
  }

  updateDataFormatOptions(t);
}

function updateDataFormatOptions(modelType) {
  const formatSelect = document.getElementById('f-data-format');
  if (!formatSelect) return;

  const formats = {
    cnn: [{ value: 'image_folder', label: 'Image Folder' }],
    vision: [{ value: 'image_folder', label: 'Image Folder' }],
    llm: [
      { value: 'json', label: 'JSON/JSONL' },
      { value: 'parquet', label: 'Parquet' },
      { value: 'csv', label: 'CSV' }
    ],
    vlm: [
      { value: 'json', label: 'JSON/JSONL' },
      { value: 'parquet', label: 'Parquet' }
    ],
    detection: [
      { value: 'yolo', label: 'YOLO Format' },
      { value: 'coco', label: 'COCO Format' }
    ],
    embedding: [
      { value: 'json', label: 'JSON/JSONL' },
      { value: 'csv', label: 'CSV' },
      { value: 'parquet', label: 'Parquet' }
    ]
  };

  formatSelect.innerHTML = '<option value="" disabled selected hidden>Select format...</option>';
  const relevantFormats = formats[modelType] || [];
  relevantFormats.forEach(fmt => {
    const option = document.createElement('option');
    option.value = fmt.value;
    option.textContent = fmt.label;
    formatSelect.appendChild(option);
  });
}

function onFinetuneModeChange() {
  const m = getVal('f-finetune-mode');
  const modelType = getVal('f-model-type');
  const supportsQuantization = modelType === 'llm' || modelType === 'vlm';
  const isLora = m === 'lora' || m === 'qlora';
  toggle('lora-options', isLora);

  if (m === 'qlora') {
    const quantizeEl = document.getElementById('f-quantize');
    if (quantizeEl) quantizeEl.checked = true;
    setVal('f-quant-bits', '4');
    toggle('tr-quant-bits', true);
    toggle('tr-quantize', false);
  } else if ((m === 'lora' || m === 'full') && supportsQuantization) {
    toggle('tr-quantize', true);
    onQuantizeChange();
  } else {
    const quantizeEl = document.getElementById('f-quantize');
    if (quantizeEl) quantizeEl.checked = false;
    toggle('tr-quantize', false);
    toggle('tr-quant-bits', false);
  }
}

function onQuantizeChange() {
  const finetuneMode = getVal('f-finetune-mode');
  if (finetuneMode !== 'lora' && finetuneMode !== 'full') {
    const quantizeElForce = document.getElementById('f-quantize');
    if (quantizeElForce) quantizeElForce.checked = false;
    toggle('tr-quant-bits', false);
    return;
  }

  const quantizeEl = document.getElementById('f-quantize');
  const q = quantizeEl ? quantizeEl.checked : false;
  toggle('tr-quant-bits', q);

  // Full fine-tune only supports 8-bit (4-bit requires PEFT/QLoRA)
  const bitsSelect = document.getElementById('f-quant-bits');
  if (bitsSelect) {
    if (finetuneMode === 'full' && q) {
      setVal('f-quant-bits', '8');
      Array.from(bitsSelect.options).forEach(opt => { opt.disabled = opt.value !== '8'; });
    } else {
      Array.from(bitsSelect.options).forEach(opt => { opt.disabled = false; });
    }
  }
}

// =========================================================================
// W&B Toggle
// =========================================================================
function onWandbChange() {
  const enabled = document.getElementById('f-wandb-enabled')?.checked;
  toggle('fg-wandb-project', !!enabled);
  toggle('fg-wandb-entity', !!enabled);
  toggle('fg-wandb-run', !!enabled);
}

// =========================================================================
// Strategy / GPU compatibility warning
// =========================================================================
function checkStrategyGpuCompatibility() {
  const strategy = getVal('f-strategy');
  const gpuCount = parseInt(getVal('f-gpu-count') || '1');
  const warnEl = document.getElementById('strategy-gpu-warning');
  if (!warnEl) return;

  if (strategy === 'fsdp' && gpuCount < 2) {
    warnEl.textContent = '⚠ FSDP requires multiple GPUs to shard parameters. Increase GPU count or switch to Solo/DDP.';
    warnEl.style.display = 'block';
  } else if (strategy === 'ddp' && gpuCount < 2) {
    warnEl.textContent = '⚠ DDP with 1 GPU provides no throughput benefit. Use 2+ GPUs or switch to Solo mode.';
    warnEl.style.display = 'block';
  } else {
    warnEl.style.display = 'none';
  }
}

function syncGpuOptionsWithStrategy() {
  const strategy = getVal('f-strategy');
  const gpuSelect = document.getElementById('f-gpu-count');
  if (!gpuSelect) return;

  const multiGpuAllowed = strategy === 'ddp' || strategy === 'fsdp' || strategy === 'hybrid';

  Array.from(gpuSelect.options).forEach(opt => {
    const gpuValue = parseInt(opt.value, 10);
    opt.disabled = !multiGpuAllowed && gpuValue > 1;
  });

  const selectedGpu = parseInt(gpuSelect.value || '1', 10);
  if (!multiGpuAllowed && selectedGpu > 1) {
    gpuSelect.value = '1';
  }
}

function onModelSourceChange() {
  const source = getVal('f-model-source');
  const modelType = getVal('f-model-type');

  toggle('fg-model-name', false);
  toggle('fg-model-url', false);
  toggle('fg-model-path', false);
  toggle('fg-model-upload', false);
  // Detection model dropdown is shown/hidden by onModelTypeChange, not by source
  if (modelType !== 'detection') {
    toggle('fg-yolo-model', false);
  }

  if (source === 'huggingface') {
    if (modelType !== 'detection') {
      toggle('fg-model-name', true);
    }
    const el = document.getElementById('f-model-name');
    if (el) el.placeholder = 'e.g. facebook/opt-125m';
  } else if (source === 'torchvision') {
    toggle('fg-model-name', true);
    const el = document.getElementById('f-model-name');
    if (el) el.placeholder = 'e.g. resnet50, vit_b_16, efficientnet_b0';
  } else if (source === 'url') {
    toggle('fg-model-url', true);
  } else if (source === 'local') {
    toggle('fg-model-path', true);
  } else if (source === 'upload') {
    toggle('fg-model-upload', true);
  }
}

function onModelFileSelect(input) {
  if (input.files && input.files[0]) {
    const file = input.files[0];
    document.getElementById('upload-filename').textContent = file.name;
    document.getElementById('upload-area').style.display = 'none';
    document.getElementById('upload-selected').style.display = 'flex';
  }
}

function clearModelUpload() {
  const fileInput = document.getElementById('f-model-file');
  const uploadArea = document.getElementById('upload-area');
  const uploadSelected = document.getElementById('upload-selected');
  if (fileInput) fileInput.value = '';
  if (uploadArea) uploadArea.style.display = 'block';
  if (uploadSelected) uploadSelected.style.display = 'none';
}

function onDataSourceChange() {
  const source = getVal('f-data-source');

  toggle('fg-data-name', false);
  toggle('fg-data-subset', false);
  toggle('fg-data-split', false);
  toggle('fg-data-kaggle', false);
  toggle('fg-data-torchvision', false);
  toggle('fg-data-path', false);
  toggle('fg-data-url', false);
  toggle('fg-data-upload', false);
  toggle('fg-data-format', false);

  if (source === 'huggingface') {
    toggle('fg-data-name', true);
    toggle('fg-data-subset', true);
    toggle('fg-data-split', true);
    const el = document.getElementById('f-data-name');
    if (el) el.placeholder = 'e.g. wikitext, tatsu-lab/alpaca';
    const subsetEl = document.getElementById('f-data-subset');
    if (subsetEl) subsetEl.placeholder = 'e.g. wikitext-2-raw-v1';
  } else if (source === 'kaggle') {
    toggle('fg-data-kaggle', true);
  } else if (source === 'torchvision') {
    toggle('fg-data-torchvision', true);
  } else if (source === 'local') {
    toggle('fg-data-path', true);
    toggle('fg-data-format', true);
  } else if (source === 'url') {
    toggle('fg-data-url', true);
    toggle('fg-data-format', true);
  } else if (source === 'upload') {
    toggle('fg-data-upload', true);
    toggle('fg-data-format', true);
  }
}

function onDataFileSelect(input) {
  if (input.files && input.files[0]) {
    const file = input.files[0];
    document.getElementById('data-upload-filename').textContent = file.name;
    document.getElementById('data-upload-area').style.display = 'none';
    document.getElementById('data-upload-selected').style.display = 'flex';
  }
}

function clearDataUpload() {
  const fileInput = document.getElementById('f-data-file');
  const uploadArea = document.getElementById('data-upload-area');
  const uploadSelected = document.getElementById('data-upload-selected');
  if (fileInput) fileInput.value = '';
  if (uploadArea) uploadArea.style.display = 'block';
  if (uploadSelected) uploadSelected.style.display = 'none';
}

// =========================================================================
// Side Panel
// =========================================================================
function toggleSidePanel() {
  const panel = document.getElementById('side-panel');
  const toggle = document.getElementById('side-panel-toggle');
  const icon = document.getElementById('panel-toggle-icon');
  const actionBar = document.querySelector('.form-action-bar');
  const configLayout = document.querySelector('.config-layout');

  if (!panel) return;

  const isCollapsed = panel.classList.toggle('collapsed');
  if (toggle) toggle.classList.toggle('collapsed', isCollapsed);

  if (isCollapsed) {
    if (icon) icon.textContent = '>';
    if (actionBar) actionBar.style.left = '0';
    if (configLayout) configLayout.style.marginLeft = '0';
  } else {
    if (icon) icon.textContent = '<';
    if (actionBar) actionBar.style.left = '272px';
    if (configLayout) configLayout.style.marginLeft = '272px';
  }
}

function switchSideTab(name) {
  document.querySelectorAll('.side-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.side-content').forEach(c => c.classList.remove('active'));

  if (name === 'templates') {
    document.querySelectorAll('.side-tab')[0].classList.add('active');
    const el = document.getElementById('side-templates');
    if (el) el.classList.add('active');
  } else {
    document.querySelectorAll('.side-tab')[1].classList.add('active');
    const el = document.getElementById('side-logs');
    if (el) el.classList.add('active');
  }
}

// =========================================================================
// Side Templates
// =========================================================================
async function loadTemplates() {
  try {
    const res = await fetch('/api/configs');
    const data = await res.json();
    allTemplates = data.configs;
    renderSideTemplates(sideTemplatesFilter);
  } catch (e) {
    console.error('Failed to load templates:', e);
  }
}

function filterSideTemplates(filter, btn) {
  document.querySelectorAll('.side-filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  sideTemplatesFilter = filter;
  renderSideTemplates(filter);
}

function renderSideTemplates(filter = 'all') {
  const list = document.getElementById('template-list');
  if (!list) return;
  list.innerHTML = '';

  const seen = new Set();
  const filtered = allTemplates.filter(name => {
    if (hiddenTemplates.includes(name)) return false;
    const displayName = templateDisplayNames[name] || name;
    if (seen.has(displayName)) return false;
    seen.add(displayName);
    return true;
  });

  const items = filtered.map(name => {
    let type = 'llm';
    if (name.startsWith('cnn')) type = 'cnn';
    else if (name.startsWith('vlm')) type = 'vlm';
    else if (name.startsWith('detection')) type = 'detection';
    else if (name.startsWith('embedding')) type = 'embedding';
    return {
      name,
      display: templateDisplayNames[name] || name.replace(/_/g, ' '),
      type,
      isExtra: false
    };
  });

  extraTemplates.forEach(t => {
    if (!seen.has(t.display)) {
      seen.add(t.display);
      items.push({ ...t, isExtra: true });
    }
  });

  const finalItems = filter === 'all' ? items : items.filter(t => {
    if (filter === 'cnn') return t.type === 'cnn' || t.type === 'vision';
    return t.type === filter;
  });
  const typeOrder = { llm: 0, vlm: 1, cnn: 2, vision: 2, detection: 3, embedding: 4 };
  finalItems.sort((a, b) => typeOrder[a.type] - typeOrder[b.type]);

  finalItems.forEach(item => {
    const tag = item.type === 'cnn' ? 'CNN' :
                item.type === 'vision' ? 'VIS-T' :
                item.type === 'vlm' ? 'VLM' :
                item.type === 'detection' ? 'OBJ DET' :
                item.type === 'embedding' ? 'EMBED' : 'LLM';

    const div = document.createElement('div');
    div.className = 'template-item';
    div.setAttribute('data-type', item.type);
    div.setAttribute('data-name', item.name);
    div.title = item.display;
    div.innerHTML = `
      <span class="template-tag ${item.type}">${tag}</span>
      <span class="template-name">${item.display}</span>
    `;
    div.onclick = (e) => selectSideTemplate(item, e);
    list.appendChild(div);
  });

  // Highlight pending template selection from landing page
  if (pendingTemplateSelection) {
    const templateItems = list.querySelectorAll('.template-item');
    templateItems.forEach(el => {
      if (el.getAttribute('data-name') === pendingTemplateSelection) {
        el.classList.add('active');
      }
    });
    pendingTemplateSelection = null;
  }
}

async function selectSideTemplate(item, event) {

  // Snap to top before any DOM changes — cover both window scroll and panel scroll
  window.scrollTo(0, 0);
  document.documentElement.scrollTop = 0;
  document.body.scrollTop = 0;
  const panel = document.querySelector('.form-panel');
  if (panel) panel.scrollTop = 0;

  document.querySelectorAll('.template-item').forEach(t => t.classList.remove('active'));
  if (event && event.currentTarget) event.currentTarget.classList.add('active');

  selectedModelType = item.type;
  setVal('f-model-type', item.type);
  onModelTypeChange();

  if (item.isExtra) {
    applyExtraTemplate(item);
    showToast(`Template loaded: ${item.display}`, 'success');
  } else {
    try {
      const res = await fetch(`/api/configs/${item.name}`);
      const data = await res.json();
      currentConfig = data.config;
      applyConfigToForm(data.config);
      showToast(`Template loaded: ${item.display}`, 'success');
    } catch (e) {
      console.error('Failed to load template:', e);
      showToast('Failed to load template', 'info');
    }
  }
}

function applyExtraTemplate(template) {
  const name = template.name;

  setVal('f-model-source', 'huggingface');
  toggle('fg-model-name', true);
  toggle('fg-model-url', false);
  toggle('fg-model-path', false);
  toggle('fg-model-upload', false);

  setVal('f-strategy', 'none');
  onStrategyChange();
  setVal('f-mixed-precision', 'true');
  setVal('f-epochs', '3');
  setVal('f-batch-size', '8');
  setVal('f-lr', '2e-5');
  setVal('f-lr-schedule', 'cosine');
  setVal('f-grad-accum', '1');
  setVal('f-warmup-steps', '100');
  setVal('f-weight-decay', '0.01');
  setVal('f-grad-clip', '1.0');
  setVal('f-checkpoint-dir', 'checkpoints');

  if (template.type === 'llm') {
    let modelName = 'facebook/opt-125m';
    let finetuneMode = 'lora';
    let maxSeqLen = '2048';

    // Top 2026 LLMs
    if (name.includes('glm5')) modelName = 'THUDM/glm-5-9b-chat';
    else if (name.includes('kimi_k25')) modelName = 'moonshotai/Kimi-K2.5-Instruct';
    else if (name.includes('deepseek_v3')) modelName = 'deepseek-ai/DeepSeek-V3';
    else if (name.includes('deepseek_r1')) modelName = 'deepseek-ai/DeepSeek-R1';
    else if (name.includes('qwen3_235b')) { modelName = 'Qwen/Qwen3-235B-Instruct-2507'; maxSeqLen = '32768'; }
    else if (name.includes('llama33_70b')) { modelName = 'meta-llama/Llama-3.3-70B-Instruct'; maxSeqLen = '8192'; }
    else if (name.includes('mimo_v2_flash')) { modelName = 'Xiaomi/MiMo-V2-Flash'; maxSeqLen = '32768'; }
    // Other LLMs
    else if (name.includes('llama2_7b')) modelName = 'meta-llama/Llama-2-7b-hf';
    else if (name.includes('mistral_7b')) modelName = 'mistralai/Mistral-7B-v0.1';
    else if (name.includes('phi3')) modelName = 'microsoft/Phi-3-mini-4k-instruct';
    // Small/test models
    else if (name.includes('opt_125m')) { modelName = 'facebook/opt-125m'; maxSeqLen = '512'; }
    else if (name.includes('opt_350m')) { modelName = 'facebook/opt-350m'; maxSeqLen = '512'; }
    else if (name.includes('gpt2_medium')) { modelName = 'gpt2-medium'; maxSeqLen = '1024'; }
    else if (name.includes('gpt2')) { modelName = 'gpt2'; maxSeqLen = '1024'; }

    if (name.includes('qlora')) finetuneMode = 'qlora';
    else if (name.includes('full')) finetuneMode = 'full';

    setVal('f-model-name', modelName);
    setVal('f-finetune-mode', finetuneMode);
    setVal('f-max-seq-len', maxSeqLen);
    setVal('f-lora-r', '16');
    setVal('f-lora-alpha', '32');
    setVal('f-lora-dropout', '0.05');

    // Default dataset: wikitext with small split for all LLMs
    setVal('f-data-source', 'huggingface');
    onDataSourceChange();
    setVal('f-data-name', 'wikitext');
    setVal('f-data-subset', 'wikitext-2-raw-v1');
    // Use a tiny slice for small/test models, full train split for large ones
    const isSmall = name.includes('opt_125m') || name.includes('opt_350m') || name.includes('gpt2');
    setVal('f-data-split', '');

    onFinetuneModeChange();
  } else if (template.type === 'vlm') {
    let modelName = 'llava-hf/llava-1.5-7b-hf';
    let finetuneMode = 'lora';
    let maxSeqLen = '2048';

    // Top 2026 VLMs
    if (name.includes('qwen2_vl_72b')) { modelName = 'Qwen/Qwen2-VL-72B-Instruct'; maxSeqLen = '32768'; }
    else if (name.includes('internvl2_26b')) { modelName = 'OpenGVLab/InternVL2-26B'; maxSeqLen = '8192'; }
    else if (name.includes('llava_next_34b')) { modelName = 'llava-hf/llava-v1.6-34b-hf'; maxSeqLen = '4096'; }
    else if (name.includes('cogvlm2')) { modelName = 'THUDM/cogvlm2-llama3-chat-19B'; maxSeqLen = '8192'; }
    else if (name.includes('phi3_vision')) { modelName = 'microsoft/Phi-3-vision-128k-instruct'; maxSeqLen = '8192'; }
    else if (name.includes('idefics2_8b')) { modelName = 'HuggingFaceM4/idefics2-8b'; maxSeqLen = '4096'; }
    else if (name.includes('paligemma_3b')) { modelName = 'google/paligemma-3b-mix-448'; maxSeqLen = '2048'; }
    // Other VLMs
    else if (name.includes('llava_7b')) modelName = 'llava-hf/llava-1.5-7b-hf';
    else if (name.includes('blip2')) modelName = 'Salesforce/blip2-flan-t5-xl';

    if (name.includes('qlora')) finetuneMode = 'qlora';
    else if (name.includes('full')) finetuneMode = 'full';

    setVal('f-model-name', modelName);
    setVal('f-finetune-mode', finetuneMode);
    setVal('f-max-seq-len', maxSeqLen);
    setVal('f-image-size', '448');
    setVal('f-lora-r', '16');
    setVal('f-lora-alpha', '32');
    setVal('f-lora-dropout', '0.05');

    // Default dataset: small image-text pairs from HuggingFace
    setVal('f-data-source', 'huggingface');
    onDataSourceChange();
    setVal('f-data-name', 'HuggingFaceM4/the_cauldron');
    setVal('f-data-subset', 'ai2d');
    setVal('f-data-split', '');

    onFinetuneModeChange();
  } else if (template.type === 'cnn') {
    let modelName = 'resnet50';
    if (name.includes('resnet18')) modelName = 'resnet18';
    else if (name.includes('vit_base')) modelName = 'vit_b_16';
    else if (name.includes('efficientnet_b0')) modelName = 'efficientnet_b0';

    setVal('f-model-name', modelName);
    setVal('f-num-classes', '10');
    setVal('f-image-size', '224');

    // Default dataset: CIFAR-10 via torchvision
    setVal('f-data-source', 'torchvision');
    onDataSourceChange();
    setVal('f-data-torchvision', 'cifar10');
  } else if (template.type === 'vision') {
    let modelName = 'google/vit-base-patch16-224';
    let finetuneMode = 'lora';

    if (name.includes('vit_base')) { modelName = 'google/vit-base-patch16-224'; }
    else if (name.includes('vit_large')) { modelName = 'google/vit-large-patch16-224'; }
    else if (name.includes('vit_huge')) { modelName = 'google/vit-huge-patch14-224-in21k'; }
    else if (name.includes('swin_tiny')) { modelName = 'microsoft/swin-tiny-patch4-window7-224'; }
    else if (name.includes('swin_base')) { modelName = 'microsoft/swin-base-patch4-window7-224'; }
    else if (name.includes('swin_large')) { modelName = 'microsoft/swin-large-patch4-window7-224'; }
    else if (name.includes('deit_tiny')) { modelName = 'facebook/deit-tiny-patch16-224'; }
    else if (name.includes('deit_small')) { modelName = 'facebook/deit-small-patch16-224'; }
    else if (name.includes('deit_base')) { modelName = 'facebook/deit-base-patch16-224'; }
    else if (name.includes('beit_base')) { modelName = 'microsoft/beit-base-patch16-224'; }
    else if (name.includes('beit_large')) { modelName = 'microsoft/beit-large-patch16-224'; }
    else if (name.includes('convnext_tiny')) { modelName = 'facebook/convnext-tiny-224'; }
    else if (name.includes('convnext_base')) { modelName = 'facebook/convnext-base-224'; }
    else if (name.includes('convnext_large')) { modelName = 'facebook/convnext-large-224'; }
    else if (name.includes('efficientnet_b4')) { modelName = 'google/efficientnet-b4'; }
    else if (name.includes('efficientnet_b7')) { modelName = 'google/efficientnet-b7'; }

    if (name.includes('full')) finetuneMode = 'full';

    setVal('f-model-source', 'huggingface');
    toggle('fg-model-name', true);
    setVal('f-model-name', modelName);
    setVal('f-finetune-mode', finetuneMode);
    setVal('f-image-size', '224');
    setVal('f-lora-r', '16');
    setVal('f-lora-alpha', '32');
    setVal('f-lora-dropout', '0.05');

    // Default dataset: ImageNet-1k subset via HuggingFace
    setVal('f-data-source', 'huggingface');
    onDataSourceChange();
    setVal('f-data-name', 'ILSVRC/imagenet-1k');
    setVal('f-data-split', 'train[:5%]');

    onFinetuneModeChange();
  } else if (template.type === 'detection') {
    let detectionModel = 'hustvl/yolos-small';
    if      (name.includes('yolos_tiny'))        detectionModel = 'hustvl/yolos-tiny';
    else if (name.includes('yolos_base'))        detectionModel = 'hustvl/yolos-base';
    else if (name.includes('rtdetr_r18'))        detectionModel = 'PekingU/rtdetr_r18vd';
    else if (name.includes('rtdetr_r50'))        detectionModel = 'PekingU/rtdetr_r50vd';
    else if (name.includes('rtdetr_r101'))       detectionModel = 'PekingU/rtdetr_r101vd';
    else if (name.includes('detr_r101'))         detectionModel = 'facebook/detr-resnet-101';
    else if (name.includes('conditional_detr'))  detectionModel = 'microsoft/conditional-detr-resnet-50';
    else if (name.includes('deformable_detr'))   detectionModel = 'SenseTime/deformable-detr';
    else if (name.includes('dab_detr'))          detectionModel = 'IDEA-Research/dab-detr-r50';
    else if (name.includes('owlvit_large'))      detectionModel = 'google/owlvit-large-patch14';
    else if (name.includes('owlvit_base'))       detectionModel = 'google/owlvit-base-patch32';
    else if (name.includes('detr_r50'))          detectionModel = 'facebook/detr-resnet-50';

    setVal('f-model-source', 'huggingface');
    setVal('f-yolo-model', detectionModel);
    setVal('f-image-size', '640');

    // Default dataset: COCO128 sample — small enough for quick runs
    setVal('f-data-source', 'huggingface');
    onDataSourceChange();
    setVal('f-data-name', 'keremberke/coco128-object-detection');
    setVal('f-data-split', '');
  } else if (template.type === 'embedding') {
    let modelName = 'sentence-transformers/all-MiniLM-L6-v2';
    let finetuneMode = 'lora';
    let maxSeqLen = '128';

    // Text embedding models
    if (name.includes('e5_large_v2')) { modelName = 'intfloat/e5-large-v2'; maxSeqLen = '512'; }
    else if (name.includes('e5_mistral')) { modelName = 'intfloat/e5-mistral-7b-instruct'; maxSeqLen = '4096'; }
    else if (name.includes('gte_qwen2')) { modelName = 'Alibaba-NLP/gte-Qwen2-7B-instruct'; maxSeqLen = '8192'; }
    else if (name.includes('bge_large_en')) { modelName = 'BAAI/bge-large-en-v1.5'; maxSeqLen = '512'; }
    else if (name.includes('bge_m3')) { modelName = 'BAAI/bge-m3'; maxSeqLen = '8192'; }
    else if (name.includes('minilm_l6')) { modelName = 'sentence-transformers/all-MiniLM-L6-v2'; maxSeqLen = '256'; }
    else if (name.includes('mpnet_base')) { modelName = 'sentence-transformers/all-mpnet-base-v2'; maxSeqLen = '384'; }
    else if (name.includes('gte_large')) { modelName = 'thenlper/gte-large'; maxSeqLen = '512'; }
    else if (name.includes('nomic_text')) { modelName = 'nomic-ai/nomic-embed-text-v1.5'; maxSeqLen = '8192'; }
    // Vision embedding models
    else if (name.includes('dinov2_large')) { modelName = 'facebook/dinov2-large'; finetuneMode = 'full'; }
    else if (name.includes('clip_vit_l14')) { modelName = 'openai/clip-vit-large-patch14'; }
    else if (name.includes('siglip_so400m')) { modelName = 'google/siglip-so400m-patch14-384'; }
    else if (name.includes('imagebind')) { modelName = 'facebook/imagebind-huge'; }
    // Legacy / YAML-backed
    else if (name.includes('bert')) { modelName = 'bert-base-uncased'; }
    else if (name.includes('clip')) { modelName = 'openai/clip-vit-base-patch32'; }
    else if (name.includes('vision')) { modelName = 'microsoft/resnet-50'; }

    if (name.includes('full')) finetuneMode = 'full';

    setVal('f-model-name', modelName);
    setVal('f-finetune-mode', finetuneMode);
    setVal('f-max-seq-len', maxSeqLen);
    setVal('f-lora-r', '8');
    setVal('f-lora-alpha', '16');
    setVal('f-lora-dropout', '0.05');

    // Default dataset: all-nli — NLI sentence pairs, widely used for contrastive/triplet embedding training
    setVal('f-data-source', 'huggingface');
    onDataSourceChange();
    setVal('f-data-name', 'sentence-transformers/all-nli');
    setVal('f-data-subset', 'triplet');
    setVal('f-data-split', '');

    onFinetuneModeChange();
  }
}

/**
 * Normalize the flat LLM schema (used by llm_*.yaml configs) to the standard
 * nested schema expected by applyConfigToForm.
 *
 * Flat schema keys: model_name, dataset, strategy, num_gpus, checkpoint_dir,
 *   dist_parameters, peft, quantization, training.
 * Standard schema keys: model.{type,name,...}, data.{name,...},
 *   distributed.{strategy,...}, training.{...}.
 */
function normalizeLlmFlatConfig(cfg) {
  const peft = cfg.peft || {};
  const quant = cfg.quantization || {};
  const distParams = cfg.dist_parameters || {};
  const dataset = cfg.dataset || {};
  const t = cfg.training || {};

  // Determine finetune mode from peft block
  let finetuneMode = 'full';
  if (peft.enabled) {
    finetuneMode = quant.enabled ? 'qlora' : 'lora';
  }

  // Map strategy: 'solo' → 'none' (UI uses 'none' for single-GPU)
  let strategy = cfg.strategy || 'none';
  if (strategy === 'solo') strategy = 'none';

  // Determine data source from dataset.name value
  const datasetName = dataset.name || '';
  let dataSource = 'huggingface';
  let dataName = datasetName;
  let dataPath = '';
  let dataUrl = '';
  if (datasetName.startsWith('s3://') || datasetName.startsWith('http://') || datasetName.startsWith('https://')) {
    dataSource = 'url';
    dataUrl = datasetName;
    dataName = '';
  } else if (datasetName.startsWith('/') || datasetName.startsWith('./') || datasetName.startsWith('../')) {
    dataSource = 'local';
    dataPath = datasetName;
    dataName = '';
  }

  return {
    seed: cfg.seed || 42,
    num_gpus: cfg.num_gpus || 1,
    _flatLlm: true,
    model: {
      type: 'llm',
      name: cfg.model_name || '',
      finetune_mode: finetuneMode === 'qlora' ? 'lora' : finetuneMode,
      quantize: quant.enabled === true,
      quant_bits: quant.bits || 4,
      lora_r: peft.r || 16,
      lora_alpha: peft.alpha || 32,
      lora_dropout: peft.dropout || 0.05,
    },
    data: {
      _source: dataSource,
      name: dataName,
      subset: dataset.subset || '',
      split: dataset.split || '',
      _path: dataPath,
      _url: dataUrl,
      max_seq_len: t.max_length || 2048,
    },
    distributed: {
      strategy: strategy,
      mixed_precision: distParams.mixed_precision !== false,
    },
    training: {
      epochs: t.epochs || 3,
      batch_size: t.batch_size || 8,
      lr: t.learning_rate || t.lr || '2e-4',
      weight_decay: t.weight_decay || 0.01,
      grad_accum_steps: t.grad_accum_steps || 1,
      warmup_steps: t.warmup_steps || 100,
      grad_clip: t.grad_clip || 1.0,
      lr_schedule: t.lr_schedule || 'cosine',
      checkpoint_dir: cfg.checkpoint_dir || 'checkpoints',
    },
  };
}

function applyConfigToForm(cfg) {
  if (!cfg) return;

  // Detect flat LLM schema (model_name at top level, no model.type)
  if (cfg.model_name != null && cfg.model == null) {
    cfg = normalizeLlmFlatConfig(cfg);
  }

  const m = cfg.model || {};
  const d = cfg.data || {};
  const dist = cfg.distributed || {};
  const t = cfg.training || {};

  const modelType = m.type || 'llm';
  const modelName = m.name || '';

  // Step 1: Set model type and trigger visibility updates (this resets values)
  setVal('f-model-type', modelType);
  onModelTypeChange();

  // Step 2: Now set all the specific values AFTER the reset
  // Model source and name
  if (modelType === 'detection') {
    setVal('f-model-source', 'huggingface');
  } else if (modelType === 'cnn') {
    setVal('f-model-source', 'torchvision');
    setVal('f-model-name', modelName);
    toggle('fg-model-name', true);
  } else if (modelType === 'vision' || modelType === 'llm' || modelType === 'vlm' || modelType === 'embedding') {
    // Always default to HuggingFace for Vision-Transformer/LLM/VLM/Embedding
    setVal('f-model-source', 'huggingface');
    toggle('fg-model-name', true);
    if (modelName) {
      setVal('f-model-name', modelName);
    }
  }

  // Model-specific settings
  setVal('f-num-classes', m.num_classes || 10);
  setVal('f-yolo-model', m.yolo_model || 'hustvl/yolos-small');
  const freezeEl = document.getElementById('f-freeze-backbone');
  if (freezeEl) freezeEl.checked = m.freeze_backbone === true;

  // Finetune mode and related settings
  let finetuneMode = m.finetune_mode || 'full';
  if (finetuneMode === 'lora' && m.quantize === true) {
    finetuneMode = 'qlora';
  }
  setVal('f-finetune-mode', finetuneMode);
  onFinetuneModeChange();  // Update visibility for lora options

  // Quantization (set after finetune mode)
  const quantizeEl = document.getElementById('f-quantize');
  if (quantizeEl) quantizeEl.checked = m.quantize === true;
  setVal('f-quant-bits', m.quant_bits || 4);
  onQuantizeChange();  // Update quant bits visibility

  // LoRA settings
  setVal('f-lora-r', m.lora_r || 16);
  setVal('f-lora-alpha', m.lora_alpha || 32);
  setVal('f-lora-dropout', m.lora_dropout || 0.05);

  // Data source and fields
  // d._source is set by normalizeLlmFlatConfig; for standard configs derive from d.type
  const rawDataType = d.type || '';
  let dataSource = d._source || '';
  if (!dataSource) {
    if (rawDataType === 'torchvision') dataSource = 'torchvision';
    else if (rawDataType === 'hf_dataset') dataSource = 'huggingface';
    else if (rawDataType === 'local_file' || rawDataType === 'image_folder') dataSource = 'local';
    else if (rawDataType === 'yolo' || rawDataType === 'coco') dataSource = 'url';
    else if (d.name) dataSource = 'huggingface';
  }
  if (dataSource) {
    setVal('f-data-source', dataSource);
    onDataSourceChange();
  }

  if (dataSource === 'huggingface') {
    setVal('f-data-name', d.name || '');
    setVal('f-data-subset', d.subset || d.dataset_full_name || '');
    setVal('f-data-split', d.split || '');
  } else if (dataSource === 'torchvision') {
    const tvName = (d.name || '').toLowerCase();
    setVal('f-data-torchvision', tvName || 'cifar10');
  } else if (dataSource === 'local') {
    setVal('f-data-path', d._path || d.name || d.data_yaml || '');
  } else if (dataSource === 'url') {
    // For detection configs the YAML paths are local paths, not download URLs — leave blank
    const isDetectionFmt = rawDataType === 'coco' || rawDataType === 'yolo';
    setVal('f-data-url', isDetectionFmt ? '' : (d._url || d.name || ''));
  }

  // Pre-select data format for detection templates
  if (rawDataType === 'coco' || rawDataType === 'yolo') {
    setVal('f-data-format', rawDataType);
  }

  setVal('f-image-size', d.image_size || '');
  setVal('f-max-seq-len', d.max_seq_len || '');

  // Distributed settings
  setVal('f-strategy', dist.strategy || 'none');
  setVal('f-gpu-count', cfg.num_gpus || dist.gpu_count || 1);
  onStrategyChange();
  setVal('f-mixed-precision', String(dist.mixed_precision !== false));

  // Launch mode & SLURM
  setVal('f-launch-mode', cfg.launch_mode || 'torchrun');
  onLaunchModeChange();
  if (cfg.slurm) {
    setVal('f-slurm-nodes', cfg.slurm.nodes || 1);
    setVal('f-slurm-gpus-per-node', cfg.slurm.gpus_per_node || 4);
    setVal('f-slurm-partition', cfg.slurm.partition || '');
    setVal('f-slurm-time', cfg.slurm.time || '');
  }

  // 3D topology
  const topo = cfg.topology || {};
  if (topo.parallelism_mode) {
    setVal('f-parallelism-mode', topo.parallelism_mode);
    onParallelismModeChange();
  }
  if (topo.data_parallel_size) setVal('f-dp-size', topo.data_parallel_size);
  if (topo.tensor_parallel_size) setVal('f-tp-size', topo.tensor_parallel_size);
  if (topo.pipeline_parallel_size) setVal('f-pp-size', topo.pipeline_parallel_size);
  if (dist.strategy === 'hybrid') onTopologyChange();

  // Training settings
  setVal('f-epochs', t.epochs || 3);
  setVal('f-batch-size', t.batch_size || 8);
  setVal('f-lr', t.lr || '2e-5');
  setVal('f-weight-decay', t.weight_decay || 0.01);
  setVal('f-grad-accum', t.grad_accum_steps || 1);
  setVal('f-warmup-steps', t.warmup_steps || 100);
  setVal('f-grad-clip', t.grad_clip || 1.0);
  setVal('f-lr-schedule', t.lr_schedule || 'cosine');
  setVal('f-checkpoint-dir', t.checkpoint_dir || 'checkpoints');

  // W&B settings
  const wandb = cfg.wandb || {};
  const wandbEl = document.getElementById('f-wandb-enabled');
  if (wandbEl) wandbEl.checked = !!(wandb.wandb_log_with_train);
  setVal('f-wandb-project', wandb.wandb_project || '');
  setVal('f-wandb-entity', wandb.wandb_entity || '');
  setVal('f-wandb-run', wandb.wandb_run_name || '');
  onWandbChange();
}

function buildConfigFromForm() {
  const modelType = getVal('f-model-type');
  const finetuneMode = getVal('f-finetune-mode');

  const cfg = { seed: 42 };

  const modelSource = getVal('f-model-source');
  let modelName = '';
  if (modelSource === 'huggingface') {
    modelName = modelType === 'detection' ? getVal('f-yolo-model') : getVal('f-model-name');
  } else if (modelSource === 'torchvision') {
    modelName = getVal('f-model-name');
  } else if (modelSource === 'url') {
    modelName = getVal('f-model-url');
  } else if (modelSource === 'local') {
    modelName = getVal('f-model-path');
  } else if (modelSource === 'upload') {
    const fileInput = document.getElementById('f-model-file');
    if (fileInput && fileInput.files && fileInput.files[0]) {
      modelName = 'uploaded:' + fileInput.files[0].name;
    }
  }
  cfg.model = { type: modelType, name: modelName, source: modelSource };

  if (modelType === 'cnn') {
    cfg.model.num_classes = parseInt(getVal('f-num-classes'));
    cfg.model.pretrained = true;
    const freezeEl = document.getElementById('f-freeze-backbone');
    cfg.model.freeze_backbone = freezeEl ? freezeEl.checked : false;
  } else if (modelType === 'llm') {
    const isQlora = finetuneMode === 'qlora';
    cfg.model.finetune_mode = isQlora ? 'lora' : finetuneMode;
    cfg.model.use_flash_attention = true;

    const quantizeEl = document.getElementById('f-quantize');
    if (isQlora || (quantizeEl && quantizeEl.checked)) {
      cfg.model.quantize = true;
      cfg.model.quant_bits = isQlora ? 4 : parseInt(getVal('f-quant-bits'));
    }

    if (finetuneMode === 'lora' || isQlora) {
      cfg.model.lora_r = parseInt(getVal('f-lora-r'));
      cfg.model.lora_alpha = parseInt(getVal('f-lora-alpha'));
      cfg.model.lora_dropout = parseFloat(getVal('f-lora-dropout'));
      cfg.model.lora_target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj'];
    }
  } else if (modelType === 'vlm') {
    const isQlora = finetuneMode === 'qlora';
    cfg.model.finetune_mode = isQlora ? 'lora' : finetuneMode;
    cfg.model.use_flash_attention = true;
    cfg.model.lora_target = 'llm_only';

    const quantizeEl = document.getElementById('f-quantize');
    if (isQlora || (quantizeEl && quantizeEl.checked)) {
      cfg.model.quantize = true;
      cfg.model.quant_bits = isQlora ? 4 : parseInt(getVal('f-quant-bits'));
    }

    if (finetuneMode === 'lora' || isQlora) {
      cfg.model.lora_r = parseInt(getVal('f-lora-r'));
      cfg.model.lora_alpha = parseInt(getVal('f-lora-alpha'));
      cfg.model.lora_dropout = parseFloat(getVal('f-lora-dropout'));
      cfg.model.lora_target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj'];
    }
  } else if (modelType === 'vision') {
    cfg.model.finetune_mode = finetuneMode;
    if (finetuneMode === 'lora') {
      cfg.model.lora_r = parseInt(getVal('f-lora-r'));
      cfg.model.lora_alpha = parseInt(getVal('f-lora-alpha'));
      cfg.model.lora_dropout = parseFloat(getVal('f-lora-dropout'));
      cfg.model.lora_target_modules = 'all-linear';
    }
  } else if (modelType === 'detection') {
    cfg.model.yolo_model = getVal('f-yolo-model');
  } else if (modelType === 'custom_transformer') {
    cfg.model.source = 'custom';
    cfg.model.arch = {
      n_layers: parseInt(getVal('f-custom-n-layers')) || 6,
      vocab_size: parseInt(getVal('f-custom-vocab-size')) || 8192,
      max_seq_len: parseInt(getVal('f-custom-max-seq-len')) || 512,
      dim: parseInt(getVal('f-custom-dim')) || 512,
      n_heads: parseInt(getVal('f-custom-n-heads')) || 8,
      dropout_p: parseFloat(getVal('f-custom-dropout-p')) || 0.1,
    };
  }

  cfg.data = { type: 'dummy', num_workers: 4 };
  const dataSource = getVal('f-data-source');
  cfg.data._source = dataSource || 'huggingface';
  const dataName = getVal('f-data-name');
  const dataSubset = getVal('f-data-subset');
  const dataSplit = getVal('f-data-split');
  if (dataSource === 'local') {
    const dataPath = getVal('f-data-path');
    const dataFormat = getVal('f-data-format');
    if (dataPath) cfg.data._path = dataPath;
    if (dataFormat) cfg.data.format = dataFormat;
  } else if (dataSource === 'torchvision') {
    cfg.data.name = getVal('f-data-torchvision') || 'cifar10';
  } else if (dataSource === 'url') {
    const dataUrl = getVal('f-data-url');
    if (dataUrl) cfg.data._url = dataUrl;
    if (dataName) cfg.data.name = dataName;
  } else {
    if (dataName) cfg.data.name = dataName;
  }
  if (dataSubset) cfg.data.subset = dataSubset;
  if (dataSplit) cfg.data.split = dataSplit;
  if (modelType === 'cnn' || modelType === 'vision' || modelType === 'vlm' || modelType === 'detection') {
    cfg.data.image_size = parseInt(getVal('f-image-size'));
  }
  if (modelType === 'llm' || modelType === 'vlm') {
    cfg.data.max_seq_len = parseInt(getVal('f-max-seq-len'));
  }

  cfg.distributed = {
    strategy: getVal('f-strategy'),
    mixed_precision: getVal('f-mixed-precision') === 'true',
  };

  cfg.num_gpus = parseInt(getVal('f-gpu-count') || '1');

  // Launch mode
  cfg.launch_mode = getVal('f-launch-mode') || 'torchrun';
  if (cfg.launch_mode === 'slurm') {
    cfg.slurm = {
      nodes: parseInt(getVal('f-slurm-nodes')) || 1,
      gpus_per_node: parseInt(getVal('f-slurm-gpus-per-node')) || 4,
      partition: getVal('f-slurm-partition') || 'gpu',
      time: getVal('f-slurm-time') || '2:00:00',
    };
  }

  // 3D topology (hybrid strategy only)
  if (cfg.distributed.strategy === 'hybrid') {
    const ppMode = getVal('f-parallelism-mode') || '2d';
    cfg.topology = {
      parallelism_mode: ppMode,
      data_parallel_size: parseInt(getVal('f-dp-size')) || 2,
      tensor_parallel_size: parseInt(getVal('f-tp-size')) || 2,
      pipeline_parallel_size: ppMode === '3d' ? (parseInt(getVal('f-pp-size')) || 1) : 1,
      tensor_parallel_auto_plan: true,
    };
  }

  cfg.training = {
    epochs: parseInt(getVal('f-epochs')) || 3,
    batch_size: parseInt(getVal('f-batch-size')) || 8,
    lr: getVal('f-lr') || '2e-5',
    weight_decay: parseFloat(getVal('f-weight-decay')) || 0.01,
    grad_clip: parseFloat(getVal('f-grad-clip')) || 1.0,
    grad_accum_steps: parseInt(getVal('f-grad-accum')) || 1,
    warmup_steps: parseInt(getVal('f-warmup-steps')) || 100,
    lr_schedule: getVal('f-lr-schedule') || 'cosine',
    checkpoint_dir: getVal('f-checkpoint-dir'),
    log_interval: 10,
  };

  // W&B logging
  const wandbEnabled = document.getElementById('f-wandb-enabled')?.checked || false;
  cfg.wandb = {
    wandb_log_with_train: wandbEnabled,
    wandb_project: getVal('f-wandb-project') || 'dist-train-project',
    wandb_entity: getVal('f-wandb-entity') || '',
    wandb_run_name: getVal('f-wandb-run') || '',
  };

  return cfg;
}

// =========================================================================
// Custom Transformer live parameter count
// =========================================================================
function updateCustomParamCount() {
  const el = document.getElementById('custom-param-count');
  if (!el) return;

  const nLayers   = parseInt(getVal('f-custom-n-layers'))   || 0;
  const dim       = parseInt(getVal('f-custom-dim'))         || 0;
  const nHeads    = parseInt(getVal('f-custom-n-heads'))     || 0;
  const vocabSize = parseInt(getVal('f-custom-vocab-size'))  || 0;
  const maxSeqLen = parseInt(getVal('f-custom-max-seq-len')) || 0;

  if (!nLayers || !dim || !nHeads || !vocabSize || !maxSeqLen) {
    el.textContent = '⚙️ Estimated parameters: fill all fields';
    return;
  }

  if (dim % nHeads !== 0) {
    el.style.color = 'var(--danger, #e74c3c)';
    el.textContent = `⚠️ dim (${dim}) must be divisible by n_heads (${nHeads})`;
    return;
  }
  el.style.color = '';

  const embedParams  = vocabSize * dim + maxSeqLen * dim;
  const attnParams   = 4 * dim * dim;
  const ffnParams    = dim * (4 * dim) + (4 * dim) * dim;
  const normParams   = 2 * dim * 2 + dim * 2;
  const blockParams  = attnParams + ffnParams + normParams;
  const outputParams = dim * vocabSize;
  const total        = embedParams + nLayers * blockParams + outputParams;

  let label;
  if (total >= 1e9)      label = (total / 1e9).toFixed(2) + 'B';
  else if (total >= 1e6) label = (total / 1e6).toFixed(1) + 'M';
  else if (total >= 1e3) label = (total / 1e3).toFixed(0) + 'K';
  else                   label = total.toString();

  el.textContent = `⚙️ Estimated parameters: ${label}`;
}

// =========================================================================
// FSDP Check
// =========================================================================
function resetFsdpCheck() {
  const resultEl = document.getElementById('fsdp-check-result');
  if (resultEl) resultEl.style.display = 'none';
}

async function checkGpuAvailability() {
  const btn = document.getElementById('btn-gpu-check');
  const resultEl = document.getElementById('gpu-check-result');
  const verdictEl = document.getElementById('gpu-check-verdict');
  const pillEl = document.getElementById('gpu-check-pill');
  const breakdownEl = document.getElementById('gpu-check-breakdown');

  btn.disabled = true;
  btn.textContent = '⏳ Checking...';
  resultEl.style.display = 'none';

  try {
    const res = await fetch('/api/system/gpus');
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();

    if (!data.available) {
      verdictEl.textContent = 'No GPUs available';
      verdictEl.className = 'fsdp-verdict fsdp-unknown';
      pillEl.textContent = data.error || 'CUDA not available';
      breakdownEl.innerHTML = '';
    } else {
      verdictEl.textContent = `${data.count} GPU${data.count > 1 ? 's' : ''} detected`;
      verdictEl.className = 'fsdp-verdict fsdp-ok';
      pillEl.textContent = `CUDA ${data.cuda_version || '?'} · PyTorch ${data.pytorch_version || '?'}`;

      breakdownEl.innerHTML = data.gpus.map(g => {
        const usedPct = g.total_memory_gb > 0 ? Math.round((g.used_memory_gb / g.total_memory_gb) * 100) : 0;
        const barColor = usedPct > 80 ? '#ef4444' : usedPct > 50 ? '#f59e0b' : '#22c55e';
        return `<div class="fsdp-breakdown-row">
          <span class="fsdp-breakdown-label">🖥️ GPU ${g.index} — ${g.name}</span>
          <span class="fsdp-breakdown-value">${g.free_memory_gb} / ${g.total_memory_gb} GB free</span>
        </div>
        <div class="gpu-check-bar-row">
          <div class="gpu-check-bar-bg">
            <div class="gpu-check-bar-fill" style="width:${usedPct}%; background:${barColor};"></div>
          </div>
          <span class="gpu-check-bar-label">${usedPct}% used · SM ${g.compute_capability}</span>
        </div>`;
      }).join('');
    }

    resultEl.style.display = 'block';
  } catch (e) {
    verdictEl.textContent = 'Error';
    verdictEl.className = 'fsdp-verdict fsdp-unknown';
    pillEl.textContent = e.message;
    breakdownEl.innerHTML = '';
    resultEl.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = '🖥️ Check available GPUs';
  }
}

async function checkFsdpNeeded() {
  const btn = document.getElementById('btn-fsdp-check');
  const resultEl = document.getElementById('fsdp-check-result');
  const verdictEl = document.getElementById('fsdp-verdict');
  const reasonEl = document.getElementById('fsdp-reason');
  const breakdownEl = document.getElementById('fsdp-breakdown');

  btn.disabled = true;
  btn.textContent = '⏳ Checking...';
  resultEl.style.display = 'none';

  try {
    const cfg = buildConfigFromForm();
    const modelType = getVal('f-model-type');
    const usesSeqLen = modelType === 'llm' || modelType === 'vlm' || modelType === 'embedding';

    // Helper to show an early-exit warning in the result panel
    const showWarning = (msg) => {
      verdictEl.textContent = 'Missing Input';
      verdictEl.className = 'fsdp-verdict fsdp-unknown';
      const vramPillEl = document.getElementById('fsdp-vram-pill');
      if (vramPillEl) vramPillEl.textContent = '';
      reasonEl.textContent = msg;
      breakdownEl.innerHTML = '';
      resultEl.style.display = 'block';
    };

    // Must have a model type selected
    if (!modelType) {
      showWarning('Please select a Model Type before running this check.');
      return;
    }

    // Must have a model name for types where the estimate depends on it
    const modelName = (getVal('f-model-name') || '').trim();
    const requiresModelName = modelType === 'llm' || modelType === 'vlm' || modelType === 'vision' || modelType === 'embedding';
    if (requiresModelName && !modelName) {
      showWarning('Please enter a Model Name before running this check — the VRAM estimate is based on the number of parameters inferred from the model name.');
      return;
    }

    // Read Max Sequence Length from the form field and inject into training config
    // so the backend FSDP estimator uses the user-specified value.
    const rawSeqLen = getVal('f-max-seq-len');
    const seqLen = parseInt(rawSeqLen);

    if (usesSeqLen) {
      if (!rawSeqLen || isNaN(seqLen) || seqLen <= 0) {
        showWarning('Please specify Max Sequence Length before running this check — it significantly affects VRAM estimation for sequence-based models.');
        return;
      }
      // Ensure the value reaches the backend (which reads from training.max_length)
      if (!cfg.training) cfg.training = {};
      cfg.training.max_length = seqLen;
    }

    const res = await fetch('/api/fsdp-check', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: cfg }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();

    // Set verdict badge style
    verdictEl.textContent = data.verdict;
    verdictEl.className = 'fsdp-verdict';
    if (data.fsdp_needed === true && data.verdict && data.verdict.includes('more GPUs')) {
      verdictEl.classList.add('fsdp-warn');
    } else if (data.fsdp_needed === true) {
      verdictEl.classList.add('fsdp-needed');
    } else if (data.fsdp_needed === false) {
      verdictEl.classList.add('fsdp-ok');
    } else {
      verdictEl.classList.add('fsdp-unknown');
    }

    const vramPillEl = document.getElementById('fsdp-vram-pill');
    if (vramPillEl) {
      vramPillEl.textContent = data.gpu_memory_gb
        ? `${data.vram_needed_gb} GB needed · ${data.gpu_memory_gb} GB available`
        : `${data.vram_needed_gb} GB estimated`;
    }

    reasonEl.textContent = data.reason;

    // Build breakdown table
    const b = data.vram_breakdown || {};
    const infoRows = [
      { label: 'Model', value: data.model_params_b ? `~${data.model_params_b}B params` : '—', icon: '🧠' },
      { label: 'GPU', value: data.gpu_memory_gb ? `${data.gpu_name} · ${data.gpu_memory_gb} GB` : 'Not detected', icon: '🖥️' },
    ];
    const memRows = [
      { label: 'Weights', value: `${b.weights_gb} GB`, icon: '⚖️' },
      { label: 'Gradients', value: `${b.gradients_gb} GB`, icon: '📐' },
      { label: 'Optimizer', value: `${b.optimizer_gb} GB`, icon: '🔧' },
      { label: 'Activations', value: `${b.activations_gb} GB`, icon: '⚡' },
    ];
    const totalRows = [
      { label: 'Total VRAM', value: `${data.vram_needed_gb} GB`, icon: '📊' },
    ];
    if (data.fsdp_needed) {
      totalRows.push({ label: `FSDP per-GPU (${data.gpu_count || '?'} GPUs)`, value: `~${data.fsdp_per_gpu_gb} GB`, icon: '🔀' });
    }
    const renderRows = (rows, cls = '') =>
      rows.map(r => `<tr class="${cls}"><td class="fsdp-table-label">${r.icon} ${r.label}</td><td class="fsdp-table-value">${r.value}</td></tr>`).join('');
    breakdownEl.innerHTML = `
      <table class="fsdp-table">
        <thead><tr><th>Component</th><th>Value</th></tr></thead>
        <tbody>
          ${renderRows(infoRows)}
          <tr class="fsdp-table-section-divider"><td colspan="2"></td></tr>
          ${renderRows(memRows)}
          ${renderRows(totalRows, 'fsdp-table-total')}
        </tbody>
      </table>`;

    resultEl.style.display = 'block';
  } catch (e) {
    verdictEl.textContent = 'Error';
    verdictEl.className = 'fsdp-verdict fsdp-unknown';
    const vramPillElErr = document.getElementById('fsdp-vram-pill');
    if (vramPillElErr) vramPillElErr.textContent = '';
    reasonEl.textContent = e.message;
    breakdownEl.innerHTML = '';
    resultEl.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = '🔍 Distributed Training needed?';
  }
}

// =========================================================================
// Validation
// =========================================================================
async function validateConfig() {
  const cfg = buildConfigFromForm();

  try {
    const res = await fetch('/api/config/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: cfg }),
    });
    const data = await res.json();

    if (data.valid) {
      showMessage('Validation Successful', 'Configuration is valid!', 'success');
    } else {
      showMessage('Validation Error', data.error || 'Unknown error', 'error');
    }
  } catch (e) {
    showMessage('Validation Failed', e.message, 'error');
  }
}

// =========================================================================
// Environment Variables Page
// =========================================================================
function getFormStateValue(fieldId) {
  // First try to get from DOM (if on config page)
  const el = document.getElementById(fieldId);
  if (el) return el.value;

  // Otherwise get from localStorage (if on env page)
  const data = localStorage.getItem('omni_form_state');
  if (!data) return '';
  try {
    const formData = JSON.parse(data);
    return formData[fieldId] || '';
  } catch (e) {
    return '';
  }
}

function getRequiredApiKeys() {
  const keys = [];
  const dataSource = getFormStateValue('f-data-source');

  if (dataSource === 'kaggle') {
    keys.push({
      name: 'KAGGLE_USERNAME',
      description: 'Kaggle username',
      url: 'https://www.kaggle.com/settings'
    });
    keys.push({
      name: 'KAGGLE_KEY',
      description: 'Kaggle API key',
      url: 'https://www.kaggle.com/settings'
    });
  }
  return keys;
}

function generateEnvTemplate(keys) {
  if (keys.length === 0) return '';

  let template = '# OMNI-Train Environment Variables\n';
  template += '# Copy this to a .env file in your project root\n\n';

  const groupedKeys = {};
  keys.forEach(key => {
    const group = key.url || 'other';
    if (!groupedKeys[group]) groupedKeys[group] = [];
    groupedKeys[group].push(key);
  });

  Object.entries(groupedKeys).forEach(([url, groupKeys]) => {
    groupKeys.forEach((key, idx) => {
      template += `# ${key.description}\n`;
      if (idx === 0 && url !== 'other') {
        template += `# Get yours at: ${url}\n`;
      }
      template += `${key.name}=your_${key.name.toLowerCase()}_here\n\n`;
    });
  });

  return template.trim();
}

function showEnvPage() {
  loadFormState();
  const cfg = buildConfigFromForm();
  localStorage.setItem('yaml_from_form', 'true');
  localStorage.setItem('omni_yaml_config', JSON.stringify(cfg));
  saveFormState();
  navigateTo('yaml.html');
}

function proceedToEnvPage() {
  loadFormState();
  const cfg = buildConfigFromForm();
  localStorage.setItem('yaml_from_form', 'true');
  localStorage.setItem('omni_yaml_config', JSON.stringify(cfg));
  saveFormState();
  navigateTo('yaml.html');
}


async function initEnvPage() {
  loadFormState();
  const keys = getRequiredApiKeys();
  const container = document.getElementById('env-keys-container');
  if (!container) return;

  // Get the parent form-section to update title and description
  const formSection = container.closest('.form-section');
  const titleEl = formSection ? formSection.querySelector('h3') : null;
  const descEl = formSection ? formSection.querySelector('.env-page-description') : null;
  const privacyEl = formSection ? formSection.querySelector('.env-privacy-notice') : null;

  if (keys.length === 0) {
    // Update title and hide unnecessary elements for "no keys needed" state
    if (titleEl) titleEl.innerHTML = '<span class="icon" style="background: var(--green-dim); color: var(--green);">✅</span> No API Keys Required';
    if (descEl) descEl.textContent = 'Your configuration doesn\'t require any external API keys. You can proceed directly to the YAML editor.';
    if (privacyEl) privacyEl.style.display = 'none';

    container.innerHTML = `
      <div class="env-no-keys">
        <p>You're using local files or built-in datasets that don't require authentication.</p>
      </div>
    `;
    return;
  }

  // Try to load keys from the project's .env file
  let dotenvKeys = {};
  try {
    const keyNames = keys.map(k => k.name).join(',');
    const resp = await fetch(`/api/env?keys=${encodeURIComponent(keyNames)}`);
    if (resp.ok) {
      const data = await resp.json();
      dotenvKeys = data.keys || {};
    }
  } catch (e) {
    // .env fetch failed — fall through to manual entry
  }

  // If all required keys are present in .env, store and auto-proceed
  const allFound = keys.every(k => dotenvKeys[k.name]);
  if (allFound) {
    localStorage.setItem('env_api_keys', JSON.stringify(dotenvKeys));
    navigateTo('yaml.html');
    return;
  }

  // Merge .env values over any previously saved keys; show inputs for missing ones
  const savedKeys = JSON.parse(localStorage.getItem('env_api_keys') || '{}');
  const merged = Object.assign({}, savedKeys, Object.fromEntries(
    Object.entries(dotenvKeys).filter(([, v]) => v)
  ));

  const anyFromDotenv = keys.some(k => dotenvKeys[k.name]);
  if (anyFromDotenv && descEl) {
    descEl.textContent = 'Some keys were loaded from your .env file. Please fill in any missing ones.';
  }

  container.innerHTML = keys.map(key => `
    <div class="env-key-group">
      <label for="env-${key.name}">${key.name}</label>
      <div class="key-description">${key.description}</div>
      ${key.url ? `<a href="${key.url}" target="_blank" class="key-link">Get your key here →</a>` : ''}
      <input
        type="password"
        id="env-${key.name}"
        name="${key.name}"
        placeholder="Enter your ${key.name}..."
        value="${merged[key.name] || ''}"
        autocomplete="off"
      >
    </div>
  `).join('');
}

function getEnteredApiKeys() {
  const keys = getRequiredApiKeys();
  const enteredKeys = {};

  keys.forEach(key => {
    const input = document.getElementById(`env-${key.name}`);
    if (input && input.value.trim()) {
      enteredKeys[key.name] = input.value.trim();
    }
  });

  return enteredKeys;
}

function proceedToYaml() {
  const keys = getRequiredApiKeys();
  const enteredKeys = getEnteredApiKeys();

  // Check if all required keys are provided
  const missingKeys = keys.filter(key => !enteredKeys[key.name]);

  if (missingKeys.length > 0) {
    const missing = missingKeys.map(k => k.name).join(', ');
    showMessage('Missing API Keys', `Please enter the following keys: ${missing}`, 'warning');
    return;
  }

  // Store keys temporarily in localStorage (will be used to generate .env content)
  localStorage.setItem('env_api_keys', JSON.stringify(enteredKeys));

  navigateTo('yaml.html');
}

// =========================================================================
// YAML Editor
// =========================================================================
function generateYamlTemplate(config = null) {
  if (config) {
    return configToYaml(config);
  }

  return `# ===================== MODEL =====================
model_name: facebook/opt-350m  # Model name (e.g., opt-125m, opt-350m, llama-3-8b, gpt2 variants)

# ===================== DATASET =====================
dataset:
  name: wikitext              # Dataset name (wikitext, wikitext-103, PennTreebank, c4, squad)
  subset: wikitext-2-raw-v1   # Dataset subset (e.g., wikitext-2-raw-v1, wikitext-103-raw-v1)
  split: train[:1%]           # Data split (train, validation, or slice like train[:1%])

# ===================== TRAINING =====================
training:
  epochs: 4                     # Number of training epochs
  batch_size: 32                # Batch size per step
  max_length: 128               # Maximum sequence length
  learning_rate: 1e-5           # Optimizer learning rate
  warmup_steps: 100             # LR warmup steps before reaching target learning rate
  weight_decay: 0.01            # AdamW weight decay coefficient
  grad_clip: 1.0                # Max gradient norm for clipping (0 to disable)
  gradient_checkpointing: true  # Enable checkpointing to save memory (adds compute cost)

# ===================== SYSTEM / DISTRIBUTION =====================
strategy: fsdp        # Training mode: solo (1 GPU), ddp (multi-GPU), fsdp (sharded)
num_gpus: 1           # Number of GPUs (used with torchrun for ddp/fsdp)
checkpoint_dir: checkpoints  # Directory to save checkpoints
save: true            # Enable checkpoint saving

# ===================== FSDP / DISTRIBUTED SETTINGS =====================
dist_parameters:
  mixed_precision: true        # Enable mixed precision training
  param_dtype: bfloat16        # Parameter dtype (used mainly in FSDP)
  reduce_dtype: float32        # Gradient reduction dtype (FSDP)
  output_dtype: bfloat16       # Output dtype (FSDP)
  cast_forward_inputs: false   # Cast inputs to module dtype before forward pass
  dcp_api: true                # Use Distributed Checkpoint API (FSDP)
  dtensor_api: false           # Use DTensor API (experimental)

# ===================== CHECKPOINT LOAD / RESUME =====================
save_load:
  resume: false                # Resume training from checkpoint
  resume_path: ""              # Path to checkpoint (if resume=true)
  load_model_from_hf: true     # Load pretrained model from Hugging Face

# ===================== PREFETCH (FSDP PERFORMANCE) =====================
prefetch:
  explicit: true    # Enable explicit prefetching (FSDP only)
  forward: 2        # Number of forward prefetch steps
  backward: 2       # Number of backward prefetch steps

# ===================== PEFT (LoRA / QLoRA) =====================
peft:
  enabled: false              # Enable parameter-efficient fine-tuning
  type: lora                  # Adapter type: lora or qlora (qlora implies quantization)
  r: 16                       # LoRA rank (typical: 4, 8, 16)
  alpha: 32                   # LoRA scaling factor (usually 2–4× r)
  dropout: 0.05               # LoRA dropout rate
  target_modules: all-linear  # Target modules ("all-linear" or list of module names)
  bias: none                  # Bias mode: none, all, or lora_only

# ===================== QUANTIZATION (QLoRA ONLY) =====================
quantization:
  enabled: false              # Enable quantization (requires PEFT; not supported with FSDP)
  bits: 4                     # Quantization bits (4 or 8)
  quant_type: nf4             # 4-bit type: nf4 or fp4
  compute_dtype: bfloat16     # Compute dtype for quantized ops
  double_quant: true          # Enable nested (double) quantization

# ===================== LOGGING (W&B) =====================
wandb:
  wandb_log_with_train: false        # Enable Weights & Biases logging
  wandb_entity: "dist-train-project"  # W&B entity (user or team)
  wandb_project: "dist-train-project" # W&B project name
`;
}

function configToYaml(cfg) {
  const modelName = cfg.model?.name || 'facebook/opt-350m';
  const dataSource = cfg.data?._source || 'huggingface';
  const dataPath = cfg.data?._path || '';
  const dataUrl = cfg.data?._url || '';
  const dataName = dataSource === 'local' ? '' : (cfg.data?.name || 'wikitext');
  const dataSubset = cfg.data?.subset || 'wikitext-2-raw-v1';
  const dataSplit = cfg.data?.split || cfg.data?.train_split || 'train[:1%]';

  const epochs = cfg.training?.epochs ?? 4;
  const batchSize = cfg.training?.batch_size ?? 32;
  // For custom_transformer, the toy model has no tokenizer — its positional embedding
  // table caps seq_len at arch.max_seq_len, so the YAML's max_length must match it.
  const customArch = (cfg.model?.type || '').toLowerCase() === 'custom_transformer' ? (cfg.model?.arch || {}) : null;
  const maxLength = customArch?.max_seq_len ?? cfg.training?.max_length ?? cfg.data?.max_seq_len ?? 128;
  const learningRate = cfg.training?.learning_rate ?? cfg.training?.lr ?? '1e-5';
  const warmupSteps = cfg.training?.warmup_steps ?? 100;
  const weightDecay = cfg.training?.weight_decay ?? 0.01;
  const gradClip = cfg.training?.grad_clip ?? 1.0;
  const gradientCheckpointing = cfg.training?.gradient_checkpointing ?? cfg.distributed?.activation_checkpointing ?? true;

  let strategy = cfg.distributed?.strategy || cfg.strategy || 'fsdp';
  if (strategy === 'none') strategy = 'solo';
  const numGpus = cfg.num_gpus ?? 1;

  const checkpointDir = cfg.training?.checkpoint_dir || cfg.checkpoint_dir || 'checkpoints';
  const save = cfg.save ?? true;

  const mixedPrecision = cfg.distributed?.mixed_precision ?? cfg.dist_parameters?.mixed_precision ?? true;
  const paramDtype = cfg.dist_parameters?.param_dtype || 'bfloat16';
  const reduceDtype = cfg.dist_parameters?.reduce_dtype || 'float32';
  const outputDtype = cfg.dist_parameters?.output_dtype || 'bfloat16';
  const castForwardInputs = cfg.dist_parameters?.cast_forward_inputs ?? false;
  const dcpApi = cfg.dist_parameters?.dcp_api ?? true;
  const dtensorApi = cfg.dist_parameters?.dtensor_api ?? false;

  const resume = cfg.save_load?.resume ?? false;
  const resumePath = cfg.save_load?.resume_path ?? '';
  const loadModelFromHf = cfg.save_load?.load_model_from_hf ?? true;

  const explicit = cfg.prefetch?.explicit ?? true;
  const forward = cfg.prefetch?.forward ?? 2;
  const backward = cfg.prefetch?.backward ?? 2;

  const finetuneMode = (cfg.model?.finetune_mode || '').toLowerCase();
  const peftEnabled = cfg.peft?.enabled ?? (finetuneMode === 'lora' || finetuneMode === 'qlora') ?? true;
  const peftType = cfg.peft?.type || (finetuneMode === 'qlora' ? 'qlora' : 'lora') || 'lora';
  const peftR = cfg.peft?.r ?? cfg.model?.lora_r ?? 16;
  const peftAlpha = cfg.peft?.alpha ?? cfg.model?.lora_alpha ?? 32;
  const peftDropout = cfg.peft?.dropout ?? cfg.model?.lora_dropout ?? 0.05;
  const peftTargetModules = cfg.peft?.target_modules || 'all-linear';
  const peftBias = cfg.peft?.bias || 'none';

  const quantEnabled = cfg.quantization?.enabled ?? cfg.model?.quantize ?? (finetuneMode === 'qlora') ?? true;
  const quantBits = cfg.quantization?.bits ?? cfg.model?.quant_bits ?? 4;
  const quantType = cfg.quantization?.quant_type || 'nf4';
  const computeDtype = cfg.quantization?.compute_dtype || 'bfloat16';
  const doubleQuant = cfg.quantization?.double_quant ?? true;

  const wandbLog = cfg.wandb?.wandb_log_with_train ?? false;
  const wandbEntity = cfg.wandb?.wandb_entity || 'dist-train-project';
  const wandbProject = cfg.wandb?.wandb_project || 'dist-train-project';

  const datasetNameLine = dataSource === 'local'
    ? `  path: ${dataPath}  # local image folder`
    : dataSource === 'url'
    ? `  name: ${dataUrl}`
    : dataSource === 'torchvision'
    ? `  name: ${dataName}  # torchvision built-in dataset`
    : `  name: ${dataName} # other datasets: wikitext-103, PennTreebank, c4, squad`;
  const datasetSubsetLine = dataSource === 'local' || dataSource === 'torchvision' || dataSource === 'url'
    ? ''
    : `  subset: ${dataSubset} # other subsets: wikitext-103-raw-v1, PennTreebank (no subset)\n`;

  // Map UI model type to backend model_type (mirrors config_adapter.py _ui_to_model_type)
  const uiModelType = (cfg.model?.type || 'llm').toLowerCase();
  const _uiToMiniType = { llm: 'llm', vlm: 'vlm', vision: 'vision', embedding: 'encoder',
    detection: 'yolo', cnn: 'llm', custom_transformer: 'custom_transformer' };
  const miniModelType = _uiToMiniType[uiModelType] || 'llm';

  // Build model-type-specific extra lines for the YAML
  let modelTypeExtras = '';
  if (uiModelType === 'cnn') {
    const numClasses = cfg.model?.num_classes || 10;
    const imgSize = cfg.data?.image_size || 224;
    modelTypeExtras = `model_type: vision  # cnn model loaded via torchvision\nnum_classes: ${numClasses}\nimage_size: ${imgSize}\n`;
  } else if (uiModelType === 'vision') {
    const imgSize = cfg.data?.image_size || 224;
    modelTypeExtras = `model_type: ${miniModelType}\nimage_size: ${imgSize}\n`;
  } else if (uiModelType === 'vlm') {
    const imgSize = cfg.data?.image_size || 448;
    modelTypeExtras = `model_type: ${miniModelType}\nimage_size: ${imgSize}\n`;
  } else if (uiModelType === 'detection') {
    const imgSize = cfg.data?.image_size || 640;
    modelTypeExtras = `model_type: ${miniModelType}\nimage_size: ${imgSize}\n`;
  } else if (uiModelType !== 'llm') {
    modelTypeExtras = `model_type: ${miniModelType}\n`;
  }

  const modelNameLine = customArch
    ? ''
    : `model_name: ${modelName} # other models: facebook/opt-125m, facebook/opt-350m,  llama-3-8b, gpt2, gpt2-medium, gpt2-large, gpt2-xl\n`;
  return `${modelNameLine}${modelTypeExtras}
dataset:
${datasetNameLine}
${datasetSubsetLine}  split: ${dataSplit}  ##

training:
  epochs: ${epochs}                     ## Number of epochs to train
  batch_size: ${batchSize}                ## Batch size for training 
  max_length: ${maxLength}               ## Maximum sequence length for training
  learning_rate: ${learningRate}           ## Learning rate for training
  warmup_steps: ${warmupSteps}               ## LR warmup steps before target LR is reached
  weight_decay: ${weightDecay}               ## AdamW weight decay
  grad_clip: ${gradClip}               ## max gradient norm (set 0 to disable clipping)
  gradient_checkpointing: ${gradientCheckpointing}  ## whether to use gradient checkpointing

strategy: ${strategy} ## choose from {ddp, fsdp, solo}
num_gpus: ${numGpus} ## number of GPUs to launch with torchrun for ddp/fsdp
checkpoint_dir: ${checkpointDir} ## directory to save checkpoints
save: ${save}  ## whether to save checkpoints

dist_parameters:
  mixed_precision: ${mixedPrecision}       ## Use Mixed Precision? 
  param_dtype: ${paramDtype}       ## what dtype to use for parameters?  
  reduce_dtype: ${reduceDtype}       ## what dtype to use for reduction?  
  output_dtype: ${outputDtype}      ## what dtype to use for output? 
  cast_forward_inputs: ${castForwardInputs}  ## whether to cast forward inputs to the dtype of the wrapped module
  dcp_api: ${dcpApi}             ## whether to use DCP API
  dtensor_api: ${dtensorApi}        ## whether to use DTensor API

save_load:
  resume: ${resume}  ## whether to resume from a checkpoint
  resume_path: "${resumePath}"  ## path to checkpoint to resume from
  load_model_from_hf: ${loadModelFromHf}  ## whether to load  from HFace 

prefetch:
  explicit: ${explicit}  ## explicit prefetching?
  forward: ${forward}  ## number of forward passes to prefetch
  backward: ${backward}   ## number of backward passes to prefetch

peft: ## Parameter Efficient Fine Tuning
  enabled: ${peftEnabled}  ## whether to apply PEFT 
  type: ${peftType}  ## LoRA or QLoRA
  r: ${peftR}  ## rank for LoRA, often set to a small value like 4, 8, 16, etc.
  alpha: ${peftAlpha} ## scaling factor for LoRA, often set to r*2 or r*4
  dropout: ${peftDropout} ## dropout rate for LoRA layers
  target_modules: ${peftTargetModules} ## "all-linear" or list/comma-separated module names
  bias: ${peftBias} ## whether to include bias in LoRA, options: "none", "all", "lora_only"

quantization:
  enabled: ${quantEnabled}  ## whether to apply quantization
  bits: ${quantBits}  ## number of bits for quantization, typically 4 or 8
  quant_type: ${quantType} ## nf4 or fp4 for 4-bit
  compute_dtype: ${computeDtype} ## dtype for computations with quantized models
  double_quant: ${doubleQuant} ## whether to apply double quantization (for 4-bit)
  
wandb:
  wandb_log_with_train: ${wandbLog}  ## whether to log training metrics to Weights & Biases
  wandb_entity: "${wandbEntity}"   ## your W&B entity (username or team)
  wandb_project: "${wandbProject}"  ## your W&B project name
${customArch ? `
custom_transformer_args:
  n_layers: ${customArch.n_layers ?? 6}
  vocab_size: ${customArch.vocab_size ?? 8192}
  max_seq_len: ${customArch.max_seq_len ?? 512}
  dim: ${customArch.dim ?? 512}
  n_heads: ${customArch.n_heads ?? 8}
  dropout_p: ${customArch.dropout_p ?? 0.1}
` : ''}`;
}

function initYamlPage() {
  const fromForm = localStorage.getItem('yaml_from_form') === 'true';
  const editor = document.getElementById('yaml-editor');
  if (!editor) return;

  setupYamlCodeEditor();

  if (fromForm) {
    let yamlContent = null;
    const savedConfig = localStorage.getItem('omni_yaml_config');
    if (savedConfig) {
      try {
        const cfg = JSON.parse(savedConfig);
        yamlContent = configToYaml(cfg);
      } catch (e) {
        console.error('Failed to parse saved config:', e);
      }
    }
    setYamlEditorValue(yamlContent || generateYamlTemplate());
    yamlFromForm = true;
  } else {
    setYamlEditorValue(generateYamlTemplate());
    yamlFromForm = false;
  }

  // Keyboard shortcut: Ctrl/Cmd+Enter → Start Training
  document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      const yamlPage = document.querySelector('.yaml-page');
      if (yamlPage) startTrainingFromYaml();
    }
  });
}

function resetYamlTemplate() {
  const editor = document.getElementById('yaml-editor');
  if (!editor) return;
  setYamlEditorValue(generateYamlTemplate());
}

function parseYaml(yamlStr) {
  // Use js-yaml when available for accurate, spec-compliant parsing.
  if (typeof window.jsyaml !== 'undefined') {
    return window.jsyaml.load(yamlStr) || {};
  }
  // Fallback: hand-rolled parser (no js-yaml CDN loaded).
  const lines = yamlStr.split('\n');
  const result = {};
  const stack = [{ obj: result, indent: -1 }];

  for (let line of lines) {
    if (line.trim().startsWith('#') || line.trim() === '') continue;

    // Remove inline comments like "## note" and "# note" while preserving hashes inside quoted values.
    line = line.replace(/\s+#.*$/, '');
    if (line.trim() === '') continue;

    const match = line.match(/^(\s*)([^:]+):\s*(.*)$/);
    if (!match) {
      const arrayMatch = line.match(/^(\s*)-\s*(.+)$/);
      if (arrayMatch) {
        const indent = arrayMatch[1].length;
        const value = arrayMatch[2].trim();
        while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
          stack.pop();
        }
        const parent = stack[stack.length - 1];
        if (Array.isArray(parent.currentArray)) {
          parent.currentArray.push(value);
        }
      }
      continue;
    }

    const indent = match[1].length;
    const key = match[2].trim();
    let value = match[3].trim();

    while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
      stack.pop();
    }

    const parent = stack[stack.length - 1].obj;

    if (value === '') {
      parent[key] = {};
      stack.push({ obj: parent[key], indent: indent });
    } else {
      if (value === 'true') value = true;
      else if (value === 'false') value = false;
      else if (value === 'null') value = null;
      else if (!isNaN(value) && value !== '') value = parseFloat(value);

      parent[key] = value;

      const nextLineIdx = lines.indexOf(line) + 1;
      if (nextLineIdx < lines.length) {
        const nextLine = lines[nextLineIdx];
        if (nextLine.trim().startsWith('-')) {
          parent[key] = [];
          stack.push({ obj: parent, indent: indent, currentArray: parent[key] });
        }
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/**
 * Format a raw server-side validation error into a clean, readable message.
 * Strips Python exception class prefixes and trims overly long strings.
 */
function formatValidationError(raw) {
  let msg = String(raw).replace(/^(ValueError|TypeError|KeyError|AttributeError|RuntimeError|Exception):\s*/i, '');
  // Collapse any nested "caused by" chain to just the outermost message.
  const firstLine = msg.split('\n')[0].trim();
  msg = firstLine || msg;
  if (msg.length > 300) msg = msg.slice(0, 297) + '…';
  return msg;
}

/**
 * Show a rich, field-by-field summary of a successfully validated config.
 * Safely builds the content with textContent (no innerHTML).
 */
function showValidationSuccess(cfg) {
  // Support both UI-format (model/distributed/data) and mini-project format.
  const model    = (typeof cfg.model === 'object' && cfg.model)    || {};
  const dist     = (typeof cfg.distributed === 'object' && cfg.distributed) || {};
  const training = (typeof cfg.training === 'object' && cfg.training) || {};
  const data     = (typeof cfg.data === 'object' && cfg.data)
                || (typeof cfg.dataset === 'object' && cfg.dataset) || {};

  const modelName  = model.name        || cfg.model_name          || '—';
  const strategy   = (dist.strategy    || cfg.strategy            || 'solo').toUpperCase();
  const gpus       = dist.gpu_count    || cfg.num_gpus            || 1;
  const epochs     = training.epochs                              || '—';
  const batchSize  = training.batch_size                          || '—';
  const lr         = training.learning_rate || training.lr        || '—';
  const dataName   = data.name         || cfg.dataset?.name       || '—';
  const ftMode     = model.finetune_mode
                  || (cfg.peft?.enabled ? (cfg.peft.type || 'lora') : 'full');

  const pad = (s, n) => String(s).padEnd(n);
  const lines = [
    `${pad('Model',     12)}${modelName}`,
    `${pad('Strategy',  12)}${strategy}  ·  ${gpus} GPU${gpus > 1 ? 's' : ''}`,
    `${pad('Dataset',   12)}${dataName}`,
    `${pad('Training',  12)}${epochs} epoch(s)  ·  batch ${batchSize}  ·  lr ${lr}`,
    `${pad('Fine-tune', 12)}${ftMode}`,
  ];

  const iconEl  = document.getElementById('message-modal-icon');
  const titleEl = document.getElementById('message-modal-title');
  const textEl  = document.getElementById('message-modal-text');
  const modal   = document.getElementById('message-modal');

  if (iconEl)  iconEl.textContent  = 'check';
  if (titleEl) titleEl.textContent = 'Validation Passed';
  if (textEl) {
    textEl.textContent = lines.join('\n');
    textEl.classList.add('validation-summary');
  }
  if (modal) modal.classList.add('active');
}

async function validateYaml() {
  const yamlStr = getYamlEditorValue().trim();
  if (!yamlStr) {
    showMessage('Empty Configuration', 'The editor is empty. Please enter a YAML configuration.', 'warning');
    return;
  }

  // Button loading state
  const btn = document.querySelector('.yaml-action-bar .btn-secondary');
  const origLabel = btn ? btn.textContent : null;
  if (btn) { btn.disabled = true; btn.textContent = 'Validating…'; }

  // ── Step 1: client-side YAML syntax check ───────────────────────────────
  let cfg;
  try {
    cfg = parseYaml(yamlStr);
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = origLabel; }
    // js-yaml provides e.mark with 0-based line/column
    const loc = (e.mark != null)
      ? ` at line ${e.mark.line + 1}, column ${e.mark.column + 1}`
      : '';
    const reason = e.reason || e.message || 'Syntax error';
    showMessage('YAML Syntax Error', `${reason}${loc}`, 'error');
    return;
  }

  if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) {
    if (btn) { btn.disabled = false; btn.textContent = origLabel; }
    showMessage('Invalid YAML', 'The configuration must be a YAML mapping (key: value pairs), not a scalar or list.', 'error');
    return;
  }

  // ── Step 2: server-side schema + logic validation ────────────────────────
  try {
    const res = await fetch('/api/config/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: cfg }),
    });
    const data = await res.json();
    if (btn) { btn.disabled = false; btn.textContent = origLabel; }

    if (data.valid) {
      showValidationSuccess(cfg);
    } else {
      showMessage('Validation Error', formatValidationError(data.error || 'Unknown error'), 'error');
    }
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = origLabel; }
    showMessage('Connection Error', 'Could not reach the server. Make sure the app is running.', 'error');
  }
}

// Track if we're currently polling on yaml page
let yamlPollInterval = null;

// Loss tracking for chart
let epochLossPoints = [];   // [{epoch, loss}] from "Epoch N complete | avg loss: X"
let stepLossPoints  = [];   // [{step, loss}]  from step-level progress lines

async function startTrainingFromYaml() {
  const editor = document.getElementById('yaml-editor');
  if (!editor) return;

  const yamlStr = getYamlEditorValue();
  let cfg;
  try {
    cfg = parseYaml(yamlStr);
  } catch (e) {
    showMessage('YAML Syntax Error', e.message || String(e), 'error');
    return;
  }

  // Reset loss accumulators for this run
  epochLossPoints = [];
  stepLossPoints  = [];

  startedFromYaml = true;
  const strategy = cfg.distributed?.strategy || 'none';
  showTrainingOverlay(strategy);
  startTrainingTimer(cfg);

  try {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ config: cfg }),
    });

    if (!res.ok) {
      hideTrainingOverlay();
      stopTrainingTimer();
      const err = await res.json().catch(() => ({}));
      showMessage('Training Error', err.detail || 'Failed to start training', 'error');
      return;
    }

    // Start polling with setInterval (doesn't block)
    startYamlPagePolling();

  } catch (e) {
    hideTrainingOverlay();
    stopTrainingTimer();
    showMessage('Training Error', e.message || String(e), 'error');
  }
}

function startYamlPagePolling() {
  // Clear any existing interval
  if (yamlPollInterval) {
    clearInterval(yamlPollInterval);
  }

  // Poll every second
  yamlPollInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/train/status');
      const data = await res.json();

      const title = document.getElementById('training-title');
      const subtitle = document.getElementById('training-subtitle');
      const progressFill = document.getElementById('progress-fill');
      const progressText = document.getElementById('progress-text');

      console.log('Training status:', data.status, 'Logs:', data.logs?.length || 0);
      renderOverlayLogs(data.logs || []);

      if (data.status === 'running') {
        if (title) title.textContent = 'Training in Progress...';

        // Show last log line in subtitle
        if (data.logs && data.logs.length > 0) {
          const lastLog = data.logs[data.logs.length - 1];
          if (subtitle) subtitle.textContent = lastLog.substring(0, 100);

          // Parse epoch-level losses: "Epoch N complete | avg loss: X.XXXX"
          for (const line of data.logs) {
            const em = line.match(/Epoch (\d+) complete \| avg loss: ([\d.]+)/);
            if (em) {
              const ep = parseInt(em[1]);
              const lv = parseFloat(em[2]);
              if (!epochLossPoints.find(p => p.epoch === ep)) {
                epochLossPoints.push({ epoch: ep, loss: lv });
              }
            }
          }

          // Parse step-level losses from progress bar lines when no epoch points yet
          if (epochLossPoints.length === 0) {
            for (const line of data.logs) {
              const sm = line.match(/step (\d+)\/\d+ \| batch_loss: [\d.]+ \| avg: ([\d.]+)/);
              if (sm) {
                const st = parseInt(sm[1]);
                const lv = parseFloat(sm[2]);
                if (!stepLossPoints.find(p => p.step === st)) {
                  stepLossPoints.push({ step: st, loss: lv });
                }
              }
            }
          }

          // Try to parse epoch progress
          const epochMatch = lastLog.match(/Epoch (\d+)\/(\d+)/);
          if (epochMatch) {
            const current = parseInt(epochMatch[1]);
            const total = parseInt(epochMatch[2]);
            const progress = (current / total) * 100;
            if (progressFill) progressFill.style.width = progress + '%';
            if (progressText) progressText.textContent = `Epoch ${current} of ${total}`;
          }

          // Update timer from logs
          updateTimerFromLogs(data.logs);
        }

      } else if (data.status === 'error' || data.status === 'stopped') {
        stopYamlPagePolling();
        stopTrainingTimer();
        showTrainingError(data.logs || [], data.error_summary, data.exit_code);

      } else if (data.status === 'finished') {
        stopYamlPagePolling();
        hideTrainingOverlay();
        stopTrainingTimer();

        showTrainingSuccess(data.logs || []);

      } else if (data.status === 'idle') {
        // Check if we have error logs (training failed very quickly)
        if (data.logs && data.logs.length > 0) {
          const hasError = data.logs.some(l =>
            l.includes('Error') || l.includes('Traceback') || l.includes('Exception') || l.includes('exited with code')
          );
          if (hasError) {
            stopYamlPagePolling();
            stopTrainingTimer();
            showTrainingError(data.logs, data.error_summary, data.exit_code);
          }
        }
      }

    } catch (e) {
      console.error('Poll error:', e);
    }
  }, 1000);
}

function stopYamlPagePolling() {
  if (yamlPollInterval) {
    clearInterval(yamlPollInterval);
    yamlPollInterval = null;
  }
}

/**
 * Extract the most meaningful error from a list of log lines.
 * Returns { headline, contextLines } where headline is the exception message
 * and contextLines is the relevant traceback/error block.
 */
function extractTrainingError(logs, serverErrorSummary) {
  // Prefer the pre-extracted summary from the backend
  const headline = serverErrorSummary && serverErrorSummary.trim() ? serverErrorSummary.trim() : null;

  // Find the last "Traceback (most recent call last)" in all logs
  let tracebackStart = -1;
  for (let i = logs.length - 1; i >= 0; i--) {
    if (logs[i].includes('Traceback (most recent call last)')) {
      tracebackStart = i;
      break;
    }
  }

  if (tracebackStart !== -1) {
    // Show from traceback start to end (up to 80 lines)
    const contextLines = logs.slice(tracebackStart, tracebackStart + 80);
    // If no backend headline, derive it from the last non-indented line in the traceback
    if (!headline) {
      for (let i = contextLines.length - 1; i >= 0; i--) {
        const line = contextLines[i].trim();
        if (line && !/^\s/.test(contextLines[i]) && contextLines[i] !== contextLines[tracebackStart]) {
          return { headline: line, contextLines };
        }
      }
    }
    return { headline, contextLines };
  }

  // No traceback — look for any line with a known error keyword
  const errorPattern = /Error:|Exception:|CUDA error|RuntimeError|ValueError|TypeError|KeyError|ImportError|ModuleNotFoundError/i;
  let fallbackHeadline = headline;
  let fallbackContext = logs.slice(-30);
  if (!fallbackHeadline) {
    for (let i = logs.length - 1; i >= 0; i--) {
      if (errorPattern.test(logs[i])) {
        fallbackHeadline = logs[i].trim();
        // Show surrounding context
        fallbackContext = logs.slice(Math.max(0, i - 5), Math.min(logs.length, i + 15));
        break;
      }
    }
  }
  return { headline: fallbackHeadline, contextLines: fallbackContext };
}

function showTrainingError(logs, serverErrorSummary, exitCode) {
  const { headline } = extractTrainingError(logs, serverErrorSummary);

  // Turn the existing overlay red — keep it open with all output visible
  const modal = document.querySelector('.training-modal');
  if (modal) modal.classList.add('training-modal--error');

  // Replace spinner/animation with a failure icon
  const animation = document.getElementById('training-animation');
  if (animation) {
    animation.innerHTML = '<div style="font-size:56px;line-height:1;">✖</div>';
  }

  // Update title
  const title = document.getElementById('training-title');
  const titleSuffix = exitCode != null ? ` (exit ${exitCode})` : '';
  if (title) {
    title.textContent = `Training Failed${titleSuffix}`;
    title.style.color = '#e74c3c';
  }

  // Show headline under the title if we have one
  const subtitle = document.getElementById('training-subtitle');
  if (subtitle && headline) {
    subtitle.textContent = headline;
    subtitle.style.color = '#ff8080';
    subtitle.style.fontFamily = 'monospace';
    subtitle.style.fontSize = '12px';
  } else if (subtitle) {
    subtitle.textContent = 'See console output above for details.';
    subtitle.style.color = '#e74c3c';
  }

  // Hide progress bar and timer
  const progress = document.querySelector('.training-progress');
  if (progress) progress.style.display = 'none';
  const timer = document.getElementById('training-timer');
  if (timer) timer.style.display = 'none';

  // Swap the Stop button into a Close button
  const btnStop = document.getElementById('btn-stop-training');
  if (btnStop) {
    btnStop.textContent = 'Close';
    btnStop.style.background = '#e74c3c';
    btnStop.onclick = hideTrainingOverlay;
  } else {
    // Fallback: add a Close button after the live log panel
    const livePanel = document.querySelector('.training-live-panel');
    if (livePanel && !document.getElementById('training-error-close-btn')) {
      const closeBtn = document.createElement('button');
      closeBtn.id = 'training-error-close-btn';
      closeBtn.className = 'btn btn-danger';
      closeBtn.textContent = 'Close';
      closeBtn.style.marginTop = '20px';
      closeBtn.onclick = hideTrainingOverlay;
      livePanel.insertAdjacentElement('afterend', closeBtn);
    }
  }
}

function copyTrainingError(btn) {
  const pre = document.getElementById('training-error-pre');
  if (!pre) return;
  navigator.clipboard.writeText(pre.textContent).then(() => {
    btn.textContent = '✅ Copied';
    setTimeout(() => { btn.textContent = '📋 Copy'; }, 2000);
  }).catch(() => {
    btn.textContent = '❌ Failed';
    setTimeout(() => { btn.textContent = '📋 Copy'; }, 2000);
  });
}

function extractWandbUrl(logs) {
  for (const line of logs) {
    const m = line.match(/https:\/\/wandb\.ai\/\S+\/runs\/\S+/);
    if (m) return m[0];
  }
  return null;
}

function buildLossChartSvg(points, xKey, xLabel) {
  if (!points || points.length < 2) return '';

  const W = 460, H = 200;
  const pad = { top: 18, right: 20, bottom: 44, left: 52 };
  const iW = W - pad.left - pad.right;
  const iH = H - pad.top  - pad.bottom;

  const xs = points.map(p => p[xKey]);
  const ys = points.map(p => p.loss);
  const xMin = xs[0], xMax = xs[xs.length - 1];
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const yRange = yMax - yMin || 1;

  const px = x => pad.left + ((x - xMin) / (xMax - xMin || 1)) * iW;
  const py = y => pad.top  + (1 - (y - yMin) / yRange) * iH;

  // Build polyline
  const pts = points.map(p => `${px(p[xKey]).toFixed(1)},${py(p.loss).toFixed(1)}`).join(' ');

  // Area fill path
  const areaPath = [
    `M ${px(xs[0]).toFixed(1)},${(pad.top + iH).toFixed(1)}`,
    ...points.map(p => `L ${px(p[xKey]).toFixed(1)},${py(p.loss).toFixed(1)}`),
    `L ${px(xs[xs.length-1]).toFixed(1)},${(pad.top + iH).toFixed(1)}`,
    'Z'
  ].join(' ');

  // Y-axis ticks (5 levels)
  const yTicks = Array.from({length: 5}, (_, i) => yMin + (yRange * i / 4));
  const yTickLines = yTicks.map(v => {
    const y = py(v).toFixed(1);
    return `
      <line x1="${pad.left}" y1="${y}" x2="${pad.left + iW}" y2="${y}" stroke="#2a2a4a" stroke-width="1"/>
      <text x="${pad.left - 6}" y="${y}" fill="#888" font-size="9" text-anchor="end" dominant-baseline="middle">${v.toFixed(3)}</text>`;
  }).join('');

  // X-axis ticks (up to 8)
  const step = Math.max(1, Math.ceil(points.length / 8));
  const xTickLines = points.filter((_, i) => i % step === 0 || i === points.length - 1).map(p => {
    const x = px(p[xKey]).toFixed(1);
    return `
      <line x1="${x}" y1="${pad.top + iH}" x2="${x}" y2="${pad.top + iH + 4}" stroke="#555" stroke-width="1"/>
      <text x="${x}" y="${pad.top + iH + 14}" fill="#888" font-size="9" text-anchor="middle">${p[xKey]}</text>`;
  }).join('');

  // Dots on data points
  const dots = points.map(p =>
    `<circle cx="${px(p[xKey]).toFixed(1)}" cy="${py(p.loss).toFixed(1)}" r="3" fill="#4ade80" stroke="#1a1a2e" stroke-width="1.5"/>`
  ).join('');

  return `
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px;display:block;margin:0 auto;">
    <defs>
      <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%"  stop-color="#4ade80" stop-opacity="0.25"/>
        <stop offset="100%" stop-color="#4ade80" stop-opacity="0"/>
      </linearGradient>
    </defs>
    <rect width="${W}" height="${H}" fill="#0d0d1a" rx="8"/>
    ${yTickLines}
    ${xTickLines}
    <path d="${areaPath}" fill="url(#lossGrad)"/>
    <polyline points="${pts}" fill="none" stroke="#4ade80" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
    ${dots}
    <text x="${pad.left + iW / 2}" y="${H - 4}" fill="#666" font-size="10" text-anchor="middle">${xLabel}</text>
    <text x="12" y="${pad.top + iH / 2}" fill="#666" font-size="10" text-anchor="middle" transform="rotate(-90,12,${pad.top + iH / 2})">Loss</text>
    <text x="${pad.left + iW / 2}" y="12" fill="#aaa" font-size="11" text-anchor="middle" font-weight="600">Training Loss</text>
  </svg>`;
}

function showTrainingSuccess(logs = []) {
  const container = document.querySelector('.yaml-page') || document.body;

  const wandbUrl = extractWandbUrl(logs);

  // Final pass to catch any epoch losses that arrived in the last batch of logs
  for (const line of logs) {
    const em = line.match(/Epoch (\d+) complete \| avg loss: ([\d.]+)/);
    if (em) {
      const ep = parseInt(em[1]);
      const lv = parseFloat(em[2]);
      if (!epochLossPoints.find(p => p.epoch === ep)) {
        epochLossPoints.push({ epoch: ep, loss: lv });
      }
    }
  }
  epochLossPoints.sort((a, b) => a.epoch - b.epoch);
  stepLossPoints.sort((a, b) => a.step - b.step);

  // Choose which points to chart: prefer epoch-level; fall back to step-level
  const chartPoints = epochLossPoints.length >= 2 ? epochLossPoints
                    : stepLossPoints.length  >= 2 ? stepLossPoints
                    : null;
  const chartSvg = chartPoints
    ? buildLossChartSvg(
        chartPoints,
        epochLossPoints.length >= 2 ? 'epoch' : 'step',
        epochLossPoints.length >= 2 ? 'Epoch'  : 'Step'
      )
    : '';

  const successDiv = document.createElement('div');
  successDiv.id = 'training-error-display';
  successDiv.style.cssText = `
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #1a1a2e;
    border: 2px solid #27ae60;
    border-radius: 12px;
    padding: 24px 28px;
    z-index: 10000;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    text-align: center;
    min-width: 340px;
    max-width: 90vw;
  `;

  // Get checkpoint directory from config
  const checkpointDir = (typeof getVal === 'function' ? getVal('f-checkpoint-dir') : '') || 'checkpoints';
  const strategy = (typeof getVal === 'function' ? getVal('f-strategy') : '') || 'none';
  const strategyFolder = strategy === 'fsdp' ? 'fsdp/dcp_api' : (strategy === 'ddp' ? 'ddp' : 'solo');
  const fullCheckpointPath = `${checkpointDir}/${strategyFolder}/`;

  const checkpointSection = `
    <div style="margin: 12px 0; padding: 12px 16px; background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 8px; text-align: left;">
      <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
        <span style="font-size: 16px;">💾</span>
        <span style="color: #a5b4fc; font-weight: 600; font-size: 13px;">Model saved to:</span>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <code id="checkpoint-path-text" style="flex: 1; background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 6px; font-size: 12px; color: #e0e7ff; word-break: break-all; font-family: 'JetBrains Mono', 'Fira Code', monospace;">${fullCheckpointPath}</code>
        <button onclick="openCheckpointFolder()" style="padding: 8px 12px; background: rgba(99, 102, 241, 0.5); border: 1px solid rgba(99, 102, 241, 0.7); border-radius: 6px; color: #fff; cursor: pointer; font-size: 12px; white-space: nowrap; font-weight: 500;" title="Open checkpoints">📂 Open</button>
      </div>
    </div>`;

  const wandbSection = wandbUrl ? `
    <div style="margin: 12px 0; padding: 10px 16px; background: rgba(255,149,0,0.08); border: 1px solid rgba(255,149,0,0.25); border-radius: 8px; display: flex; align-items: center; gap: 10px; justify-content: center;">
      <span style="font-size: 18px;">📊</span>
      <a href="${wandbUrl}" target="_blank" rel="noopener noreferrer"
        style="color: #f59e0b; font-weight: 600; font-size: 13px; text-decoration: none; word-break: break-all;"
        onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">
        View run on Weights &amp; Biases ↗
      </a>
    </div>` : '';

  const chartSection = chartSvg ? `
    <div style="margin: 14px 0 4px 0; border-radius: 8px; overflow: hidden;">
      ${chartSvg}
    </div>` : '';

  successDiv.innerHTML = `
    <h3 style="color: #27ae60; margin: 0 0 8px 0; font-size: 18px;">✅ Training Completed Successfully!</h3>
    ${checkpointSection}
    ${wandbSection}
    ${chartSection}
    <button onclick="closeTrainingError()" style="margin-top: 14px; padding: 10px 24px; background: #27ae60; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px;">Close</button>
  `;

  container.appendChild(successDiv);

  const backdrop = document.createElement('div');
  backdrop.id = 'training-error-backdrop';
  backdrop.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.7);
    z-index: 9999;
  `;
  backdrop.onclick = closeTrainingError;
  container.appendChild(backdrop);
}

function closeTrainingError() {
  const modal   = document.getElementById('training-error-display');
  const backdrop = document.getElementById('training-error-backdrop');
  if (modal)   modal.remove();
  if (backdrop) backdrop.remove();
  hideTrainingOverlay();
}

function openCheckpointFolder() {
  const pathEl = document.getElementById('checkpoint-path-text');
  if (!pathEl) return;
  const path = pathEl.textContent.replace(/\/+$/, ''); // Remove trailing slash
  const dir = path.split('/')[0] || 'checkpoints'; // Get base checkpoint dir
  closeTrainingError();
  window.location.href = `/static/checkpoints.html?dir=${encodeURIComponent(dir)}`;
}

let _toastTimer = null;
function showToast(message, type = 'info') {
  let el = document.getElementById('app-toast');
  if (!el) {
    el = document.createElement('div');
    el.id = 'app-toast';
    el.className = 'toast';
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.className = `toast toast--${type}`;
  // Force reflow so the transition fires even on rapid repeat calls
  void el.offsetWidth;
  el.classList.add('toast--visible');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('toast--visible'), 2200);
}

// =========================================================================
// Training
// =========================================================================
async function startTraining() {
  // Use queue-based submission
  await submitJobToQueue();
}

async function stopTraining() {
  try {
    // If we have a current job ID, cancel it via queue
    if (currentJobId) {
      await fetch(`/api/queue/jobs/${currentJobId}/cancel`, { method: 'POST' });
      stopJobPolling();
      currentJobId = null;
    } else {
      // Fallback to legacy stop
      await fetch('/api/train/stop', { method: 'POST' });
    }
    hideTrainingOverlay();
    stopTrainingTimer();
    isJobQueued = false;
  } catch (e) {
    console.error('Failed to stop:', e);
  }
}

function showTrainingOverlay(strategy, gpuCount = null) {
  const overlay = document.getElementById('training-overlay');
  const animation = document.getElementById('training-animation');
  const title = document.getElementById('training-title');
  const subtitle = document.getElementById('training-subtitle');
  const queueBox = document.getElementById('queue-status-box');
  const btnStop = document.getElementById('btn-stop-training');
  const btnCancel = document.getElementById('btn-cancel-queued');
  const progressContainer = document.querySelector('.training-progress');
  const timerContainer = document.getElementById('training-timer');

  if (!overlay) return;

  // Reset error state from previous run
  const modal = document.querySelector('.training-modal');
  if (modal) modal.classList.remove('training-modal--error');
  if (title) title.style.color = '';
  if (subtitle) { subtitle.style.color = ''; subtitle.style.fontFamily = ''; subtitle.style.fontSize = ''; }
  const oldCloseBtn = document.getElementById('training-error-close-btn');
  if (oldCloseBtn) oldCloseBtn.remove();

  // Reset to training mode (not queued)
  if (queueBox) queueBox.style.display = 'none';
  if (progressContainer) progressContainer.style.display = 'block';
  if (timerContainer) timerContainer.style.display = 'block';
  if (btnStop) { btnStop.style.display = 'inline-block'; btnStop.textContent = 'Stop Training'; btnStop.style.background = ''; btnStop.onclick = stopTraining; }
  if (btnCancel) btnCancel.style.display = 'none';

  // Use provided gpuCount or infer from strategy
  const numGpus = gpuCount || (strategy === 'fsdp' ? 4 : (strategy === 'ddp' ? 2 : 1));

  if (numGpus > 1) {
    if (animation) {
      animation.innerHTML = `
        <div class="gpu-animation">
          ${Array.from({length: numGpus}, (_, i) => `
            <div class="gpu-box" data-id="${i}">
              <div class="pulse"></div>
            </div>
          `).join('')}
        </div>
        <div class="connection-lines">
          ${Array.from({length: numGpus - 1}, () => '<div class="connection-line"></div>').join('')}
        </div>
      `;
    }
    if (title) title.textContent = strategy === 'fsdp' ? 'Starting FSDP Training...' : 'Starting DDP Training...';
    if (subtitle) subtitle.textContent = `Synchronizing ${numGpus} GPUs`;
  } else {
    if (animation) animation.innerHTML = '<div class="spinner"></div>';
    if (title) title.textContent = 'Starting Training...';
    if (subtitle) subtitle.textContent = 'Initializing model and data loaders';
  }

  renderOverlayLogs([]);
  overlay.classList.add('active');
}

function hideTrainingOverlay() {
  const overlay = document.getElementById('training-overlay');
  if (overlay) overlay.classList.remove('active');
  // Reset error state so next run starts clean
  const modal = document.querySelector('.training-modal');
  if (modal) modal.classList.remove('training-modal--error');
  const title = document.getElementById('training-title');
  if (title) title.style.color = '';
  const subtitle = document.getElementById('training-subtitle');
  if (subtitle) { subtitle.style.color = ''; subtitle.style.fontFamily = ''; subtitle.style.fontSize = ''; }
  const closeBtn = document.getElementById('training-error-close-btn');
  if (closeBtn) closeBtn.remove();
}

// =========================================================================
// Queue Management
// =========================================================================

/**
 * Fetch GPU availability and queue status from backend
 */
async function fetchQueueStatus() {
  try {
    const res = await fetch('/api/queue/status');
    if (res.ok) {
      return await res.json();
    }
  } catch (e) {
    console.error('Failed to fetch queue status:', e);
  }
  return null;
}

/**
 * Update GPU availability display
 */
async function updateGpuAvailability() {
  const status = await fetchQueueStatus();
  if (!status) return;

  const el = document.getElementById('gpu-availability');
  if (el) {
    const available = status.available_gpus;
    const total = status.total_gpus;
    const statusClass = available > 0 ? 'available' : 'busy';
    el.innerHTML = `<span class="gpu-status-text ${statusClass}">${available} of ${total} GPUs available</span>`;

    // Update GPU count dropdown to show which options will queue
    const gpuSelect = document.getElementById('f-gpu-count');
    if (gpuSelect) {
      Array.from(gpuSelect.options).forEach(opt => {
        const count = parseInt(opt.value);
        if (count > available) {
          opt.text = `${count} GPU${count > 1 ? 's' : ''} (will queue)`;
        } else {
          opt.text = `${count} GPU${count > 1 ? 's' : ''}`;
        }
      });
    }
  }

  // Update queue indicator
  updateQueueIndicator(status);
}

/**
 * Update queue indicator in header
 */
function updateQueueIndicator(status) {
  const indicator = document.getElementById('queue-indicator');
  const text = document.getElementById('queue-text');

  if (!indicator || !text) return;

  if (status.pending_jobs > 0 || status.running_jobs > 0) {
    indicator.style.display = 'flex';
    text.textContent = `${status.running_jobs} running, ${status.pending_jobs} queued`;
  } else {
    indicator.style.display = 'none';
  }
}

/**
 * Submit job to queue instead of direct start
 */
async function submitJobToQueue() {
  const cfg = buildConfigFromForm();
  const strategy = getVal('f-strategy');
  const gpuCount = parseInt(cfg.num_gpus || 1);

  try {
    const res = await fetch('/api/queue/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: cfg,
        gpu_count: gpuCount,
        priority: 0
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(err.detail || 'Failed to submit job');
      return null;
    }

    const data = await res.json();
    currentJobId = data.job_id;

    if (data.status === 'running') {
      // Job started immediately
      isJobQueued = false;
      showTrainingOverlay(strategy, gpuCount);
      startTrainingTimer(cfg);
      switchSideTab('logs');
      startJobPolling(data.job_id);
    } else {
      // Job is queued
      isJobQueued = true;
      showQueuedOverlay(data, gpuCount);
      startJobPolling(data.job_id);
    }

    return data;
  } catch (e) {
    alert('Failed to submit job: ' + e.message);
    return null;
  }
}

/**
 * Show overlay for queued job
 */
function showQueuedOverlay(jobData, gpuCount) {
  const overlay = document.getElementById('training-overlay');
  const animation = document.getElementById('training-animation');
  const title = document.getElementById('training-title');
  const subtitle = document.getElementById('training-subtitle');
  const queueBox = document.getElementById('queue-status-box');
  const positionEl = document.getElementById('queue-position-value');
  const waitEl = document.getElementById('queue-wait-value');
  const btnStop = document.getElementById('btn-stop-training');
  const btnCancel = document.getElementById('btn-cancel-queued');
  const progressContainer = document.querySelector('.training-progress');
  const timerContainer = document.getElementById('training-timer');

  if (!overlay) return;

  // Show queue animation
  if (animation) {
    animation.innerHTML = `
      <div class="queue-animation">
        <div class="queue-icon-large">&#128203;</div>
        <div class="queue-dots">
          <span class="dot"></span>
          <span class="dot"></span>
          <span class="dot"></span>
        </div>
      </div>
    `;
  }

  if (title) title.textContent = 'Job Queued';
  if (subtitle) subtitle.textContent = `Waiting for ${gpuCount} GPU${gpuCount > 1 ? 's' : ''} to become available`;

  // Show queue status box
  if (queueBox) queueBox.style.display = 'flex';
  if (positionEl) positionEl.textContent = `#${jobData.queue_position || '?'}`;
  if (waitEl) waitEl.textContent = formatTime(jobData.estimated_wait || 0);

  // Hide progress and timer for queued jobs
  if (progressContainer) progressContainer.style.display = 'none';
  if (timerContainer) timerContainer.style.display = 'none';

  // Show cancel button instead of stop
  if (btnStop) btnStop.style.display = 'none';
  if (btnCancel) btnCancel.style.display = 'inline-block';

  renderOverlayLogs([
    `Job queued for ${gpuCount} GPU${gpuCount > 1 ? 's' : ''}`,
    `Queue position: #${jobData.queue_position || '?'}`,
    `Estimated wait: ${formatTime(jobData.estimated_wait || 0)}`,
  ]);
  overlay.classList.add('active');
}

/**
 * Transition from queued to running state
 */
function transitionToRunning(jobData) {
  isJobQueued = false;

  const animation = document.getElementById('training-animation');
  const title = document.getElementById('training-title');
  const subtitle = document.getElementById('training-subtitle');
  const queueBox = document.getElementById('queue-status-box');
  const btnStop = document.getElementById('btn-stop-training');
  const btnCancel = document.getElementById('btn-cancel-queued');
  const progressContainer = document.querySelector('.training-progress');
  const timerContainer = document.getElementById('training-timer');

  // Update animation
  const gpuCount = jobData.gpu_indices?.length || 1;
  if (animation) {
    if (gpuCount > 1) {
      animation.innerHTML = `
        <div class="gpu-animation">
          ${Array.from({length: gpuCount}, (_, i) => `
            <div class="gpu-box" data-id="${i}">
              <div class="pulse"></div>
            </div>
          `).join('')}
        </div>
        <div class="connection-lines">
          ${Array.from({length: gpuCount - 1}, () => '<div class="connection-line"></div>').join('')}
        </div>
      `;
    } else {
      animation.innerHTML = '<div class="spinner"></div>';
    }
  }

  if (title) title.textContent = 'Training in Progress...';
  if (subtitle) subtitle.textContent = `Running on ${gpuCount} GPU${gpuCount > 1 ? 's' : ''}`;

  // Hide queue status
  if (queueBox) queueBox.style.display = 'none';

  // Show progress and timer
  if (progressContainer) progressContainer.style.display = 'block';
  if (timerContainer) timerContainer.style.display = 'block';

  // Show stop button
  if (btnStop) btnStop.style.display = 'inline-block';
  if (btnCancel) btnCancel.style.display = 'none';

  // Start timer
  startTrainingTimer(jobData.config || {});
}

/**
 * Cancel a queued job
 */
async function cancelQueuedJob() {
  if (!currentJobId) return;

  try {
    const res = await fetch(`/api/queue/jobs/${currentJobId}/cancel`, {
      method: 'POST'
    });

    if (res.ok) {
      hideTrainingOverlay();
      stopJobPolling();
      currentJobId = null;
      isJobQueued = false;
    } else {
      const err = await res.json();
      alert(err.detail || 'Failed to cancel job');
    }
  } catch (e) {
    alert('Failed to cancel job: ' + e.message);
  }
}

/**
 * Start polling for job status
 */
function startJobPolling(jobId) {
  if (queuePollInterval) clearInterval(queuePollInterval);
  queuePollInterval = setInterval(() => pollJobStatus(jobId), 2000);
  pollJobStatus(jobId);
}

/**
 * Stop job polling
 */
function stopJobPolling() {
  if (queuePollInterval) {
    clearInterval(queuePollInterval);
    queuePollInterval = null;
  }
}

/**
 * Poll job status from queue
 */
async function pollJobStatus(jobId) {
  try {
    const res = await fetch(`/api/queue/jobs/${jobId}`);
    if (!res.ok) {
      stopJobPolling();
      return;
    }

    const job = await res.json();
    updateStatusBadge(job.status);

    if (job.status === 'pending') {
      // Update queue position
      const positionEl = document.getElementById('queue-position-value');
      const waitEl = document.getElementById('queue-wait-value');
      if (positionEl) positionEl.textContent = `#${job.queue_position || '?'}`;
      if (waitEl) waitEl.textContent = formatTime(job.estimated_wait || 0);
    } else if (job.status === 'running') {
      // Transition to running state if was queued
      if (isJobQueued) {
        transitionToRunning(job);
        switchSideTab('logs');
      }
      // Update logs and progress
      renderLogs(job.logs || []);
      updateTrainingProgress(job.status, job.logs || []);
    } else if (job.status === 'completed') {
      hideTrainingOverlay();
      stopTrainingTimer();
      stopJobPolling();
      renderLogs(job.logs || []);
      showTrainingSuccess(job.logs || []);
      setTimeout(() => updateStatusBadge('idle'), 3000);
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      hideTrainingOverlay();
      stopTrainingTimer();
      stopJobPolling();
      renderLogs(job.logs || []);
      if (job.error_message) {
        alert('Job ' + job.status + ': ' + job.error_message);
      }
      setTimeout(() => updateStatusBadge('idle'), 3000);
    }
  } catch (e) {
    console.error('Failed to poll job status:', e);
  }
}

/**
 * Handle strategy change - adjust GPU count options
 */
function onStrategyChange() {
  const strategy = getVal('f-strategy');
  const gpuSelect = document.getElementById('f-gpu-count');

  syncGpuOptionsWithStrategy();

  // Solo always forces 1 GPU; hybrid starts at 1 (topology product drives it from there).
  // DDP and FSDP leave the count unchanged so the user stays in control.
  if (strategy === 'none' || strategy === 'hybrid') {
    if (gpuSelect) gpuSelect.value = '1';
  }

  // Show / hide 3D topology section
  const topoSection = document.getElementById('topology-section');
  if (topoSection) topoSection.style.display = strategy === 'hybrid' ? '' : 'none';
  if (strategy === 'hybrid') onTopologyChange();

  // Solo needs no launch mode — hide it (and SLURM sub-section)
  const launchModeGroup = document.getElementById('f-launch-mode')?.closest('.form-group');
  const isSolo = strategy === 'none';
  if (launchModeGroup) launchModeGroup.style.display = isSolo ? 'none' : '';
  if (isSolo) {
    const slurmSection = document.getElementById('slurm-section');
    if (slurmSection) slurmSection.style.display = 'none';
  } else {
    onLaunchModeChange();
  }

  // Refresh availability display
  updateGpuAvailability();
  checkStrategyGpuCompatibility();
}

function onLaunchModeChange() {
  const mode = getVal('f-launch-mode') || 'torchrun';
  const slurmSection = document.getElementById('slurm-section');
  if (slurmSection) slurmSection.style.display = mode === 'slurm' ? '' : 'none';
}

function onParallelismModeChange() {
  const mode = getVal('f-parallelism-mode') || '2d';
  const ppGroup = document.getElementById('fg-pp-size');
  if (ppGroup) ppGroup.style.display = mode === '3d' ? '' : 'none';
  // Force pp=1 in 2D mode so the mesh math and hint don't carry a stale value
  if (mode !== '3d') {
    const ppInput = document.getElementById('f-pp-size');
    if (ppInput) ppInput.value = '1';
  }
  onTopologyChange();
}

function onTopologyChange() {
  const mode = getVal('f-parallelism-mode') || '2d';
  const dp = parseInt(getVal('f-dp-size')) || 1;
  const tp = parseInt(getVal('f-tp-size')) || 1;
  const pp = mode === '3d' ? (parseInt(getVal('f-pp-size')) || 1) : 1;
  const total = dp * tp * pp;

  const gpuSelect = document.getElementById('f-gpu-count');

  // Auto-sync GPU count if an exact matching option exists
  if (gpuSelect) {
    const opt = Array.from(gpuSelect.options).find(o => parseInt(o.value) === total);
    if (opt) gpuSelect.value = String(total);
  }

  updateTopologyHint();
}

function updateTopologyHint() {
  const mode = getVal('f-parallelism-mode') || '2d';
  const dp = parseInt(getVal('f-dp-size')) || 1;
  const tp = parseInt(getVal('f-tp-size')) || 1;
  const pp = mode === '3d' ? (parseInt(getVal('f-pp-size')) || 1) : 1;
  const total = dp * tp * pp;
  const selected = parseInt(getVal('f-gpu-count')) || 1;

  const hintEl = document.getElementById('fg-topology-hint');
  const hintText = document.getElementById('topology-hint-text');
  if (!hintEl || !hintText) return;

  const meshStr = mode === '3d'
    ? `${dp} DP × ${tp} TP × ${pp} PP`
    : `${dp} DP × ${tp} TP`;

  hintEl.style.display = '';
  if (selected === total) {
    hintText.textContent = `✓ Mesh: ${meshStr} = ${total} GPUs — matches GPU Count.`;
    hintText.style.color = 'var(--green)';
  } else {
    hintText.textContent = `⚠️ Mesh: ${meshStr} = ${total} GPUs, but GPU Count is set to ${selected}. Update one to match.`;
    hintText.style.color = 'var(--orange)';
  }
}

/**
 * Handle GPU count change
 */
function onGpuCountChange() {
  syncGpuOptionsWithStrategy();
  updateGpuAvailability();
  checkStrategyGpuCompatibility();
  // Refresh topology hint if hybrid strategy is active
  const strategy = getVal('f-strategy');
  if (strategy === 'hybrid') updateTopologyHint();
}

function classifyLogLine(line) {
  if (/error|Error|ERROR|Traceback|Exception|failed|FAILED/i.test(line)) return 'log-line-error';
  if (/✅|success|completed|Training Complete|Saved/i.test(line)) return 'log-line-success';
  if (/⚠️|warning|Warning/i.test(line)) return 'log-line-warning';
  if (/rank=|local_rank=|world_size=|CUDA_VISIBLE_DEVICES/i.test(line)) return 'log-line-rank';
  if (/Step \d+\/\d+|Optimizer step|ETA:|\| Loss:/i.test(line)) return 'log-line-progress';
  if (/loss|Loss/i.test(line)) return 'log-line-loss';
  if (/^[=\-]{5,}|Starting|Loading|OMNI-Train|Stage \d+\/\d+|Launch Summary|Training Configuration|Distributed/i.test(line)) return 'log-line-info';
  return 'log-line';
}

function getLatestMeaningfulLog(logs) {
  if (!Array.isArray(logs) || logs.length === 0) return '';

  for (let i = logs.length - 1; i >= 0; i--) {
    const line = (logs[i] || '').trim();
    if (!line) continue;
    if (/^[=\-]{5,}$/.test(line)) continue;
    return logs[i];
  }

  return logs[logs.length - 1] || '';
}

function renderOverlayLogs(lines) {
  const el = document.getElementById('training-live-log');
  const metaEl = document.getElementById('training-log-meta');
  const hintEl = document.getElementById('training-live-hint');
  if (!el) return;

  const safeLines = Array.isArray(lines) ? lines : [];
  const wasNearBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) < 24;

  if (safeLines.length === 0) {
    el.innerHTML = '<div class="training-live-log-empty">Waiting for training output...</div>';
  } else {
    el.innerHTML = safeLines.map(line => {
      const cls = `training-live-log-line ${classifyLogLine(line)}`;
      return `<div class="${cls}">${escapeHtml(line)}</div>`;
    }).join('');
  }

  if (metaEl) {
    metaEl.textContent = `${safeLines.length} line${safeLines.length === 1 ? '' : 's'} streamed`;
  }

  if (hintEl) {
    hintEl.textContent = document.getElementById('side-logs')
      ? 'Full output is also available in the Logs tab.'
      : 'Latest training output appears here in real time.';
  }

  if (wasNearBottom) {
    el.scrollTop = el.scrollHeight;
  }
}

function updateTrainingProgress(status, logs) {
  const title = document.getElementById('training-title');
  const subtitle = document.getElementById('training-subtitle');
  const progressFill = document.getElementById('progress-fill');
  const progressText = document.getElementById('progress-text');
  const overlay = document.getElementById('training-overlay');
  const safeLogs = Array.isArray(logs) ? logs : [];

  renderOverlayLogs(safeLogs);

  if (status === 'running') {
    // Ensure overlay is visible when training is running
    if (overlay && !overlay.classList.contains('active')) {
      showTrainingOverlay('none');
    }
    if (title) title.textContent = 'Training in Progress...';

    const lastLog = getLatestMeaningfulLog(safeLogs);
    if (lastLog && subtitle) {
      subtitle.textContent = lastLog.substring(0, 140);
    }

    const recentLog = [...safeLogs].reverse().find(line =>
      /Stage \d+\/\d+|Step \d+\/\d+|Optimizer step \d+|Epoch \d+\/\d+/.test(line || '')
    ) || lastLog;

    const stepMatch = recentLog.match(/Step (\d+)\/(\d+)/i);
    const stageMatch = recentLog.match(/Stage (\d+)\/(\d+)/i);
    const epochMatch = recentLog.match(/Epoch (\d+)\/(\d+)/i);
    const optMatch = recentLog.match(/Optimizer step (\d+)/i);

    if (stepMatch) {
      const current = parseInt(stepMatch[1]);
      const total = parseInt(stepMatch[2]);
      const progress = total > 0 ? (current / total) * 100 : 0;
      if (progressFill) progressFill.style.width = progress + '%';
      if (progressText) progressText.textContent = `Step ${current} of ${total}`;
    } else if (stageMatch) {
      const current = parseInt(stageMatch[1]);
      const total = parseInt(stageMatch[2]);
      const progress = total > 0 ? (current / total) * 100 : 0;
      if (progressFill) progressFill.style.width = progress + '%';
      if (progressText) progressText.textContent = `Stage ${current} of ${total}`;
    } else if (epochMatch) {
      const current = parseInt(epochMatch[1]);
      const total = parseInt(epochMatch[2]);
      const progress = total > 0 ? (current / total) * 100 : 0;
      if (progressFill) progressFill.style.width = progress + '%';
      if (progressText) progressText.textContent = `Epoch ${current} of ${total}`;
    } else if (optMatch && progressText) {
      progressText.textContent = `Optimizer step ${optMatch[1]}`;
    }

    // Update timer from training logs
    updateTimerFromLogs(safeLogs);
  } else if (status === 'finished') {
    hideTrainingOverlay();
    stopTrainingTimer();
    startedFromYaml = false;
  } else if (status === 'error' || status === 'stopped') {
    hideTrainingOverlay();
    stopTrainingTimer();
  }
}

// =========================================================================
// Polling
// =========================================================================
function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(pollStatus, 1000);
  pollStatus();
}

async function pollStatus() {
  try {
    const res = await fetch('/api/train/status');
    const data = await res.json();

    let displayStatus = data.status;
    if (!pollInterval && data.status !== 'running') {
      displayStatus = 'idle';
    }

    updateStatusBadge(displayStatus);
    updateButtons(data.status);
    renderLogs(data.logs);
    updateTrainingProgress(data.status, data.logs);

    if (data.status !== 'running' && pollInterval) {
      setTimeout(() => {
        if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
        updateStatusBadge('idle');
      }, 3000);
    }
  } catch (e) {
    updateStatusBadge('idle');
  }
}

function updateStatusBadge(status) {
  const el = document.getElementById('status-badge');
  if (el) {
    el.textContent = status.toUpperCase();
    el.className = 'status-badge status-' + status;
  }
}

function updateButtons(status) {
  const start = document.getElementById('btn-start-main');
  const stop = document.getElementById('btn-stop-main');
  if (status === 'running') {
    if (start) start.style.display = 'none';
    if (stop) stop.style.display = 'flex';
  } else {
    if (start) start.style.display = 'flex';
    if (stop) stop.style.display = 'none';
  }
}

function renderLogs(lines) {
  const el = document.getElementById('log-output');
  const autoscrollEl = document.getElementById('autoscroll');
  if (!el) return;

  const safeLines = Array.isArray(lines) ? lines : [];
  const autoScroll = autoscrollEl ? autoscrollEl.checked : true;

  el.innerHTML = safeLines.map(line => {
    const cls = classifyLogLine(line);
    return `<div class="${cls}">${escapeHtml(line)}</div>`;
  }).join('');

  const countEl = document.getElementById('log-count');
  if (countEl) countEl.textContent = safeLines.length + ' lines';

  if (autoScroll) {
    el.scrollTop = el.scrollHeight;
  }
}

// =========================================================================
// Helpers
// =========================================================================
function setVal(id, val) { const el = document.getElementById(id); if (el) el.value = val; }
function getVal(id) { const el = document.getElementById(id); return el ? el.value : ''; }
function toggle(id, show) { const el = document.getElementById(id); if (el) el.style.display = show ? '' : 'none'; }
function escapeHtml(t) { return String(t ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

/**
 * Toggle the live log panel between normal and expanded height.
 */
function toggleLogExpand(btn) {
  const logEl = document.getElementById('training-live-log');
  if (!logEl) return;
  const isExpanded = logEl.classList.toggle('expanded');
  if (btn) btn.textContent = isExpanded ? '⤡' : '⤢';
  if (isExpanded) logEl.scrollTop = logEl.scrollHeight;
}

// =========================================================================
// Drag and Drop
// =========================================================================
function setupDragDrop(area, inputId, callback) {
  if (!area) return;

  area.addEventListener('dragover', (e) => {
    e.preventDefault();
    area.classList.add('dragover');
  });

  area.addEventListener('dragleave', () => {
    area.classList.remove('dragover');
  });

  area.addEventListener('drop', (e) => {
    e.preventDefault();
    area.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const input = document.getElementById(inputId);
      if (input) {
        input.files = files;
        callback(input);
      }
    }
  });
}

// =========================================================================
// Page Initialization
// =========================================================================
async function initConfigPage() {
  loadFormState();
  loadTemplates();

  const uploadArea = document.getElementById('upload-area');
  if (uploadArea) {
    setupDragDrop(uploadArea, 'f-model-file', onModelFileSelect);
  }

  const dataUploadArea = document.getElementById('data-upload-area');
  if (dataUploadArea) {
    setupDragDrop(dataUploadArea, 'f-data-file', onDataFileSelect);
  }

  // Check for selected template from landing page
  const templateData = localStorage.getItem('selected_template');
  if (templateData) {
    try {
      const template = JSON.parse(templateData);
      localStorage.removeItem('selected_template');

      // Set pending selection so sidebar highlights it after rendering
      pendingTemplateSelection = template.name;

      if (template.isExtra) {
        setVal('f-model-type', template.type);
        onModelTypeChange();
        applyExtraTemplate(template.extra);
      } else {
        fetch(`/api/configs/${template.name}`)
          .then(res => res.json())
          .then(data => {
            currentConfig = data.config;
            applyConfigToForm(data.config);
          })
          .catch(e => console.error('Failed to load template:', e));
      }
    } catch (e) {
      console.error('Failed to parse template data:', e);
    }
  }

  // Check URL for tab parameter
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get('tab') === 'logs') {
    switchSideTab('logs');
    startPolling();

    // Check if training is already running and show overlay
    try {
      const res = await fetch('/api/train/status');
      const data = await res.json();
      if (data.status === 'running') {
        // Determine strategy from config if available
        const strategy = data.config?.distributed?.strategy || 'none';
        showTrainingOverlay(strategy);
        startTrainingTimer(data.config || {});
        renderLogs(data.logs || []);
      }
    } catch (e) {
      console.error('Failed to check training status:', e);
    }
  }

  // Initialize GPU availability and queue status
  onStrategyChange();
  updateGpuAvailability();
  // Refresh availability every 10 seconds
  setInterval(updateGpuAvailability, 10000);

  pollStatus();

  // Keyboard shortcut: Ctrl/Cmd+Enter → Next (go to YAML editor)
  document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      const btn = document.getElementById('btn-review-yaml');
      if (btn && !btn.disabled) btn.click();
    }
  });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  loadTheme();

  // Reset FSDP check result when any config field that affects VRAM changes
  const fsdpResetIds = [
    'f-model-name', 'f-model-url', 'f-model-path',
    'f-finetune-mode', 'f-lora-r', 'f-lora-alpha',
    'f-max-seq-len', 'f-batch-size', 'f-mixed-precision',
    'f-quantize', 'f-quant-bits', 'f-model-source',
  ];
  fsdpResetIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', resetFsdpCheck);
    if (el && (el.tagName === 'INPUT')) el.addEventListener('input', resetFsdpCheck);
  });
});
