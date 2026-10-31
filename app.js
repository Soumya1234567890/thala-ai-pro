/**
 * THALA-AI PRO™ - Universal In-Browser & API Machine Learning Controller
 * Supports dual-mode execution:
 * 1. Client-Side Mode: 100% in-browser Support Vector Machine (RBF) inference, Kernel SHAP attributions,
 *    concordance matrix, and jsPDF report generation for zero-backend GitHub Pages hosting.
 * 2. API Mode: Seamlessly connects to FastAPI backend if available.
 */

// --- STATE MANAGEMENT ---
let currentAnalysis = null;
let debounceTimer = null;
let presetsList = [];
let decisionSpaceData = null;
let currentProjection = 'MCV_RBC'; // 'MCV_RBC' or 'MCV_RDW'
let apiAvailable = null; // null = untried, true = connected, false = fallback to client

// Universal Model Bundle Accessor
function getModelBundle() {
  if (typeof window !== 'undefined' && window.THALA_MODEL_BUNDLE) return window.THALA_MODEL_BUNDLE;
  if (typeof THALA_MODEL_BUNDLE !== 'undefined') return THALA_MODEL_BUNDLE;
  if (typeof globalThis !== 'undefined' && globalThis.THALA_MODEL_BUNDLE) return globalThis.THALA_MODEL_BUNDLE;
  return null;
}

const PARAM_DEFAULTS = {
  rbc_count: 5.82,
  hemoglobin: 9.8,
  hct: 32.5,
  mcv: 56.8,
  mch: 17.1,
  mchc: 30.1,
  rdw_cv: 14.1,
  mentzer_index: 9.76
};

const REF_RANGES = {
  rbc_count: { min: 3.80, max: 5.80 },
  hemoglobin: { min: 11.5, max: 16.5 },
  hct: { min: 35.0, max: 48.0 },
  mcv: { min: 80.0, max: 100.0 },
  mch: { min: 27.0, max: 33.0 },
  mchc: { min: 31.0, max: 36.0 },
  rdw_cv: { min: 11.5, max: 15.0 },
  mentzer_index: { min: 13.0, max: 35.0 },
  // Display name aliases for CBC report evaluation
  'RBC Count': { min: 3.80, max: 5.80 },
  'Hemoglobin': { min: 11.5, max: 16.5 },
  'HCT': { min: 35.0, max: 48.0 },
  'MCV': { min: 80.0, max: 100.0 },
  'MCH': { min: 27.0, max: 33.0 },
  'MCHC': { min: 31.0, max: 36.0 },
  'RDW-CV': { min: 11.5, max: 15.0 },
  'Mentzer Index': { min: 13.0, max: 35.0 }
};

const FEATURE_DESCRIPTIONS = {
  'RBC Count': {
    thal: 'Elevated erythrocyte count with microcytosis is a hallmark of ineffective erythropoiesis compensation in BTT.',
    ida: 'Subnormal or normal RBC count with microcytosis aligns with impaired erythropoiesis due to depleted iron stores.'
  },
  'MCV': {
    thal: 'Marked microcytosis (< 70 fL) disproportionate to mild anemia strongly favors Thalassemia trait.',
    ida: 'Microcytosis typically correlates linearly with severity of anemia in IDA; higher MCV pulls away from BTT.'
  },
  'RDW-CV': {
    thal: 'Uniformly small red cells with lower RDW (normal or mild elevation) strongly supports BTT over IDA.',
    ida: 'Marked anisocytosis (elevated RDW > 16%) is characteristic of progressive iron depletion.'
  },
  'Mentzer Index': {
    thal: 'Mentzer Index < 13 reflects microcytosis accompanied by high red cell count, characteristic of BTT.',
    ida: 'Mentzer Index > 13 reflects microcytosis with inadequate red cell expansion, characteristic of IDA.'
  },
  'Hemoglobin': {
    thal: 'Mild to moderate hemoglobin reduction without severe tissue hypoxia signs fits stable heterozygous trait.',
    ida: 'Hemoglobin decline directly driven by iron exhaustion.'
  },
  'HCT': {
    thal: 'Preserved hematocrit relative to very low MCV indicates microcytic polycythemia.',
    ida: 'Reduced packed cell volume proportional to iron-restricted erythropoiesis.'
  },
  'MCH': {
    thal: 'Severe hypochromia per erythrocyte reflects impaired globin chain synthesis.',
    ida: 'Hypochromia driven by lack of heme synthesis.'
  },
  'MCHC': {
    thal: 'Relative preservation of intracellular hemoglobin concentration in microcytic cells.',
    ida: 'Progressive decrease in cellular hemoglobin concentration.'
  }
};

// --- DOM ELEMENTS ---
const elements = {
  themeToggleBtn: document.getElementById('themeToggleBtn'),
  themeIcon: document.getElementById('themeIcon'),
  previewReportBtn: document.getElementById('previewReportBtn'),
  downloadPdfBtn: document.getElementById('downloadPdfBtn'),
  btnRandomCohort: document.getElementById('btnRandomCohort'),
  autoSyncMentzer: document.getElementById('autoSyncMentzer'),
  mentzerSyncStatus: document.getElementById('mentzerSyncStatus'),
  recalculateBtn: document.getElementById('recalculateBtn'),
  lastInferenceTime: document.getElementById('lastInferenceTime'),
  toastContainer: document.getElementById('toastContainer'),

  // Patient inputs
  sliderAge: document.getElementById('slider_age'),
  numAge: document.getElementById('num_age'),

  // Verdict outputs
  verdictHeroCard: document.getElementById('verdictHeroCard'),
  verdictRiskFlag: document.getElementById('verdictRiskFlag'),
  verdictTitle: document.getElementById('verdictTitle'),
  verdictSummary: document.getElementById('verdictSummary'),
  probThalText: document.getElementById('probThalText'),
  probIdaText: document.getElementById('probIdaText'),
  probFillThal: document.getElementById('probFillThal'),
  probFillIda: document.getElementById('probFillIda'),
  dialMeter: document.getElementById('dialMeter'),
  dialPercentage: document.getElementById('dialPercentage'),
  dialConfidence: document.getElementById('dialConfidence'),

  // SHAP
  shapBaseVal: document.getElementById('shapBaseVal'),
  shapTopFeature: document.getElementById('shapTopFeature'),
  shapBarsContainer: document.getElementById('shapBarsContainer'),

  // 2D Decision Space
  btnAxisMCVRBC: document.getElementById('btnAxisMCV_RBC'),
  btnAxisMCVRDW: document.getElementById('btnAxisMCV_RDW'),
  decisionCanvas: document.getElementById('decisionCanvas'),
  patientCoordReadout: document.getElementById('patientCoordReadout'),
  patientZoneReadout: document.getElementById('patientZoneReadout'),
  patientMentzerReadout: document.getElementById('patientMentzerReadout'),

  // Concordance
  concordanceScore: document.getElementById('concordanceScore'),
  concordanceStatusPill: document.getElementById('concordanceStatusPill'),
  concordanceTableBody: document.getElementById('concordanceTableBody'),
  divergenceAlert: document.getElementById('divergenceAlert'),
  divergenceAlertText: document.getElementById('divergenceAlertText'),

  // Recommendations
  recommendationsGrid: document.getElementById('recommendationsGrid'),

  // Modal
  previewModal: document.getElementById('previewModal'),
  closeModalBtn: document.getElementById('closeModalBtn'),
  modalPrintArea: document.getElementById('modalPrintArea'),
  modalPrintActionBtn: document.getElementById('modalPrintActionBtn'),
  modalDownloadPdfActionBtn: document.getElementById('modalDownloadPdfActionBtn')
};

// --- HELPER FOR ROBUST PARAMETER EXTRACTION ---
function getParamVal(param) {
  const numInput = document.getElementById(`num_${param}`);
  const slider = document.getElementById(`slider_${param}`);
  
  if (numInput && numInput.value !== '' && !isNaN(parseFloat(numInput.value))) {
    return parseFloat(numInput.value);
  }
  if (slider && slider.value !== '' && !isNaN(parseFloat(slider.value))) {
    return parseFloat(slider.value);
  }
  return PARAM_DEFAULTS[param] !== undefined ? PARAM_DEFAULTS[param] : 0;
}

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  setupEventListeners();
  setupParamSync();
  setupAgeSync();
  setupDecisionSpaceControls();

  // Load client presets & decision space cohort synchronously first so UI is ready immediately
  const bundle = getModelBundle();
  if (bundle) {
    if (bundle.presets) {
      presetsList = bundle.presets;
      bindPresetChips();
    }
    if (bundle.cohort) {
      decisionSpaceData = {
        btt_points: bundle.cohort.btt_points,
        ida_points: bundle.cohort.ida_points
      };
      renderDecisionSpace();
    }
  }

  // Trigger initial inference immediately
  await triggerInference();

  // Background check for API if hosted locally
  loadPresets();
  loadDecisionSpaceData();
});

// --- THEME MANAGEMENT ---
function initTheme() {
  const savedTheme = localStorage.getItem('thala_theme') || 'dark';
  if (savedTheme === 'light') {
    document.body.classList.add('light-theme');
    updateThemeIcon(true);
  } else {
    updateThemeIcon(false);
  }

  elements.themeToggleBtn.addEventListener('click', () => {
    const isLight = document.body.classList.toggle('light-theme');
    localStorage.setItem('thala_theme', isLight ? 'light' : 'dark');
    updateThemeIcon(isLight);
    renderDecisionSpace();
    showToast(isLight ? 'Clinical Day Theme Activated' : 'Medical Dark Theme Activated');
  });
}

function updateThemeIcon(isLight) {
  if (isLight) {
    elements.themeIcon.innerHTML = `
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
    `;
  } else {
    elements.themeIcon.innerHTML = `
      <circle cx="12" cy="12" r="5"/>
      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
    `;
  }
}

// --- AGE INPUT SYNCHRONIZATION (Without reference range defined) ---
function setupAgeSync() {
  if (!elements.sliderAge || !elements.numAge) return;

  const handleAgeChange = (val) => {
    const num = parseFloat(val);
    if (!isNaN(num)) {
      elements.sliderAge.value = num;
      elements.numAge.value = num;
      debouncedInference();
    }
  };

  elements.sliderAge.addEventListener('input', (e) => {
    elements.numAge.value = e.target.value;
    debouncedInference();
  });

  elements.numAge.addEventListener('input', (e) => {
    handleAgeChange(e.target.value);
  });

  elements.numAge.addEventListener('change', (e) => {
    handleAgeChange(e.target.value);
  });
}

// --- DUAL SLIDER & NUMBER INPUT SYNCHRONIZATION ---
function setupParamSync() {
  const params = Object.keys(PARAM_DEFAULTS);

  params.forEach(param => {
    const slider = document.getElementById(`slider_${param}`);
    const numInput = document.getElementById(`num_${param}`);

    if (!slider || !numInput) return;

    slider.addEventListener('input', (e) => {
      numInput.value = e.target.value;
      const val = parseFloat(e.target.value);
      updateParamBadge(param, val);

      if (param === 'mentzer_index' && elements.autoSyncMentzer && elements.autoSyncMentzer.checked) {
        elements.autoSyncMentzer.checked = false;
        if (elements.mentzerSyncStatus) {
          elements.mentzerSyncStatus.textContent = 'Manual override';
        }
      }

      handleAutoMentzer(param);
      renderDecisionSpace();
      debouncedInference();
    });

    numInput.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (!isNaN(val)) {
        slider.value = e.target.value;
        updateParamBadge(param, val);
      }

      if (param === 'mentzer_index' && elements.autoSyncMentzer && elements.autoSyncMentzer.checked) {
        elements.autoSyncMentzer.checked = false;
        if (elements.mentzerSyncStatus) {
          elements.mentzerSyncStatus.textContent = 'Manual override';
        }
      }

      handleAutoMentzer(param);
      renderDecisionSpace();
      debouncedInference();
    });

    numInput.addEventListener('change', (e) => {
      const val = parseFloat(e.target.value);
      if (!isNaN(val)) {
        slider.value = e.target.value;
        updateParamBadge(param, val);
      }

      if (param === 'mentzer_index' && elements.autoSyncMentzer && elements.autoSyncMentzer.checked) {
        elements.autoSyncMentzer.checked = false;
        if (elements.mentzerSyncStatus) {
          elements.mentzerSyncStatus.textContent = 'Manual override';
        }
      }

      handleAutoMentzer(param);
      renderDecisionSpace();
      triggerInference();
    });

    updateParamBadge(param, getParamVal(param));
  });

  if (elements.autoSyncMentzer) {
    elements.autoSyncMentzer.addEventListener('change', (e) => {
      const isAuto = e.target.checked;
      if (elements.mentzerSyncStatus) {
        elements.mentzerSyncStatus.textContent = isAuto ? 'Auto-calculated' : 'Manual override';
      }
      if (isAuto) {
        calculateAndSetMentzer();
        renderDecisionSpace();
        debouncedInference();
      }
    });
  }
}

function handleAutoMentzer(changedParam) {
  if (elements.autoSyncMentzer && elements.autoSyncMentzer.checked && (changedParam === 'mcv' || changedParam === 'rbc_count')) {
    calculateAndSetMentzer();
  }
}

function calculateAndSetMentzer() {
  const mcv = getParamVal('mcv');
  const rbc = getParamVal('rbc_count');
  const calculatedMentzer = (mcv / Math.max(rbc, 0.01)).toFixed(2);

  const slider = document.getElementById('slider_mentzer_index');
  const numInput = document.getElementById('num_mentzer_index');

  if (slider && numInput) {
    slider.value = calculatedMentzer;
    numInput.value = calculatedMentzer;
    updateParamBadge('mentzer_index', parseFloat(calculatedMentzer));
  }
}

function updateParamBadge(param, val) {
  const badge = document.getElementById(`badge_${param}`);
  if (!badge) return;

  const ref = REF_RANGES[param];
  if (!ref) return;

  badge.className = 'param-badge';

  if (param === 'mentzer_index') {
    if (val < 13.0) {
      badge.textContent = '< 13.0 (Thal)';
      badge.classList.add('badge-very-low');
    } else {
      badge.textContent = '> 13.0 (IDA/Norm)';
      badge.classList.add('badge-normal');
    }
    return;
  }

  if (val < ref.min) {
    if (val < ref.min * 0.8) {
      badge.textContent = 'Very Low';
      badge.classList.add('badge-very-low');
    } else {
      badge.textContent = 'Low';
      badge.classList.add('badge-low');
    }
  } else if (val > ref.max) {
    if (val > ref.max * 1.2) {
      badge.textContent = 'Very High';
      badge.classList.add('badge-high');
    } else {
      badge.textContent = 'High';
      badge.classList.add('badge-high');
    }
  } else {
    badge.textContent = 'Normal';
    badge.classList.add('badge-normal');
  }
}

// --- SETUP EVENT LISTENERS ---
function setupEventListeners() {
  elements.recalculateBtn.addEventListener('click', () => {
    triggerInference();
    renderDecisionSpace();
    showToast('Inference updated');
  });

  elements.downloadPdfBtn.addEventListener('click', () => {
    downloadReportPdf();
  });

  elements.previewReportBtn.addEventListener('click', () => {
    openPrintPreview();
  });

  elements.closeModalBtn.addEventListener('click', () => {
    elements.previewModal.classList.remove('active');
  });

  elements.previewModal.addEventListener('click', (e) => {
    if (e.target === elements.previewModal) {
      elements.previewModal.classList.remove('active');
    }
  });

  elements.modalPrintActionBtn.addEventListener('click', () => {
    window.print();
  });

  elements.modalDownloadPdfActionBtn.addEventListener('click', () => {
    downloadReportPdf();
  });

  elements.btnRandomCohort.addEventListener('click', async () => {
    await loadRandomCohortPatient();
  });

  window.addEventListener('resize', () => {
    renderDecisionSpace();
  });
}

// --- 2D DECISION SPACE CONTROLS & CANVAS RENDERING ---
function setupDecisionSpaceControls() {
  if (!elements.btnAxisMCVRBC || !elements.btnAxisMCVRDW) return;

  elements.btnAxisMCVRBC.addEventListener('click', () => {
    currentProjection = 'MCV_RBC';
    elements.btnAxisMCVRBC.classList.add('active');
    elements.btnAxisMCVRDW.classList.remove('active');
    renderDecisionSpace();
  });

  elements.btnAxisMCVRDW.addEventListener('click', () => {
    currentProjection = 'MCV_RDW';
    elements.btnAxisMCVRDW.classList.add('active');
    elements.btnAxisMCVRBC.classList.remove('active');
    renderDecisionSpace();
  });
}

async function loadDecisionSpaceData() {
  const isStaticHost = window.location.protocol === 'file:' || 
                       window.location.hostname.includes('github.io');

  if (!isStaticHost && apiAvailable !== false) {
    try {
      const res = await fetch('/api/decision-space', { signal: AbortSignal.timeout(600) });
      if (res.ok) {
        decisionSpaceData = await res.json();
        renderDecisionSpace();
        return;
      }
    } catch (_) {
      // API not reachable
    }
  }

  // Client-side fallback from pre-compiled model bundle
  const bundle = getModelBundle();
  if (bundle && bundle.cohort) {
    decisionSpaceData = {
      btt_points: bundle.cohort.btt_points,
      ida_points: bundle.cohort.ida_points
    };
    renderDecisionSpace();
  }
}

function renderDecisionSpace() {
  const canvas = elements.decisionCanvas;
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  const containerW = canvas.parentElement ? canvas.parentElement.clientWidth : 800;
  const W = (rect.width && rect.width > 50) ? rect.width : (containerW || 800);
  const H = 340;

  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const padLeft = 60;
  const padRight = 30;
  const padTop = 25;
  const padBottom = 45;
  const plotW = Math.max(100, W - padLeft - padRight);
  const plotH = Math.max(100, H - padTop - padBottom);

  const isLight = document.body.classList.contains('light-theme');

  // Background Fill
  ctx.fillStyle = isLight ? '#f8fafc' : '#080d1a';
  ctx.fillRect(0, 0, W, H);

  // Axes Limits
  const xMin = 44.0;
  const xMax = 105.0; // MCV (fL)

  let yMin, yMax, yLabel, currentYVal;
  const currentMCV = getParamVal('mcv');
  const currentRBC = getParamVal('rbc_count');
  const currentRDW = getParamVal('rdw_cv');
  const currentMI = (currentMCV / Math.max(currentRBC, 0.01)).toFixed(2);

  if (currentProjection === 'MCV_RBC') {
    yMin = 2.0;
    yMax = 7.5;
    yLabel = 'RBC Count (10¹²/L)';
    currentYVal = currentRBC;
  } else {
    yMin = 10.0;
    yMax = 30.0;
    yLabel = 'RDW-CV (%)';
    currentYVal = currentRDW;
  }

  function toX(val) {
    return padLeft + ((val - xMin) / (xMax - xMin)) * plotW;
  }

  function toY(val) {
    return padTop + plotH - ((val - yMin) / (yMax - yMin)) * plotH;
  }

  // --- 1. RENDER 2D DECISION REGION GRADIENT ---
  const grad = ctx.createLinearGradient(padLeft, padTop, padLeft + plotW, padTop + plotH);
  if (currentProjection === 'MCV_RBC') {
    if (isLight) {
      grad.addColorStop(0, 'rgba(239, 68, 68, 0.16)');
      grad.addColorStop(0.48, 'rgba(239, 68, 68, 0.04)');
      grad.addColorStop(0.52, 'rgba(16, 185, 129, 0.04)');
      grad.addColorStop(1, 'rgba(16, 185, 129, 0.16)');
    } else {
      grad.addColorStop(0, 'rgba(239, 68, 68, 0.22)');
      grad.addColorStop(0.48, 'rgba(239, 68, 68, 0.05)');
      grad.addColorStop(0.52, 'rgba(16, 185, 129, 0.05)');
      grad.addColorStop(1, 'rgba(16, 185, 129, 0.22)');
    }
  } else {
    if (isLight) {
      grad.addColorStop(0, 'rgba(239, 68, 68, 0.15)');
      grad.addColorStop(0.5, 'rgba(239, 68, 68, 0.05)');
      grad.addColorStop(1, 'rgba(16, 185, 129, 0.16)');
    } else {
      grad.addColorStop(0, 'rgba(239, 68, 68, 0.22)');
      grad.addColorStop(0.5, 'rgba(239, 68, 68, 0.06)');
      grad.addColorStop(1, 'rgba(16, 185, 129, 0.22)');
    }
  }
  ctx.fillStyle = grad;
  ctx.fillRect(padLeft, padTop, plotW, plotH);

  // --- 2. GRID LINES & LABELS ---
  ctx.strokeStyle = isLight ? 'rgba(0, 0, 0, 0.07)' : 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillStyle = isLight ? '#475569' : '#64748b';

  // Vertical Grid (MCV)
  for (let m = 50; m <= 100; m += 10) {
    const x = toX(m);
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, padTop + plotH);
    ctx.stroke();
    ctx.fillText(`${m}`, x - 8, padTop + plotH + 16);
  }

  // Horizontal Grid (RBC or RDW)
  const yStep = currentProjection === 'MCV_RBC' ? 1.0 : 5.0;
  for (let y = Math.ceil(yMin); y <= yMax; y += yStep) {
    const py = toY(y);
    ctx.beginPath();
    ctx.moveTo(padLeft, py);
    ctx.lineTo(padLeft + plotW, py);
    ctx.stroke();
    ctx.fillText(currentProjection === 'MCV_RBC' ? y.toFixed(1) : y.toString(), 16, py + 3);
  }

  // Axis Title Labels
  ctx.font = '11px Plus Jakarta Sans, sans-serif';
  ctx.fillStyle = isLight ? '#334155' : '#94a3b8';
  ctx.fillText('Mean Corpuscular Volume - MCV (fL)', padLeft + plotW / 2 - 90, padTop + plotH + 34);

  ctx.save();
  ctx.translate(14, padTop + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  // --- 3. MENTZER INDEX DISCRIMINATION LINE (MCV / RBC = 13.0) ---
  if (currentProjection === 'MCV_RBC') {
    ctx.save();
    ctx.strokeStyle = '#f59e0b';
    if (ctx.setLineDash) ctx.setLineDash([6, 4]);

    ctx.beginPath();
    const mcvStart = Math.max(xMin, yMin * 13.0);
    const mcvEnd = Math.min(xMax, yMax * 13.0);
    ctx.moveTo(toX(mcvStart), toY(mcvStart / 13.0));
    ctx.lineTo(toX(mcvEnd), toY(mcvEnd / 13.0));
    ctx.stroke();

    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = '#f59e0b';
    ctx.fillText('Mentzer Threshold Line (MI = 13.0)', toX(68) + 8, toY(68 / 13.0) - 8);
    ctx.restore();
  }

  // --- 4. COHORT SCATTER POINTS (1,190-record synthetic dataset, Google Gemini-generated; not real patients) ---
  if (!decisionSpaceData) {
    const bundle = getModelBundle();
    if (bundle && bundle.cohort) {
      decisionSpaceData = {
        btt_points: bundle.cohort.btt_points,
        ida_points: bundle.cohort.ida_points
      };
    }
  }

  if (decisionSpaceData && decisionSpaceData.btt_points && decisionSpaceData.ida_points) {
    // A. Beta Thalassemia Trait Points (Red)
    ctx.fillStyle = 'rgba(239, 68, 68, 0.78)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 0.5;
    decisionSpaceData.btt_points.forEach(pt => {
      const mcv = parseFloat(pt.mcv);
      const yVal = parseFloat(currentProjection === 'MCV_RBC' ? pt.rbc : pt.rdw);
      if (mcv >= xMin && mcv <= xMax && yVal >= yMin && yVal <= yMax) {
        const px = toX(mcv);
        const py = toY(yVal);
        ctx.beginPath();
        ctx.arc(px, py, 3.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    });

    // B. Iron Deficiency Anemia Points (Emerald)
    ctx.fillStyle = 'rgba(16, 185, 129, 0.78)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 0.5;
    decisionSpaceData.ida_points.forEach(pt => {
      const mcv = parseFloat(pt.mcv);
      const yVal = parseFloat(currentProjection === 'MCV_RBC' ? pt.rbc : pt.rdw);
      if (mcv >= xMin && mcv <= xMax && yVal >= yMin && yVal <= yMax) {
        const px = toX(mcv);
        const py = toY(yVal);
        ctx.beginPath();
        ctx.arc(px, py, 3.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    });
  }

  // --- 5. CURRENT PATIENT BEACON (🎯 Target Locator) ---
  const ptX = toX(Math.max(xMin, Math.min(xMax, currentMCV)));
  const ptY = toY(Math.max(yMin, Math.min(yMax, currentYVal)));

  ctx.save();
  ctx.strokeStyle = 'rgba(0, 229, 255, 0.45)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(ptX, ptY, 15, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(ptX, ptY, 8, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(ptX, ptY, 4, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(ptX - 18, ptY); ctx.lineTo(ptX - 10, ptY);
  ctx.moveTo(ptX + 10, ptY); ctx.lineTo(ptX + 18, ptY);
  ctx.moveTo(ptX, ptY - 18); ctx.lineTo(ptX, ptY - 10);
  ctx.moveTo(ptX, ptY + 10); ctx.lineTo(ptX, ptY + 18);
  ctx.stroke();

  // Floating Patient Tag Badge
  ctx.font = 'bold 10px Plus Jakarta Sans, sans-serif';
  const tagText = `PATIENT: MCV ${currentMCV.toFixed(1)} | ${currentProjection === 'MCV_RBC' ? 'RBC ' + currentRBC.toFixed(2) : 'RDW ' + currentRDW.toFixed(1) + '%'}`;
  const tagW = ctx.measureText(tagText).width + 12;

  let tagX = ptX - tagW / 2;
  tagX = Math.max(padLeft + 4, Math.min(padLeft + plotW - tagW - 4, tagX));
  const tagY = ptY - 24;

  ctx.fillStyle = isLight ? 'rgba(255, 255, 255, 0.95)' : 'rgba(10, 16, 30, 0.95)';
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(tagX, tagY, tagW, 18, 4);
  else ctx.rect(tagX, tagY, tagW, 18);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = isLight ? '#0891b2' : '#00e5ff';
  ctx.fillText(tagText, tagX + 6, tagY + 13);
  ctx.restore();

  // --- 6. UPDATE DECISION META READOUTS ---
  const isThalZone = (currentMCV / Math.max(currentRBC, 0.01)) < 13.0;
  if (elements.patientCoordReadout) {
    elements.patientCoordReadout.textContent = `MCV: ${currentMCV.toFixed(1)} fL | ${currentProjection === 'MCV_RBC' ? 'RBC: ' + currentRBC.toFixed(2) + ' × 10¹²/L' : 'RDW: ' + currentRDW.toFixed(1) + '%'}`;
  }
  if (elements.patientZoneReadout) {
    if (isThalZone) {
      elements.patientZoneReadout.textContent = 'Inside Beta Thalassemia Trait Space';
      elements.patientZoneReadout.style.color = '#f87171';
    } else {
      elements.patientZoneReadout.textContent = 'Inside IDA / Non-Thal Space';
      elements.patientZoneReadout.style.color = '#34d399';
    }
  }
  if (elements.patientMentzerReadout) {
    elements.patientMentzerReadout.textContent = `MI = ${currentMI} (${isThalZone ? '< 13.0 indicates Thal' : '> 13.0 indicates IDA'})`;
  }
}

// --- PRESET MANAGEMENT ---
async function loadPresets() {
  const isStaticHost = window.location.protocol === 'file:' || 
                       window.location.hostname.includes('github.io');

  if (!isStaticHost && apiAvailable !== false) {
    try {
      const res = await fetch('/api/presets', { signal: AbortSignal.timeout(600) });
      if (res.ok) {
        presetsList = await res.json();
        bindPresetChips();
        return;
      }
    } catch (_) {
      // API not reachable
    }
  }

  // Client-side fallback from bundle
  const bundle = getModelBundle();
  if (bundle && bundle.presets) {
    presetsList = bundle.presets;
    bindPresetChips();
  }
}

function bindPresetChips() {
  document.querySelectorAll('.preset-chip[data-preset]').forEach(chip => {
    chip.addEventListener('click', () => {
      const presetId = chip.getAttribute('data-preset');
      applyPreset(presetId);

      document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
    });
  });
}

function applyPreset(presetId) {
  const preset = presetsList.find(p => p.id === presetId);
  if (!preset) return;

  const d = preset.data;
  if (elements.sliderAge && elements.numAge) {
    const ageNum = parseFloat(d.age) || 28;
    elements.sliderAge.value = ageNum;
    elements.numAge.value = ageNum;
  }

  setFormValues({
    rbc_count: d.rbc_count,
    hemoglobin: d.hemoglobin,
    hct: d.hct,
    mcv: d.mcv,
    mch: d.mch,
    mchc: d.mchc,
    rdw_cv: d.rdw_cv,
    mentzer_index: d.mentzer_index
  });

  renderDecisionSpace();
  triggerInference();
  showToast(`Loaded ${preset.title}`);
}

async function loadRandomCohortPatient() {
  const isStaticHost = window.location.protocol === 'file:' || 
                       window.location.hostname.includes('github.io');

  if (!isStaticHost && apiAvailable !== false) {
    try {
      const res = await fetch('/api/random-patient', { signal: AbortSignal.timeout(600) });
      if (res.ok) {
        const p = await res.json();
        applyPatientObject(p);
        return;
      }
    } catch (_) {}
  }

  // Client-side pool from bundle
  const bundle = getModelBundle();
  if (bundle && bundle.random_pool) {
    const pool = bundle.random_pool;
    if (pool.length > 0) {
      const p = pool[Math.floor(Math.random() * pool.length)];
      applyPatientObject(p);
    }
  }
}

function applyPatientObject(p) {
  if (elements.sliderAge && elements.numAge) {
    const match = String(p.age).match(/\d+/);
    const parsedAge = match ? parseInt(match[0], 10) : 30;
    elements.sliderAge.value = parsedAge;
    elements.numAge.value = parsedAge;
  }

  setFormValues({
    rbc_count: p.rbc_count,
    hemoglobin: p.hemoglobin,
    hct: p.hct,
    mcv: p.mcv,
    mch: p.mch,
    mchc: p.mchc,
    rdw_cv: p.rdw_cv,
    mentzer_index: p.mentzer_index
  });

  document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  if (elements.btnRandomCohort) {
    elements.btnRandomCohort.classList.add('active');
  }

  renderDecisionSpace();
  triggerInference();
  showToast(`Loaded synthetic record (dataset label: ${p.ground_truth_diagnosis}) - not a real patient`);
}

function setFormValues(vals) {
  Object.keys(vals).forEach(k => {
    const slider = document.getElementById(`slider_${k}`);
    const numInput = document.getElementById(`num_${k}`);
    if (slider && numInput) {
      slider.value = vals[k];
      numInput.value = vals[k];
      updateParamBadge(k, vals[k]);
    }
  });
}

function getFormValues() {
  let ageVal = 28.0;
  if (elements.numAge && elements.numAge.value !== '' && !isNaN(parseFloat(elements.numAge.value))) {
    ageVal = parseFloat(elements.numAge.value);
  } else if (elements.sliderAge && elements.sliderAge.value !== '' && !isNaN(parseFloat(elements.sliderAge.value))) {
    ageVal = parseFloat(elements.sliderAge.value);
  }

  return {
    age: ageVal,
    patient_id: 'DEMO-INPUT',
    patient_name: 'Demo input (not a patient)',
    gender: 'Unspecified',
    referring_physician: 'Not applicable (educational demo)',
    clinical_notes: '',

    rbc_count: getParamVal('rbc_count'),
    hemoglobin: getParamVal('hemoglobin'),
    hct: getParamVal('hct'),
    mcv: getParamVal('mcv'),
    mch: getParamVal('mch'),
    mchc: getParamVal('mchc'),
    rdw_cv: getParamVal('rdw_cv'),
    mentzer_index: getParamVal('mentzer_index')
  };
}

// --- DEBOUNCED LIVE INFERENCE ---
function debouncedInference() {
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    triggerInference();
  }, 80);
}

// --- PURE IN-BROWSER CLIENT-SIDE SUPPORT VECTOR MACHINE (RBF KERNEL) & EXPLAINABILITY ENGINE ---
function computeRbfKernelDecision(scaledVals, bundle) {
  const sv = bundle.support_vectors_scaled;
  const dual = bundle.dual_coef;
  const gamma = bundle.gamma;
  const numSv = sv.length;
  let sum = 0.0;
  for (let i = 0; i < numSv; i++) {
    const s = sv[i];
    let distSq = 0.0;
    for (let j = 0; j < 7; j++) {
      const diff = scaledVals[j] - s[j];
      distSq += diff * diff;
    }
    sum += dual[i] * Math.exp(-gamma * distSq);
  }
  return sum + bundle.intercept;
}

function runClientSideInference(patientData) {
  const bundle = getModelBundle();
  if (!bundle) throw new Error("Model bundle not loaded");

  const featOrder = bundle.features; // ['RBC', 'Hb', 'Hct', 'MCV', 'MCH', 'MCHC', 'RDW-CV']
  const mean = bundle.scaler.mean;
  const scale = bundle.scaler.scale;

  const rbc = patientData.rbc_count;
  const hb = patientData.hemoglobin;
  const hct = patientData.hct;
  const mcv = patientData.mcv;
  const mch = patientData.mch;
  const mchc = patientData.mchc;
  const rdw = patientData.rdw_cv;
  const mentzer = patientData.mentzer_index !== undefined ? patientData.mentzer_index : (mcv / Math.max(rbc, 0.001));

  // 7 scaler features mapping
  const rawMap = {
    'RBC': rbc,
    'Hb': hb,
    'Hct': hct,
    'MCV': mcv,
    'MCH': mch,
    'MCHC': mchc,
    'RDW-CV': rdw
  };

  const scaled = featOrder.map((f, i) => (rawMap[f] - mean[i]) / scale[i]);

  // Exact RBF SVM Decision Function
  const z = computeRbfKernelDecision(scaled, bundle);

  // Platt Calibrated Probability (scikit-learn convention: P(class 1) = 1 / (1 + exp(probA * f - probB)),
  // where f is the SVC decision_function value; note the MINUS sign before probB)
  const probThal = 1.0 / (1.0 + Math.exp(bundle.probA * z - bundle.probB));
  const probIda = 1.0 - probThal;
  const predClass = probThal >= 0.5 ? 1 : 0;

  // SHAP Feature Attribution via Marginal Kernel Decomposition
  const bgScaled = bundle.background_scaled;
  const shapFeatures = featOrder.map((fName, j) => {
    const perturbed = [...scaled];
    perturbed[j] = bgScaled[j];
    const z_perturbed = computeRbfKernelDecision(perturbed, bundle);
    const phi = z - z_perturbed;

    const metaLabel = (bundle.feature_meta && bundle.feature_meta[fName]) ? bundle.feature_meta[fName].label : fName;
    const desc = FEATURE_DESCRIPTIONS[fName] || FEATURE_DESCRIPTIONS[metaLabel] || {
      thal: 'Pushes probability towards Beta Thalassemia Trait.',
      ida: 'Pushes probability towards Iron Deficiency Anemia.'
    };

    return {
      name: metaLabel,
      actual_value: round(rawMap[fName], 2),
      unit: (bundle.feature_meta && bundle.feature_meta[fName]) ? bundle.feature_meta[fName].unit : '',
      shap_value: round(phi, 4),
      abs_shap: Math.abs(phi),
      direction: phi > 0 ? 'Thalassemia' : 'IDA',
      explanation: phi > 0 ? desc.thal : desc.ida
    };
  });

  shapFeatures.sort((a, b) => b.abs_shap - a.abs_shap);
  shapFeatures.forEach((f, idx) => { f.rank = idx + 1; });

  // Classical Discrimination Indices
  const { indices, summary } = calculateClientIndices(
    rbc, hct, hb, mcv, mch, rdw, predClass
  );

  // CBC Evaluation Flags
  const cbcEval = evaluateClientCbcFlags({
    'RBC Count': rbc,
    'Hemoglobin': hb,
    'HCT': hct,
    'MCV': mcv,
    'MCH': mch,
    'MCHC': mchc,
    'RDW-CV': rdw,
    'Mentzer Index': mentzer
  });

  const maxProb = Math.max(probThal, probIda);
  const confidenceTier = maxProb >= 0.90 ? "Very High Confidence" : (maxProb >= 0.70 ? "High Confidence" : "Borderline Confidence");

  return {
    prediction: {
      predicted_class: predClass,
      predicted_label: predClass === 1 ? 'Beta Thalassemia Trait / Carrier' : 'Iron Deficiency Anemia (IDA)',
      prob_thalassemia: round(probThal, 4),
      prob_ida: round(probIda, 4),
      decision_margin: round(z, 2),
      confidence_tier: confidenceTier,
      primary_risk_flag: predClass === 1 ? 'HIGH RISK - THALASSEMIA TRAIT' : 'LOW RISK - IDA SUSPECTED'
    },
    shap_analysis: {
      base_value: bundle.base_value || 0.461,
      features: shapFeatures,
      top_driver: shapFeatures[0]?.name || 'MCV'
    },
    indices: indices,
    concordance_summary: summary,
    cbc_evaluated: cbcEval,
    recommendations: getClientRecommendations(predClass),
    divergence_notes: summary.agreement_pct < 75 ? [
      `Diagnostic Divergence Noted: ${summary.total_indices - summary.agreement_count} of 8 classical formulas conflict with the AI. Classical heuristic rules (like Mentzer Index >= 13) rely on single ratios that often fail in overlapping microcytosis. The RBF Support Vector Machine integrates all 7 red cell indices simultaneously (97.7% accuracy on a held-out synthetic test set of 220 records; not clinically validated).`
    ] : [],
    patient_metadata: {
      age: `${patientData.age || 28} Years`,
      patient_id: 'DEMO-INPUT'
    }
  };
}

function calculateClientIndices(rbc, hct, hb, mcv, mch, rdw, aiPredClass) {
  const indices = [];

  // 1. Mentzer
  const miVal = mcv / Math.max(rbc, 0.001);
  const miThal = miVal < 13.0;
  indices.push({
    id: 'mentzer', name: 'Mentzer Index', formula: 'MCV / RBC', value: round(miVal, 2),
    cutoff_rule: '< 13.0 indicates Thal; > 13.0 indicates IDA', threshold: 13.0,
    interpretation: miThal ? 'Thalassemia Trait' : 'Iron Deficiency Anemia',
    indicates_thal: miThal, agrees_with_ai: (miThal ? 1 : 0) === aiPredClass,
    clinical_note: 'Classic ratio. Vulnerable when RBC is depressed in severe iron deficiency.'
  });

  // 2. Srivastava
  const siVal = mch / Math.max(rbc, 0.001);
  const siThal = siVal < 3.8;
  indices.push({
    id: 'srivastava', name: 'Srivastava Index', formula: 'MCH / RBC', value: round(siVal, 2),
    cutoff_rule: '< 3.8 indicates Thal; > 3.8 indicates IDA', threshold: 3.8,
    interpretation: siThal ? 'Thalassemia Trait' : 'Iron Deficiency Anemia',
    indicates_thal: siThal, agrees_with_ai: (siThal ? 1 : 0) === aiPredClass,
    clinical_note: 'Emphasizes low MCH per red cell.'
  });

  // 3. Green & King
  const gkVal = ((mcv ** 2) * rdw) / Math.max(hb * 100.0, 0.001);
  const gkThal = gkVal < 65.0;
  indices.push({
    id: 'green_king', name: 'Green & King Index', formula: '(MCV² × RDW) / (Hb × 100)', value: round(gkVal, 2),
    cutoff_rule: '< 65.0 indicates Thal; > 72.0 indicates IDA', threshold: 65.0,
    interpretation: gkThal ? 'Thalassemia Trait' : (gkVal <= 72.0 ? 'Borderline' : 'Iron Deficiency Anemia'),
    indicates_thal: gkThal, agrees_with_ai: (gkThal ? 1 : 0) === aiPredClass,
    clinical_note: 'Incorporates RDW to penalize wide red cell distribution.'
  });

  // 4. Shine & Lal
  const slVal = ((mcv ** 2) * mch) / 100.0;
  const slThal = slVal < 1530.0;
  indices.push({
    id: 'shine_lal', name: 'Shine & Lal Index', formula: '(MCV² × MCH) / 100', value: round(slVal, 2),
    cutoff_rule: '< 1530.0 indicates Thal; > 1530.0 indicates IDA', threshold: 1530.0,
    interpretation: slThal ? 'Thalassemia Trait' : 'Iron Deficiency Anemia',
    indicates_thal: slThal, agrees_with_ai: (slThal ? 1 : 0) === aiPredClass,
    clinical_note: 'Highly sensitive for microcytosis, but often over-predicts Thalassemia in IDA.'
  });

  // 5. England & Fraser
  const efVal = mcv - rbc - (5.0 * hb) - 3.4;
  const efThal = efVal < 0.0;
  indices.push({
    id: 'england_fraser', name: 'England & Fraser Index', formula: 'MCV - RBC - (5 × Hb) - 3.4', value: round(efVal, 2),
    cutoff_rule: '< 0 indicates Thal; > 0 indicates IDA', threshold: 0.0,
    interpretation: efThal ? 'Thalassemia Trait' : 'Iron Deficiency Anemia',
    indicates_thal: efThal, agrees_with_ai: (efThal ? 1 : 0) === aiPredClass,
    clinical_note: 'Classical linear discriminant function.'
  });

  // 6. Ricerca
  const ricVal = rdw / Math.max(rbc, 0.001);
  const ricThal = ricVal < 4.4;
  indices.push({
    id: 'ricerca', name: 'Ricerca Index', formula: 'RDW / RBC', value: round(ricVal, 2),
    cutoff_rule: '< 4.4 indicates Thal; > 4.4 indicates IDA', threshold: 4.4,
    interpretation: ricThal ? 'Thalassemia Trait' : 'Iron Deficiency Anemia',
    indicates_thal: ricThal, agrees_with_ai: (ricThal ? 1 : 0) === aiPredClass,
    clinical_note: 'Measures RDW relative to total erythrocyte count.'
  });

  // 7. Ehsani
  const ehVal = mcv - (10.0 * rbc);
  const ehThal = ehVal < 15.0;
  indices.push({
    id: 'ehsani', name: 'Ehsani Index', formula: 'MCV - (10 × RBC)', value: round(ehVal, 2),
    cutoff_rule: '< 15.0 indicates Thal; > 15.0 indicates IDA', threshold: 15.0,
    interpretation: ehThal ? 'Thalassemia Trait' : 'Iron Deficiency Anemia',
    indicates_thal: ehThal, agrees_with_ai: (ehThal ? 1 : 0) === aiPredClass,
    clinical_note: 'Rapid mathematical screening formula.'
  });

  // 8. RDWI
  const rdwiVal = (mcv * rdw) / Math.max(rbc, 0.001);
  const rdwiThal = rdwiVal < 220.0;
  indices.push({
    id: 'rdwi', name: 'RDWI Index', formula: '(MCV × RDW) / RBC', value: round(rdwiVal, 2),
    cutoff_rule: '< 220.0 indicates Thal; > 220.0 indicates IDA', threshold: 220.0,
    interpretation: rdwiThal ? 'Thalassemia Trait' : 'Iron Deficiency Anemia',
    indicates_thal: rdwiThal, agrees_with_ai: (rdwiThal ? 1 : 0) === aiPredClass,
    clinical_note: 'Highest single-index concordance with AI.'
  });

  const agreeCount = indices.filter(x => x.agrees_with_ai).length;
  const agreementPct = round((agreeCount / indices.length) * 100.0, 1);

  return {
    indices,
    summary: {
      agreement_count: agreeCount,
      total_indices: indices.length,
      agreement_pct: agreementPct,
      consensus_verdict: agreementPct >= 75 ? 'High Index Concordance' : (agreementPct >= 50 ? 'Moderate Index Concordance' : 'Diagnostic Divergence / Mixed Profile')
    }
  };
}

function evaluateClientCbcFlags(rawMap) {
  const units = {
    'RBC Count': '10^12/L', 'Hemoglobin': 'g/dL', 'HCT': '%', 'MCV': 'fL',
    'MCH': 'pg', 'MCHC': 'g/dL', 'RDW-CV': '%', 'Mentzer Index': 'ratio'
  };

  return Object.keys(rawMap).map(param => {
    const val = rawMap[param];
    const ref = REF_RANGES[param] || { min: 0, max: 100 };
    let flag = 'Normal';
    let comment = 'Within standard biological reference interval.';

    if (param === 'Mentzer Index') {
      flag = val < 13.0 ? 'Low (<13)' : 'High (>13)';
      comment = val < 13.0 ? 'Typical BTT range (<13)' : 'Typical IDA/normal range (>13)';
    } else if (val < ref.min) {
      flag = val < ref.min * 0.8 ? 'VERY LOW' : 'LOW';
      comment = param === 'MCV' ? 'Marked microcytosis' : (param === 'Hemoglobin' ? 'Anemia' : 'Below reference range');
    } else if (val > ref.max) {
      flag = val > ref.max * 1.25 ? 'VERY HIGH' : 'HIGH';
      comment = param === 'RBC Count' ? 'Erythrocytosis (microcytic polycythemia)' : 'Above reference range';
    }

    return {
      name: param,
      value: round(val, 2),
      unit: units[param] || '',
      ref_range: `${ref.min} - ${ref.max}`,
      flag,
      comment
    };
  });
}

function getClientRecommendations(predClass) {
  if (predClass === 1) {
    return [
      { title: 'Cation-Exchange HPLC / Capillary Electrophoresis', desc: 'Mandatory confirmatory quantification of HbA2 and HbF. An elevated HbA2 > 3.5% confirms Beta Thalassemia Trait.' },
      { title: 'Concomitant Iron Deficiency Screening', desc: 'Measure Serum Ferritin, Serum Iron, and TIBC. Co-existing iron deficiency can artificially suppress HbA2 levels, masking BTT.' },
      { title: 'Cascade Family Screening & Genetic Counseling', desc: 'Screen partner and first-degree relatives before conception to prevent Thalassemia Major in offspring.' },
      { title: 'Avoid Inappropriate Iron Supplementation', desc: 'Patients with isolated BTT should NOT receive empirical iron therapy as it confers risk of secondary hemochromatosis.' }
    ];
  } else {
    return [
      { title: 'Complete Iron Kinetic Studies', desc: 'Evaluate Serum Ferritin (<15-30 ng/mL diagnostic of IDA), Transferrin Saturation (<16%), and Soluble Transferrin Receptor.' },
      { title: 'Etiological Investigation for Iron Loss', desc: 'Investigate occult gastrointestinal bleeding, celiac disease/malabsorption, or menorrhagia.' },
      { title: 'Therapeutic Iron Trial with CBC Re-evaluation', desc: 'Initiate oral iron therapy (e.g. ferrous sulfate) and monitor reticulocyte response at 7-10 days and Hb recovery at 4 weeks.' },
      { title: 'Post-Treatment Hemoglobin Re-testing', desc: 'If microcytosis persists after normal ferritin repletion, re-screen with HPLC to uncover latent masked Thalassemia.' }
    ];
  }
}

function round(val, dec) {
  return Number(Math.round(val + 'e' + dec) + 'e-' + dec);
}

// --- DUAL API / CLIENT-SIDE TRIGGER ---
async function triggerInference() {
  const payload = getFormValues();
  const startTime = performance.now();
  let data = null;

  const isStaticHost = window.location.protocol === 'file:' || 
                       window.location.hostname.includes('github.io') ||
                       window.location.hostname.includes('pages.dev') ||
                       window.location.hostname.includes('netlify.app') ||
                       window.location.hostname.includes('vercel.app');

  // 1. Attempt API call (only if not on static host and API not known unavailable)
  if (!isStaticHost && apiAvailable !== false) {
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(600)
      });
      if (res.ok) {
        data = await res.json();
        apiAvailable = true;
      } else {
        apiAvailable = false;
      }
    } catch (_) {
      apiAvailable = false;
    }
  }

  // 2. Client-side fallback: 100% in-browser Support Vector Machine (RBF) & Kernel Attribution engine
  const bundle = getModelBundle();
  if (!data && bundle) {
    try {
      data = runClientSideInference(payload);
    } catch (err) {
      console.error("Client inference error:", err);
    }
  }

  if (data) {
    currentAnalysis = data;
    const elapsed = Math.round(performance.now() - startTime);
    if (elements.lastInferenceTime) {
      elements.lastInferenceTime.textContent = `Live Inference: ${elapsed} ms`;
    }
    renderDiagnosticResults(data);
    renderDecisionSpace();
  }
}

// --- RENDER COMPLETE RESULTS ---
function renderDiagnosticResults(data) {
  renderVerdictHero(data.prediction);
  renderShapAttributions(data.shap_analysis);
  renderConcordanceMatrix(data.indices, data.concordance_summary, data.divergence_notes);
  renderRecommendations(data.recommendations);
}

// 1. Verdict & Probability Dial
function renderVerdictHero(pred) {
  const isThal = pred.predicted_class === 1;
  const probThalPct = (pred.prob_thalassemia * 100).toFixed(1);
  const probIdaPct = (pred.prob_ida * 100).toFixed(1);

  elements.verdictHeroCard.className = `verdict-hero glass-panel ${isThal ? 'is-thal' : 'is-ida'}`;
  elements.verdictRiskFlag.textContent = pred.primary_risk_flag;
  elements.verdictTitle.textContent = pred.predicted_label;

  if (isThal) {
    elements.verdictSummary.textContent = 
      `The Support Vector Machine (RBF) classifier predicts a ${probThalPct}% probability of Beta Thalassemia Trait. The blood profile demonstrates significant microcytosis alongside compensatory erythrocytosis, with relatively homogeneous red blood cell volume.`;
  } else {
    elements.verdictSummary.textContent = 
      `The Support Vector Machine (RBF) classifier predicts a ${probIdaPct}% probability of Iron Deficiency Anemia / Non-Thalassemia. The microcytic hypochromic indices are accompanied by marked anisocytosis (elevated RDW) and non-compensatory erythropoiesis.`;
  }

  elements.probThalText.textContent = `${probThalPct}%`;
  elements.probIdaText.textContent = `${probIdaPct}%`;
  elements.probFillThal.style.width = `${probThalPct}%`;
  elements.probFillIda.style.width = `${probIdaPct}%`;

  const mainProb = isThal ? pred.prob_thalassemia : pred.prob_ida;
  const circumference = 314.16;
  const offset = circumference * (1 - mainProb);

  elements.dialMeter.style.strokeDashoffset = offset;
  elements.dialMeter.setAttribute('stroke', isThal ? '#ef4444' : '#10b981');
  elements.dialPercentage.textContent = `${Math.round(mainProb * 100)}%`;
  elements.dialConfidence.textContent = pred.confidence_tier.replace(' Confidence', ' Conf.');
}

// 2. SHAP Waterfall & Force Attribution Bars
function renderShapAttributions(shap) {
  if (!shap) return;
  if (elements.shapBaseVal) {
    elements.shapBaseVal.textContent = `${(shap.base_value * 100).toFixed(1)}%`;
  }
  if (elements.shapTopFeature) {
    elements.shapTopFeature.textContent = shap.top_driver || 'MCV';
  }

  const container = elements.shapBarsContainer;
  if (!container) return;
  container.innerHTML = '';

  if (!shap.features || shap.features.length === 0) return;

  const maxAbs = Math.max(...shap.features.map(f => f.abs_shap), 0.15);

  shap.features.forEach(feat => {
    const isPos = feat.shap_value > 0;
    const barWidthPct = Math.min(48, Math.max(2, (feat.abs_shap / maxAbs) * 45));

    const row = document.createElement('div');
    row.className = 'shap-bar-row';

    row.innerHTML = `
      <div class="shap-feat-name">
        <span>${feat.name}</span>
        <span class="shap-feat-val">${feat.actual_value}</span>
      </div>
      <div class="shap-bar-canvas" title="${feat.explanation || ''}">
        <div class="shap-center-axis"></div>
        <div class="shap-bar-fill ${isPos ? 'positive' : 'negative'}"
             style="width: ${barWidthPct.toFixed(1)}%;"></div>
      </div>
      <div class="shap-val-text ${isPos ? 'pos' : 'neg'}">
        ${isPos ? '+' : ''}${feat.shap_value.toFixed(4)}
      </div>
    `;

    container.appendChild(row);
  });
}

// 3. Classical Indices Concordance Matrix
function renderConcordanceMatrix(indices, summary, divergenceNotes) {
  elements.concordanceScore.textContent = 
    `${summary.agreement_count} of ${summary.total_indices} Indices Agree (${summary.agreement_pct}% Concordant)`;

  elements.concordanceStatusPill.textContent = summary.consensus_verdict;
  if (summary.agreement_pct >= 75) {
    elements.concordanceStatusPill.style.background = 'rgba(16, 185, 129, 0.15)';
    elements.concordanceStatusPill.style.color = '#6ee7b7';
    elements.concordanceStatusPill.style.borderColor = 'rgba(16, 185, 129, 0.3)';
  } else {
    elements.concordanceStatusPill.style.background = 'rgba(245, 158, 11, 0.15)';
    elements.concordanceStatusPill.style.color = '#fde68a';
    elements.concordanceStatusPill.style.borderColor = 'rgba(245, 158, 11, 0.3)';
  }

  const tbody = elements.concordanceTableBody;
  tbody.innerHTML = '';

  indices.forEach(idx => {
    const tr = document.createElement('tr');
    const isAgree = idx.agrees_with_ai;

    tr.innerHTML = `
      <td>
        <b style="color: var(--text-main);">${idx.name}</b>
        <div style="font-size: 0.68rem; color: var(--text-dim);">${idx.clinical_note}</div>
      </td>
      <td><code style="font-family: var(--font-mono); color: var(--text-muted);">${idx.formula}</code></td>
      <td><b style="font-family: var(--font-mono); color: var(--cyan-accent);">${idx.value}</b></td>
      <td style="color: var(--text-muted); font-size: 0.74rem;">${idx.cutoff_rule}</td>
      <td>
        <span style="font-weight: 700; color: ${idx.indicates_thal ? '#f87171' : '#34d399'};">
          ${idx.interpretation}
        </span>
      </td>
      <td>
        <span class="${isAgree ? 'concordance-agree' : 'concordance-diverge'}">
          ${isAgree ? '✓ Concordant' : '⚠ Divergent'}
        </span>
      </td>
    `;
    tbody.appendChild(tr);
  });

  if (divergenceNotes && divergenceNotes.length > 0) {
    elements.divergenceAlert.style.display = 'block';
    elements.divergenceAlertText.textContent = divergenceNotes[0];
  } else {
    elements.divergenceAlert.style.display = 'none';
  }
}

// 4. Clinical Recommendations
function renderRecommendations(recs) {
  const grid = elements.recommendationsGrid;
  grid.innerHTML = '';

  recs.forEach(r => {
    const card = document.createElement('div');
    card.className = 'rec-item';
    card.innerHTML = `
      <div class="rec-title">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="12" cy="12" r="10"/>
          <polyline points="12 6 12 12 16 14"/>
        </svg>
        ${r.title}
      </div>
      <div class="rec-desc">${r.desc}</div>
    `;
    grid.appendChild(card);
  });
}

// --- UNIVERSAL PDF DOWNLOAD (Client-side jsPDF + API fallback) ---
async function downloadReportPdf() {
  if (!currentAnalysis) {
    showToast('Inference data not ready', 'error');
    return;
  }

  showToast('Generating educational demo PDF (not for clinical use)...');

  const ageVal = elements.sliderAge ? `${elements.sliderAge.value} Years` : '28 Years';

  // 1. Try FastAPI backend if running locally
  try {
    const payload = {
      patient: {
        patient_id: 'DEMO-INPUT',
        patient_name: 'Demo input (not a patient)',
        age: ageVal,
        gender: 'Unspecified',
        referring_physician: 'Not applicable (educational demo)'
      },
      analysis: currentAnalysis
    };

    const res = await fetch('/api/download-report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(1000)
    });

    if (res.ok) {
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Thalassemia_AI_Diagnostic_Report.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      showToast('Demo PDF downloaded (educational use only, not for clinical use)');
      return;
    }
  } catch (_) {
    // API not reachable (GitHub Pages)
  }

  // 2. Client-side PDF generation using jsPDF
  try {
    generateClientPdfReport(ageVal);
    showToast('Demo PDF downloaded (educational use only, not for clinical use)');
  } catch (err) {
    console.error('jsPDF error:', err);
    // 3. Fallback: print preview
    openPrintPreview();
    window.print();
  }
}

function generateClientPdfReport(ageVal) {
  const payload = getFormValues();
  try {
    const bundle = getModelBundle();
    if (bundle) {
      currentAnalysis = runClientSideInference(payload);
    }
  } catch (err) {
    console.warn("PDF generation sync inference fallback:", err);
  }
  if (!window.jspdf || !window.jspdf.jsPDF) {
    openPrintPreview();
    window.print();
    return;
  }

  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ unit: 'pt', format: 'letter' });
  const pred = currentAnalysis.prediction;
  const isThal = pred.predicted_class === 1;

  // Header Banner
  doc.setFillColor(15, 23, 42); // slate 900
  doc.rect(36, 36, 540, 48, 'F');

  doc.setTextColor(255, 255, 255);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(14);
  doc.text('THALA-AI PRO - EDUCATIONAL DEMO (NOT A CLINICAL REPORT)', 48, 58);
  doc.setFontSize(8);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(148, 163, 184);
  doc.text('Educational research prototype trained on synthetic data generated with Google Gemini.', 48, 70);
  doc.text('Not a medical device; not for clinical use; no laboratory accreditation.', 48, 79);

  // Meta bar
  doc.setFontSize(8);
  doc.setTextColor(15, 23, 42);
  doc.text(`Demo output   |   Date: ${new Date().toLocaleDateString()}   |   Age entered: ${ageVal}`, 36, 102);

  // Diagnosis Banner
  if (isThal) {
    doc.setFillColor(254, 242, 242);
    doc.setDrawColor(239, 68, 68);
    doc.rect(36, 114, 540, 42, 'FD');
    doc.setTextColor(185, 28, 28);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.text(`MODEL PREDICTION (DEMO): ${pred.predicted_label.toUpperCase()} (${(pred.prob_thalassemia * 100).toFixed(1)}% Probability)`, 48, 132);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(100, 116, 139);
    doc.text(`Confidence Level: ${pred.confidence_tier}   |   Primary Risk Flag: ${pred.primary_risk_flag}`, 48, 146);
  } else {
    doc.setFillColor(240, 253, 244);
    doc.setDrawColor(16, 185, 129);
    doc.rect(36, 114, 540, 42, 'FD');
    doc.setTextColor(4, 120, 87);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.text(`MODEL PREDICTION (DEMO): ${pred.predicted_label.toUpperCase()} (${(pred.prob_ida * 100).toFixed(1)}% Probability)`, 48, 132);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(100, 116, 139);
    doc.text(`Confidence Level: ${pred.confidence_tier}   |   Primary Risk Flag: ${pred.primary_risk_flag}`, 48, 146);
  }

  // Section 1: CBC Table
  let y = 175;
  doc.setTextColor(30, 64, 175);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.text('1. Complete Blood Count & Red Cell Indices', 36, y);

  y += 12;
  doc.setFillColor(30, 41, 59);
  doc.rect(36, y, 540, 16, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(8);
  doc.text('Parameter', 42, y + 11);
  doc.text('Observed Result', 150, y + 11);
  doc.text('Units', 240, y + 11);
  doc.text('Biological Ref Range', 320, y + 11);
  doc.text('Evaluation Flag', 460, y + 11);

  y += 16;
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(15, 23, 42);

  currentAnalysis.cbc_evaluated.forEach((r, idx) => {
    if (idx % 2 === 1) {
      doc.setFillColor(248, 250, 252);
      doc.rect(36, y, 540, 14, 'F');
    }
    doc.text(r.name, 42, y + 10);
    doc.text(String(r.value), 150, y + 10);
    doc.text(r.unit, 240, y + 10);
    doc.text(r.ref_range, 320, y + 10);
    if (r.flag !== 'Normal') {
      doc.setTextColor(185, 28, 28);
      doc.text(r.flag, 460, y + 10);
      doc.setTextColor(15, 23, 42);
    } else {
      doc.setTextColor(4, 120, 87);
      doc.text(r.flag, 460, y + 10);
      doc.setTextColor(15, 23, 42);
    }
    y += 14;
  });

  // Section 2: Classical Indices Concordance Table
  y += 12;
  doc.setTextColor(30, 64, 175);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.text(`2. Classical Discrimination Indices (${currentAnalysis.concordance_summary.agreement_count}/8 Concordant - ${currentAnalysis.concordance_summary.agreement_pct}%)`, 36, y);

  y += 12;
  doc.setFillColor(51, 65, 85);
  doc.rect(36, y, 540, 16, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(8);
  doc.text('Index Name', 42, y + 11);
  doc.text('Formula', 150, y + 11);
  doc.text('Value', 280, y + 11);
  doc.text('Verdict', 350, y + 11);
  doc.text('AI Agreement', 460, y + 11);

  y += 16;
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(15, 23, 42);

  currentAnalysis.indices.forEach((idxItem, idx) => {
    if (idx % 2 === 1) {
      doc.setFillColor(248, 250, 252);
      doc.rect(36, y, 540, 14, 'F');
    }
    doc.text(idxItem.name, 42, y + 10);
    doc.text(idxItem.formula, 150, y + 10);
    doc.text(String(idxItem.value), 280, y + 10);
    doc.text(idxItem.interpretation, 350, y + 10);
    if (idxItem.agrees_with_ai) {
      doc.setTextColor(4, 120, 87);
      doc.text('Concordant', 460, y + 10);
    } else {
      doc.setTextColor(217, 119, 6);
      doc.text('Discordant', 460, y + 10);
    }
    doc.setTextColor(15, 23, 42);
    y += 14;
  });

  // Section 3: Recommendations
  y += 12;
  doc.setTextColor(30, 64, 175);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10);
  doc.text('3. Confirmatory Clinical Recommendations', 36, y);

  y += 12;
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7.5);
  doc.setTextColor(51, 65, 85);

  currentAnalysis.recommendations.forEach(rec => {
    doc.setFont('helvetica', 'bold');
    doc.text(`• ${rec.title}: `, 42, y);
    doc.setFont('helvetica', 'normal');
    doc.text(rec.desc, 42 + doc.getTextWidth(`• ${rec.title}: `), y);
    y += 11;
  });

  // Footer Signature Block
  y += 14;
  doc.setDrawColor(203, 213, 225);
  doc.line(36, y, 576, y);
  y += 14;
  doc.setFontSize(7.5);
  doc.setTextColor(100, 116, 139);
  doc.text('Generated by: THALA-AI PRO educational demo (SVM, RBF kernel, 7 inputs; trained on synthetic data)', 36, y);
  y += 11;
  doc.text('Not a medical device; not for clinical use; no laboratory accreditation. Not reviewed or validated for patient care.', 36, y);

  doc.save('THALA-AI_Demo_Output_NOT_FOR_CLINICAL_USE.pdf');
}

// --- PRINT PREVIEW MODAL ---
function openPrintPreview() {
  const payload = getFormValues();
  try {
    const bundle = getModelBundle();
    if (bundle) {
      currentAnalysis = runClientSideInference(payload);
    }
  } catch (err) {
    console.warn("Print preview sync inference fallback:", err);
  }
  if (!currentAnalysis) {
    showToast('Inference data not ready', 'error');
    return;
  }

  const ageVal = elements.sliderAge ? `${elements.sliderAge.value} Years` : '28 Years';
  const pred = currentAnalysis.prediction;
  const isThal = pred.predicted_class === 1;

  let cbcRowsHtml = currentAnalysis.cbc_evaluated.map(r => `
    <tr>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1;"><b>${r.name}</b></td>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1; font-family: monospace;"><b>${r.value}</b></td>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1;">${r.unit}</td>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1;">${r.ref_range}</td>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1; color: ${r.flag !== 'Normal' ? '#b91c1c' : '#059669'}; font-weight: bold;">${r.flag}</td>
    </tr>
  `).join('');

  let indicesRowsHtml = currentAnalysis.indices.map(idx => `
    <tr>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1;"><b>${idx.name}</b></td>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1; font-family: monospace;">${idx.formula}</td>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1; font-family: monospace;"><b>${idx.value}</b></td>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1;">${idx.cutoff_rule}</td>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1; font-weight: bold;">${idx.interpretation}</td>
      <td style="padding: 6px; border-bottom: 1px solid #cbd5e1; color: ${idx.agrees_with_ai ? '#059669' : '#d97706'}; font-weight: bold;">
        ${idx.agrees_with_ai ? 'Concordant' : 'Discordant'}
      </td>
    </tr>
  `).join('');

  elements.modalPrintArea.innerHTML = `
    <div style="background: #ffffff; color: #0f172a; padding: 24px; border-radius: 8px; font-family: 'Plus Jakarta Sans', sans-serif;">
      
      <!-- Report Header -->
      <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #1e40af; padding-bottom: 12px; margin-bottom: 16px;">
        <div>
          <h2 style="font-size: 1.3rem; font-weight: 800; color: #0f172a; margin: 0;">THALA-AI PRO - Educational Demo Output</h2>
          <div style="font-size: 0.75rem; color: #64748b;">Research prototype trained on synthetic data generated with Google Gemini. Not a medical device; not for clinical use; no laboratory accreditation.</div>
        </div>
        <div style="text-align: right; font-size: 0.8rem;">
          <div><b>Demo output</b></div>
          <div><b>Date:</b> ${new Date().toLocaleDateString()}</div>
          <div style="color: #b91c1c; font-weight: bold;">NOT FOR CLINICAL USE</div>
        </div>
      </div>

      <!-- Patient Demographics -->
      <table style="width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 0.82rem; background: #f8fafc; border: 1px solid #cbd5e1;">
        <tr>
          <td style="padding: 6px 10px;"><b>Age:</b> ${ageVal}</td>
          <td style="padding: 6px 10px;"><b>Model:</b> Support Vector Machine (RBF Kernel, 102 Support Vectors)</td>
          <td style="padding: 6px 10px;"><b>Input:</b> Values entered manually (demo)</td>
        </tr>
      </table>

      <!-- AI Diagnosis Banner -->
      <div style="background: ${isThal ? '#fef2f2' : '#f0fdf4'}; border: 1px solid ${isThal ? '#ef4444' : '#10b981'}; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
        <div style="font-size: 0.75rem; font-weight: bold; color: ${isThal ? '#b91c1c' : '#047857'}; text-transform: uppercase;">
          SVM Model Prediction (educational demo, not a diagnosis):
        </div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin: 4px 0;">
          ${pred.predicted_label} (${(pred.prob_thalassemia * 100).toFixed(1)}% Thalassemia Probability)
        </div>
        <div style="font-size: 0.78rem; color: #475569;">
          Confidence Level: <b>${pred.confidence_tier}</b> | Primary Risk Flag: <b>${pred.primary_risk_flag}</b>
        </div>
      </div>

      <!-- CBC Table -->
      <h4 style="font-size: 0.95rem; margin: 12px 0 6px; color: #1e40af;">1. Complete Blood Count & Red Cell Indices</h4>
      <table style="width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-bottom: 16px;">
        <thead>
          <tr style="background: #1e293b; color: #ffffff; text-align: left;">
            <th style="padding: 6px;">Parameter</th>
            <th style="padding: 6px;">Result</th>
            <th style="padding: 6px;">Unit</th>
            <th style="padding: 6px;">Ref Interval</th>
            <th style="padding: 6px;">Flag</th>
          </tr>
        </thead>
        <tbody>
          ${cbcRowsHtml}
        </tbody>
      </table>

      <!-- Indices Concordance Table -->
      <h4 style="font-size: 0.95rem; margin: 12px 0 6px; color: #1e40af;">
        2. Classical Discrimination Indices (${currentAnalysis.concordance_summary.agreement_count} / 8 Agree)
      </h4>
      <table style="width: 100%; border-collapse: collapse; font-size: 0.76rem; margin-bottom: 16px;">
        <thead>
          <tr style="background: #334155; color: #ffffff; text-align: left;">
            <th style="padding: 6px;">Index</th>
            <th style="padding: 6px;">Formula</th>
            <th style="padding: 6px;">Value</th>
            <th style="padding: 6px;">Cutoff</th>
            <th style="padding: 6px;">Verdict</th>
            <th style="padding: 6px;">AI Agreement</th>
          </tr>
        </thead>
        <tbody>
          ${indicesRowsHtml}
        </tbody>
      </table>

      <!-- Clinical Recommendations -->
      <h4 style="font-size: 0.95rem; margin: 12px 0 6px; color: #1e40af;">3. Recommended Confirmatory Actions</h4>
      <ul style="font-size: 0.78rem; color: #334155; padding-left: 20px; line-height: 1.5;">
        ${currentAnalysis.recommendations.map(r => `<li><b>${r.title}:</b> ${r.desc}</li>`).join('')}
      </ul>

      <!-- Disclaimer -->
      <div style="margin-top: 16px; padding: 10px 12px; border: 1px solid #f59e0b; background: #fffbeb; color: #78350f; font-size: 0.76rem; border-radius: 6px;">
        <b>Disclaimer:</b> Educational research prototype trained on synthetic data generated with Google Gemini. Not a medical device; not for clinical use; no laboratory accreditation. Model performance (97.7% accuracy, 96.0% BTT sensitivity, 99.2% IDA specificity, ROC-AUC 0.994) was measured on a held-out synthetic test set of 220 records only.
      </div>
    </div>
  `;

  elements.previewModal.classList.add('active');
}

// --- TOAST HELPER ---
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = message;

  if (type === 'error') {
    toast.style.borderColor = 'var(--thal-red)';
  }

  elements.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 2800);
}
