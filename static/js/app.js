/* CureHelp+ frontend controller */

const pages = {
  landing: document.getElementById("landing-page"),
  patient: document.getElementById("patient-page"),
  dashboard: document.getElementById("dashboard"),
};

const patientForm = document.getElementById("patient-form");
const patientSummary = document.getElementById("patient-summary");
const startButton = document.getElementById("start-onboarding");
const backButton = document.getElementById("back-to-landing");
const resetSessionButton = document.getElementById("reset-session");
const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
const tabTriggers = Array.from(document.querySelectorAll("[data-tab]"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
const diseasesDropdownToggle = document.getElementById("dropdownDefaultButton");
const diseasesDropdownMenu = document.getElementById("dropdown");
const diseasesDropdownLabel = diseasesDropdownToggle?.querySelector(".goo-button-label");
const medicalReportInput = document.getElementById("medical-report");
const reportLoadingModal = document.getElementById("report-loading-modal");
const loaderWrapper = document.getElementById("css3-spinner-svg-pulse-wrapper");
const patientCard = document.querySelector("#patient-page .card");
const testInputButtons = Array.from(document.querySelectorAll(".test-inputs-button"));
const testInputsModal = document.getElementById("test-inputs-modal");
const testInputsTitle = document.getElementById("test-inputs-title");
const testInputsMessage = document.getElementById("test-inputs-message");
const testInputsNormal = document.getElementById("test-inputs-normal");
const testInputsAbnormal = document.getElementById("test-inputs-abnormal");
const themeToggle = document.getElementById("theme-toggle");
const rootElement = document.documentElement;
const THEME_STORAGE_KEY = "curehelp-theme";
const prefersDarkScheme = typeof window !== "undefined" && typeof window.matchMedia === "function"
  ? window.matchMedia("(prefers-color-scheme: dark)")
  : null;
const themeMenu = document.getElementById("theme-menu");
let isThemeMenuOpen = false;
const landingVideo = document.querySelector(".video-embed");

if (landingVideo) {
  landingVideo.loop = true;
  landingVideo.play().catch(() => {});
}

const FORM_CONSTRAINTS = {
  "patient-form": {
    age: { min: 1, max: 120 },
  },
  "diabetes-form": {
    age: { min: 1, max: 120 },
    bmi: { min: 10, max: 70, step: 0.1 },
    glucose: { min: 40, max: 400 },
    blood_pressure: { min: 60, max: 250 },
    pregnancies: { min: 0, max: 20 },
    skin_thickness: { min: 5, max: 80 },
    insulin: { min: 15, max: 900 },
    diabetes_pedigree_function: { min: 0, max: 3, step: 0.01 },
  },
  "heart-form": {
    age: { min: 18, max: 100 },
    resting_bp: { min: 80, max: 220 },
    cholesterol: { min: 100, max: 600 },
    max_heart_rate: { min: 60, max: 220 },
    st_depression: { min: 0, max: 10, step: 0.1 },
    major_vessels: { min: 0, max: 3 },
  },
  "fever-form": {
    age: { min: 1, max: 110 },
    bmi: { min: 10, max: 60, step: 0.1 },
    temperature: { min: 30, max: 43, step: 0.1 },
    humidity: { min: 0, max: 100 },
    air_quality: { min: 0, max: 500 },
    heart_rate: { min: 40, max: 220 },
  },
  "anemia-form": {
    rbc: { min: 2, max: 8, step: 0.01 },
    hemoglobin: { min: 6, max: 20, step: 0.1 },
    hematocrit: { min: 20, max: 60, step: 0.1 },
    mcv: { min: 60, max: 110, step: 0.1 },
    mch: { min: 15, max: 40, step: 0.1 },
    mchc: { min: 25, max: 38, step: 0.1 },
    wbc: { min: 2, max: 30, step: 0.1 },
    platelets: { min: 50, max: 1000, step: 1 },
    rdw: { min: 10, max: 20, step: 0.1 },
    pdw: { min: 5, max: 25, step: 0.1 },
    pct: { min: 0.05, max: 0.6, step: 0.01 },
    lymphocytes: { min: 5, max: 60, step: 0.1 },
    neutrophils_pct: { min: 20, max: 80, step: 0.1 },
    neutrophils_num: { min: 1, max: 10, step: 0.1 },
  },
};

const TEST_INPUT_PRESETS = {
  diabetes: {
    normal: {
      gender: "Female",
      age: 32,
      bmi: 23.5,
      glucose: 95,
      blood_pressure: 118,
      pregnancies: 1,
      skin_thickness: 22,
      insulin: 85,
      diabetes_pedigree_function: 0.45,
    },
    abnormal: {
      gender: "Female",
      age: 48,
      bmi: 34.2,
      glucose: 185,
      blood_pressure: 145,
      pregnancies: 4,
      skin_thickness: 35,
      insulin: 210,
      diabetes_pedigree_function: 0.92,
    },
  },
  heart: {
    normal: {
      gender: "Male",
      age: 45,
      resting_bp: 120,
      cholesterol: 190,
      chest_pain_type: "1",
      fasting_bs: "No",
      resting_ecg: "0",
      max_heart_rate: 160,
      exercise_angina: "No",
      st_depression: 0.8,
      slope: "1",
      major_vessels: 0,
      thal: "3",
    },
    abnormal: {
      gender: "Female",
      age: 62,
      resting_bp: 165,
      cholesterol: 280,
      chest_pain_type: "4",
      fasting_bs: "Yes",
      resting_ecg: "2",
      max_heart_rate: 120,
      exercise_angina: "Yes",
      st_depression: 2.6,
      slope: "3",
      major_vessels: 2,
      thal: "7",
    },
  },
  fever: {
    normal: {
      age: 28,
      bmi: 22.1,
      temperature: 36.7,
      humidity: 45,
      air_quality: 40,
      heart_rate: 72,
      gender: "Female",
      headache: "No",
      body_ache: "No",
      fatigue: "No",
      chronic_conditions: "No",
      allergies: "No",
      smoking_history: "No",
      alcohol_consumption: "No",
      physical_activity: "Active",
      diet_type: "Vegetarian",
      blood_pressure: "Normal",
      previous_medication: "None",
    },
    abnormal: {
      age: 35,
      bmi: 27.5,
      temperature: 39.5,
      humidity: 75,
      air_quality: 110,
      heart_rate: 105,
      gender: "Male",
      headache: "Yes",
      body_ache: "Yes",
      fatigue: "Yes",
      chronic_conditions: "Yes",
      allergies: "Yes",
      smoking_history: "Yes",
      alcohol_consumption: "Yes",
      physical_activity: "Sedentary",
      diet_type: "Non-Vegetarian",
      blood_pressure: "High",
      previous_medication: "Other",
    },
  },
  anemia: {
    normal: {
      gender: "Female",
      rbc: 4.8,
      hemoglobin: 13.5,
      hematocrit: 40,
      mcv: 88,
      mch: 29,
      mchc: 33,
      wbc: 7.2,
      platelets: 260,
      rdw: 12.8,
      pdw: 11.5,
      pct: 0.25,
      lymphocytes: 32,
      neutrophils_pct: 55,
      neutrophils_num: 4.1,
    },
    abnormal: {
      gender: "Female",
      rbc: 3.1,
      hemoglobin: 8.6,
      hematocrit: 28,
      mcv: 74,
      mch: 24,
      mchc: 30,
      wbc: 10.8,
      platelets: 420,
      rdw: 17.5,
      pdw: 16.2,
      pct: 0.12,
      lymphocytes: 20,
      neutrophils_pct: 72,
      neutrophils_num: 6.5,
    },
  },
};

const diabetesForm = document.getElementById("diabetes-form");
const heartForm = document.getElementById("heart-form");
const feverForm = document.getElementById("fever-form");
const anemiaForm = document.getElementById("anemia-form");

const chatForm = document.getElementById("chat-form");
const chatHistory = document.getElementById("chat-history");
const chatInput = document.getElementById("chat-input");
const chatbotLauncher = document.getElementById("chatbot-launcher");
const chatbotModal = document.getElementById("chatbot-modal");
const chatbotOverlay = document.getElementById("chatbot-overlay");
const chatbotClose = document.getElementById("chatbot-close");

const profileSearch = document.getElementById("profile-search");
const refreshProfiles = document.getElementById("refresh-profiles");
const profilesGrid = document.getElementById("profiles-grid");


const consultantSearch = document.getElementById("consultant-search");
const refreshConsultants = document.getElementById("refresh-consultants");
const hospitalList = document.getElementById("hospital-list");
const doctorList = document.getElementById("doctor-list");
const consultantTabButtons = Array.from(document.querySelectorAll(".consultant-tab"));
const consultantViews = Array.from(document.querySelectorAll(".consultant-view"));

const state = {
  profile: null,
  predictions: {},
  normals: {},
  chatHistory: [],
};

let pendingTestInputsDisease = null;
let historyInitialized = false;
let isRestoringHistory = false;
let isDiseasesDropdownOpen = false;
const diseaseTabKeys = new Set(["diabetes", "heart", "fever", "anemia"]);
const diseaseTabLabels = {
  diabetes: "Diabetes",
  heart: "Heart Disease",
  fever: "Fever",
  anemia: "Anemia",
};
let modalOpenCount = 0;
let chatTypingIndicator = null;
const REPORT_ALLOWED_EXTENSIONS = new Set([".csv", ".pdf", ".xls", ".xlsx"]);
const REPORT_ALLOWED_MIME_TYPES = new Set([
  "text/csv",
  "application/pdf",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/octet-stream",
]);
const MAX_REPORT_SIZE_BYTES = 200 * 1024 * 1024;

function getStoredTheme() {
  try {
    const storedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    if (storedTheme === "light" || storedTheme === "dark") {
      return storedTheme;
    }
  } catch (error) {
    console.warn("Unable to read theme preference:", error);
  }
  return null;
}

function persistTheme(theme) {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (error) {
    console.warn("Unable to persist theme preference:", error);
  }
}

function resetThemePreference() {
  try {
    localStorage.removeItem(THEME_STORAGE_KEY);
  } catch (error) {
    console.warn("Unable to clear theme preference:", error);
  }
  applyTheme(prefersDarkScheme?.matches ? "dark" : "light");
}

function resolveInitialTheme() {
  const storedTheme = getStoredTheme();
  if (storedTheme) {
    return storedTheme;
  }
  return prefersDarkScheme?.matches ? "dark" : "light";
}

function reflectTheme(theme) {
  if (!themeToggle) {
    return;
  }
  const isDark = theme === "dark";
  const nextLabel = isDark ? "Switch to light theme" : "Switch to dark theme";
  themeToggle.setAttribute("aria-pressed", String(isDark));
  themeToggle.setAttribute("aria-label", nextLabel);
  themeToggle.setAttribute("title", nextLabel);
  themeToggle.dataset.theme = theme;
  themeToggle.classList.toggle("is-dark", isDark);
}

function applyTheme(theme) {
  const resolvedTheme = theme === "dark" ? "dark" : "light";
  rootElement.dataset.theme = resolvedTheme;
  if (document.body) {
    document.body.dataset.theme = resolvedTheme;
  }
  reflectTheme(resolvedTheme);
}

function closeThemeMenu(options = {}) {
  if (!isThemeMenuOpen || !themeMenu) {
    return;
  }
  themeMenu.classList.remove("show");
  themeMenu.style.left = "";
  themeMenu.style.top = "";
  isThemeMenuOpen = false;
  themeToggle?.setAttribute("aria-expanded", "false");
  if (options.focusToggle) {
    themeToggle?.focus({ preventScroll: true });
  }
}

function openThemeMenu(position = {}) {
  if (!themeMenu || !themeToggle) {
    return;
  }

  closeThemeMenu();

  themeMenu.classList.add("show");
  themeMenu.style.left = "0px";
  themeMenu.style.top = "0px";

  const menuRect = themeMenu.getBoundingClientRect();
  const toggleRect = themeToggle.getBoundingClientRect();
  const margin = 8;
  let left;
  let top;

  if (typeof position.x === "number" && typeof position.y === "number") {
    left = position.x;
    top = position.y + margin;
  } else {
    left = toggleRect.left;
    top = toggleRect.bottom + margin;
  }

  const maxLeft = window.innerWidth - menuRect.width - margin;
  const maxTop = window.innerHeight - menuRect.height - margin;
  left = Math.max(margin, Math.min(maxLeft, left));
  top = Math.max(margin, Math.min(maxTop, top));

  themeMenu.style.left = `${left}px`;
  themeMenu.style.top = `${top}px`;

  themeToggle.setAttribute("aria-expanded", "true");
  isThemeMenuOpen = true;

  window.requestAnimationFrame(() => {
    themeMenu.querySelector("button")?.focus({ preventScroll: true });
  });
}

function isWithinThemeControls(target) {
  if (!target) {
    return false;
  }
  if (themeToggle && themeToggle.contains(target)) {
    return true;
  }
  if (themeMenu && themeMenu.contains(target)) {
    return true;
  }
  return false;
}

function handleThemeControlsPointerDown(event) {
  if (!isThemeMenuOpen) {
    return;
  }

  const target = event.target;
  if (!isWithinThemeControls(target)) {
    closeThemeMenu();
  }
}

function handleThemeControlsKeydown(event) {
  if (event.key === "Escape" && isThemeMenuOpen) {
    event.preventDefault();
    closeThemeMenu({ focusToggle: true });
  }
}

function handleThemeControlsFocusIn(event) {
  if (!isThemeMenuOpen) {
    return;
  }

  if (!isWithinThemeControls(event.target)) {
    closeThemeMenu();
  }
}

function initializeTheme() {
  applyTheme(resolveInitialTheme());

  if (themeToggle) {
    themeToggle.setAttribute("aria-expanded", "false");

    themeToggle.addEventListener("click", () => {
      closeThemeMenu();
      const currentTheme = rootElement.dataset.theme === "dark" ? "dark" : "light";
      const nextTheme = currentTheme === "dark" ? "light" : "dark";
      applyTheme(nextTheme);
      persistTheme(nextTheme);
    });

    themeToggle.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      openThemeMenu({ x: event.clientX, y: event.clientY });
    });

    themeToggle.addEventListener("keydown", (event) => {
      const key = event.key;
      const openViaKeyboard =
        key === "ArrowDown" ||
        key === "ContextMenu" ||
        (key === "F10" && event.shiftKey) ||
        (key === "Enter" && event.altKey);
      if (openViaKeyboard) {
        event.preventDefault();
        openThemeMenu();
      }
    });
  }

  if (themeMenu) {
    themeMenu.addEventListener("click", (event) => {
      const actionButton = event.target?.closest("button[data-theme-action]");
      if (!actionButton) {
        return;
      }

      const { themeAction } = actionButton.dataset;
      if (themeAction === "system") {
        resetThemePreference();
        closeThemeMenu({ focusToggle: true });
      }
    });
  }

  if (prefersDarkScheme) {
    const handleSystemThemeChange = (event) => {
      const storedTheme = getStoredTheme();
      if (storedTheme) {
        return;
      }
      applyTheme(event.matches ? "dark" : "light");
      closeThemeMenu();
    };

    if (typeof prefersDarkScheme.addEventListener === "function") {
      prefersDarkScheme.addEventListener("change", handleSystemThemeChange);
    } else if (typeof prefersDarkScheme.addListener === "function") {
      prefersDarkScheme.addListener(handleSystemThemeChange);
    }
  }

  if (themeToggle || themeMenu) {
    document.addEventListener("pointerdown", handleThemeControlsPointerDown);
    document.addEventListener("keydown", handleThemeControlsKeydown);
    document.addEventListener("focusin", handleThemeControlsFocusIn);
    window.addEventListener("resize", closeThemeMenu);
  }
}

function lockBodyScroll() {
  if (typeof window === "undefined" || !document?.body) {
    return;
  }

  if (modalOpenCount === 0) {
    const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;
    if (scrollbarWidth > 0) {
      document.body.style.paddingRight = `${scrollbarWidth}px`;
      document.body.dataset.scrollbarComp = "1";
    }
    document.body.classList.add("modal-open");
  }

  modalOpenCount += 1;
}

function unlockBodyScroll() {
  if (!document?.body || modalOpenCount === 0) {
    return;
  }

  modalOpenCount -= 1;
  if (modalOpenCount === 0) {
    document.body.classList.remove("modal-open");
    if (document.body.dataset.scrollbarComp) {
      document.body.style.paddingRight = "";
      delete document.body.dataset.scrollbarComp;
    }
  }
}

function sanitizeDispositionValue(value) {
  return String(value ?? "")
    .replace(/"/g, "%22")
    .replace(/[\r\n]/g, " ");
}

function createMultipartRequest(formData) {
  const boundary = `----CureHelpFormBoundary${Math.random().toString(16).slice(2)}`;
  const dashBoundary = `--${boundary}`;
  const chunks = [];

  formData.forEach((value, key) => {
    const safeKey = sanitizeDispositionValue(key);
    if (value instanceof File) {
      if (!value.name && value.size === 0) {
        return;
      }
      const safeFilename = sanitizeDispositionValue(value.name || "file");
      const contentType = value.type || "application/octet-stream";
      const fileHeader =
        `${dashBoundary}\r\n` +
        `Content-Disposition: form-data; name="${safeKey}"; filename="${safeFilename}"\r\n` +
        `Content-Type: ${contentType}\r\n\r\n`;
      chunks.push(fileHeader);
      chunks.push(value);
      chunks.push("\r\n");
    } else {
      const stringValue = value == null ? "" : String(value);
      const fieldPart =
        `${dashBoundary}\r\n` +
        `Content-Disposition: form-data; name="${safeKey}"\r\n\r\n` +
        `${stringValue}\r\n`;
      chunks.push(fieldPart);
    }
  });

  chunks.push(`${dashBoundary}--\r\n`);

  const body = new Blob(chunks, { type: `multipart/form-data; boundary=${boundary}` });
  return { body, contentType: `multipart/form-data; boundary=${boundary}` };
}

function startLoaderDelay(duration = 3000) {
  const hasWindow = typeof window !== "undefined";

  if (hasWindow && typeof window.startLoader === "function") {
    return window.startLoader(duration);
  }

  const shouldHideManually = !hasWindow || typeof window.showLoader3Sec !== "function";

  if (hasWindow && typeof window.showLoader3Sec === "function") {
    window.showLoader3Sec();
  } else if (loaderWrapper) {
    loaderWrapper.style.display = "block";
  }

  return new Promise((resolve) => {
    setTimeout(() => {
      if (shouldHideManually && loaderWrapper) {
        loaderWrapper.style.display = "none";
      }
      resolve();
    }, duration);
  });
}

function clampValue(value, { min, max }) {
  const raw = value == null ? "" : String(value).trim();
  if (raw === "") {
    return { numeric: value, clamped: value, changed: false, empty: true };
  }
  const numeric = Number(raw);
  if (Number.isNaN(numeric)) {
    return { numeric: value, clamped: value, changed: false, invalid: true };
  }
  let clamped = numeric;
  if (min !== undefined && numeric < min) {
    clamped = min;
  }
  if (max !== undefined && numeric > max) {
    clamped = max;
  }
  return { numeric, clamped, changed: clamped !== numeric };
}

function formatConstraintMessage({ min, max }) {
  if (min !== undefined && max !== undefined) {
    return `Value must be between ${min} and ${max}.`;
  }
  if (min !== undefined) {
    return `Value must be at least ${min}.`;
  }
  if (max !== undefined) {
    return `Value must be at most ${max}.`;
  }
  return "Invalid value.";
}

function getConstraintHint(input, { createIfMissing = true } = {}) {
  if (!input) return null;
  const hintHost = input.closest("label") || input.parentElement;
  if (!hintHost) return null;

  let hint = hintHost.querySelector(".input-constraint-hint");
  if (!hint && createIfMissing) {
    hint = document.createElement("span");
    hint.className = "input-constraint-hint";
    hintHost.appendChild(hint);
  }
  return hint;
}

function flagInputOutOfRange(input, message) {
  if (!input) return;
  input.classList.add("input-out-of-range");
  const hint = getConstraintHint(input);
  if (hint) {
    hint.textContent = message;
    hint.classList.add("visible");
    const previousTimer = hint.dataset.timerId ? Number(hint.dataset.timerId) : null;
    if (previousTimer) {
      window.clearTimeout(previousTimer);
    }
    const timerId = window.setTimeout(() => {
      hint.classList.remove("visible");
      input.classList.remove("input-out-of-range");
      hint.textContent = "";
      hint.dataset.timerId = "";
    }, 2400);
    hint.dataset.timerId = String(timerId);
  }
}

function clearInputRangeState(input) {
  if (!input) return;
  input.classList.remove("input-out-of-range");
  const hint = getConstraintHint(input, { createIfMissing: false });
  if (hint) {
    const previousTimer = hint.dataset.timerId ? Number(hint.dataset.timerId) : null;
    if (previousTimer) {
      window.clearTimeout(previousTimer);
      hint.dataset.timerId = "";
    }
    hint.classList.remove("visible");
    hint.textContent = "";
  }
}

function applyConstraintAttributes(input, constraints) {
  if (!input || !constraints) return;
  const { min, max, step } = constraints;
  if (min !== undefined) {
    input.min = String(min);
  }
  if (max !== undefined) {
    input.max = String(max);
  }
  if (step !== undefined) {
    input.step = String(step);
  }
}

function validateInputValue(input, constraints, { clampOnBlur = false } = {}) {
  if (!input || !constraints) {
    return true;
  }

  const result = clampValue(input.value, constraints);
  const { numeric, clamped, changed, invalid, empty } = result;
  if (invalid) {
    if (clampOnBlur) {
      input.value = "";
      flagInputOutOfRange(input, "Enter a valid number.");
    } else {
      input.classList.add("input-out-of-range");
    }
    return false;
  }

  if (empty) {
    clearInputRangeState(input);
    return true;
  }

  const outOfRange = changed;
  if (!outOfRange) {
    clearInputRangeState(input);
    return true;
  }

  const message = formatConstraintMessage(constraints);
  if (clampOnBlur) {
    input.value = clamped;
    flagInputOutOfRange(input, message);
    return true;
  }

  input.classList.add("input-out-of-range");
  return false;
}

function registerInputConstraints(form, constraints) {
  if (!form || !constraints) return;

  const numericInputs = Array.from(form.querySelectorAll("input[type='number']"));
  numericInputs.forEach((input) => {
    const fieldConstraints = constraints[input.name];
    if (!fieldConstraints) {
      return;
    }

    applyConstraintAttributes(input, fieldConstraints);
    clearInputRangeState(input);

    input.addEventListener("input", () => {
      validateInputValue(input, fieldConstraints, { clampOnBlur: false });
    });

    input.addEventListener("blur", () => {
      validateInputValue(input, fieldConstraints, { clampOnBlur: true });
    });

    input.addEventListener("change", () => {
      validateInputValue(input, fieldConstraints, { clampOnBlur: true });
    });
  });
}

function enforceFormConstraints(form, constraints) {
  if (!form || !constraints) {
    return true;
  }

  let valid = true;
  let firstInvalidInput = null;
  const numericInputs = Array.from(form.querySelectorAll("input[type='number']"));
  numericInputs.forEach((input) => {
    const fieldConstraints = constraints[input.name];
    if (!fieldConstraints) {
      return;
    }

    const fieldValid = validateInputValue(input, fieldConstraints, { clampOnBlur: true });
    if (!fieldValid) {
      valid = false;
      if (!firstInvalidInput) {
        firstInvalidInput = input;
      }
    }
  });

  if (!valid && firstInvalidInput) {
    firstInvalidInput.focus({ preventScroll: true });
  }

  return valid;
}

function initializeFormConstraints() {
  registerInputConstraints(patientForm, FORM_CONSTRAINTS["patient-form"]);
  registerInputConstraints(diabetesForm, FORM_CONSTRAINTS["diabetes-form"]);
  registerInputConstraints(heartForm, FORM_CONSTRAINTS["heart-form"]);
  registerInputConstraints(feverForm, FORM_CONSTRAINTS["fever-form"]);
  registerInputConstraints(anemiaForm, FORM_CONSTRAINTS["anemia-form"]);
}

function setSessionValue(key, value) {
  try {
    if (value === null || value === undefined) {
      sessionStorage.removeItem(key);
    } else {
      sessionStorage.setItem(key, value);
    }
  } catch (_) {
    /* storage disabled - ignore */
  }
}

function getSessionValue(key) {
  try {
    return sessionStorage.getItem(key);
  } catch (_) {
    return null;
  }
}

function getCurrentAppState() {
  return {
    page: getSessionValue("currentPage") || "landing",
    tab: getSessionValue("currentTab") || "diabetes",
  };
}

function openDiseasesDropdown() {
  if (!diseasesDropdownMenu) return;
  diseasesDropdownMenu.classList.remove("hidden");
  diseasesDropdownToggle?.setAttribute("aria-expanded", "true");
  isDiseasesDropdownOpen = true;
}

function closeDiseasesDropdown() {
  if (!diseasesDropdownMenu) return;
  diseasesDropdownMenu.classList.add("hidden");
  diseasesDropdownToggle?.setAttribute("aria-expanded", "false");
  isDiseasesDropdownOpen = false;
}

function toggleDiseasesDropdown() {
  if (!diseasesDropdownMenu) return;
  if (isDiseasesDropdownOpen) {
    closeDiseasesDropdown();
  } else {
    openDiseasesDropdown();
  }
}

function getFileExtension(filename = "") {
  const index = filename.lastIndexOf(".");
  return index === -1 ? "" : filename.slice(index).toLowerCase();
}

function isValidReportFile(file) {
  if (!file) {
    return { valid: true, message: "" };
  }

  if (file.size > MAX_REPORT_SIZE_BYTES) {
    return { valid: false, message: "Report exceeds the 200 MB size limit." };
  }

  const extension = getFileExtension(file.name || "");
  const mimeType = (file.type || "").toLowerCase();
  if (!REPORT_ALLOWED_EXTENSIONS.has(extension) && !REPORT_ALLOWED_MIME_TYPES.has(mimeType)) {
    return {
      valid: false,
      message: "Unsupported report type. Please upload a CSV, PDF, XLS, or XLSX file.",
    };
  }

  if (!REPORT_ALLOWED_EXTENSIONS.has(extension)) {
    return {
      valid: false,
      message: "Unsupported report extension. Allowed extensions: .csv, .pdf, .xls, .xlsx.",
    };
  }

  return { valid: true, message: "" };
}

function showReportLoadingModal() {
  if (!reportLoadingModal) return;
  reportLoadingModal.hidden = false;
  reportLoadingModal.classList.add("open");
  lockBodyScroll();
}

function hideReportLoadingModal() {
  if (!reportLoadingModal) return;
  if (!reportLoadingModal.classList.contains("open")) {
    reportLoadingModal.hidden = true;
    return;
  }
  reportLoadingModal.classList.remove("open");
  reportLoadingModal.hidden = true;
  unlockBodyScroll();
}

function updateHistoryState(partialState = {}, { replace = false } = {}) {
  if (typeof window === "undefined" || !window.history || isRestoringHistory) {
    return;
  }

  const baseState = getCurrentAppState();
  const nextState = {
    page: partialState.page || baseState.page,
    tab: partialState.tab || baseState.tab,
  };

  if (!historyInitialized || replace) {
    window.history.replaceState(nextState, "", window.location.pathname);
    historyInitialized = true;
    return;
  }

  const currentState = window.history.state || {};
  if (currentState.page === nextState.page && currentState.tab === nextState.tab) {
    window.history.replaceState(nextState, "", window.location.pathname);
    return;
  }

  window.history.pushState(nextState, "", window.location.pathname);
}

function isChatbotModalOpen() {
  return Boolean(chatbotModal?.classList.contains("open"));
}

function openChatbotModal() {
  closeDiseasesDropdown();
  if (!chatbotModal || !chatbotOverlay) return;
  if (isChatbotModalOpen()) {
    chatInput?.focus({ preventScroll: true });
    return;
  }
  chatbotModal.hidden = false;
  chatbotOverlay.hidden = false;
  chatbotModal.classList.add("open");
  chatbotOverlay.classList.add("open");
  lockBodyScroll();
  chatbotLauncher?.setAttribute("aria-expanded", "true");
  window.setTimeout(() => {
    chatInput?.focus({ preventScroll: true });
  }, 100);
}

function closeChatbotModal({ returnFocus = false } = {}) {
  if (!chatbotModal || !chatbotOverlay) return;
  if (!isChatbotModalOpen()) return;
  chatbotModal.classList.remove("open");
  chatbotOverlay.classList.remove("open");
  chatbotModal.hidden = true;
  chatbotOverlay.hidden = true;
  unlockBodyScroll();
  chatbotLauncher?.setAttribute("aria-expanded", "false");
  if (returnFocus && chatbotLauncher && !chatbotLauncher.hidden && chatbotLauncher.classList.contains("visible")) {
    chatbotLauncher.focus({ preventScroll: true });
  }
}

function updateThemeToggleVisibility(activePage) {
  if (!themeToggle) {
    return;
  }
  const isLanding = activePage === "landing";
  if (!isLanding) {
    closeThemeMenu();
  }
  themeToggle.hidden = !isLanding;
  themeToggle.style.display = isLanding ? "" : "none";
  themeToggle.setAttribute("aria-hidden", isLanding ? "false" : "true");
  themeToggle.tabIndex = isLanding ? 0 : -1;
  themeToggle.disabled = !isLanding;
  if (themeMenu && !isLanding) {
    themeMenu.hidden = true;
  } else if (themeMenu && isLanding && themeMenu.hidden) {
    themeMenu.hidden = false;
  }
}

function showPage(key, { recordHistory = true } = {}) {
  Object.entries(pages).forEach(([name, element]) => {
    const isActive = name === key;
    element.classList.toggle("active", isActive);
    element.classList.toggle("hidden", !isActive);
    element.hidden = !isActive;
  });
  const isDashboard = key === "dashboard";
  if (!isDashboard) {
    closeChatbotModal({ returnFocus: false });
  }
  if (chatbotLauncher) {
    chatbotLauncher.hidden = !isDashboard;
    chatbotLauncher.classList.toggle("visible", isDashboard);
    chatbotLauncher.tabIndex = isDashboard ? 0 : -1;
    if (!isChatbotModalOpen()) {
      chatbotLauncher.setAttribute("aria-expanded", "false");
    }
  }
  updateThemeToggleVisibility(key);
  setSessionValue("currentPage", key);
  if (recordHistory) {
    const { tab } = getCurrentAppState();
    updateHistoryState({ page: key, tab });
  }
}

function setPatientSummary(profile) {
  if (!profile) {
    patientSummary.textContent = "No patient selected.";
    return;
  }
  patientSummary.textContent = `Current Patient: ${profile.name} • Age ${profile.age} • ${profile.gender}`;
}

async function fetchCurrentProfile() {
  try {
    const response = await fetch("/api/profile");
    const payload = await response.json();
    if (!payload.success) throw new Error(payload.error || "Unable to fetch profile");
    state.profile = payload.profile || null;
    setPatientSummary(state.profile);
    applyProfileDemographics(state.profile);
    return state.profile;
  } catch (error) {
    console.warn("Failed to restore profile", error);
    state.profile = null;
    setPatientSummary(null);
    return null;
  }
}

async function fetchConfig() {
  try {
    const response = await fetch("/api/config");
    const payload = await response.json();
    if (payload.success) {
      state.normals = payload.normals || {};
    }
  } catch (error) {
    console.warn("Failed to fetch config", error);
  }
}

function activateTab(tabName, { recordHistory = true } = {}) {
  closeDiseasesDropdown();
  const previousTab = getSessionValue("currentTab");
  tabButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  if (diseasesDropdownToggle) {
    diseasesDropdownToggle.classList.toggle("active", diseaseTabKeys.has(tabName));
    if (diseasesDropdownLabel) {
      if (diseaseTabKeys.has(tabName)) {
        diseasesDropdownLabel.textContent = diseaseTabLabels[tabName] || "Diseases";
      } else {
        diseasesDropdownLabel.textContent = "Diseases";
      }
    }
  }
  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${tabName}-panel`);
  });
  setSessionValue("currentTab", tabName);
  if (recordHistory && previousTab !== tabName) {
    const { page } = getCurrentAppState();
    updateHistoryState({ page, tab: tabName });
  }
}

function activateConsultantView(targetId) {
  if (!targetId) return;
  consultantTabButtons.forEach((button) => {
    const isActive = button.dataset.target === targetId;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", isActive ? "true" : "false");
    button.setAttribute("tabindex", isActive ? "0" : "-1");
  });
  consultantViews.forEach((view) => {
    const isActive = view.id === targetId;
    view.classList.toggle("active", isActive);
    if (isActive) {
      view.removeAttribute("hidden");
      view.setAttribute("aria-hidden", "false");
    } else {
      view.setAttribute("hidden", "");
      view.setAttribute("aria-hidden", "true");
    }
  });
  setSessionValue("consultantTab", targetId);
}

async function handleTabChange(tabName) {
  if (tabName === "profiles") {
    const query = profileSearch ? profileSearch.value.trim() : "";
    await loadProfiles(query);
  } else if (tabName === "consultants") {
    const query = consultantSearch ? consultantSearch.value.trim() : "";
    await loadConsultants(query);
  }
}

function getFormData(form) {
  const formData = new FormData(form);
  const data = {};
  for (const [key, value] of formData.entries()) {
    data[key] = value;
  }
  return data;
}

function buildBarChart(container, disease, inputs = {}, normals = {}) {
  // Determine gender context from profile or provided inputs
  const rawGender = (state?.profile?.gender || inputs?.Gender || "").toString();
  const isMale = rawGender.toLowerCase() === "male";

  // Consider only numeric input fields
  let keys = Object.keys(inputs).filter(
    (key) => typeof inputs[key] === "number" || !Number.isNaN(Number(inputs[key]))
  );

  // Hide pregnancy bar for male context within Diabetes chart
  if (disease && String(disease).toLowerCase() === "diabetes" && isMale) {
    keys = keys.filter((k) => k.toLowerCase() !== "pregnancies");
  }

  const userValues = keys.map((key) => Number(inputs[key]));
  const normalValues = keys.map((key) => (key in normals ? Number(normals[key]) : 0));

  const traceUser = {
    type: "bar",
    name: "Your Value",
    x: keys,
    y: userValues,
    marker: { color: "#2e70ffff" },
  };

  const traceNormal = {
    type: "bar",
    name: "Normal",
    x: keys,
    y: normalValues,
    marker: { color: "#9e9e9eff" },
  };

  const layout = {
    barmode: "group",
    height: 400,
    margin: { t: 60, r: 10, l: 40, b: 80 },
    legend: {
      orientation: "h",
      x: 1,
      y: 1.15,
      xanchor: "right",
      yanchor: "top",
      font: { size: 12 },
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    title: undefined,
    bargap: 0.15,
    bargroupgap: 0,
  };

  Plotly.newPlot(container, [traceUser, traceNormal], layout, { displayModeBar: false, responsive: true });
}

function buildGauge(container, value) {
  const gauge = {
    type: "indicator",
    mode: "gauge+number",
    value,
    gauge: {
      axis: { range: [0, 100] },
      bar: { color: "#f97316", thickness: 0.25 },
      steps: [
        { range: [0, 40], color: "#16a34a" },
        { range: [40, 60], color: "#22c55e" },
        { range: [60, 80], color: "#f59e0b" },
        { range: [80, 100], color: "#ef4444" },
      ],
    },
  };

  Plotly.newPlot(container, [gauge], {
    height: 380,
    margin: { t: 40, r: 10, l: 10, b: 10 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
  }, { displayModeBar: false, responsive: true });
}

function riskBadge(probability) {
  if (probability < 35) return { text: "Low", className: "badge low" };
  if (probability < 70) return { text: "Moderate", className: "badge medium" };
  return { text: "High", className: "badge high" };
}

function renderRecommendations(list = [], title = "") {
  if (!list.length) return "<p class=\"muted\">No items available.</p>";
  const items = list.map((item) => `<li>${item}</li>`).join("");
  return `<div><h4>${title}</h4><ul>${items}</ul></div>`;
}

function renderRiskView(targetId, payload) {
  const container = document.getElementById(targetId);
  if (!container) return;

  const { disease, probability, prob, score, severity, inputs = {}, normal_values = {}, recommendations = {} } = payload;

  let probabilityValue = [probability, prob, score]
    .map((value) => (value === null || value === undefined ? null : Number(value)))
    .find((value) => Number.isFinite(value));

  if (!Number.isFinite(probabilityValue)) {
    probabilityValue = 0;
  } else if (probabilityValue > 0 && probabilityValue <= 1) {
    probabilityValue *= 100;
  }

  probabilityValue = Math.min(Math.max(probabilityValue, 0), 100);

  const badge = riskBadge(probabilityValue);
  const barId = `${targetId}-bar-${Date.now()}`;
  const gaugeId = `${targetId}-gauge-${Date.now()}`;

  const diseaseLabel = disease || "Selected disease";

  container.innerHTML = `
    <div class="result-card">
      <div class="result-actions">
        <button type="button" class="result-action-button result-back-button" data-back="${targetId}" aria-label="Back to inputs">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M15 18L9 12L15 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
        <button type="button" class="result-action-button result-download-button" data-disease="${disease || ""}" aria-label="Download ${diseaseLabel} report">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 3v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
            <path d="M6 11l6 6 6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
            <path d="M5 19h14" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
          </svg>
        </button>
      </div>
      <div class="result-head">
        <span class="${badge.className}"> ${badge.text} Risk • ${probabilityValue.toFixed(1)}%</span>
      </div>
      ${severity ? `<p class="muted">Predicted Severity: <strong>${severity}</strong></p>` : ""}
      <div class="charts-row">
        <div class="chart-box"><div id="${barId}"></div></div>
        <div class="gauge-box"><div id="${gaugeId}"></div></div>
      </div>
      <div class="recommendations">
        ${renderRecommendations(recommendations.prevention_measures, "Risk Reduction Protocols")}
        ${renderRecommendations(recommendations.medicine_suggestions, "Clinical Interventions")}
      </div>
    </div>
  `;

  buildBarChart(barId, disease, inputs || {}, normal_values || state.normals[disease?.toLowerCase()] || {});
  buildGauge(gaugeId, probabilityValue);
}

function hidePanelInputs(resultId) {
  const container = document.getElementById(resultId);
  if (!container) {
    return;
  }
  const panel = container.closest(".tab-panel");
  if (panel) {
    panel.classList.add("result-only");
  }
}

function restorePanelInputs(panel) {
  panel.classList.remove("result-only");
  const resultContainer = panel.querySelector(".result-container");
  if (resultContainer) {
    resultContainer.innerHTML = "";
  }
}

async function submitPrediction(form, url, resultId) {
  if (typeof form?.reportValidity === "function" && !form.reportValidity()) {
    return;
  }
  const constraints = form?.id && FORM_CONSTRAINTS[form.id];
  if (constraints && !enforceFormConstraints(form, constraints)) {
    return;
  }
  const data = getFormData(form);
  const panel = form.closest(".tab-panel");
  panel?.classList.add("loader-hidden");
  const loaderDelay = startLoaderDelay();
  try {
    Object.keys(data).forEach((key) => {
      if (data[key] === "") delete data[key];
    });

    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    let payload;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      try {
        payload = await response.json();
      } catch (err) {
        throw new Error("Invalid JSON response from server.");
      }
    } else {
      const responseText = await response.text();
      throw new Error(responseText || "Unexpected server response.");
    }
    if (!response.ok || !payload.success) {
      throw new Error(payload.error || "Prediction failed");
    }

    state.predictions = state.predictions || {};
    const diseaseKey = (payload.disease || "").toLowerCase() || resultId;
    state.predictions[diseaseKey] = payload;
    await loaderDelay;
    renderRiskView(resultId, payload);
    hidePanelInputs(resultId);
  } catch (error) {
    await loaderDelay;
    const container = document.getElementById(resultId);
    if (container) {
      container.innerHTML = `<div class="result-card"><p class="muted">${error.message}</p></div>`;
    }
    panel?.classList.remove("result-only");
  } finally {
    panel?.classList.remove("loader-hidden");
  }
}

async function downloadDiseaseReport(disease) {
  if (!disease) return;
  try {
    const params = new URLSearchParams({ disease });
    const response = await fetch(`/api/report/pdf?${params.toString()}`);
    if (!response.ok) {
      let message = "Unable to download report";
      try {
        const payload = await response.json();
        if (payload?.error) message = payload.error;
      } catch (_) {
        /* non-JSON error */
      }
      throw new Error(message);
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    const safeName = disease.replace(/[^a-z0-9]+/gi, "_").replace(/^_+|_+$/g, "");
    a.href = url;
    a.download = `CureHelp_${safeName || "Report"}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (error) {
    alert(error.message);
  }
}

function renderChatMessage(role, content) {
  const bubble = document.createElement("div");
  bubble.className = `chat-message ${role}`;
  bubble.innerHTML = content;
  chatHistory.appendChild(bubble);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function showChatTyping() {
  if (chatTypingIndicator) {
    return;
  }
  const bubble = document.createElement("div");
  bubble.className = "chat-message bot typing";
  bubble.innerHTML = `
    <span class="typing-dots" aria-hidden="true">
      <span></span>
      <span></span>
      <span></span>
    </span>
    <span class="sr-only">Assistant is typing…</span>
  `;
  chatHistory.appendChild(bubble);
  chatTypingIndicator = bubble;
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function hideChatTyping() {
  if (!chatTypingIndicator) {
    return;
  }
  chatTypingIndicator.remove();
  chatTypingIndicator = null;
}

function formatChatAnalysis(analysis) {
  if (analysis.type === "question" && analysis.faq_answer) {
    return `
      <strong>FAQ Answer</strong><br />
      <strong>Q:</strong> ${analysis.faq_question}<br />
      <strong>A:</strong> ${analysis.faq_answer}
    `;
  }

  if (analysis.type === "disease" || analysis.type === "symptoms") {
    const disease = analysis.disease ? `<strong>Predicted Condition:</strong> ${analysis.disease}` : "";

    let symptomsPrecautions = "";
    if (analysis.symptoms?.length || analysis.precautions?.length) {
      const symptomsList = (analysis.symptoms || []).map((s) => `<li>${s}</li>`).join("");
      const precautionsList = (analysis.precautions || []).map((p) => `<li>${p}</li>`).join("");
      symptomsPrecautions = `
        <div class="chat-analysis-grid">
          <div>
            <strong>Associated Symptoms</strong>
            <ul>${symptomsList || "<li class=\"muted\">None provided</li>"}</ul>
          </div>
          <div>
            <strong>Precautions</strong>
            <ul>${precautionsList || "<li class=\"muted\">None provided</li>"}</ul>
          </div>
        </div>
      `;
    }
    const description = analysis.description ? `<strong>Description:</strong><br />${analysis.description}` : "";
    return [disease, symptomsPrecautions, description].filter(Boolean).join("<br /><br />") || "I could not interpret that input.";
  }

  return analysis.message || "I could not find a relevant answer. Please try rephrasing.";
}

async function submitChat(event) {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  renderChatMessage("user", message);
  chatInput.value = "";
  showChatTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const payload = await response.json();
    if (!payload.success) throw new Error(payload.error || "Assistant unavailable.");
    const analysis = payload.response?.analysis || {};
    hideChatTyping();
    renderChatMessage("bot", formatChatAnalysis(analysis));
  } catch (error) {
    hideChatTyping();
    renderChatMessage("bot", `<span class="muted">${error.message}</span>`);
  }
}

function renderProfileCard(profile, currentProfileId = state.profile?.id) {
  const predictions = profile.predictions || {};
  const summary = Object.entries(predictions)
    .map(([disease, data = {}]) => {
      const probabilitySource = [data.prob, data.probability, data.score]
        .map((value) => (value === null || value === undefined ? null : Number(value)))
        .find((value) => Number.isFinite(value));
      let probabilityValue = Number.isFinite(probabilitySource) ? probabilitySource : 0;
      if (probabilityValue > 0 && probabilityValue <= 1) {
        probabilityValue *= 100;
      }
      probabilityValue = Math.min(Math.max(probabilityValue, 0), 100);
      return `${disease}: ${probabilityValue.toFixed(1)}%${data.severity ? ` (${data.severity})` : ""}`;
    })
    .join("<br />");

  const isCurrent = Boolean(currentProfileId && profile.id && profile.id === currentProfileId);
  const currentNote = "";

  return `
    <div class="profile-card${isCurrent ? " current" : ""}" data-profile-id="${profile.id || ""}">
      <h4>${profile.name || "Unknown"}</h4>
      <p>ID: ${profile.id || "-"}</p>
      <p>${profile.age ? `Age: ${profile.age}` : ""} ${profile.gender ? `• ${profile.gender}` : ""}</p>
      <p>Contact: ${profile.contact || "-"}</p>
      <p class="muted">${profile.address || ""}</p>
      <p><strong>Predictions:</strong><br />${summary || "No predictions"}</p>
      <p class="muted">Updated: ${profile.last_updated || profile.created_at || "-"}</p>
      ${currentNote}
    </div>
  `;
}

async function loadProfiles(query = "") {
  try {
    const url = query ? `/api/profiles?q=${encodeURIComponent(query)}` : "/api/profiles";
    const response = await fetch(url);
    const payload = await response.json();
    if (!payload.success) throw new Error(payload.error || "Unable to load profiles");
    const profiles = payload.profiles || [];
    const currentProfileId = state.profile?.id || null;
    profilesGrid.innerHTML = profiles.length
      ? profiles.map((profile) => renderProfileCard(profile, currentProfileId)).join("")
      : '<div class="profile-card"><p class="muted">No saved profiles yet.</p></div>';
  } catch (error) {
    profilesGrid.innerHTML = `<div class="profile-card"><p class="muted">${error.message}</p></div>`;
  }
}

function getConsultantInitials(name = "") {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "DR";
  const initials = parts.slice(0, 2).map((part) => part[0].toUpperCase()).join("");
  return initials || "DR";
}

function renderConsultantCard(item, { isDoctor = false, isHospital = false, imageUrl = "" } = {}) {
  if (!isDoctor) {
    const initials = getConsultantInitials(item.name || "Hospital");
    const avatarMarkup = imageUrl
      ? `<img src="${imageUrl}" alt="${item.name || "Hospital"}" loading="lazy" />`
      : initials || "H";
    return `
      <div class="consultant-card hospital-card">
        <div class="hospital-card-header">
          <div class="hospital-avatar" aria-hidden="true">${avatarMarkup}</div>
          <div class="hospital-info">
            <h4>${item.name}</h4>
            <p class="hospital-speciality">${item.speciality || item.specialization || ""}</p>
          </div>
        </div>
        <div class="hospital-meta">
          <p>${item.address || ""}</p>
          <p>Contact: ${item.contact || "-"}</p>
          <p class="muted">${item.distance ? `Distance: ${item.distance}` : ""}</p>
        </div>
        <div class="form-actions">
          ${item.website_url ? `<a href="${item.website_url}" target="_blank">Website</a>` : ""}
          ${item.location_url ? `<a href="${item.location_url}" target="_blank">Location</a>` : ""}
        </div>
      </div>
    `;
  }

  const speciality = item.speciality || item.specialization || "";
  const address = item.address || "";
  const distance = item.distance ? `Distance: ${item.distance}` : "";
  const initials = getConsultantInitials(item.name || "Doctor");
  const avatarMarkup = imageUrl
    ? `<img src="${imageUrl}" alt="${item.name || "Doctor"}" loading="lazy" />`
    : initials;

  return `
    <div class="consultant-card doctor-card">
      <div class="doctor-card-header">
        <div class="doctor-avatar" aria-hidden="true">${avatarMarkup}</div>
        <div class="doctor-info">
          <h4>${item.name}</h4>
          <p class="doctor-speciality">${speciality}</p>
        </div>
      </div>
      <div class="doctor-meta">
        <p>${address}</p>
        <p>Contact: ${item.contact || "-"}</p>
        <p class="muted">${distance}</p>
      </div>
      <div class="form-actions">
        ${item.website_url ? `<a href="${item.website_url}" target="_blank">Website</a>` : ""}
        ${item.location_url ? `<a href="${item.location_url}" target="_blank">Location</a>` : ""}
      </div>
    </div>
  `;
}

async function loadConsultants(query = "") {
  try {
    const url = query ? `/api/consultants?q=${encodeURIComponent(query)}` : "/api/consultants";
    const response = await fetch(url);
    const payload = await response.json();
    if (!payload.success) throw new Error(payload.error || "Unable to load consultants");
    const { hospitals = [], doctors = [] } = payload.data || {};
    const hospitalImages = [
      "/static/assets/hospitals/hos1.jpg",
      "/static/assets/hospitals/hos2.jpg",
      "/static/assets/hospitals/hos3.jpg",
      "/static/assets/hospitals/hos4.jpg",
      "/static/assets/hospitals/hos5.jpg",
      "/static/assets/hospitals/hos6.jpg",
      "/static/assets/hospitals/hos7.jpg",
      "/static/assets/hospitals/hos8.jpg",
      "/static/assets/hospitals/hos9.jpg",
      "/static/assets/hospitals/hos10.jpg",
      "/static/assets/hospitals/hos11.jpg",
      "/static/assets/hospitals/hos12.jpg",
    ];
    hospitalList.innerHTML = hospitals.length
      ? hospitals.map((item, index) => renderConsultantCard(item, {
          isDoctor: false,
          isHospital: true,
          imageUrl: hospitalImages[index] || "",
        })).join("")
      : '<div class="consultant-card"><p class="muted">No hospitals found.</p></div>';
    const doctorImages = [
      "/static/assets/doctors/doc1.jpg",
      "/static/assets/doctors/doc2.jpg",
      "/static/assets/doctors/doc3.jpg",
      "/static/assets/doctors/doc4.jpg",
      "/static/assets/doctors/doc5.jpg",
      "/static/assets/doctors/doc6.jpg",
      "/static/assets/doctors/doc7.jpg",
      "/static/assets/doctors/doc8.jpg",
      "/static/assets/doctors/doc9.jpg",
      "/static/assets/doctors/doc10.jpg",
      "/static/assets/doctors/doc11.jpg",
      "/static/assets/doctors/doc12.jpg",
      "/static/assets/doctors/doc13.jpg",
      "/static/assets/doctors/doc14.jpg",
    ];
    doctorList.innerHTML = doctors.length
      ? doctors.map((item, index) => renderConsultantCard(item, {
          isDoctor: true,
          imageUrl: doctorImages[index] || "",
        })).join("")
      : '<div class="consultant-card"><p class="muted">No doctors found.</p></div>';
  } catch (error) {
    const errorMarkup = `<div class="consultant-card"><p class="muted">${error.message}</p></div>`;
    hospitalList.innerHTML = errorMarkup;
    doctorList.innerHTML = errorMarkup;
  }
}

function applyAutofillValues(values = {}) {
  const formMap = {
    diabetes: diabetesForm,
    heart: heartForm,
    fever: feverForm,
    anemia: anemiaForm,
  };

  Object.entries(values).forEach(([diseaseKey, fieldValues = {}]) => {
    const form = formMap[diseaseKey];
    if (!form) return;

    Object.entries(fieldValues).forEach(([fieldName, fieldValue]) => {
      if (fieldValue === null || fieldValue === undefined) return;
      const elements = form.querySelectorAll(`[name="${fieldName}"]`);
      if (!elements.length) return;

      const stringValue = typeof fieldValue === "string" ? fieldValue : String(fieldValue);

      elements.forEach((element) => {
        if (element instanceof HTMLInputElement) {
          if (element.type === "radio" || element.type === "checkbox") {
            element.checked = element.value === stringValue;
          } else {
            element.value = stringValue;
          }
        } else if (element instanceof HTMLSelectElement) {
          const optionExists = Array.from(element.options).some((option) => option.value === stringValue);
          if (optionExists) {
            element.value = stringValue;
          }
        } else if (element instanceof HTMLTextAreaElement) {
          element.value = stringValue;
        }
      });
    });
  });
}

function isTestInputsModalOpen() {
  return Boolean(testInputsModal?.classList.contains("open"));
}

function closeTestInputsModal() {
  if (!testInputsModal) return;
  if (!testInputsModal.classList.contains("open")) {
    testInputsModal.hidden = true;
    pendingTestInputsDisease = null;
    return;
  }
  pendingTestInputsDisease = null;
  testInputsModal.classList.remove("open");
  testInputsModal.hidden = true;
  unlockBodyScroll();
}

function openTestInputsModal(diseaseKey) {
  if (!testInputsModal) return;
  const presets = TEST_INPUT_PRESETS[diseaseKey];
  if (!presets) {
    console.warn(`No test input presets defined for ${diseaseKey}.`);
    return;
  }

  pendingTestInputsDisease = diseaseKey;

  const label = diseaseTabLabels[diseaseKey] || diseaseKey.charAt(0).toUpperCase() + diseaseKey.slice(1);
  if (testInputsTitle) {
    testInputsTitle.textContent = `${label} Sample Inputs`;
  }
  if (testInputsMessage) {
    testInputsMessage.textContent = "All the input values are real and authentic.";
  }

  lockBodyScroll();
  testInputsModal.hidden = false;
  testInputsModal.classList.add("open");

  if (testInputsNormal) {
    testInputsNormal.focus();
  }
}

function handleTestInputsSelection(variant) {
  if (!pendingTestInputsDisease) {
    return;
  }

  const presets = TEST_INPUT_PRESETS[pendingTestInputsDisease];
  if (!presets) {
    closeTestInputsModal();
    return;
  }

  const values = presets[variant];
  if (!values) {
    console.warn(`Missing ${variant} preset for ${pendingTestInputsDisease}.`);
    return;
  }

  applyAutofillValues({ [pendingTestInputsDisease]: values });
  closeTestInputsModal();
}

function applyProfileDemographics(profile) {
  if (!profile) return;

  const genderValue = typeof profile.gender === "string" ? profile.gender.trim() : "";
  const hasGender = Boolean(genderValue);

  const rawAge = profile.age;
  let hasAge = false;
  let ageFieldValue = null;

  if (typeof rawAge === "number" && Number.isFinite(rawAge)) {
    hasAge = true;
    ageFieldValue = rawAge;
  } else if (typeof rawAge === "string") {
    const trimmed = rawAge.trim();
    if (trimmed) {
      const parsed = Number(trimmed);
      if (Number.isFinite(parsed)) {
        ageFieldValue = parsed;
      } else {
        ageFieldValue = trimmed;
      }
      hasAge = true;
    }
  }

  if (!hasGender && !hasAge) {
    return;
  }

  const defaults = {};

  const addDemographics = (diseaseKey, { includeAge = true, includeGender = true } = {}) => {
    if (!includeAge && !includeGender) return;
    if (!defaults[diseaseKey]) {
      defaults[diseaseKey] = {};
    }
    if (includeGender && hasGender) {
      defaults[diseaseKey].gender = genderValue;
    }
    if (includeAge && hasAge) {
      defaults[diseaseKey].age = ageFieldValue;
    }
    if (Object.keys(defaults[diseaseKey]).length === 0) {
      delete defaults[diseaseKey];
    }
  };

  addDemographics("diabetes");
  addDemographics("heart");
  addDemographics("fever");
  addDemographics("anemia", { includeAge: false });

  if (Object.keys(defaults).length > 0) {
    applyAutofillValues(defaults);
  }
}

async function createProfile(event) {
  event.preventDefault();
  if (typeof patientForm?.reportValidity === "function" && !patientForm.reportValidity()) {
    return;
  }
  if (!enforceFormConstraints(patientForm, FORM_CONSTRAINTS["patient-form"])) {
    return;
  }
  const reportFile = medicalReportInput?.files?.[0] || null;
  const validation = isValidReportFile(reportFile);
  if (!validation.valid) {
    alert(validation.message);
    return;
  }

  const formData = new FormData(patientForm);
  const multipart = createMultipartRequest(formData);
  const loaderDelay = startLoaderDelay();
  patientCard?.classList.add("loader-hidden");

  if (reportFile) {
    showReportLoadingModal();
  }

  try {
    const response = await fetch("/api/profile", {
      method: "POST",
      headers: {
        "Content-Type": multipart.contentType,
      },
      body: multipart.body,
    });
    let payload;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      try {
        payload = await response.json();
      } catch (err) {
        throw new Error("Invalid JSON response from server.");
      }
    } else {
      const responseText = await response.text();
      throw new Error(responseText || "Unexpected server response.");
    }
    if (!response.ok || !payload.success) throw new Error(payload.error || "Unable to create profile");
    state.profile = payload.profile;
    setPatientSummary(state.profile);
    if (payload.autofill) {
      applyAutofillValues(payload.autofill);
    }
    applyProfileDemographics(state.profile);
    const loadProfilesPromise = loadProfiles();
    await Promise.all([loaderDelay, loadProfilesPromise]);
    await enterDashboard("diabetes");
  } catch (error) {
    await loaderDelay;
    alert(error.message);
  } finally {
    patientCard?.classList.remove("loader-hidden");
    if (reportFile) {
      hideReportLoadingModal();
    }
  }
}

async function resetSession(options = {}) {
  const { redirect = true, recordHistory = true } = options;
  await fetch("/api/reset", { method: "POST" });
  state.profile = null;
  state.predictions = {};
  patientForm.reset();
  diabetesForm.reset();
  heartForm.reset();
  feverForm.reset();
  anemiaForm.reset();
  chatHistory.innerHTML = "";
  profilesGrid.innerHTML = "";
  setPatientSummary(null);
  setSessionValue("currentPage", "landing");
  setSessionValue("currentTab", "diabetes");
  hideReportLoadingModal();
  if (redirect) {
    showPage("landing", { recordHistory });
  }
}

async function enterDashboard(tabName, { recordHistory = true } = {}) {
  const targetTab = tabName === "assistant" ? "diabetes" : tabName || "diabetes";
  showPage("dashboard", { recordHistory });
  activateTab(targetTab, { recordHistory });
  await handleTabChange(targetTab);
}

async function restorePageState() {
  const storedPage = getSessionValue("currentPage");
  let storedTab = getSessionValue("currentTab") || "diabetes";
  if (storedTab === "assistant") {
    storedTab = "diabetes";
    setSessionValue("currentTab", storedTab);
  }
  const profile = await fetchCurrentProfile();

  if (storedPage === "patient") {
    showPage("patient", { recordHistory: false });
    return;
  }

  if (storedPage === "dashboard" && profile) {
    await enterDashboard(storedTab, { recordHistory: false });
    return;
  }

  if (storedPage === "dashboard" && !profile) {
    setSessionValue("currentPage", null);
    setSessionValue("currentTab", null);
  }

  showPage("landing", { recordHistory: false });
}

function setupGooButton(button) {
  if (!button) return;

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

  const resetButton = () => {
    button.style.setProperty("--x", 50);
    button.style.setProperty("--y", 50);
    button.style.removeProperty("--a");
  };

  resetButton();

  const syncPointerPosition = (clientX, clientY, rect) => {
    const normalizedX = ((clientX - rect.left) / rect.width) * 100;
    const normalizedY = ((clientY - rect.top) / rect.height) * 100;
    button.style.setProperty("--x", clamp(normalizedX, 0, 100));
    button.style.setProperty("--y", clamp(normalizedY, 0, 100));
  };

  const handleAmbientPointerMove = (event) => {
    const rect = button.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    const dx = event.clientX - centerX;
    const dy = event.clientY - centerY;
    const distance = Math.hypot(dx, dy);
    const influenceRadius = Math.max(rect.width, rect.height) * 0.55;

    if (distance <= influenceRadius) {
      button.style.setProperty("--a", "100%");
      syncPointerPosition(event.clientX, event.clientY, rect);
    } else {
      resetButton();
    }
  };

  const pointerMoveHandler = (event) => {
    const rect = button.getBoundingClientRect();
    button.style.setProperty("--a", "100%");
    syncPointerPosition(event.clientX, event.clientY, rect);
  };

  const pointerLeaveHandler = () => {
    resetButton();
  };

  button.addEventListener("pointermove", pointerMoveHandler);
  button.addEventListener("pointerleave", pointerLeaveHandler);
  window.addEventListener("pointermove", handleAmbientPointerMove, { passive: true });
}

function bindEvents() {
  startButton?.addEventListener("click", () => showPage("patient"));
  backButton?.addEventListener("click", () => showPage("landing"));
  resetSessionButton?.addEventListener("click", resetSession);
  patientForm?.addEventListener("submit", createProfile);
  chatbotLauncher?.setAttribute("aria-expanded", "false");

  tabTriggers.forEach((trigger) => {
    trigger.addEventListener("click", async (event) => {
      event?.preventDefault();
      const tabName = trigger.dataset.tab;
      if (!tabName) return;

      if (diseasesDropdownMenu?.contains(trigger)) {
        closeDiseasesDropdown();
      }

      if (tabName === "assistant") {
        openChatbotModal();
        return;
      }

      activateTab(tabName);
      try {
        await handleTabChange(tabName);
      } catch (error) {
        console.warn(`Failed to load ${tabName} data`, error);
      }
    });
  });

  if (diseasesDropdownToggle && diseasesDropdownMenu) {
    diseasesDropdownToggle.setAttribute("aria-expanded", "false");
    diseasesDropdownToggle.addEventListener("click", (event) => {
      event.preventDefault();
      toggleDiseasesDropdown();
    });

    diseasesDropdownMenu.addEventListener("click", (event) => {
      const tabTrigger = event.target.closest("[data-tab]");
      if (tabTrigger) {
        closeDiseasesDropdown();
      }
    });

    document.addEventListener("click", (event) => {
      if (!isDiseasesDropdownOpen) return;
      if (diseasesDropdownToggle.contains(event.target)) return;
      if (diseasesDropdownMenu.contains(event.target)) return;
      closeDiseasesDropdown();
    });
  }

  diabetesForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrediction(diabetesForm, "/api/diabetes", "diabetes-result");
  });

  heartForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrediction(heartForm, "/api/heart", "heart-result");
  });

  feverForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrediction(feverForm, "/api/fever", "fever-result");
  });

  anemiaForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrediction(anemiaForm, "/api/anemia", "anemia-result");
  });

  chatForm?.addEventListener("submit", submitChat);
  chatbotLauncher?.addEventListener("click", () => openChatbotModal());
  chatbotClose?.addEventListener("click", () => closeChatbotModal({ returnFocus: true }));
  chatbotOverlay?.addEventListener("click", () => closeChatbotModal({ returnFocus: true }));

  tabPanels.forEach((panel) => {
    panel.addEventListener("click", (event) => {
      const downloadTarget = event.target.closest(".result-download-button");
      if (downloadTarget) {
        const { disease } = downloadTarget.dataset;
        downloadDiseaseReport(disease);
        return;
      }

      const backTarget = event.target.closest(".result-back-button");
      if (backTarget) {
        restorePanelInputs(panel);
      }
    });
  });

  testInputButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const diseaseKey = button.dataset.disease;
      if (!diseaseKey) return;
      openTestInputsModal(diseaseKey);
    });
  });

  testInputsNormal?.addEventListener("click", () => handleTestInputsSelection("normal"));
  testInputsAbnormal?.addEventListener("click", () => handleTestInputsSelection("abnormal"));
  testInputsModal?.addEventListener("click", (event) => {
    if (event.target === testInputsModal) {
      closeTestInputsModal();
    }
  });

  profileSearch?.addEventListener("input", (event) => {
    loadProfiles(event.target.value.trim());
  });
  refreshProfiles?.addEventListener("click", () => loadProfiles(profileSearch.value.trim()));

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;

    let handled = false;

    if (isDiseasesDropdownOpen) {
      closeDiseasesDropdown();
      handled = true;
    }

    if (isChatbotModalOpen()) {
      closeChatbotModal({ returnFocus: true });
      handled = true;
    }

    if (isTestInputsModalOpen()) {
      closeTestInputsModal();
      handled = true;
    }

    if (handled) {
      event.preventDefault();
    }
  });

  consultantSearch?.addEventListener("input", (event) => {
    loadConsultants(event.target.value.trim());
  });
  refreshConsultants?.addEventListener("click", () => loadConsultants(consultantSearch.value.trim()));
  consultantTabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const targetId = button.dataset.target;
      activateConsultantView(targetId);
    });
  });

  window.addEventListener("popstate", async (event) => {
    if (isRestoringHistory) return;

    closeDiseasesDropdown();

    const stateData = event.state || {};
    const targetPage = stateData.page || "landing";
    const targetTab = stateData.tab || "diabetes";

    const currentState = getCurrentAppState();
    if (currentState.page === "dashboard" && targetPage !== "dashboard") {
      await resetSession({ recordHistory: false });
      updateHistoryState({ page: "landing", tab: "diabetes" }, { replace: true });
      return;
    }

    isRestoringHistory = true;
    try {
      if (targetPage === "patient") {
        showPage("patient", { recordHistory: false });
        return;
      }

      if (targetPage === "dashboard") {
        if (!state.profile) {
          await fetchCurrentProfile();
        }

        if (!state.profile) {
          showPage("landing", { recordHistory: false });
          return;
        }

        await enterDashboard(targetTab, { recordHistory: false });
        return;
      }

      showPage("landing", { recordHistory: false });
    } finally {
      isRestoringHistory = false;
    }
  });
}

(async function init() {
  initializeTheme();
  await fetchConfig();
  bindEvents();
  initializeFormConstraints();
  setupGooButton(startButton);
  setupGooButton(patientForm?.querySelector(".goo-button"));
  tabButtons.forEach((button) => setupGooButton(button));
  if (diseasesDropdownToggle) {
    setupGooButton(diseasesDropdownToggle);
  }
  const storedConsultantTab = getSessionValue("consultantTab");
  const defaultConsultantTab = storedConsultantTab && consultantViews.some((view) => view.id === storedConsultantTab)
    ? storedConsultantTab
    : consultantTabButtons[0]?.dataset.target;
  if (defaultConsultantTab) {
    activateConsultantView(defaultConsultantTab);
  }
  await restorePageState();
  updateHistoryState(getCurrentAppState(), { replace: true });
})();

const footerTypingText = document.getElementById("footer-typing-text");

if (footerTypingText) {
  const footerTips = [
    "Verify uploaded lab values and vitals before auto-triage, then summarize key outliers so clinicians can review the most urgent risks first.",
    "Confirm patient contact details and demographics early, so follow-ups, alerts, and referrals reach the right person without delays or rework.",
    "Use consistent symptom scoring across visits to compare trends, detect escalation sooner, and align handoffs with clinical expectations.",
    "Upload prior reports or histories with intake to speed screening, reduce duplicate questions, and deliver a clearer summary to the care team.",
    "Flag high-risk vitals immediately, annotate possible causes, and prioritize next steps to keep urgent cases moving through the workflow.",
  ];

  const TIP_INTERVAL = 7000;
  let index = 0;

  const showTip = (tip) => {
    footerTypingText.classList.remove("tip-slide");
    void footerTypingText.offsetWidth;
    footerTypingText.textContent = tip;
    footerTypingText.classList.add("tip-slide");
  };

  showTip(footerTips[index]);
  window.setInterval(() => {
    index = (index + 1) % footerTips.length;
    showTip(footerTips[index]);
  }, TIP_INTERVAL);
}
