/* CureHelp+ frontend controller */

const pages = {
  landing: document.getElementById("landing-page"),
  patient: document.getElementById("patient-page"),
  dashboard: document.getElementById("dashboard"),
};

const authForm = document.getElementById("auth-form");
const authEmailInput = document.getElementById("auth-email");
const authPasswordInput = document.getElementById("auth-password");
const authNameInput = document.getElementById("auth-name");
const authNewPasswordInput = document.getElementById("auth-new-password");
const authResetTokenInput = document.getElementById("auth-reset-token");
const authOtpInput = document.getElementById("auth-otp");
const authEmailLabel = document.getElementById("auth-email-label");
const authPasswordEyeButton = document.getElementById("auth-password-eye");
const authNewPasswordEyeButton = document.getElementById("auth-new-password-eye");
const authSubmitButton = document.getElementById("auth-submit-button");
const authInlineStatus = document.getElementById("auth-inline-status");
const authModeButtons = Array.from(document.querySelectorAll("[data-auth-mode]"));
const forgotPasswordButton = document.getElementById("forgot-password-button");
const resendVerificationButton = document.getElementById("resend-verification-button");

const patientForm = null;
const patientDobInput = null;
const patientSummary = document.getElementById("patient-summary");
const logoutButton = document.getElementById("logout-button");
const startButton = document.getElementById("start-onboarding");
const backButton = document.getElementById("back-to-landing");
const landingLoginLink = document.getElementById("landing-login-link");
const landingSignupLink = document.getElementById("landing-signup-link");
const landingDashboardLink = document.getElementById("landing-dashboard-link");
const authGoogleSigninButton = document.getElementById("auth-google-signin");
const userDashboardTabButton = document.getElementById("user-dashboard-tab-button");
const resetSessionButton = document.getElementById("reset-session");
const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
const tabTriggers = Array.from(document.querySelectorAll("[data-tab]"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
const diseasesDropdownToggle = document.getElementById("dropdownDefaultButton");
const diseasesDropdownMenu = document.getElementById("dropdown");
const diseasesDropdownLabel = diseasesDropdownToggle?.querySelector(".goo-button-label");
const resourcesDropdownContainer = document.getElementById("landing-resources");
const resourcesDropdownToggle = document.getElementById("resources-toggle");
const resourcesDropdownMenu = document.getElementById("resources-menu");
const contactFormTrigger = document.querySelector("[data-contact-popup='open']");
const contactFormMenu = document.getElementById("contact-form-menu");
const resourceContactForm = document.getElementById("resource-contact-form");
const contactFormCloseButton = document.getElementById("contact-form-close");
const resourceContactStatus = document.getElementById("resource-contact-status");
const docsSectionTriggers = Array.from(document.querySelectorAll("[data-docs-section]"));
const docsPopup = document.getElementById("docs-popup");
const docsPopupClose = document.getElementById("docs-popup-close");
const docsPopupBody = document.getElementById("docs-popup-body");
const docsPopupMain = document.getElementById("docs-popup-main");
const docsPopupSections = Array.from(document.querySelectorAll(".docs-popup-section"));
const footerTip = document.getElementById("footer-tip");
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
const logoutConfirmModal = document.getElementById("logout-confirm-modal");
const logoutConfirmOk = document.getElementById("logout-confirm-ok");
const logoutConfirmCancel = document.getElementById("logout-confirm-cancel");
const editProfileButton = document.getElementById("edit-profile-button");
const profileEditPanel = document.getElementById("profile-edit-panel");
const profileEditForm = document.getElementById("profile-edit-form");
const profileEditStatus = document.getElementById("profile-edit-status");
const userProfilePhoto = document.getElementById("user-profile-photo");
const userNameLine = document.getElementById("user-name");
const userDobLine = document.getElementById("user-dob");
const userGenderLine = document.getElementById("user-gender");
const userAbhaIdLine = document.getElementById("user-abha-id");
const userAddressLine = document.getElementById("user-address");
const userMobileLine = document.getElementById("user-mobile");
const summaryTotalPredictions = document.getElementById("summary-total-predictions");
const summaryDiseasesCount = document.getElementById("summary-diseases-count");
const summaryReportsCount = document.getElementById("summary-reports-count");
const profileCompletenessFill = document.getElementById("profile-completeness-fill");
const profileCompletenessText = document.getElementById("profile-completeness-text");
const profileCompletenessPrompts = document.getElementById("profile-completeness-prompts");
const riskAlertsPanel = document.getElementById("risk-alerts-panel");
const riskAlertsList = document.getElementById("risk-alerts-list");
const snapshotLastPrediction = document.getElementById("snapshot-last-prediction");
const snapshotHighestRisk = document.getElementById("snapshot-highest-risk");
const snapshotActiveDisease = document.getElementById("snapshot-active-disease");
const medAdherenceStreak = document.getElementById("med-adherence-streak");
const medAdherenceToday = document.getElementById("med-adherence-today");
const medAdherenceYes = document.getElementById("med-adherence-yes");
const medAdherenceNo = document.getElementById("med-adherence-no");
const shortcutNearestHospital = document.getElementById("shortcut-nearest-hospital");
const shortcutFindDoctor = document.getElementById("shortcut-find-doctor");
const fileSummaryTotal = document.getElementById("file-summary-total");
const fileSummaryLastFile = document.getElementById("file-summary-last-file");
const fileSummaryLastDate = document.getElementById("file-summary-last-date");
const previousMedicationsInput = document.getElementById("previous-medications-input");
const savePreviousMedicationsButton = document.getElementById("save-previous-medications");
const previousMedicationsStatus = document.getElementById("previous-medications-status");
const dataQualityScore = document.getElementById("data-quality-score");
const dataQualityMissing = document.getElementById("data-quality-missing");
const dataQualityFixProfile = document.getElementById("data-quality-fix-profile");
const dataQualityFixHistory = document.getElementById("data-quality-fix-history");
const downloadLatestReport = document.getElementById("download-latest-report");
const downloadFullHistory = document.getElementById("download-full-history");
const reportUploadForm = document.getElementById("report-upload-form");
const userReportFileInput = document.getElementById("user-report-file");
const userReportFileName = document.getElementById("user-report-file-name");
const reportUploadStatus = document.getElementById("report-upload-status");
const reportHistoryList = document.getElementById("report-history-list");
const pastHealthInputsList = document.getElementById("past-health-inputs-list");
const profilePhotoFileInput = document.getElementById("profile-photo-file");
const profilePhotoPreview = document.getElementById("profile-photo-preview");
const historyImageModal = document.getElementById("history-image-modal");
const historyImageClose = document.getElementById("history-image-close");
const historyImagePreview = document.getElementById("history-image-preview");
const pastHistoryFilters = Array.from(document.querySelectorAll(".past-history-filter"));
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
const anemiaForm = document.getElementById("anemia-form");
const pneumoniaForm = document.getElementById("pneumonia-form");
const pneumoniaImageInput = document.getElementById("pneumonia-image");
const tuberculosisForm = document.getElementById("tuberculosis-form");
const tuberculosisImageInput = document.getElementById("tuberculosis-image");

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
  historyFilter: "all",
  auth: {
    authenticated: false,
    user: null,
  },
};

let activeAuthMode = "login";
let authStatusHideTimeoutId = null;
let resendOtpTimerId = null;
let resendOtpSecondsRemaining = 0;
const RESEND_OTP_COOLDOWN_SECONDS = 30;

function getMedicationStorageKey() {
  const userId = String(state.auth?.user?.id || state.profile?.id || "guest").trim() || "guest";
  return `curehelp-medication-adherence-${userId}`;
}

function getPreviousMedicationsStorageKey() {
  const userId = String(state.auth?.user?.id || state.profile?.id || "guest").trim() || "guest";
  return `curehelp-previous-medications-${userId}`;
}

function getTodayKey() {
  const now = new Date();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${month}-${day}`;
}

function getYesterdayKey() {
  const now = new Date();
  now.setDate(now.getDate() - 1);
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${month}-${day}`;
}

function loadMedicationAdherenceState() {
  const key = getMedicationStorageKey();
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return { streak: 0, lastYesDate: "", lastActionDate: "", lastAction: "none" };
    const parsed = JSON.parse(raw);
    return {
      streak: Number(parsed?.streak) || 0,
      lastYesDate: String(parsed?.lastYesDate || ""),
      lastActionDate: String(parsed?.lastActionDate || ""),
      lastAction: String(parsed?.lastAction || "none"),
    };
  } catch (_) {
    return { streak: 0, lastYesDate: "", lastActionDate: "", lastAction: "none" };
  }
}

function saveMedicationAdherenceState(payload) {
  const key = getMedicationStorageKey();
  localStorage.setItem(key, JSON.stringify(payload || {}));
}

function loadPreviousMedications() {
  const key = getPreviousMedicationsStorageKey();
  try {
    return String(localStorage.getItem(key) || "");
  } catch (_) {
    return "";
  }
}

function savePreviousMedications(value) {
  const key = getPreviousMedicationsStorageKey();
  localStorage.setItem(key, String(value || ""));
}

function renderMedicationAdherenceState() {
  const adherence = loadMedicationAdherenceState();
  if (medAdherenceStreak) {
    medAdherenceStreak.textContent = `${adherence.streak || 0} day(s)`;
  }
  if (medAdherenceToday) {
    const today = getTodayKey();
    if (adherence.lastActionDate === today) {
      medAdherenceToday.textContent = adherence.lastAction === "yes" ? "Yes" : "No";
    } else {
      medAdherenceToday.textContent = "Not marked";
    }
  }
}

function renderPreviousMedications() {
  if (!previousMedicationsInput) return;
  previousMedicationsInput.value = loadPreviousMedications();
}

function clearAuthStatusHideTimer() {
  if (authStatusHideTimeoutId) {
    window.clearTimeout(authStatusHideTimeoutId);
    authStatusHideTimeoutId = null;
  }
}

function formatResendOtpCountdown(totalSeconds) {
  const seconds = Math.max(0, Number(totalSeconds) || 0);
  const minutesPart = String(Math.floor(seconds / 60)).padStart(2, "0");
  const secondsPart = String(seconds % 60).padStart(2, "0");
  return `${minutesPart}:${secondsPart}`;
}

function updateResendOtpButtonState() {
  if (!resendVerificationButton) return;

  if (activeAuthMode !== "verify") {
    resendVerificationButton.disabled = false;
    resendVerificationButton.textContent = "Resent OTP";
    return;
  }

  if (resendOtpSecondsRemaining > 0) {
    resendVerificationButton.disabled = true;
    resendVerificationButton.textContent = formatResendOtpCountdown(resendOtpSecondsRemaining);
    return;
  }

  resendVerificationButton.disabled = false;
  resendVerificationButton.textContent = "Resent OTP";
}

function clearResendOtpTimer() {
  if (resendOtpTimerId) {
    window.clearInterval(resendOtpTimerId);
    resendOtpTimerId = null;
  }
}

function startResendOtpTimer(seconds = RESEND_OTP_COOLDOWN_SECONDS) {
  clearResendOtpTimer();
  resendOtpSecondsRemaining = Math.max(0, Number(seconds) || 0);
  updateResendOtpButtonState();

  if (resendOtpSecondsRemaining <= 0) {
    return;
  }

  resendOtpTimerId = window.setInterval(() => {
    resendOtpSecondsRemaining = Math.max(0, resendOtpSecondsRemaining - 1);
    updateResendOtpButtonState();
    if (resendOtpSecondsRemaining <= 0) {
      clearResendOtpTimer();
    }
  }, 1000);
}

function mapAuthErrorMessage(rawMessage = "", mode = "") {
  const normalized = String(rawMessage || "").trim();
  const lowered = normalized.toLowerCase();
  const currentMode = String(mode || activeAuthMode || "").toLowerCase();

  if (currentMode === "verify" && (lowered.includes("otp") || lowered.includes("invalid or expired"))) {
    return "Incorrect OTP";
  }

  if (currentMode === "signup" && lowered.includes("already exists")) {
    return "Email Already Exists";
  }

  if (currentMode === "login" && (lowered.includes("invalid email or password") || lowered.includes("incorrect password"))) {
    return "Incorrect Password";
  }

  return normalized || "Authentication failed";
}

function isLogoutConfirmModalOpen() {
  return Boolean(logoutConfirmModal?.classList.contains("open"));
}

function closeLogoutConfirmModal() {
  if (!logoutConfirmModal) return;
  if (!logoutConfirmModal.classList.contains("open")) {
    logoutConfirmModal.hidden = true;
    return;
  }
  logoutConfirmModal.classList.remove("open");
  logoutConfirmModal.hidden = true;
  unlockBodyScroll();
}

function openLogoutConfirmModal() {
  if (!logoutConfirmModal) return;
  if (isLogoutConfirmModalOpen()) return;
  if (logoutConfirmOk) {
    logoutConfirmOk.disabled = false;
    logoutConfirmOk.textContent = "Logout";
  }
  if (logoutConfirmCancel) {
    logoutConfirmCancel.disabled = false;
  }
  lockBodyScroll();
  logoutConfirmModal.hidden = false;
  logoutConfirmModal.classList.add("open");
  logoutConfirmOk?.focus?.({ preventScroll: true });
}

async function logoutUser() {
  try {
    await fetch("/api/auth/logout-all", { method: "POST" });
  } catch (_) {
    /* noop fallback */
  }
  await resetSession({ redirect: true, recordHistory: false });
  closeLogoutConfirmModal();
  updateHistoryState({ page: "landing", tab: "diabetes" }, { replace: true });
}

let pendingTestInputsDisease = null;
let historyInitialized = false;
let isRestoringHistory = false;
let isDiseasesDropdownOpen = false;
let isResourcesDropdownOpen = false;
let isContactFormPopupOpen = false;
let isDocsPopupOpen = false;
let isLoginPopupOpen = false;
let shouldReopenResourcesMenu = false;
const diseaseTabKeys = new Set(["pneumonia", "tuberculosis", "diabetes", "heart", "anemia"]);
const diseaseTabLabels = {
  pneumonia: "Pneumonia",
  tuberculosis: "Tuberculosis(TB)",
  diabetes: "Type-2 Diabetes",
  heart: "Coronary Artery Disease",
  anemia: "Anemia",
};
const PNEUMONIA_ALLOWED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png"]);
const PNEUMONIA_ALLOWED_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/jpg", "application/octet-stream"]);
const MAX_XRAY_IMAGE_SIZE_BYTES = 10 * 1024 * 1024;
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

function startLoaderDelay(duration = 900) {
  const startTime = (typeof performance !== "undefined" && typeof performance.now === "function")
    ? performance.now()
    : Date.now();

  if (loaderWrapper) {
    loaderWrapper.style.display = "block";
  }

  let finished = false;
  return {
    finish() {
      if (finished) {
        return Promise.resolve();
      }
      finished = true;

      const now = (typeof performance !== "undefined" && typeof performance.now === "function")
        ? performance.now()
        : Date.now();
      const elapsed = now - startTime;
      const remaining = Math.max(0, duration - elapsed);

      return new Promise((resolve) => {
        window.setTimeout(() => {
          if (loaderWrapper) {
            loaderWrapper.style.display = "none";
          }
          resolve();
        }, remaining);
      });
    },
  };
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
  registerInputConstraints(anemiaForm, FORM_CONSTRAINTS["anemia-form"]);
}

function isValidPneumoniaImage(file) {
  if (!file) {
    return { valid: false, message: "Please upload a chest X-ray image." };
  }

  const extension = getFileExtension(file.name || "");
  const mimeType = (file.type || "").toLowerCase();

  if (!PNEUMONIA_ALLOWED_EXTENSIONS.has(extension) && !PNEUMONIA_ALLOWED_MIME_TYPES.has(mimeType)) {
    return { valid: false, message: "Unsupported image type. Please upload JPG, JPEG, or PNG." };
  }

  if (!PNEUMONIA_ALLOWED_EXTENSIONS.has(extension)) {
    return { valid: false, message: "Unsupported image extension. Allowed: .jpg, .jpeg, .png." };
  }

  if ((file.size || 0) > MAX_XRAY_IMAGE_SIZE_BYTES) {
    return { valid: false, message: "Image size must be 10MB or smaller." };
  }

  return { valid: true, message: "" };
}

function isValidTuberculosisImage(file) {
  if (!file) {
    return { valid: false, message: "Please upload a chest X-ray image." };
  }

  const extension = getFileExtension(file.name || "");
  const mimeType = (file.type || "").toLowerCase();

  if (!PNEUMONIA_ALLOWED_EXTENSIONS.has(extension) && !PNEUMONIA_ALLOWED_MIME_TYPES.has(mimeType)) {
    return { valid: false, message: "Unsupported image type. Please upload JPG, JPEG, or PNG." };
  }

  if (!PNEUMONIA_ALLOWED_EXTENSIONS.has(extension)) {
    return { valid: false, message: "Unsupported image extension. Allowed: .jpg, .jpeg, .png." };
  }

  if ((file.size || 0) > MAX_XRAY_IMAGE_SIZE_BYTES) {
    return { valid: false, message: "Image size must be 10MB or smaller." };
  }

  return { valid: true, message: "" };
}

function bindFileDropZone(input, validateFile) {
  if (!input) {
    return;
  }

  const dropZone = input.closest(".report-upload-field");
  if (!dropZone) {
    return;
  }

  let dragDepth = 0;

  const stopEvent = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const setDragState = (active) => {
    dropZone.classList.toggle("is-dragover", Boolean(active));
  };

  dropZone.addEventListener("dragenter", (event) => {
    stopEvent(event);
    dragDepth += 1;
    setDragState(true);
  });

  dropZone.addEventListener("dragover", (event) => {
    stopEvent(event);
    setDragState(true);
  });

  dropZone.addEventListener("dragleave", (event) => {
    stopEvent(event);
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) {
      setDragState(false);
    }
  });

  dropZone.addEventListener("drop", (event) => {
    stopEvent(event);
    dragDepth = 0;
    setDragState(false);

    const droppedFile = event.dataTransfer?.files?.[0];
    if (!droppedFile) {
      return;
    }

    if (typeof validateFile === "function") {
      const validation = validateFile(droppedFile);
      if (!validation?.valid) {
        alert(validation?.message || "Unsupported file.");
        return;
      }
    }

    if (typeof DataTransfer !== "undefined") {
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(droppedFile);
      input.files = dataTransfer.files;
    }

    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

function formatFileSize(bytes) {
  const size = Number(bytes);
  if (!Number.isFinite(size) || size < 0) return "0 B";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

function initReferenceFileUpload(input) {
  if (!input) return;

  const root = input.closest("[data-hs-file-upload]");
  if (!root) return;

  const trigger = root.querySelector("[data-hs-file-upload-trigger]");
  const browse = root.querySelector(".file-upload-browse");
  const prompt = root.querySelector(".file-upload-prompt");
  const meta = root.querySelector(".file-upload-meta");
  const browseDefaultText = browse?.textContent?.trim() || "browse";
  const promptDefaultText = prompt?.textContent?.trim() || "Drop your file here or";
  const metaDefaultText = meta?.textContent?.trim() || "Pick a file up to 2MB.";

  const openPicker = () => input.click();

  trigger?.addEventListener("click", (event) => {
    if (event.target.closest("[data-hs-file-upload-remove]")) return;
    openPicker();
  });

  browse?.addEventListener("click", openPicker);
  browse?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPicker();
    }
  });

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) {
      if (prompt) prompt.textContent = promptDefaultText;
      if (browse) browse.textContent = browseDefaultText;
      if (meta) meta.textContent = metaDefaultText;
      return;
    }

    if (prompt) prompt.textContent = file.name || "Selected file";
    if (browse) browse.textContent = "change";
    if (meta) meta.textContent = formatFileSize(file.size);
  });
}

function parseDobDdMmYyyy(value) {
  const trimmed = (value || "").toString().trim();
  const match = /^(\d{2})\/(\d{2})\/(\d{4})$/.exec(trimmed);
  if (!match) {
    return { date: null, error: "DOB must be in dd/mm/yyyy format." };
  }

  const day = Number(match[1]);
  const month = Number(match[2]);
  const year = Number(match[3]);

  if (!Number.isFinite(day) || !Number.isFinite(month) || !Number.isFinite(year)) {
    return { date: null, error: "DOB must be in dd/mm/yyyy format." };
  }

  const candidate = new Date(year, month - 1, day, 12, 0, 0, 0);
  if (
    candidate.getFullYear() !== year ||
    candidate.getMonth() !== month - 1 ||
    candidate.getDate() !== day
  ) {
    return { date: null, error: "DOB is not a valid calendar date." };
  }

  const today = new Date();
  if (candidate.getTime() > today.getTime()) {
    return { date: null, error: "DOB cannot be in the future." };
  }

  return { date: candidate, error: null };
}

function calculateAgeFromDob(dobDate, referenceDate = new Date()) {
  const dob = dobDate instanceof Date ? dobDate : null;
  if (!dob || Number.isNaN(dob.getTime())) return null;

  const today = referenceDate instanceof Date ? referenceDate : new Date();
  let age = today.getFullYear() - dob.getFullYear();
  const monthDiff = today.getMonth() - dob.getMonth();
  const dayDiff = today.getDate() - dob.getDate();
  if (monthDiff < 0 || (monthDiff === 0 && dayDiff < 0)) {
    age -= 1;
  }
  return age;
}

function clearPatientDobDerivedAge() {
  if (patientDobInput) {
    patientDobInput.dataset.computedAge = "";
    patientDobInput.setCustomValidity("");
  }
}

function syncPatientDobDerivedAge({ setInputValidity = false } = {}) {
  if (!patientDobInput) {
    return null;
  }

  const rawDob = patientDobInput.value.trim();
  if (!rawDob) {
    if (setInputValidity) {
      patientDobInput.setCustomValidity("");
    }
    patientDobInput.dataset.computedAge = "";
    return null;
  }

  const parsed = parseDobDdMmYyyy(rawDob);
  if (parsed.error) {
    if (setInputValidity) {
      patientDobInput.setCustomValidity(parsed.error);
    }
    patientDobInput.dataset.computedAge = "";
    return null;
  }

  const age = calculateAgeFromDob(parsed.date);
  if (!Number.isFinite(age) || age < 1 || age > 120) {
    const message = "Calculated age must be between 1 and 120.";
    if (setInputValidity) {
      patientDobInput.setCustomValidity(message);
    }
    patientDobInput.dataset.computedAge = "";
    return null;
  }

  if (setInputValidity) {
    patientDobInput.setCustomValidity("");
  }
  patientDobInput.dataset.computedAge = String(age);
  return age;
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

function buildAdminSourceToken() {
  const appState = getCurrentAppState();
  const page = String(appState.page || "landing");
  const tab = String(appState.tab || "diabetes");

  if (page === "landing") {
    return isResourcesDropdownOpen ? "resources" : "landing";
  }

  if (page === "dashboard") {
    return `dashboard:${tab}`;
  }

  return "landing";
}

function applyAdminReturnToken(token) {
  const value = String(token || "").trim().toLowerCase();
  if (!value) return;

  if (value === "landing") {
    setSessionValue("currentPage", "landing");
    setSessionValue("currentTab", "diabetes");
    shouldReopenResourcesMenu = false;
    return;
  }

  if (value === "resources") {
    setSessionValue("currentPage", "landing");
    setSessionValue("currentTab", "diabetes");
    shouldReopenResourcesMenu = true;
    return;
  }

  if (value.startsWith("dashboard:")) {
    const tab = value.split(":")[1] || "diabetes";
    setSessionValue("currentPage", "dashboard");
    setSessionValue("currentTab", tab);
    shouldReopenResourcesMenu = false;
  }
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

function openResourcesDropdown() {
  if (!resourcesDropdownMenu) return;
  resourcesDropdownMenu.classList.remove("hidden");
  resourcesDropdownToggle?.setAttribute("aria-expanded", "true");
  resourcesDropdownToggle?.classList.add("open");
  isResourcesDropdownOpen = true;
}

function closeResourcesDropdown() {
  if (!resourcesDropdownMenu) return;
  resourcesDropdownMenu.classList.add("hidden");
  resourcesDropdownToggle?.setAttribute("aria-expanded", "false");
  resourcesDropdownToggle?.classList.remove("open");
  isResourcesDropdownOpen = false;
}

function toggleResourcesDropdown() {
  if (!resourcesDropdownMenu) return;
  if (isResourcesDropdownOpen) {
    closeResourcesDropdown();
  } else {
    openResourcesDropdown();
  }
}

function openContactFormPopup() {
  if (!contactFormMenu) return;
  closeResourcesDropdown();
  contactFormMenu.classList.remove("hidden");
  isContactFormPopupOpen = true;
}

function closeContactFormPopup() {
  if (!contactFormMenu) return;
  contactFormMenu.classList.add("hidden");
  isContactFormPopupOpen = false;
}

function toggleContactFormPopup() {
  if (!contactFormMenu) return;
  if (isContactFormPopupOpen) {
    closeContactFormPopup();
  } else {
    openContactFormPopup();
  }
}

function openLoginPopup() {
  if (!pages.patient || isLoginPopupOpen) return;
  closeResourcesDropdown();
  closeContactFormPopup();
  closeDocsPopup({ returnFocus: false });
  lockBodyScroll();
  pages.patient.hidden = false;
  pages.patient.classList.remove("hidden");
  pages.patient.classList.add("active");
  document.body.classList.add("login-popup-open");
  isLoginPopupOpen = true;
  authEmailInput?.focus?.({ preventScroll: true });
  updateThemeToggleVisibility("landing");
  updateFooterTipVisibility("landing");
}

function closeLoginPopup({ returnFocus = false } = {}) {
  if (!pages.patient || !isLoginPopupOpen) {
    if (pages.patient) {
      pages.patient.classList.remove("active");
      pages.patient.classList.add("hidden");
      pages.patient.hidden = true;
    }
    document.body.classList.remove("login-popup-open");
    return;
  }
  pages.patient.classList.remove("active");
  pages.patient.classList.add("hidden");
  pages.patient.hidden = true;
  document.body.classList.remove("login-popup-open");
  isLoginPopupOpen = false;
  unlockBodyScroll();
  updateThemeToggleVisibility("landing");
  updateFooterTipVisibility("landing");
  if (returnFocus) {
    startButton?.focus?.({ preventScroll: true });
  }
}

function openDocsPopup(sectionKey = "overview") {
  if (!docsPopup) return;
  lockBodyScroll();
  docsPopup.hidden = false;
  docsPopup.classList.add("open");
  isDocsPopupOpen = true;
  updateThemeToggleVisibility(getCurrentAppState().page);
  updateFooterTipVisibility(getCurrentAppState().page);

  if (docsPopupMain) {
    docsPopupMain.scrollTop = 0;
  }

  const section = docsPopup.querySelector(`#docs-${sectionKey}`) || docsPopup.querySelector("#docs-overview");
  if (section) {
    section.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveDocsSection(sectionKey);
  } else if (docsPopupMain) {
    docsPopupMain.scrollTop = 0;
    setActiveDocsSection("overview");
  }
}

function closeDocsPopup({ returnFocus = false } = {}) {
  if (!docsPopup || !isDocsPopupOpen) return;
  docsPopup.classList.remove("open");
  docsPopup.hidden = true;
  isDocsPopupOpen = false;
  updateThemeToggleVisibility(getCurrentAppState().page);
  updateFooterTipVisibility(getCurrentAppState().page);
  unlockBodyScroll();

  if (returnFocus) {
    docsSectionTriggers[0]?.focus?.({ preventScroll: true });
  }
}

function setActiveDocsSection(sectionKey) {
  if (!docsPopup) return;
  const key = sectionKey || "overview";
  const jumpButtons = docsPopup.querySelectorAll("[data-docs-jump]");
  jumpButtons.forEach((button) => {
    const isActive = button.dataset.docsJump === key;
    button.classList.toggle("active", isActive);
  });
}

function syncDocsActiveFromScroll() {
  if (!docsPopupMain || !docsPopupSections.length) return;
  const scrollTop = docsPopupMain.scrollTop;
  let nearestSectionKey = "overview";
  let smallestDistance = Number.POSITIVE_INFINITY;

  docsPopupSections.forEach((section) => {
    const distance = Math.abs(section.offsetTop - scrollTop - 30);
    if (distance < smallestDistance) {
      smallestDistance = distance;
      nearestSectionKey = section.id.replace("docs-", "");
    }
  });

  setActiveDocsSection(nearestSectionKey);
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
  const isLanding = activePage === "landing" && !isDocsPopupOpen && !isLoginPopupOpen;
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

function updateFooterTipVisibility(activePage) {
  if (!footerTip) {
    return;
  }
  const isLanding = activePage === "landing" && !isDocsPopupOpen && !isLoginPopupOpen;
  footerTip.hidden = !isLanding;
  footerTip.style.display = isLanding ? "" : "none";
  footerTip.setAttribute("aria-hidden", isLanding ? "false" : "true");
}

function showPage(key, { recordHistory = true } = {}) {
  if (key === "landing" && state.auth.authenticated) {
    key = "dashboard";
  }
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
  if (key !== "landing") {
    closeResourcesDropdown();
    closeContactFormPopup();
    closeLoginPopup();
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
  updateFooterTipVisibility(key);
  setSessionValue("currentPage", key);
  if (recordHistory) {
    const { tab } = getCurrentAppState();
    updateHistoryState({ page: key, tab });
  }
}

function updateAuthUi(isAuthenticated) {
  const authenticated = Boolean(isAuthenticated);

  if (landingLoginLink) {
    landingLoginLink.classList.toggle("hidden", authenticated);
    landingLoginLink.setAttribute("aria-hidden", authenticated ? "true" : "false");
  }
  if (landingSignupLink) {
    landingSignupLink.classList.toggle("hidden", authenticated);
    landingSignupLink.setAttribute("aria-hidden", authenticated ? "true" : "false");
  }

  if (landingDashboardLink) {
    landingDashboardLink.classList.toggle("hidden", !authenticated);
    landingDashboardLink.setAttribute("aria-hidden", authenticated ? "false" : "true");
  }

  if (userDashboardTabButton) {
    userDashboardTabButton.classList.toggle("hidden", !authenticated);
    userDashboardTabButton.setAttribute("aria-hidden", authenticated ? "false" : "true");
    userDashboardTabButton.tabIndex = authenticated ? 0 : -1;
  }

  if (startButton) {
    const label = startButton.querySelector(".goo-button-label");
    if (label) {
      label.textContent = authenticated ? "OPEN DASHBOARD" : "GET STARTED";
    }
  }
}

function setPatientSummary(profile) {
  if (!profile) {
    patientSummary.textContent = "No active user.";
    if (logoutButton) {
      logoutButton.hidden = true;
    }
    return;
  }
  const patientName = (profile?.name || "").trim() || "Patient";
  patientSummary.textContent = `Patient: ${patientName}`;
  if (logoutButton) {
    logoutButton.hidden = false;
  }
}

async function fetchCurrentProfile() {
  try {
    const response = await fetch("/api/auth/status");
    const payload = await response.json();
    if (!payload.success) throw new Error(payload.error || "Unable to fetch auth state");
    const authenticated = Boolean(payload.authenticated);
    state.auth.authenticated = authenticated;
    state.auth.user = authenticated ? payload.user : null;

    const profilePayload = authenticated
      ? {
          id: payload.user?.id || "",
          name: payload.user?.full_name || payload.user?.profile?.name || "",
          age: payload.user?.profile?.age || "",
          gender: payload.user?.profile?.gender || "",
          email: payload.user?.email || "",
          ...payload.user?.profile,
        }
      : null;

    state.profile = profilePayload;
    setPatientSummary(state.profile);
    applyProfileDemographics(state.profile);
    setDashboardProfilePhoto(profilePayload || {});
    updateAuthUi(authenticated);
    return state.profile;
  } catch (error) {
    console.warn("Failed to restore auth status", error);
    state.auth.authenticated = false;
    state.auth.user = null;
    state.profile = null;
    updateAuthUi(false);
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
        diseasesDropdownLabel.textContent = diseaseTabLabels[tabName] || "Type-2 Diabetes";
      } else {
        diseasesDropdownLabel.textContent = "Type-2 Diabetes";
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
  if (tabName === "user-dashboard") {
    await loadUserDashboardOverview(false, { showLoader: true, minDuration: 350 });
  } else if (tabName === "profiles") {
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

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    if (!(file instanceof File)) {
      resolve("");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : "");
    reader.onerror = () => reject(new Error("Unable to read uploaded image."));
    reader.readAsDataURL(file);
  });
}

function renderRiskView(targetId, payload) {
  const container = document.getElementById(targetId);
  if (!container) return;

  const {
    disease,
    probability,
    prob,
    score,
    severity,
    result,
    inputs = {},
    normal_values = {},
    recommendations = {},
  } = payload;

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
  const diseaseKey = String(disease || "").toLowerCase();
  const isLungsImagePrediction = diseaseKey === "pneumonia" || diseaseKey === "tuberculosis";
  const lungsPositiveLabel = diseaseKey === "tuberculosis" ? "Tuberculosis" : "Pneumonia";
  const lungsClass = String(result || payload.prediction || "").trim() || (probabilityValue >= 50 ? lungsPositiveLabel : "Normal");
  const lungsClassColor = lungsClass.toLowerCase() === lungsPositiveLabel.toLowerCase() ? "#dc2626" : "#16a34a";
  const lungsClassText = lungsClass.toUpperCase();
  const uploadedImageDataUrl = isLungsImagePrediction && typeof payload.uploaded_image_data_url === "string"
    ? payload.uploaded_image_data_url
    : "";
  const lungsImageMarkup = uploadedImageDataUrl
    ? `<img src="${uploadedImageDataUrl}" alt="Uploaded chest X-ray" style="max-width:68%;max-height:300px;border-radius:12px;object-fit:contain;border:1px solid rgba(148,163,184,0.35);" />`
    : "";
  const lungsMarkup = `
    <div style="height:100%;display:flex;flex-direction:row;align-items:center;justify-content:center;text-align:center;line-height:1.15;padding:1rem;gap:1rem;">
      ${lungsImageMarkup}
      <div style="font-size:30px;font-weight:700;color:${lungsClassColor};white-space:nowrap;letter-spacing:0.08em;">${lungsClassText}</div>
    </div>
  `;
  const meterLabel = isLungsImagePrediction
    ? `Confidence • ${probabilityValue.toFixed(1)}%`
    : `${badge.text} Risk • ${probabilityValue.toFixed(1)}%`;
  const leftColumnMarkup = isLungsImagePrediction
    ? `<div class="chart-box">${lungsMarkup}</div>`
    : `<div class="chart-box"><div id="${barId}"></div></div>`;

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
        <span class="${badge.className}"> ${meterLabel}</span>
      </div>
      ${severity ? `<p class="muted">Predicted Severity: <strong>${severity}</strong></p>` : ""}
      <div class="charts-row">
        ${leftColumnMarkup}
        <div class="gauge-box"><div id="${gaugeId}"></div></div>
      </div>
      <div class="recommendations">
        ${renderRecommendations(recommendations.prevention_measures, "Risk Reduction Protocols")}
        ${renderRecommendations(recommendations.medicine_suggestions, "Clinical Interventions")}
      </div>
    </div>
  `;

  if (!isLungsImagePrediction) {
    buildBarChart(barId, disease, inputs || {}, normal_values || state.normals[disease?.toLowerCase()] || {});
  }
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
  const loaderDelay = startLoaderDelay(2800);
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
    renderRiskView(resultId, payload);
    hidePanelInputs(resultId);
    await loaderDelay.finish();
    panel?.classList.remove("loader-hidden");
    loadUserDashboardOverview();
  } catch (error) {
    const container = document.getElementById(resultId);
    if (container) {
      container.innerHTML = `<div class="result-card"><p class="muted">${error.message}</p></div>`;
    }
    await loaderDelay.finish();
    panel?.classList.remove("loader-hidden");
    panel?.classList.remove("result-only");
  }
}

async function submitPneumoniaPrediction() {
  if (typeof pneumoniaForm?.reportValidity === "function" && !pneumoniaForm.reportValidity()) {
    return;
  }

  const imageFile = pneumoniaImageInput?.files?.[0] || null;
  const validation = isValidPneumoniaImage(imageFile);
  if (!validation.valid) {
    alert(validation.message);
    return;
  }

  const resultId = "pneumonia-result";
  const panel = pneumoniaForm?.closest(".tab-panel");
  panel?.classList.add("loader-hidden");
  const loaderDelay = startLoaderDelay(2800);

  try {
    const uploadedImageDataUrl = await fileToDataUrl(imageFile);
    const formData = new FormData(pneumoniaForm);
    const multipart = createMultipartRequest(formData);

    const response = await fetch("/api/pneumonia", {
      method: "POST",
      headers: {
        "Content-Type": multipart.contentType,
      },
      body: multipart.body,
    });

    let payload;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      payload = await response.json();
    } else {
      const responseText = await response.text();
      throw new Error(responseText || "Unexpected server response.");
    }

    if (!response.ok || !payload.success) {
      throw new Error(payload.error || "Prediction failed");
    }

    payload.uploaded_image_data_url = uploadedImageDataUrl;

    state.predictions = state.predictions || {};
    state.predictions.pneumonia = payload;
    renderRiskView(resultId, payload);
    hidePanelInputs(resultId);
    await loaderDelay.finish();
    panel?.classList.remove("loader-hidden");
    loadUserDashboardOverview();
  } catch (error) {
    const container = document.getElementById(resultId);
    if (container) {
      container.innerHTML = `<div class="result-card"><p class="muted">${error.message}</p></div>`;
    }
    await loaderDelay.finish();
    panel?.classList.remove("loader-hidden");
    panel?.classList.remove("result-only");
  }
}

async function submitTuberculosisPrediction() {
  if (typeof tuberculosisForm?.reportValidity === "function" && !tuberculosisForm.reportValidity()) {
    return;
  }

  const imageFile = tuberculosisImageInput?.files?.[0] || null;
  const validation = isValidTuberculosisImage(imageFile);
  if (!validation.valid) {
    alert(validation.message);
    return;
  }

  const resultId = "tuberculosis-result";
  const panel = tuberculosisForm?.closest(".tab-panel");
  panel?.classList.add("loader-hidden");
  const loaderDelay = startLoaderDelay(2800);

  try {
    const uploadedImageDataUrl = await fileToDataUrl(imageFile);
    const formData = new FormData(tuberculosisForm);
    const multipart = createMultipartRequest(formData);

    const response = await fetch("/api/tuberculosis", {
      method: "POST",
      headers: {
        "Content-Type": multipart.contentType,
      },
      body: multipart.body,
    });

    let payload;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      payload = await response.json();
    } else {
      const responseText = await response.text();
      throw new Error(responseText || "Unexpected server response.");
    }

    if (!response.ok || !payload.success) {
      throw new Error(payload.error || "Prediction failed");
    }

    payload.uploaded_image_data_url = uploadedImageDataUrl;

    state.predictions = state.predictions || {};
    state.predictions.tuberculosis = payload;
    renderRiskView(resultId, payload);
    hidePanelInputs(resultId);
    await loaderDelay.finish();
    panel?.classList.remove("loader-hidden");
    loadUserDashboardOverview();
  } catch (error) {
    const container = document.getElementById(resultId);
    if (container) {
      container.innerHTML = `<div class="result-card"><p class="muted">${error.message}</p></div>`;
    }
    await loaderDelay.finish();
    panel?.classList.remove("loader-hidden");
    panel?.classList.remove("result-only");
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
      return `${disease}: ${probabilityValue.toFixed(1)}%`;
    })
    .join("&nbsp;&nbsp;&nbsp;");

  const isCurrent = Boolean(currentProfileId && profile.id && profile.id === currentProfileId);
  const currentNote = "";

  const ageGenderParts = [];
  if (profile.age) ageGenderParts.push(`Age ${profile.age}`);
  if (profile.gender) ageGenderParts.push(String(profile.gender));
  const ageGender = ageGenderParts.join(" • ") || "Basic details unavailable";

  const updatedAt = profile.last_updated || profile.created_at || "-";
  const rawProfileId = String(profile.id || "").trim();
  let displayProfileId = rawProfileId || "-";
  const userIdMatch = rawProfileId.match(/^user_(\d+)$/i);
  if (userIdMatch) {
    displayProfileId = `User ${Number(userIdMatch[1])}`;
  } else if (rawProfileId.toLowerCase().startsWith("auth_")) {
    const stableShort = rawProfileId.slice(5, 9).toUpperCase();
    displayProfileId = stableShort ? `User ${stableShort}` : "User";
  }

  return `
    <div class="profile-card${isCurrent ? " current" : ""}" data-profile-id="${profile.id || ""}" tabindex="0">
      <div class="profile-card-summary">
        <div class="profile-card-heading">
          <h4 class="profile-card-name">${profile.name || "Unknown"}</h4>
          <span class="profile-card-badge">${displayProfileId}</span>
        </div>
        <p class="profile-card-meta">${ageGender}</p>
        <p class="profile-card-updated muted">Updated: ${updatedAt}</p>
      </div>
      <div class="profile-card-details">
        <p>Contact: ${profile.contact || "-"}</p>
        ${profile.address ? `<p class="muted">${profile.address}</p>` : ""}
        <p><strong>Predictions:</strong> <span class="profile-predictions-line">${summary || "No predictions"}</span></p>
        ${currentNote}
      </div>
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
    const hospitalName = item.name || "Hospital";
    const initials = getConsultantInitials(item.name || "Hospital");
    const avatarMarkup = imageUrl
      ? `<img src="${imageUrl}" alt="${item.name || "Hospital"}" loading="lazy" />`
      : initials || "H";
    const rawContact = String(item.contact || "").trim();
    const callTarget = rawContact.replace(/[^\d+]/g, "");
    const callLinkMarkup = callTarget
      ? `
      <a href="tel:${callTarget}" class="hospital-call-link" aria-label="Call ${hospitalName}" title="Call ${hospitalName}">
        <span aria-hidden="true">📞</span>
      </a>
    `
      : "";
    return `
      <div class="consultant-card hospital-card">
        ${callLinkMarkup}
        <div class="hospital-card-header">
          <div class="hospital-avatar" aria-hidden="true">${avatarMarkup}</div>
          <div class="hospital-info">
            <h4>${hospitalName}</h4>
            <p class="hospital-speciality">${item.speciality || item.specialization || ""}</p>
          </div>
        </div>
        <div class="hospital-meta">
          <p>${item.address || ""}</p>
          <p>Contact: ${item.contact || "-"}</p>
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
  const rawContact = String(item.contact || "").trim();
  const callTarget = rawContact.replace(/[^\d+]/g, "");
  const doctorName = item.name || "Doctor";
  const callLinkMarkup = callTarget
    ? `
      <a href="tel:${callTarget}" class="doctor-call-link" aria-label="Call ${doctorName}" title="Call ${doctorName}">
        <span aria-hidden="true">📞</span>
      </a>
    `
    : "";

  return `
    <div class="consultant-card doctor-card">
      ${callLinkMarkup}
      <div class="doctor-card-header">
        <div class="doctor-avatar" aria-hidden="true">${avatarMarkup}</div>
        <div class="doctor-info">
          <h4>${doctorName}</h4>
          <p class="doctor-speciality">${speciality}</p>
        </div>
      </div>
      <div class="doctor-meta">
        <p>${address}</p>
        <p>Contact: ${item.contact || "-"}</p>
        <p class="muted">${distance}</p>
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
    const limitedHospitals = hospitals;
    const limitedDoctors = doctors;
    hospitalList.innerHTML = limitedHospitals.length
      ? limitedHospitals.map((item, index) => renderConsultantCard(item, {
          isDoctor: false,
          isHospital: true,
          imageUrl: item.image_url || "",
        })).join("")
      : '<div class="consultant-card"><p class="muted">No hospitals found.</p></div>';
    doctorList.innerHTML = limitedDoctors.length
      ? limitedDoctors.map((item, index) => renderConsultantCard(item, {
          isDoctor: true,
          imageUrl: item.image_url || "",
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
  addDemographics("anemia", { includeAge: false });

  if (Object.keys(defaults).length > 0) {
    applyAutofillValues(defaults);
  }
}

function setAuthInlineStatus(message = "", { isError = false, autoHideMs = 0 } = {}) {
  if (!authInlineStatus) return;
  clearAuthStatusHideTimer();
  authInlineStatus.textContent = message;
  authInlineStatus.style.color = isError ? "var(--danger)" : "var(--muted)";

  if (message && Number(autoHideMs) > 0) {
    authStatusHideTimeoutId = window.setTimeout(() => {
      if (!authInlineStatus) return;
      authInlineStatus.textContent = "";
      authStatusHideTimeoutId = null;
    }, Number(autoHideMs));
  }
}

function getAuthSubmitLabel(mode) {
  const submitMap = {
    login: "LOGIN",
    signup: "SIGNUP",
    verify: "VERIFY OTP",
    forgot: "SEND RESET LINK",
    reset: "RESET PASSWORD",
  };
  return submitMap[mode] || submitMap.login;
}

function setAuthSubmitLabel(labelText) {
  if (!authSubmitButton) return;
  const label = authSubmitButton.querySelector(".goo-button-label");
  if (label) {
    label.textContent = labelText;
  } else {
    authSubmitButton.textContent = labelText;
  }
}

function delay(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, Math.max(0, ms || 0)));
}

function measureButtonTargetWidth(button, nextText) {
  if (!(button instanceof HTMLButtonElement)) {
    return 0;
  }

  const contentTarget = button.querySelector(".goo-button-label") || button;
  const contentStyles = window.getComputedStyle(contentTarget);
  const buttonStyles = window.getComputedStyle(button);

  const probe = document.createElement("span");
  probe.textContent = nextText;
  probe.style.position = "absolute";
  probe.style.visibility = "hidden";
  probe.style.pointerEvents = "none";
  probe.style.whiteSpace = "nowrap";
  probe.style.fontFamily = contentStyles.fontFamily;
  probe.style.fontSize = contentStyles.fontSize;
  probe.style.fontWeight = contentStyles.fontWeight;
  probe.style.letterSpacing = contentStyles.letterSpacing;
  probe.style.textTransform = contentStyles.textTransform;
  document.body.appendChild(probe);
  const textWidth = Math.ceil(probe.getBoundingClientRect().width);
  document.body.removeChild(probe);

  const paddingX =
    (parseFloat(buttonStyles.paddingLeft) || 0) +
    (parseFloat(buttonStyles.paddingRight) || 0) +
    (parseFloat(buttonStyles.borderLeftWidth) || 0) +
    (parseFloat(buttonStyles.borderRightWidth) || 0);

  const minWidth = parseFloat(buttonStyles.minWidth) || 0;
  const currentWidth = Math.ceil(button.getBoundingClientRect().width || button.offsetWidth || 0);
  return Math.ceil(Math.max(textWidth + paddingX + 4, minWidth, currentWidth));
}

async function runButtonProgress(button, loadingText, action) {
  if (!(button instanceof HTMLButtonElement)) {
    return action();
  }

  const originalText = button.textContent || "";
  const startWidth = Math.ceil(button.getBoundingClientRect().width || button.offsetWidth || 0);
  const targetWidth = measureButtonTargetWidth(button, loadingText);
  if (startWidth > 0) {
    button.style.width = `${startWidth}px`;
    button.style.transition = "width 180ms ease";
  }
  button.disabled = true;
  button.classList.add("is-loading");
  button.textContent = loadingText;
  if (targetWidth > startWidth) {
    window.requestAnimationFrame(() => {
      button.style.width = `${targetWidth}px`;
    });
  }
  const startedAt = performance.now();

  try {
    return await action();
  } finally {
    const elapsed = performance.now() - startedAt;
    if (elapsed < 280) {
      await delay(280 - elapsed);
    }
    button.disabled = false;
    button.classList.remove("is-loading");
    button.style.width = "";
    button.style.transition = "";
    button.textContent = originalText;
  }
}

async function runAuthSubmitProgress(loadingText, action) {
  if (!authSubmitButton) {
    return action();
  }

  const startWidth = Math.ceil(authSubmitButton.getBoundingClientRect().width || authSubmitButton.offsetWidth || 0);
  const targetWidth = measureButtonTargetWidth(authSubmitButton, loadingText);
  if (startWidth > 0) {
    authSubmitButton.style.width = `${startWidth}px`;
    authSubmitButton.style.transition = "width 180ms ease";
  }
  authSubmitButton.disabled = true;
  authSubmitButton.classList.add("is-loading");
  setAuthSubmitLabel(loadingText);
  if (targetWidth > startWidth) {
    window.requestAnimationFrame(() => {
      authSubmitButton.style.width = `${targetWidth}px`;
    });
  }
  const startedAt = performance.now();

  try {
    return await action();
  } finally {
    const elapsed = performance.now() - startedAt;
    if (elapsed < 300) {
      await delay(300 - elapsed);
    }
    authSubmitButton.disabled = false;
    authSubmitButton.classList.remove("is-loading");
    authSubmitButton.style.width = "";
    authSubmitButton.style.transition = "";
    setAuthSubmitLabel(getAuthSubmitLabel(activeAuthMode));
  }
}

function setAuthMode(mode, options = {}) {
  const { preserveStatus = false, startVerifyTimer = false } = options;
  const nextMode = ["login", "signup", "verify", "forgot", "reset"].includes(mode) ? mode : "login";
  activeAuthMode = nextMode;

  authModeButtons.forEach((button) => {
    const isActive = button.dataset.authMode === nextMode;
    button.classList.toggle("active", isActive);
    if (button.hasAttribute("aria-selected")) {
      button.setAttribute("aria-selected", isActive ? "true" : "false");
    }
  });

  const titleMap = {
    login: "Welcome Back",
    signup: "Create Your Account",
    verify: "Verify OTP",
    forgot: "Forgot Password",
    reset: "Reset Password",
  };

  const titleElement = document.getElementById("patient-login-title");
  if (titleElement) {
    titleElement.textContent = titleMap[nextMode];
    titleElement.classList.toggle("auth-title-small", nextMode === "signup");
  }

  if (authEmailLabel) {
    authEmailLabel.textContent = nextMode === "login" ? "Email or Username" : "Email";
  }

  if (authEmailInput) {
    authEmailInput.placeholder = nextMode === "login" ? "Enter email or username" : "Enter email";
  }

  setAuthSubmitLabel(getAuthSubmitLabel(nextMode));

  document.querySelectorAll("[data-auth-only]").forEach((element) => {
    const allowedModes = (element.getAttribute("data-auth-only") || "")
      .split(" ")
      .map((value) => value.trim())
      .filter(Boolean);
    const visible = allowedModes.includes(nextMode);
    element.classList.toggle("hidden", !visible);
    if (element instanceof HTMLElement) {
      element.hidden = !visible;
    }
  });

  if (!preserveStatus) {
    setAuthInlineStatus("");
  }

  if (nextMode === "verify") {
    if (startVerifyTimer) {
      startResendOtpTimer();
    } else {
      updateResendOtpButtonState();
    }
  } else {
    updateResendOtpButtonState();
  }
}

async function submitAuth(event) {
  event.preventDefault();
  const email = authEmailInput?.value?.trim() || "";
  const password = authPasswordInput?.value || "";
  const fullName = authNameInput?.value?.trim() || "";
  const otp = (authOtpInput?.value?.trim() || "").toUpperCase();

  try {
    if (activeAuthMode === "login") {
      await runAuthSubmitProgress("LOGGING IN", async () => {
        const response = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.success) throw new Error(payload.error || "Login failed");
        setAuthInlineStatus("");
        await fetchCurrentProfile();
        closeLoginPopup();
        await enterDashboard("diabetes");
      });
      return;
    }

    if (activeAuthMode === "signup") {
      await runAuthSubmitProgress("SENDING OTP", async () => {
        const response = await fetch("/api/auth/signup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password, name: fullName }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.success) throw new Error(payload.error || "Signup failed");
        setAuthMode("verify", { preserveStatus: true, startVerifyTimer: true });
        setAuthInlineStatus("");
      });
      return;
    }

    if (activeAuthMode === "verify") {
      await runAuthSubmitProgress("VERIFYING", async () => {
        const response = await fetch("/api/auth/verify-otp", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, otp }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.success) throw new Error(payload.error || "OTP verification failed");
        setAuthMode("login", { preserveStatus: true });
        setAuthInlineStatus(payload.message || "OTP verified. Please login.");
      });
      return;
    }

    if (activeAuthMode === "forgot") {
      await runAuthSubmitProgress("SENDING LINK", async () => {
        const response = await fetch("/api/auth/forgot-password", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.success) throw new Error(payload.error || "Request failed");
        setAuthInlineStatus(payload.message || "Temporary password sent to email.");
        setAuthMode("login", { preserveStatus: true });
      });
      return;
    }

    if (activeAuthMode === "reset") {
      await runAuthSubmitProgress("VERIFYING", async () => {
        const token = authResetTokenInput?.value?.trim() || "";
        const newPassword = authNewPasswordInput?.value || "";
        const response = await fetch("/api/auth/reset-password", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token, new_password: newPassword }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.success) throw new Error(payload.error || "Reset failed");
        setAuthInlineStatus(payload.message || "Password reset successful. Please login.");
        setAuthMode("login", { preserveStatus: true });
      });
    }
  } catch (error) {
    setAuthInlineStatus(mapAuthErrorMessage(error.message || "Authentication failed", activeAuthMode), {
      isError: true,
      autoHideMs: 3000,
    });
  }
}

const DEFAULT_PROFILE_AVATAR = userProfilePhoto?.getAttribute("src") || profilePhotoPreview?.getAttribute("src") || "";

function getProfilePhotoSource(profile = {}) {
  const photoPath = String(profile?.photo_path || "").trim();
  return photoPath || DEFAULT_PROFILE_AVATAR;
}

function setDashboardProfilePhoto(profile = {}) {
  const source = getProfilePhotoSource(profile);
  if (userProfilePhoto) userProfilePhoto.src = source;
  if (profilePhotoPreview) profilePhotoPreview.src = source;
}

function normalizeHistoryDiseaseKey(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function applyDiseaseInputs(formId, inputData = {}) {
  const form = document.getElementById(formId);
  if (!form || !inputData || typeof inputData !== "object") return;

  Object.entries(inputData).forEach(([name, value]) => {
    const field = form.querySelector(`[name="${name}"]`);
    if (!field) return;
    field.value = value == null ? "" : String(value);
  });
}

function closeHistoryImageModal() {
  if (historyImageModal) historyImageModal.hidden = true;
  if (historyImagePreview) historyImagePreview.src = "";
}

function openHistoryImageModal(imagePath) {
  if (!historyImageModal || !historyImagePreview || !imagePath) return;
  historyImagePreview.src = imagePath;
  historyImageModal.hidden = false;
}

function parseHistoryDate(value) {
  const text = String(value || "").trim();
  if (!text) return null;
  const normalized = text.endsWith("Z") ? text : `${text}Z`;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? null : date;
}

function filterHistoryEntries(entries = [], filterKey = "all") {
  if (!Array.isArray(entries) || filterKey === "all") return Array.isArray(entries) ? entries : [];

  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const ranges = {
    today: startOfToday,
    "7d": new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000),
    "30d": new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000),
  };
  const threshold = ranges[filterKey];
  if (!threshold) return entries;

  return entries.filter((entry) => {
    const createdAt = parseHistoryDate(entry?.created_at);
    if (!createdAt) return filterKey === "all";
    return createdAt >= threshold;
  });
}

function renderPastHealthInputs(entries = []) {
  if (!pastHealthInputsList) return;

  const normalizedEntries = filterHistoryEntries(Array.isArray(entries) ? entries : [], state.historyFilter || "all");
  const grouped = new Map();

  normalizedEntries.forEach((entry) => {
    const key = normalizeHistoryDiseaseKey(entry?.disease_type);
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(entry);
  });

  const topRowCards = [
    { title: "Coronary Artery Disease", key: "coronary_artery_disease", type: "form", formId: "heart-form" },
    { title: "Type 2 Diabetes", key: "type_2_diabetes", type: "form", formId: "diabetes-form" },
    { title: "Anemia", key: "anemia", type: "form", formId: "anemia-form" },
  ];

  const renderCard = (card) => {
    const cardEntries = grouped.get(card.key) || [];

    if (card.type === "form") {
      const latestEntry = cardEntries.find((entry) => entry?.input_data && typeof entry.input_data === "object");
      const inputData = latestEntry?.input_data || {};
      const lines = Object.entries(inputData)
        .slice(0, 8)
        .map(([k, v]) => `<p class="history-input-line"><strong>${k}</strong>: ${v}</p>`)
        .join("");

      const encodedPayload = encodeURIComponent(JSON.stringify(inputData));
      const hasInputs = Object.keys(inputData).length > 0;

      return `
        <article class="history-disease-card">
          <h5>${card.title}</h5>
          <div class="history-input-lines">
            ${hasInputs ? lines : '<p class="history-input-line">No history available.</p>'}
          </div>
          <button type="button" class="history-load-button" data-form-id="${card.formId}" data-inputs="${encodedPayload}" ${hasInputs ? "" : "disabled"}>Load Inputs</button>
        </article>
      `;
    }

    const imageItems = cardEntries
      .filter((entry) => String(entry?.image_path || "").trim())
      .slice(0, 12)
      .map((entry) => `<img src="${entry.image_path}" alt="${card.title} X-ray" class="history-xray-thumb" data-history-image="${entry.image_path}" />`)
      .join("");

    return `
      <article class="history-disease-card">
        <h5>${card.title}</h5>
        <div class="history-xray-grid">
          ${imageItems || '<p class="history-input-line">No X-ray history available.</p>'}
        </div>
      </article>
    `;
  };

  const xrayImages = Array.from(
    new Set(
      normalizedEntries
        .map((entry) => String(entry?.image_path || "").trim())
        .filter((path) => Boolean(path)),
    ),
  );

  const xrayImageMarkup = xrayImages
    .slice(0, 50)
    .map((path) => `<img src="${path}" alt="X-ray image" class="history-xray-thumb" data-history-image="${path}" />`)
    .join("");

  pastHealthInputsList.innerHTML = `
    <div class="past-history-row past-history-row-top">
      ${topRowCards.map((card) => renderCard(card)).join("")}
    </div>
    <div class="past-history-row past-history-row-bottom">
      <article class="history-disease-card history-disease-card-xray">
        <h5>X Ray Images</h5>
        <div class="history-xray-grid history-xray-strip">
          ${xrayImageMarkup || '<p class="history-input-line">No X-ray history available.</p>'}
        </div>
      </article>
    </div>
  `;

  pastHealthInputsList.querySelectorAll(".history-load-button").forEach((button) => {
    button.addEventListener("click", () => {
      const formId = button.getAttribute("data-form-id") || "";
      const payload = button.getAttribute("data-inputs") || "";
      if (!formId || !payload) return;
      try {
        const decoded = JSON.parse(decodeURIComponent(payload));
        applyDiseaseInputs(formId, decoded);
      } catch (_) {
        /* ignore malformed payload */
      }
    });
  });

  pastHealthInputsList.querySelectorAll("[data-history-image]").forEach((image) => {
    image.addEventListener("click", () => {
      const imagePath = image.getAttribute("data-history-image") || "";
      openHistoryImageModal(imagePath);
    });
  });
}

function hydrateUserDashboard(overview = {}) {
  const summary = overview.summary || {};
  const profile = summary.profile || {};
  const charts = overview.charts || {};
  const reports = overview.reports || [];
  const healthHistory = overview.health_history || [];
  const profileCompleteness = overview.profile_completeness || {};
  const riskAlerts = Array.isArray(overview.risk_alerts) ? overview.risk_alerts : [];
  const quickSnapshot = overview.quick_snapshot || {};
  const dataQuality = overview.data_quality || {};
  const timelineSeries = Array.isArray(overview.timeline?.series) ? overview.timeline.series : [];

  if (userNameLine) userNameLine.textContent = `Name: ${profile.name || "-"}`;
  if (userDobLine) userDobLine.textContent = `DOB: ${profile.dob || "-"}`;
  if (userGenderLine) userGenderLine.textContent = `Gender: ${profile.gender || "-"}`;
  if (userAbhaIdLine) userAbhaIdLine.textContent = `ABHA ID: ${profile.abha_id || "-"}`;
  if (userAddressLine) userAddressLine.textContent = `Address: ${profile.address || "-"}`;
  if (userMobileLine) userMobileLine.textContent = `Mobile: ${profile.mobile || "-"}`;
  setDashboardProfilePhoto(profile);

  if (summaryTotalPredictions) summaryTotalPredictions.textContent = String(summary.total_predictions || 0);
  if (summaryDiseasesCount) summaryDiseasesCount.textContent = String(summary.diseases_tracked || 0);
  if (summaryReportsCount) summaryReportsCount.textContent = String(reports.length || 0);

  const completenessScore = Math.max(0, Math.min(100, Number(profileCompleteness.score) || 0));
  const missingItems = Array.isArray(profileCompleteness.missing) ? profileCompleteness.missing : [];
  if (profileCompletenessFill) {
    profileCompletenessFill.style.width = `${completenessScore}%`;
  }
  if (profileCompletenessText) {
    profileCompletenessText.textContent = `${completenessScore}% complete`;
  }
  if (profileCompletenessPrompts) {
    profileCompletenessPrompts.textContent = missingItems.length
      ? `Missing: ${missingItems.join(", ")}`
      : "Profile context is complete.";
  }

  if (snapshotLastPrediction) {
    snapshotLastPrediction.textContent = String(quickSnapshot.last_prediction_at || "-") || "-";
  }
  if (snapshotHighestRisk) {
    const disease = String(quickSnapshot.highest_recent_risk_disease || "-");
    const riskValue = Number(quickSnapshot.highest_recent_risk) || 0;
    snapshotHighestRisk.textContent = `${riskValue.toFixed(1)}% (${disease})`;
  }
  if (snapshotActiveDisease) {
    snapshotActiveDisease.textContent = String(quickSnapshot.active_disease || "-");
  }

  if (fileSummaryTotal) {
    fileSummaryTotal.textContent = String(reports.length || 0);
  }
  const latestReport = reports.length ? reports[reports.length - 1] : null;
  if (fileSummaryLastFile) {
    fileSummaryLastFile.textContent = String(latestReport?.name || "-");
  }
  if (fileSummaryLastDate) {
    fileSummaryLastDate.textContent = String(latestReport?.at || "-");
  }

  const qualityScore = Math.max(0, Math.min(100, Number(dataQuality.score) || 0));
  const qualityMissing = Array.isArray(dataQuality.missing) ? dataQuality.missing : [];
  if (dataQualityScore) {
    dataQualityScore.textContent = `${qualityScore}%`;
  }
  if (dataQualityMissing) {
    dataQualityMissing.textContent = qualityMissing.length
      ? `Missing: ${qualityMissing.map((item) => item?.label || item?.key || "-").join(", ")}`
      : "All critical fields available.";
  }

  if (downloadLatestReport instanceof HTMLButtonElement) {
    const latestDisease = String(quickSnapshot.active_disease || "").trim();
    downloadLatestReport.disabled = !latestDisease || latestDisease === "-";
    downloadLatestReport.dataset.latestDisease = latestDisease;
  }

  renderMedicationAdherenceState();
  renderPreviousMedications();

  if (riskAlertsPanel && riskAlertsList) {
    if (!riskAlerts.length) {
      riskAlertsPanel.hidden = true;
      riskAlertsList.innerHTML = "";
    } else {
      riskAlertsPanel.hidden = false;
      riskAlertsList.innerHTML = riskAlerts
        .map(
          (alert) => `
            <article class="risk-alert-item">
              <h5>${alert.disease || "Risk alert"}</h5>
              <p>${alert.message || "Risk increased compared to previous result."}</p>
            </article>
          `,
        )
        .join("");
    }
  }

  if (reportHistoryList) {
    reportHistoryList.innerHTML = reports.length
      ? reports
          .slice()
          .reverse()
          .slice(0, 10)
          .map(
            (entry) => `
            <div class="profile-card">
              <button type="button" class="report-remove-button" data-report-id="${entry.id || ""}" aria-label="Remove report">×</button>
              <h4>${entry.name || "Report"}</h4>
              <p class="muted">Status: ${entry.status || "-"}</p>
              <p class="muted">Type: ${entry.type || "-"}</p>
              <p class="muted">Uploaded: ${entry.at || "-"}</p>
            </div>
          `,
          )
          .join("")
      : '<div class="profile-card"><p class="muted">No reports uploaded yet.</p></div>';

    reportHistoryList.querySelectorAll(".report-remove-button").forEach((button) => {
      button.addEventListener("click", async () => {
        const reportId = button.getAttribute("data-report-id") || "";
        if (!reportId) return;
        try {
          button.disabled = true;
          const response = await fetch(`/api/auth/reports/${encodeURIComponent(reportId)}`, { method: "DELETE" });
          const payload = await response.json();
          if (!response.ok || !payload.success) {
            throw new Error(payload.error || "Unable to remove report");
          }
          await loadUserDashboardOverview();
        } catch (error) {
          if (reportUploadStatus) reportUploadStatus.textContent = error.message || "Unable to remove report.";
        } finally {
          button.disabled = false;
        }
      });
    });
  }

  renderPastHealthInputs(healthHistory);

  const barData = Array.isArray(charts.bar) ? charts.bar : [];
  const donutData = Array.isArray(charts.donut) ? charts.donut : [];

  const donutImprovement = donutData.find((item) => String(item?.label || "").toLowerCase() === "improvement");
  const donutDeterioration = donutData.find((item) => String(item?.label || "").toLowerCase() === "deterioration");
  const improvementCount = Number(donutImprovement?.value) || 0;
  const deteriorationCount = Number(donutDeterioration?.value) || 0;

  const stats = barData
    .filter((item) => String(item?.label || "").trim().toUpperCase() !== "TS")
    .map((item) => ({
      short: String(item?.label || "-"),
      full: String(item?.full_label || item?.label || "-"),
      value: Number(item?.value) || 0,
      text: String(item?.display ?? item?.value ?? "0"),
    }));

  const statLabels = stats.map((item) => item.short);
  const statValues = stats.map((item) => item.value);
  const statTexts = stats.map((item) => item.text);
  const statFullForms = stats.map((item) => item.full);

  if (document.getElementById("user-progress-bar-chart")) {
    Plotly.purge("user-progress-bar-chart");
    Plotly.react(
      "user-progress-bar-chart",
      [
        {
          type: "bar",
          x: statLabels,
          y: statValues,
          text: statTexts,
          customdata: statFullForms,
          textposition: "outside",
          textfont: {
            color: "#1f2937",
            size: 14,
            family: "Inter, system-ui, sans-serif",
          },
          marker: {
            color: "#1d4ed8",
          },
          width: 0.72,
          hovertemplate: "<b>%{customdata}</b><br>%{text}<extra></extra>",
        },
      ],
      {
        margin: { t: 14, r: 10, b: 70, l: 86 },
        autosize: true,
        transition: { duration: 350, easing: "cubic-in-out" },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: {
          tickfont: { size: 14, color: "#3f3f46", family: "Inter, system-ui, sans-serif" },
          showline: true,
          linecolor: "#334155",
          linewidth: 2,
          automargin: true,
        },
        yaxis: {
          range: [0, 100],
          dtick: 10,
          tickfont: { size: 12, color: "#1f2937", family: "Inter, system-ui, sans-serif" },
          showgrid: false,
          zeroline: false,
          showline: true,
          linecolor: "#334155",
          linewidth: 2,
          ticks: "outside",
          ticklen: 5,
          automargin: true,
        },
        hoverlabel: {
          font: {
            size: 10,
            family: "Inter, system-ui, sans-serif",
          },
        },
      },
      { displayModeBar: false, responsive: true, scrollZoom: false },
    );
  }

  if (document.getElementById("user-prediction-donut-chart")) {
    Plotly.purge("user-prediction-donut-chart");
    Plotly.react(
      "user-prediction-donut-chart",
      [
        {
          type: "pie",
          labels: ["Improvement", "Deterioration"],
          values: [improvementCount > 0 ? improvementCount : 0.00001, deteriorationCount > 0 ? deteriorationCount : 0.00001],
          hole: 0.66,
          sort: false,
          direction: "clockwise",
          rotation: 90,
          textinfo: "none",
          marker: {
            colors: ["#fdba74", "#2fa74e"],
            line: {
              color: "#f8fafc",
              width: 6,
            },
          },
          hovertemplate: "%{label}<br>%{value} (%{percent})<extra></extra>",
          showlegend: true,
        },
      ],
      {
        margin: { t: 2, r: 10, b: 48, l: 10 },
        autosize: true,
        transition: { duration: 350, easing: "cubic-in-out" },
        paper_bgcolor: "rgba(0,0,0,0)",
        legend: {
          orientation: "h",
          x: 0.5,
          y: -0.04,
          xanchor: "center",
          yanchor: "top",
          font: { size: 20, color: "#1f2937", family: "Inter, system-ui, sans-serif" },
          entrywidthmode: "fraction",
          entrywidth: 0.48,
          traceorder: "normal",
        },
        hoverlabel: {
          font: {
            size: 10,
            family: "Inter, system-ui, sans-serif",
          },
        },
      },
      { displayModeBar: false, responsive: true, scrollZoom: false },
    );
  }

  if (document.getElementById("user-prediction-timeline-chart")) {
    Plotly.purge("user-prediction-timeline-chart");
    const traces = timelineSeries
      .filter((series) => Array.isArray(series?.points) && series.points.length > 0)
      .map((series) => ({
        type: "scatter",
        mode: "lines+markers",
        name: series.disease || "Disease",
        x: series.points.map((point) => point.date || point.at),
        y: series.points.map((point) => Number(point.value) || 0),
        line: { width: 2, shape: "spline", smoothing: 0.75 },
        marker: { size: 6 },
        hovertemplate: "%{x}<br>%{y:.1f}%<extra>%{fullData.name}</extra>",
      }));

    const fallbackTrace = traces.length
      ? traces
      : [
          {
            type: "scatter",
            mode: "lines",
            name: "No data",
            x: [],
            y: [],
          },
        ];

    Plotly.react(
      "user-prediction-timeline-chart",
      fallbackTrace,
      {
        margin: { t: 10, r: 12, b: 42, l: 48 },
        autosize: true,
        transition: { duration: 320, easing: "cubic-in-out" },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: {
          tickfont: { size: 11, color: "#475569" },
          automargin: true,
        },
        yaxis: {
          range: [0, 100],
          dtick: 20,
          tickfont: { size: 11, color: "#475569" },
          showgrid: true,
          gridcolor: "rgba(148, 163, 184, 0.15)",
          automargin: true,
        },
        legend: {
          orientation: "h",
          y: -0.2,
          x: 0,
        },
      },
      { displayModeBar: false, responsive: true, scrollZoom: false },
    );
  }
}

const DASHBOARD_OVERVIEW_CACHE_TTL_MS = 12000;
let dashboardOverviewCache = null;
let dashboardOverviewCacheAt = 0;
let dashboardOverviewRequest = null;

async function loadUserDashboardOverview(force = true, options = {}) {
  if (!state.auth.authenticated) return;
  const showLoader = Boolean(options?.showLoader);
  const minDuration = Math.max(200, Number(options?.minDuration) || 300);
  const loaderDelay = showLoader ? startLoaderDelay(minDuration) : null;
  const dashboardPanel = document.getElementById("user-dashboard-panel");

  if (showLoader) {
    dashboardPanel?.classList.add("loader-hidden");
  }

  try {
    const now = Date.now();
    if (!force && dashboardOverviewCache && (now - dashboardOverviewCacheAt) < DASHBOARD_OVERVIEW_CACHE_TTL_MS) {
      hydrateUserDashboard(dashboardOverviewCache);
      return;
    }

    if (dashboardOverviewRequest) {
      await dashboardOverviewRequest;
      return;
    }

    dashboardOverviewRequest = (async () => {
        const response = await fetch("/api/dashboard/overview");
        const payload = await response.json();
        if (!response.ok || !payload.success) {
          throw new Error(payload.error || "Unable to load dashboard overview");
        }
        dashboardOverviewCache = payload;
        dashboardOverviewCacheAt = Date.now();
        hydrateUserDashboard(payload);
      })();

    await dashboardOverviewRequest;
  } catch (error) {
    console.warn("Failed to load user dashboard overview", error);
  } finally {
    if (loaderDelay) {
      await loaderDelay.finish();
    }
    if (showLoader) {
      dashboardPanel?.classList.remove("loader-hidden");
    }
    dashboardOverviewRequest = null;
  }
}

async function resetSession(options = {}) {
  const { redirect = true, recordHistory = true } = options;
  await fetch("/api/auth/logout", { method: "POST" });
  await fetch("/api/reset", { method: "POST" });
  state.profile = null;
  state.auth.authenticated = false;
  state.auth.user = null;
  state.predictions = {};
  ["diabetes-result", "heart-result", "anemia-result", "pneumonia-result", "tuberculosis-result"].forEach((id) => {
    const container = document.getElementById(id);
    if (container) container.innerHTML = "";
  });
  authForm?.reset();
  clearPatientDobDerivedAge();
  diabetesForm.reset();
  heartForm.reset();
  anemiaForm.reset();
  pneumoniaForm?.reset();
  tuberculosisForm?.reset();
  chatHistory.innerHTML = "";
  profilesGrid.innerHTML = "";
  setPatientSummary(null);
  updateAuthUi(false);
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

  if (state.auth.authenticated) {
    await enterDashboard(storedTab === "landing" ? "diabetes" : storedTab, { recordHistory: false });
    return;
  }

  if (storedPage === "patient") {
    showPage("landing", { recordHistory: false });
    openLoginPopup();
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

function enhanceFormSubmitButtons() {
  const submitButtons = Array.from(document.querySelectorAll("form button[type='submit']"));
  submitButtons.forEach((button) => {
    if (!(button instanceof HTMLButtonElement)) {
      return;
    }

    if (button.classList.contains("prediction-button")) {
      return;
    }

    button.classList.add("goo-button", "form-goo-button");

    let label = button.querySelector(".goo-button-label");
    if (!label) {
      const labelText = (button.textContent || "").trim();
      button.textContent = "";
      label = document.createElement("span");
      label.className = "goo-button-label";
      label.textContent = labelText;
      button.appendChild(label);
    }

    setupGooButton(button);
  });
}

function bindEvents() {
  document.querySelectorAll('a[href^="/admin"]').forEach((link) => {
    if (!(link instanceof HTMLAnchorElement)) return;
    link.addEventListener("click", () => {
      const sourceToken = buildAdminSourceToken();
      const target = new URL(link.getAttribute("href") || "/admin/", window.location.origin);
      target.searchParams.set("from", sourceToken);
      link.setAttribute("href", `${target.pathname}${target.search}`);
    });
  });

  startButton?.addEventListener("click", async () => {
    if (state.auth.authenticated) {
      await enterDashboard("diabetes");
      return;
    }
    setAuthMode("login");
    openLoginPopup();
  });
  backButton?.addEventListener("click", () => closeLoginPopup({ returnFocus: true }));
  resetSessionButton?.addEventListener("click", resetSession);
  landingLoginLink?.addEventListener("click", (event) => {
    event.preventDefault();
    setAuthMode("login");
    openLoginPopup();
  });
  landingSignupLink?.addEventListener("click", (event) => {
    event.preventDefault();
    setAuthMode("signup");
    openLoginPopup();
  });
  landingDashboardLink?.addEventListener("click", async (event) => {
    event.preventDefault();
    if (!state.auth.authenticated) {
      setAuthMode("login");
      openLoginPopup();
      return;
    }
    await enterDashboard("user-dashboard");
  });

  logoutButton?.addEventListener("click", () => openLogoutConfirmModal());
  logoutConfirmOk?.addEventListener("click", async () => {
    if (logoutConfirmOk) {
      logoutConfirmOk.disabled = true;
      logoutConfirmOk.textContent = "Logging out...";
    }
    if (logoutConfirmCancel) {
      logoutConfirmCancel.disabled = true;
    }
    try {
      await logoutUser();
    } catch (_) {
      if (logoutConfirmOk) {
        logoutConfirmOk.disabled = false;
        logoutConfirmOk.textContent = "Logout";
      }
      if (logoutConfirmCancel) {
        logoutConfirmCancel.disabled = false;
      }
    }
  });
  logoutConfirmCancel?.addEventListener("click", () => closeLogoutConfirmModal());

  authModeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const mode = button.dataset.authMode || "login";
      setAuthMode(mode);
    });
  });

  authPasswordEyeButton?.addEventListener("click", () => {
    if (!authPasswordInput) return;
    authPasswordInput.type = authPasswordInput.type === "password" ? "text" : "password";
  });

  authNewPasswordEyeButton?.addEventListener("click", () => {
    if (!authNewPasswordInput) return;
    authNewPasswordInput.type = authNewPasswordInput.type === "password" ? "text" : "password";
  });

  forgotPasswordButton?.addEventListener("click", () => setAuthMode("forgot"));
  authGoogleSigninButton?.addEventListener("click", () => {
    window.location.assign("/api/auth/google/start");
  });
  resendVerificationButton?.addEventListener("click", async () => {
    if (resendOtpSecondsRemaining > 0) {
      return;
    }

    const email = authEmailInput?.value?.trim() || "";
    if (!email) {
      setAuthInlineStatus("Enter your email first.", { isError: true, autoHideMs: 3000 });
      return;
    }
    try {
      await runButtonProgress(resendVerificationButton, "Sending OTP", async () => {
        const response = await fetch("/api/auth/resend-verification", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email }),
        });
        const payload = await response.json();
        if (!response.ok || !payload.success) {
          throw new Error(payload.error || "Unable to resend OTP.");
        }
        startResendOtpTimer();
      });
    } catch (error) {
      setAuthInlineStatus(mapAuthErrorMessage(error.message || "Unable to resend OTP.", "verify"), {
        isError: true,
        autoHideMs: 3000,
      });
    }
  });
  authForm?.addEventListener("submit", submitAuth);
  chatbotLauncher?.setAttribute("aria-expanded", "false");

  tabTriggers.forEach((trigger) => {
    trigger.addEventListener("click", async (event) => {
      event?.preventDefault();
      if (trigger === landingDashboardLink) {
        if (!state.auth.authenticated) {
          setAuthMode("login");
          openLoginPopup();
          return;
        }
        await enterDashboard("user-dashboard");
        return;
      }

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

  if (resourcesDropdownToggle && resourcesDropdownMenu) {
    resourcesDropdownToggle.setAttribute("aria-expanded", "false");
    resourcesDropdownToggle.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      toggleResourcesDropdown();
    });

    document.addEventListener("click", (event) => {
      if (!isResourcesDropdownOpen) return;
      if (resourcesDropdownContainer?.contains(event.target)) return;
      closeResourcesDropdown();
    });
  }

  contactFormTrigger?.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    toggleContactFormPopup();
  });

  contactFormCloseButton?.addEventListener("click", () => {
    closeContactFormPopup();
  });

  resourceContactForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    const submitButton = resourceContactForm.querySelector(".resource-contact-submit");
    if (submitButton instanceof HTMLButtonElement) {
      const defaultLabel = submitButton.dataset.defaultLabel || submitButton.textContent || "Send Request";
      submitButton.dataset.defaultLabel = defaultLabel;
      submitButton.textContent = "✓";
      submitButton.disabled = true;

      window.setTimeout(() => {
        resourceContactForm.reset();
        resourceContactStatus.textContent = "";
        closeContactFormPopup();
        submitButton.textContent = defaultLabel;
        submitButton.disabled = false;
      }, 700);
      return;
    }

    resourceContactForm.reset();
    resourceContactStatus.textContent = "";
    closeContactFormPopup();
  });

  document.addEventListener("click", (event) => {
    if (!isContactFormPopupOpen) return;
    if (contactFormTrigger?.contains(event.target)) return;
    if (contactFormMenu?.contains(event.target)) return;
    closeContactFormPopup();
  });

  docsSectionTriggers.forEach((trigger) => {
    trigger.addEventListener("click", (event) => {
      event.preventDefault();
      const sectionKey = trigger.dataset.docsSection || "overview";
      closeResourcesDropdown();
      openDocsPopup(sectionKey);
    });
  });

  docsPopupClose?.addEventListener("click", () => closeDocsPopup({ returnFocus: true }));
  docsPopup?.addEventListener("click", (event) => {
    const jumpTrigger = event.target.closest("[data-docs-jump]");
    if (!jumpTrigger) return;
    const sectionKey = jumpTrigger.dataset.docsJump || "overview";
    const section = docsPopup.querySelector(`#docs-${sectionKey}`) || docsPopup.querySelector("#docs-overview");
    section?.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveDocsSection(sectionKey);
  });

  docsPopupMain?.addEventListener("scroll", syncDocsActiveFromScroll, { passive: true });

  pages.patient?.addEventListener("click", (event) => {
    if (event.target === pages.patient) {
      closeLoginPopup({ returnFocus: true });
    }
  });

  bindFileDropZone(medicalReportInput, isValidReportFile);
  bindFileDropZone(pneumoniaImageInput, isValidPneumoniaImage);
  bindFileDropZone(tuberculosisImageInput, isValidTuberculosisImage);
  initReferenceFileUpload(pneumoniaImageInput);
  initReferenceFileUpload(tuberculosisImageInput);

  diabetesForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrediction(diabetesForm, "/api/diabetes", "diabetes-result");
  });

  heartForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrediction(heartForm, "/api/heart", "heart-result");
  });

  anemiaForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPrediction(anemiaForm, "/api/anemia", "anemia-result");
  });

  pneumoniaForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPneumoniaPrediction();
  });

  tuberculosisForm?.addEventListener("submit", (event) => {
    event.preventDefault();
    submitTuberculosisPrediction();
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

    if (isResourcesDropdownOpen) {
      closeResourcesDropdown();
      handled = true;
    }

    if (isContactFormPopupOpen) {
      closeContactFormPopup();
      handled = true;
    }

    if (isDocsPopupOpen) {
      closeDocsPopup({ returnFocus: true });
      handled = true;
    }

    if (isLoginPopupOpen) {
      closeLoginPopup({ returnFocus: true });
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

    if (isLogoutConfirmModalOpen()) {
      closeLogoutConfirmModal();
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

  editProfileButton?.addEventListener("click", () => {
    if (!profileEditPanel) return;
    profileEditPanel.hidden = false;
    profileEditPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  const parseProfileDob = (value) => {
    const raw = String(value || "").trim();
    if (!raw) return null;

    let day = 0;
    let month = 0;
    let year = 0;

    const ddmmyyyy = /^(\d{2})\/(\d{2})\/(\d{4})$/;
    const yyyymmdd = /^(\d{4})-(\d{2})-(\d{2})$/;

    let match = raw.match(ddmmyyyy);
    if (match) {
      day = Number(match[1]);
      month = Number(match[2]);
      year = Number(match[3]);
    } else {
      match = raw.match(yyyymmdd);
      if (!match) return null;
      year = Number(match[1]);
      month = Number(match[2]);
      day = Number(match[3]);
    }

    const date = new Date(year, month - 1, day);
    if (
      Number.isNaN(date.getTime()) ||
      date.getFullYear() !== year ||
      date.getMonth() !== month - 1 ||
      date.getDate() !== day
    ) {
      return null;
    }

    const now = new Date();
    if (date > now || year < 1900) return null;

    return date;
  };

  profilePhotoFileInput?.addEventListener("change", async () => {
    const file = profilePhotoFileInput.files?.[0] || null;
    if (!file) {
      if (profilePhotoPreview) {
        profilePhotoPreview.src = userProfilePhoto?.src || DEFAULT_PROFILE_AVATAR;
      }
      return;
    }
    try {
      const dataUrl = await fileToDataUrl(file);
      if (profilePhotoPreview) profilePhotoPreview.src = dataUrl;
    } catch (_) {
      if (profilePhotoPreview) profilePhotoPreview.src = userProfilePhoto?.src || DEFAULT_PROFILE_AVATAR;
    }
  });

  profileEditForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const saveButton = profileEditForm.querySelector("button[type='submit']");
    const saveButtonLabel = saveButton?.querySelector(".goo-button-label");
    const saveButtonText = (saveButtonLabel?.textContent || saveButton?.textContent || "Save Changes").trim();

    if (saveButton) {
      saveButton.disabled = true;
    }
    if (saveButtonLabel) {
      saveButtonLabel.textContent = "Saving...";
    } else if (saveButton) {
      saveButton.textContent = "Saving...";
    }

    const formData = new FormData(profileEditForm);
    const payload = {};
    for (const [key, value] of formData.entries()) {
      if (key === "photo_file") continue;
      if (String(value || "").trim()) {
        payload[key] = String(value).trim();
      }
    }

    const mobileRaw = String(payload.mobile || "").trim();
    if (mobileRaw) {
      const digits = mobileRaw.replace(/\D/g, "");
      if (!/^\d{10}$/.test(digits)) {
        profileEditStatus.textContent = "Mobile number must be exactly 10 digits.";
        if (saveButton) saveButton.disabled = false;
        if (saveButtonLabel) {
          saveButtonLabel.textContent = saveButtonText;
        } else if (saveButton) {
          saveButton.textContent = saveButtonText;
        }
        return;
      }
      payload.mobile = digits;
    }

    const abhaRaw = String(payload.abha_id || "").trim();
    if (abhaRaw) {
      const digits = abhaRaw.replace(/\D/g, "");
      if (!/^\d{14}$/.test(digits)) {
        profileEditStatus.textContent = "ABHA ID must be exactly 14 digits.";
        if (saveButton) saveButton.disabled = false;
        if (saveButtonLabel) {
          saveButtonLabel.textContent = saveButtonText;
        } else if (saveButton) {
          saveButton.textContent = saveButtonText;
        }
        return;
      }
      payload.abha_id = digits;
    }

    const dobRaw = String(payload.dob || "").trim();
    if (dobRaw && !parseProfileDob(dobRaw)) {
      profileEditStatus.textContent = "Please enter a valid DOB (dd/mm/yyyy).";
      if (saveButton) saveButton.disabled = false;
      if (saveButtonLabel) {
        saveButtonLabel.textContent = saveButtonText;
      } else if (saveButton) {
        saveButton.textContent = saveButtonText;
      }
      return;
    }

    const newPasswordRaw = String(payload.new_password || "");
    const currentPasswordRaw = String(payload.current_password || "");
    if (newPasswordRaw || currentPasswordRaw) {
      if (!currentPasswordRaw || !newPasswordRaw) {
        profileEditStatus.textContent = "Enter both current and new password.";
        if (saveButton) saveButton.disabled = false;
        if (saveButtonLabel) {
          saveButtonLabel.textContent = saveButtonText;
        } else if (saveButton) {
          saveButton.textContent = saveButtonText;
        }
        return;
      }
      if (newPasswordRaw.length < 8) {
        profileEditStatus.textContent = "New password must be at least 8 characters.";
        if (saveButton) saveButton.disabled = false;
        if (saveButtonLabel) {
          saveButtonLabel.textContent = saveButtonText;
        } else if (saveButton) {
          saveButton.textContent = saveButtonText;
        }
        return;
      }
    }

    try {
      const photoFile = profilePhotoFileInput?.files?.[0] || null;
      if (photoFile) {
        const uploadForm = new FormData();
        uploadForm.append("photo", photoFile);
        const uploadResponse = await fetch("/api/profile/upload-photo", {
          method: "POST",
          body: uploadForm,
        });
        const uploadPayload = await uploadResponse.json();
        if (!uploadResponse.ok || !uploadPayload.success) {
          throw new Error(uploadPayload.error || "Unable to upload profile image");
        }
        payload.photo_path = uploadPayload.photo_path;
        if (userProfilePhoto && payload.photo_path) {
          userProfilePhoto.src = payload.photo_path;
        }
      }

      const response = await fetch("/api/auth/profile", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Unable to update profile");
      }
      profileEditStatus.textContent = "";
      if (data.require_relogin) {
        await resetSession({ redirect: true });
        return;
      }
      await loadUserDashboardOverview();
      await fetchCurrentProfile();
      if (profilePhotoFileInput) profilePhotoFileInput.value = "";
      if (profileEditPanel) {
        profileEditPanel.hidden = true;
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (error) {
      profileEditStatus.textContent = error.message;
    } finally {
      if (saveButton) {
        saveButton.disabled = false;
      }
      if (saveButtonLabel) {
        saveButtonLabel.textContent = saveButtonText;
      } else if (saveButton) {
        saveButton.textContent = saveButtonText;
      }
    }
  });

  reportUploadForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const uploadButton = reportUploadForm.querySelector("button[type='submit']");
    const uploadButtonLabel = uploadButton?.querySelector(".goo-button-label");
    const uploadButtonText = (uploadButtonLabel?.textContent || uploadButton?.textContent || "Upload").trim();

    if (uploadButton) {
      uploadButton.disabled = true;
    }
    if (uploadButtonLabel) {
      uploadButtonLabel.textContent = "Uploading...";
    } else if (uploadButton) {
      uploadButton.textContent = "Uploading...";
    }

    const file = userReportFileInput?.files?.[0] || null;
    if (!file) {
      if (reportUploadStatus) reportUploadStatus.textContent = "Select a report file first.";
      if (uploadButton) {
        uploadButton.disabled = false;
      }
      if (uploadButtonLabel) {
        uploadButtonLabel.textContent = uploadButtonText;
      } else if (uploadButton) {
        uploadButton.textContent = uploadButtonText;
      }
      return;
    }

    try {
      if (reportUploadStatus) reportUploadStatus.textContent = "";
      const formPayload = new FormData();
      formPayload.append("file", file);
      formPayload.append("status", "Uploaded");
      formPayload.append("disease_type", "pneumonia");

      const response = await fetch("/api/auth/reports", {
        method: "POST",
        body: formPayload,
      });
      const payload = await response.json();
      if (!response.ok || !payload.success) {
        throw new Error(payload.error || "Unable to upload report.");
      }
      reportUploadForm.reset();
      if (userReportFileName) userReportFileName.textContent = "No file selected";
      await loadUserDashboardOverview();
    } catch (error) {
      if (reportUploadStatus) reportUploadStatus.textContent = error.message;
    } finally {
      if (uploadButton) {
        uploadButton.disabled = false;
      }
      if (uploadButtonLabel) {
        uploadButtonLabel.textContent = uploadButtonText;
      } else if (uploadButton) {
        uploadButton.textContent = uploadButtonText;
      }
    }
  });

  userReportFileInput?.addEventListener("change", () => {
    const file = userReportFileInput.files?.[0] || null;
    if (userReportFileName) {
      userReportFileName.textContent = file?.name || "No file selected";
    }
  });

  historyImageClose?.addEventListener("click", () => closeHistoryImageModal());
  historyImageModal?.addEventListener("click", (event) => {
    if (event.target === historyImageModal || event.target.classList?.contains("history-image-backdrop")) {
      closeHistoryImageModal();
    }
  });

  pastHistoryFilters.forEach((button) => {
    button.addEventListener("click", async () => {
      const nextFilter = button.getAttribute("data-filter") || "all";
      state.historyFilter = nextFilter;
      pastHistoryFilters.forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      await loadUserDashboardOverview();
    });
  });

  medAdherenceYes?.addEventListener("click", () => {
    const today = getTodayKey();
    const yesterday = getYesterdayKey();
    const adherence = loadMedicationAdherenceState();
    let streak = Number(adherence.streak) || 0;

    if (adherence.lastYesDate !== today) {
      if (adherence.lastYesDate === yesterday) {
        streak += 1;
      } else {
        streak = 1;
      }
    }

    const next = {
      streak,
      lastYesDate: today,
      lastActionDate: today,
      lastAction: "yes",
    };
    saveMedicationAdherenceState(next);
    renderMedicationAdherenceState();
  });

  medAdherenceNo?.addEventListener("click", () => {
    const today = getTodayKey();
    const next = {
      streak: 0,
      lastYesDate: String(loadMedicationAdherenceState().lastYesDate || ""),
      lastActionDate: today,
      lastAction: "no",
    };
    saveMedicationAdherenceState(next);
    renderMedicationAdherenceState();
  });

  savePreviousMedicationsButton?.addEventListener("click", () => {
    const value = String(previousMedicationsInput?.value || "").trim();
    savePreviousMedications(value);
    if (previousMedicationsStatus) {
      previousMedicationsStatus.textContent = "Saved";
      window.setTimeout(() => {
        if (previousMedicationsStatus.textContent === "Saved") {
          previousMedicationsStatus.textContent = "";
        }
      }, 2000);
    }
  });

  shortcutNearestHospital?.addEventListener("click", async () => {
    await enterDashboard("consultants");
    activateConsultantView("hospital-view");
    if (consultantSearch) consultantSearch.value = "";
    await loadConsultants("");
  });

  shortcutFindDoctor?.addEventListener("click", async () => {
    await enterDashboard("consultants");
    activateConsultantView("doctor-view");
    if (consultantSearch) consultantSearch.value = "";
    await loadConsultants("");
  });

  dataQualityFixProfile?.addEventListener("click", () => {
    if (!profileEditPanel) return;
    profileEditPanel.hidden = false;
    profileEditPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  dataQualityFixHistory?.addEventListener("click", () => {
    reportUploadForm?.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  downloadLatestReport?.addEventListener("click", () => {
    const latestDisease = String(downloadLatestReport.dataset.latestDisease || "").trim();
    if (!latestDisease || latestDisease === "-") return;
    downloadDiseaseReport(latestDisease);
  });

  downloadFullHistory?.addEventListener("click", () => {
    const link = document.createElement("a");
    link.href = "/api/auth/history/export";
    link.download = "";
    document.body.appendChild(link);
    link.click();
    link.remove();
  });

  window.addEventListener("popstate", async (event) => {
    if (isRestoringHistory) return;

    closeDiseasesDropdown();
    closeResourcesDropdown();
    closeContactFormPopup();
    closeDocsPopup();
    closeLoginPopup();

    const stateData = event.state || {};
    const targetPage = stateData.page || "landing";
    const targetTab = stateData.tab || "diabetes";

    const currentState = getCurrentAppState();
    if (currentState.page === "dashboard" && targetPage !== "dashboard") {
      if (state.auth.authenticated) {
        const confirmed = window.confirm("You are logged in. Do you want to logout and leave the dashboard?");
        if (!confirmed) {
          updateHistoryState({ page: "dashboard", tab: currentState.tab });
          return;
        }
      }

      await resetSession({ recordHistory: false });
      updateHistoryState({ page: "landing", tab: "diabetes" }, { replace: true });
      return;
    }

    isRestoringHistory = true;
    try {
      if (targetPage === "patient") {
        showPage("landing", { recordHistory: false });
        openLoginPopup();
        return;
      }

      if (targetPage === "dashboard") {
        if (!state.profile) {
          await fetchCurrentProfile();
        }

        if (!state.auth.authenticated) {
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
  try {
    const params = new URLSearchParams(window.location.search);
    const resetToken = params.get("reset_token") || "";
    const verified = params.get("verified");
    const googleAuth = params.get("google_auth") || "";
    const returnTo = params.get("return_to") || "";

    if (returnTo) {
      applyAdminReturnToken(returnTo);
    }

    await fetchConfig();
    bindEvents();
    initializeFormConstraints();
    setAuthMode(resetToken ? "reset" : "login");
    if (authResetTokenInput) {
      authResetTokenInput.value = resetToken;
    }
    if (verified === "1") {
      setAuthInlineStatus("Email verified successfully. Please login.");
      openLoginPopup();
    }
    if (verified === "0") {
      setAuthInlineStatus("Verification link is invalid or expired.", { isError: true });
      openLoginPopup();
    }
    if (googleAuth === "success") {
      setAuthInlineStatus("Google sign-in successful.");
    } else if (googleAuth) {
      setAuthInlineStatus("Google sign-in failed. Please try again.", { isError: true, autoHideMs: 3000 });
      setAuthMode("login", { preserveStatus: true });
      openLoginPopup();
    }
    enhanceFormSubmitButtons();
    setupGooButton(startButton);
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
    if (shouldReopenResourcesMenu && getSessionValue("currentPage") === "landing") {
      openResourcesDropdown();
    }
    updateHistoryState(getCurrentAppState(), { replace: true });
  } finally {
    document.body?.classList.remove("app-booting");
  }
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
