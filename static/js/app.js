/* ── DOM refs ─────────────────────────────────────────────────────────────── */
const dropZone       = document.getElementById("dropZone");
const fileInput      = document.getElementById("fileInput");
const dropContent    = document.getElementById("dropContent");
const previewImg     = document.getElementById("previewImg");
const clearBtn       = document.getElementById("clearBtn");
const analyseBtn     = document.getElementById("analyseBtn");
const btnLabel       = document.getElementById("btnLabel");
const btnSpinner     = document.getElementById("btnSpinner");
const errorBanner    = document.getElementById("errorBanner");
const resultsSection = document.getElementById("resultsSection");

// result elements
const verdictBadge      = document.getElementById("verdictBadge");
const verdictClass      = document.getElementById("verdictClass");
const verdictConfidence = document.getElementById("verdictConfidence");
const probList          = document.getElementById("probList");
const originalImg       = document.getElementById("originalImg");
const overlayImg        = document.getElementById("overlayImg");
const heatmapImg        = document.getElementById("heatmapImg");

/* ── state ────────────────────────────────────────────────────────────────── */
let selectedFile = null;

const CLASS_COLORS = {
  Normal:    { bg: "rgba(34,197,94,.15)",  border: "#22c55e", fill: "#22c55e", emoji: "✅" },
  Benign:    { bg: "rgba(245,158,11,.15)", border: "#f59e0b", fill: "#f59e0b", emoji: "⚠️" },
  Malignant: { bg: "rgba(239,68,68,.15)",  border: "#ef4444", fill: "#ef4444", emoji: "🔴" },
};

/* ── file selection ───────────────────────────────────────────────────────── */

function setFile(file) {
  if (!file || !file.type.match(/image.*/)) {
    showError("Please select a valid image file.");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.classList.remove("hidden");
    dropContent.classList.add("hidden");
    originalImg.src = e.target.result;   // keep a copy for the results panel
  };
  reader.readAsDataURL(file);
  analyseBtn.disabled = false;
  clearBtn.disabled   = false;
  hideError();
  resultsSection.classList.add("hidden");
}

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

// drag-and-drop
dropZone.addEventListener("dragover",  (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  setFile(e.dataTransfer.files[0]);
});

/* ── clear ────────────────────────────────────────────────────────────────── */
clearBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewImg.classList.add("hidden");
  dropContent.classList.remove("hidden");
  analyseBtn.disabled = true;
  clearBtn.disabled   = true;
  resultsSection.classList.add("hidden");
  hideError();
});

/* ── analyse ──────────────────────────────────────────────────────────────── */
analyseBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  setLoading(true);
  hideError();

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const res  = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || "An unknown error occurred.");
      return;
    }

    renderResults(data);
  } catch (err) {
    showError("Network error – could not reach the server.");
  } finally {
    setLoading(false);
  }
});

/* ── render results ───────────────────────────────────────────────────────── */
function renderResults(data) {
  const col = CLASS_COLORS[data.label] || CLASS_COLORS["Normal"];

  // verdict card
  verdictBadge.textContent = col.emoji;
  verdictBadge.style.background  = col.bg;
  verdictBadge.style.border      = `2px solid ${col.border}`;
  verdictClass.textContent        = data.label;
  verdictClass.style.color        = col.fill;
  verdictConfidence.textContent   = `Confidence: ${data.confidence}%`;

  // probability bars
  probList.innerHTML = "";
  const order = ["Normal", "Benign", "Malignant"];
  order.forEach((cls) => {
    const pct   = data.probabilities[cls] ?? 0;
    const c     = CLASS_COLORS[cls];
    const row   = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-header">
        <span>${cls}</span>
        <span style="color:${c.fill}">${pct}%</span>
      </div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:0%;background:${c.fill}" data-target="${pct}"></div>
      </div>`;
    probList.appendChild(row);
  });

  // animate bars after a short delay
  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-bar-fill").forEach((el) => {
      el.style.width = el.dataset.target + "%";
    });
  });

  // images
  overlayImg.src = "data:image/png;base64," + data.overlay_b64;
  heatmapImg.src = "data:image/png;base64," + data.mask_b64;

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ── helpers ──────────────────────────────────────────────────────────────── */
function setLoading(on) {
  analyseBtn.disabled = on;
  btnLabel.classList.toggle("hidden", on);
  btnSpinner.classList.toggle("hidden", !on);
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
}

function hideError() {
  errorBanner.classList.add("hidden");
}
