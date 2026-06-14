"""
configs/settings.py — Central configuration for the DSPy SOC Assistant project.

All tunable parameters live here. Import this module at the top of every
notebook and pipeline file rather than hardcoding values.
"""

from pathlib import Path
import os

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
PROCESSED_DIR  = DATA_DIR / "processed"
OUTPUTS_DIR    = ROOT_DIR / "outputs"
REPORTS_DIR    = OUTPUTS_DIR / "reports"
OPTIMIZED_DIR  = OUTPUTS_DIR / "optimized"
SRC_DIR        = ROOT_DIR / "src"

# Create dirs if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, REPORTS_DIR, OPTIMIZED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ollama LM settings ────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Priority order — first available model wins
OLLAMA_MODEL_PRIORITY = [
    "llama3.2",     # best reasoning, 2GB
    "phi3.5",       # strong, Microsoft, 2.2GB
    "mistral",      # solid, 4GB
    "qwen2.5",      # lightest, 0.4GB
    "llama3.2:1b",  # 1B fallback
]

LM_TEMPERATURE  = 0.1    # low = more deterministic (good for classification)
LM_MAX_TOKENS   = 1024   # enough for CoT + structured output

# ── Dataset settings ──────────────────────────────────────────────────────────

# CICIDS 2017 — network intrusion dataset
# Full dataset: https://www.unb.ca/cic/datasets/ids-2017.html
# We use the Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv subset
CICIDS_URL = (
    "https://raw.githubusercontent.com/Mazzahra/CIC-IDS-2017/"
    "main/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)
CICIDS_MAX_SAMPLES    = 500    # per class
CICIDS_VAL_RATIO      = 0.15

# NVD CVE API — NIST public API, no key required
NVD_API_BASE          = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_MAX_CVES          = 200
NVD_RESULTS_PER_PAGE  = 50
NVD_RATE_LIMIT_SLEEP  = 0.7   # seconds between requests (5 req/30s limit)

# UNSW-NB15 — supplementary network anomaly data
UNSW_HF_ID            = "rdpahalavan/cyber-security-intrusion-detection"
UNSW_MAX_SAMPLES      = 300

# Alert triage — fully synthetic (generated deterministically)
ALERT_N_TRAIN         = 200
ALERT_N_TEST          = 50
ALERT_SEED            = 42

# ── Training / evaluation settings ────────────────────────────────────────────
EVAL_N_THREADS        = 1      # CPU-safe: 1 thread (set to 4+ with GPU)
BOOTSTRAP_MAX_DEMOS   = 3      # few-shot examples per predictor
BOOTSTRAP_MAX_ROUNDS  = 1      # optimization rounds (increase for better results)

# ── Output settings ───────────────────────────────────────────────────────────
REPORT_FORMAT         = "json"  # "json" | "text"
VERBOSE               = True

# ── Threat categories (used across pipelines) ─────────────────────────────────
THREAT_CATEGORIES = [
    "DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Infiltration",
    "Botnet",
    "Normal",
]

SEVERITY_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"]

CVE_SEVERITY_MAP = {
    "CRITICAL": (9.0, 10.0),
    "HIGH":     (7.0, 8.9),
    "MEDIUM":   (4.0, 6.9),
    "LOW":      (0.1, 3.9),
}
