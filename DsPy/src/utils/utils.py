"""
src/utils/ollama_setup.py — Ollama LM configuration for DSPy

src/utils/metrics.py — Evaluation metric functions for all three pipelines

Both in one file for conciseness.
"""

import sys
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from configs.settings import (
    OLLAMA_BASE_URL, OLLAMA_MODEL_PRIORITY,
    LM_TEMPERATURE, LM_MAX_TOKENS,
    THREAT_CATEGORIES, SEVERITY_LEVELS,
)


# =============================================================================
# Ollama Setup
# =============================================================================

def check_ollama() -> bool:
    try:
        import urllib.request
        r = urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status == 200
    except Exception:
        return False


def get_available_models() -> list[str]:
    try:
        import urllib.request
        r    = urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = json.loads(r.read())
        return [m["name"].split(":")[0] for m in data.get("models", [])]
    except Exception:
        return []


def configure_lm(
    model: str = None,
    temperature: float = LM_TEMPERATURE,
    max_tokens: int = LM_MAX_TOKENS,
    verbose: bool = False,
) -> str:
    """
    Configure DSPy to use a local Ollama model.
    Returns the model name selected.

    DSPy connects to Ollama via its OpenAI-compatible API endpoint.
    No API key needed. Data never leaves your machine.
    """
    print("=" * 55)
    print("DSPy Cybersecurity SOC Assistant")
    print("=" * 55)

    if not check_ollama():
        print("\nOllama is not running. Setup steps:")
        print("  1. Download from https://ollama.ai")
        print("  2. Run: ollama serve")
        print("  3. Run: ollama pull llama3.2")
        raise RuntimeError("Ollama server not reachable at " + OLLAMA_BASE_URL)

    available = get_available_models()
    print(f"\nOllama running | Available models: {available or 'none'}")

    if not available:
        raise RuntimeError("No models found. Run: ollama pull llama3.2")

    if model is None:
        for m in OLLAMA_MODEL_PRIORITY:
            if m in available:
                model = m
                break
        if model is None:
            model = available[0]
        print(f"Auto-selected model: {model}")
    elif model not in available:
        fallback = available[0]
        print(f"  '{model}' not found. Using '{fallback}'.")
        model = fallback

    lm = dspy.LM(
        model=f"ollama/{model}",
        api_base=OLLAMA_BASE_URL,
        api_key="ollama",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)

    print(f"\nDSPy configured:")
    print(f"  Model:       {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens:  {max_tokens}")
    print(f"  Local only:  yes (Ollama)\n")
    return model


# =============================================================================
# Evaluation Metrics
# =============================================================================

def threat_accuracy_metric(example: dspy.Example, pred, trace=None) -> bool:
    """
    Binary metric for threat classification.
    Returns True if predicted threat_type matches ground truth.
    Used by DSPy optimizers (BootstrapFewShot, MIPROv2).
    """
    gold = str(example.threat_type).strip()
    # pred can be a ThreatReport dataclass or dspy.Prediction
    predicted = str(getattr(pred, "threat_type", "")).strip()
    return gold == predicted


def threat_weighted_metric(example: dspy.Example, pred, trace=None) -> float:
    """
    Weighted metric: correct + confidence bonus.
    Used when we want the optimizer to maximize both accuracy and calibration.
    """
    gold      = str(example.threat_type).strip()
    predicted = str(getattr(pred, "threat_type", "")).strip()
    if gold != predicted:
        return 0.0
    confidence = getattr(pred, "confidence", 0.5)
    if isinstance(confidence, str):
        m = re.search(r"[\d.]+", confidence)
        confidence = float(m.group()) if m else 0.5
    confidence = max(0.0, min(1.0, float(confidence)))
    if confidence > 1.0:
        confidence /= 100.0
    return 0.5 + 0.5 * confidence


def cve_severity_metric(example: dspy.Example, pred, trace=None) -> bool:
    """Binary metric for CVE severity classification."""
    gold      = str(example.severity).strip().upper()
    predicted = str(getattr(pred, "severity", "")).strip().upper()
    for s in SEVERITY_LEVELS:
        if s in predicted:
            predicted = s
            break
    return gold == predicted


def cve_adjacent_metric(example: dspy.Example, pred, trace=None) -> float:
    """
    Partial credit: exact match = 1.0, adjacent level = 0.5, else 0.
    CRITICAL/HIGH adjacent, HIGH/MEDIUM adjacent, etc.
    """
    order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFORMATIONAL": 0}
    gold      = str(example.severity).strip().upper()
    predicted = str(getattr(pred, "severity", "")).strip().upper()
    for s in SEVERITY_LEVELS:
        if s in predicted:
            predicted = s
            break
    if gold == predicted:
        return 1.0
    if abs(order.get(gold, 0) - order.get(predicted, 0)) == 1:
        return 0.5
    return 0.0


def alert_triage_metric(example: dspy.Example, pred, trace=None) -> float:
    """
    Evaluate alert triage by checking:
    - Is the top-severity alert first in the priority order? (0.5 weight)
    - Is escalate_immediately correct? (0.3 weight)
    - Did the model detect that CRITICAL alerts need escalation? (0.2 weight)
    """
    score = 0.0

    # Check top-priority correctness
    try:
        expected_order  = json.loads(str(example.expected_priority_order))
        predicted_order = getattr(pred, "priority_order", [])
        if isinstance(predicted_order, str):
            m = re.findall(r'ALT-\d+', predicted_order)
            predicted_order = m
        if predicted_order and expected_order:
            if predicted_order[0] == expected_order[0]:
                score += 0.5
    except Exception:
        pass

    # Check escalation flag
    try:
        should_escalate = str(example.top_severity).upper() in ("CRITICAL", "HIGH")
        did_escalate    = bool(getattr(pred, "escalate_immediately", False))
        if should_escalate == did_escalate:
            score += 0.3
    except Exception:
        pass

    # Bonus for detecting CRITICAL
    try:
        if str(example.top_severity).upper() == "CRITICAL":
            summary = str(getattr(pred, "top_threat_summary", "")).upper()
            if "CRITICAL" in summary or "IMMEDIATELY" in summary or "ESCALAT" in summary:
                score += 0.2
    except Exception:
        pass

    return score


def evaluate_pipeline(
    module,
    test_examples: list,
    metric_fn,
    pipeline_name: str = "Pipeline",
    n_samples: int = None,
    verbose: bool = True,
) -> dict:
    """
    Run DSPy evaluation on a list of test examples.
    Returns a dict of metrics.
    """
    samples = test_examples[:n_samples] if n_samples else test_examples
    print(f"\nEvaluating {pipeline_name} on {len(samples)} examples...")

    scores, errors = [], 0
    for i, ex in enumerate(samples):
        try:
            pred  = module(**{k: getattr(ex, k) for k in ex.inputs()})
            score = metric_fn(ex, pred)
            scores.append(float(score))
            if verbose and i < 5:
                gold = {k: getattr(ex, k, "?") for k in ex.labels()}
                print(f"  [{i+1}] score={score:.2f} | gold={gold}")
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            scores.append(0.0)
            errors += 1

    avg = sum(scores) / max(len(scores), 1)
    metrics = {
        "pipeline":    pipeline_name,
        "n_samples":   len(samples),
        "avg_score":   round(avg, 3),
        "n_correct":   sum(1 for s in scores if s >= 0.9),
        "n_partial":   sum(1 for s in scores if 0.1 <= s < 0.9),
        "n_wrong":     sum(1 for s in scores if s < 0.1),
        "n_errors":    errors,
    }
    print(f"\n  Avg score: {avg:.3f} ({avg*100:.1f}%)")
    print(f"  Correct: {metrics['n_correct']} | Partial: {metrics['n_partial']} | Wrong: {metrics['n_wrong']}")
    return metrics
