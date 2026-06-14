"""
Cybersecurity dataset loader.
All datasets are free / public domain.

Datasets used:
  - SecQA        : huggingface haonan-li/secqa  (cybersec Q&A, multiple choice)
  - CyberGuard   : huggingface walledai/CyberGuard  (instruction pairs, good/bad responses)
  - NVD CVEs     : NIST NVD API (free, no key needed for basic use)
  - Phishing URLs: huggingface pirocheto/phishing-url (labelled URLs)
"""

import json
import random
import re
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Install guard — call once at top of each notebook
# ---------------------------------------------------------------------------
def install_deps():
    import subprocess, sys
    pkgs = ["datasets", "transformers", "trl", "accelerate", "torch", "requests"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
    print("All dependencies installed.")


# ---------------------------------------------------------------------------
# 1. SFT dataset  — SecQA: cybersec multiple-choice Q&A
#    Formatted as instruction-following: "Answer this security question: ..."
# ---------------------------------------------------------------------------
def load_sft_dataset(split="train", max_samples=500):
    """
    Returns list of dicts: {"prompt": str, "response": str}
    Task: given a cybersecurity question + options, output the correct answer with explanation.
    """
    from datasets import load_dataset
    print(f"Loading SecQA ({split})...")
    try:
        ds = load_dataset("haonan-li/secqa", split=split, trust_remote_code=True)
    except Exception:
        # Fallback: build synthetic cybersec QA from known facts
        print("  SecQA unavailable, using built-in synthetic fallback...")
        return _synthetic_sft_data(max_samples)

    records = []
    for row in ds:
        question = row.get("question", "")
        choices  = [row.get(f"option{i}", "") for i in range(1, 5) if row.get(f"option{i}")]
        answer   = row.get("answer", "")

        choice_str = "\n".join([f"  {chr(65+i)}) {c}" for i, c in enumerate(choices)])
        prompt = (
            "You are a cybersecurity expert. Answer the following question.\n\n"
            f"Question: {question}\n\nOptions:\n{choice_str}\n\nAnswer:"
        )
        response = f"The correct answer is {answer}."
        records.append({"prompt": prompt, "response": response})
        if len(records) >= max_samples:
            break

    print(f"  Loaded {len(records)} SFT samples.")
    return records


def _synthetic_sft_data(n=300):
    """Hard-coded cybersec Q&A pairs as fallback."""
    qa_pairs = [
        ("What does SQL injection exploit?", "SQL injection exploits unsanitized user input in database queries, allowing attackers to manipulate the SQL logic to extract, modify, or delete data."),
        ("What is a zero-day vulnerability?", "A zero-day vulnerability is a software flaw unknown to the vendor, giving attackers an advantage since no patch exists at the time of exploitation."),
        ("What does XSS stand for and what does it do?", "XSS stands for Cross-Site Scripting. It injects malicious scripts into web pages viewed by other users, potentially stealing session tokens or credentials."),
        ("What is the purpose of a firewall?", "A firewall monitors and controls incoming and outgoing network traffic based on predetermined security rules, creating a barrier between trusted internal networks and untrusted external networks."),
        ("What is phishing?", "Phishing is a social engineering attack where attackers impersonate legitimate entities via email or websites to steal credentials or install malware."),
        ("What is the CIA triad?", "The CIA triad stands for Confidentiality, Integrity, and Availability — the three core principles of information security."),
        ("What is a buffer overflow attack?", "A buffer overflow attack writes more data into a buffer than it can hold, overwriting adjacent memory and potentially allowing arbitrary code execution."),
        ("What is two-factor authentication?", "Two-factor authentication (2FA) requires two forms of verification — typically something you know (password) and something you have (token) — making unauthorized access significantly harder."),
        ("What is ransomware?", "Ransomware is malware that encrypts a victim's files and demands payment for the decryption key, often targeting organizations for maximum financial impact."),
        ("What does a SIEM system do?", "A Security Information and Event Management (SIEM) system aggregates and analyzes log data from across an infrastructure to detect anomalies and security events in real time."),
        ("What is a man-in-the-middle attack?", "A man-in-the-middle (MitM) attack intercepts communications between two parties without their knowledge, allowing the attacker to eavesdrop or alter the data in transit."),
        ("What is privilege escalation?", "Privilege escalation is gaining elevated access to resources by exploiting bugs, misconfigurations, or design flaws, allowing an attacker to perform actions beyond their authorization."),
        ("What is a DDoS attack?", "A Distributed Denial of Service (DDoS) attack overwhelms a target server or network with traffic from many compromised machines, making the service unavailable to legitimate users."),
        ("What is the difference between symmetric and asymmetric encryption?", "Symmetric encryption uses the same key to encrypt and decrypt data (fast, good for large data). Asymmetric encryption uses a public/private key pair (slower, used for key exchange and authentication)."),
        ("What is a CVE?", "CVE stands for Common Vulnerabilities and Exposures — a publicly available list of cybersecurity vulnerabilities, each assigned a unique identifier for standardized tracking."),
    ]
    templates = [
        "You are a cybersecurity expert. Answer concisely.\n\nQuestion: {q}\n\nAnswer:",
        "As a security analyst, explain the following:\n\nQ: {q}\n\nA:",
        "Cybersecurity training question:\n\n{q}\n\nExpert answer:",
    ]
    records = []
    for i in range(n):
        q, a = qa_pairs[i % len(qa_pairs)]
        tmpl = templates[i % len(templates)]
        records.append({"prompt": tmpl.format(q=q), "response": a})
    return records


# ---------------------------------------------------------------------------
# 2. DPO dataset — CyberGuard / synthetic preference pairs
#    Format: {"prompt": str, "chosen": str, "rejected": str}
# ---------------------------------------------------------------------------
def load_dpo_dataset(max_samples=300):
    """
    Returns list of dicts: {"prompt": str, "chosen": str, "rejected": str}
    chosen = detailed, accurate cybersec response
    rejected = vague, incorrect, or harmful response
    """
    from datasets import load_dataset
    print("Loading DPO preference dataset...")
    try:
        ds = load_dataset("walledai/CyberGuard", split="train", trust_remote_code=True)
        records = []
        # CyberGuard has 'prompt', 'response', 'category', 'label'
        # label=1 means harmful/bad, label=0 means safe/good
        # We create pairs: good response vs bad paraphrase
        safe_rows   = [r for r in ds if r.get("label") == 0][:max_samples]
        unsafe_rows = [r for r in ds if r.get("label") == 1][:max_samples]
        for s, u in zip(safe_rows, unsafe_rows):
            records.append({
                "prompt":   s["prompt"],
                "chosen":   s["response"],
                "rejected": u["response"],
            })
        if records:
            print(f"  Loaded {len(records)} DPO pairs from CyberGuard.")
            return records
    except Exception as e:
        print(f"  CyberGuard unavailable ({e}), using synthetic fallback...")

    return _synthetic_dpo_data(max_samples)


def _synthetic_dpo_data(n=200):
    """Synthetic preference pairs with clear quality difference."""
    pairs = [
        {
            "prompt": "Explain what SQL injection is and how to prevent it.",
            "chosen": (
                "SQL injection is an attack where malicious SQL code is inserted into a query via "
                "unsanitized user input, allowing attackers to read, modify, or delete database data. "
                "Prevention: use parameterized queries / prepared statements, ORMs, input validation, "
                "least-privilege DB accounts, and WAFs."
            ),
            "rejected": "SQL injection is bad. Don't let people put things in your database.",
        },
        {
            "prompt": "What CVSS score indicates a critical vulnerability?",
            "chosen": (
                "CVSS (Common Vulnerability Scoring System) uses a 0–10 scale. A score of 9.0–10.0 is "
                "classified as Critical. Scores 7.0–8.9 are High, 4.0–6.9 Medium, 0.1–3.9 Low. "
                "Critical vulnerabilities typically allow unauthenticated remote code execution."
            ),
            "rejected": "I'm not sure, maybe 7 or 8? It depends on the situation.",
        },
        {
            "prompt": "How does a phishing email typically trick users?",
            "chosen": (
                "Phishing emails use urgency ('Your account will be suspended'), spoofed sender addresses "
                "that mimic legitimate domains (e.g., support@paypa1.com), lookalike logos, and malicious "
                "links pointing to fake login pages that harvest credentials. Advanced spear-phishing "
                "personalizes content using OSINT about the target."
            ),
            "rejected": "Phishing emails are fake. They look real but aren't. Don't click links.",
        },
        {
            "prompt": "What is the difference between IDS and IPS?",
            "chosen": (
                "An IDS (Intrusion Detection System) monitors network traffic and generates alerts when "
                "suspicious patterns are detected — it is passive and does not block traffic. "
                "An IPS (Intrusion Prevention System) sits inline and actively blocks or drops malicious "
                "traffic in real time. IPS has higher risk of false positives disrupting legitimate traffic."
            ),
            "rejected": "IDS and IPS are both security tools that watch your network.",
        },
        {
            "prompt": "Describe the steps of a penetration test.",
            "chosen": (
                "A penetration test follows these phases: (1) Reconnaissance — passive/active info gathering; "
                "(2) Scanning — port scanning, service enumeration, vulnerability scanning; "
                "(3) Exploitation — attempting to exploit identified vulnerabilities; "
                "(4) Post-exploitation — privilege escalation, lateral movement, data exfiltration simulation; "
                "(5) Reporting — documenting findings, risk ratings, and remediation recommendations."
            ),
            "rejected": "You try to hack the system and then write a report about what you found.",
        },
        {
            "prompt": "What is certificate pinning and why is it used?",
            "chosen": (
                "Certificate pinning associates a host with its expected X.509 certificate or public key. "
                "When a client connects, it verifies the server certificate matches the pinned value, "
                "preventing MitM attacks even if a rogue CA certificate is trusted by the OS. "
                "It is commonly used in mobile apps to prevent traffic interception via proxy tools."
            ),
            "rejected": "Certificate pinning makes SSL more secure somehow.",
        },
    ]
    records = []
    for i in range(n):
        records.append(pairs[i % len(pairs)])
    return records


# ---------------------------------------------------------------------------
# 3. GRPO dataset — CVE severity reasoning (verifiable reward)
#    Format: {"prompt": str, "answer": str}  where answer is CRITICAL/HIGH/MEDIUM/LOW
# ---------------------------------------------------------------------------
def load_grpo_dataset(max_samples=200):
    """
    Returns list of dicts: {"prompt": str, "answer": str}
    Task: given a CVE description, output severity in <severity>LEVEL</severity> tags.
    Reward function checks format compliance + accuracy.
    """
    print("Loading GRPO CVE dataset (NIST NVD API)...")
    records = _fetch_nvd_cves(max_samples)
    if not records:
        print("  NVD API unavailable, using synthetic CVE fallback...")
        records = _synthetic_cve_data(max_samples)
    print(f"  Loaded {len(records)} CVE records.")
    return records


def _fetch_nvd_cves(n=200):
    """Fetch real CVEs from NIST NVD public API (no key required)."""
    import requests
    records = []
    base = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {"resultsPerPage": min(n, 50), "startIndex": 0}
    try:
        r = requests.get(base, params=params, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        for item in data.get("vulnerabilities", []):
            cve = item.get("cve", {})
            desc_list = cve.get("descriptions", [])
            desc = next((d["value"] for d in desc_list if d["lang"] == "en"), None)
            if not desc:
                continue
            metrics = cve.get("metrics", {})
            severity = None
            for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                entries = metrics.get(key, [])
                if entries:
                    severity = entries[0].get("cvssData", {}).get("baseSeverity") or \
                               entries[0].get("baseSeverity")
                    break
            if not severity:
                continue
            severity = severity.upper()
            if severity not in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}:
                continue
            prompt = (
                "You are a cybersecurity analyst. Analyze the following CVE description and "
                "classify its severity. Output your reasoning, then wrap your final answer in "
                "<severity>LEVEL</severity> tags where LEVEL is one of: CRITICAL, HIGH, MEDIUM, LOW.\n\n"
                f"CVE Description: {desc[:400]}\n\nAnalysis:"
            )
            records.append({"prompt": prompt, "answer": severity})
            if len(records) >= n:
                break
        time.sleep(0.6)  # NVD rate limit: 5 req/30s without API key
    except Exception as e:
        print(f"  NVD fetch error: {e}")
    return records


def _synthetic_cve_data(n=150):
    templates = [
        {
            "desc": "A remote code execution vulnerability exists in Apache Log4j 2.x before 2.15.0 via the JNDI lookup feature, allowing attackers to load arbitrary remote code via crafted log messages.",
            "answer": "CRITICAL"
        },
        {
            "desc": "An authentication bypass vulnerability in Cisco IOS allows a remote attacker to gain administrative access without valid credentials by sending a specially crafted HTTP request.",
            "answer": "CRITICAL"
        },
        {
            "desc": "A stored cross-site scripting (XSS) vulnerability in a web application allows authenticated users to inject persistent scripts, potentially hijacking sessions of other users.",
            "answer": "HIGH"
        },
        {
            "desc": "A privilege escalation vulnerability in the Linux kernel allows a local authenticated user to gain root privileges via a race condition in the memory management subsystem.",
            "answer": "HIGH"
        },
        {
            "desc": "An information disclosure vulnerability in a REST API endpoint exposes internal server path information in error messages, which could assist attackers in further reconnaissance.",
            "answer": "MEDIUM"
        },
        {
            "desc": "A reflected XSS vulnerability in the search parameter of a web application is exploitable only when a victim clicks a specially crafted link while authenticated.",
            "answer": "MEDIUM"
        },
        {
            "desc": "A documentation page reveals the default username used during initial installation. This information is publicly known and requires additional access to exploit.",
            "answer": "LOW"
        },
        {
            "desc": "A minor information disclosure in HTTP response headers reveals the server software version, which is widely known and does not directly enable exploitation.",
            "answer": "LOW"
        },
    ]
    system = (
        "You are a cybersecurity analyst. Analyze the following CVE description and classify its severity. "
        "Output your reasoning, then wrap your final answer in <severity>LEVEL</severity> tags "
        "where LEVEL is one of: CRITICAL, HIGH, MEDIUM, LOW.\n\n"
    )
    records = []
    for i in range(n):
        t = templates[i % len(templates)]
        prompt = system + f"CVE Description: {t['desc']}\n\nAnalysis:"
        records.append({"prompt": prompt, "answer": t["answer"]})
    return records


# ---------------------------------------------------------------------------
# 4. RLHF dataset — phishing URL explanation (reward model training)
#    Format for RM: {"prompt": str, "response": str, "label": int}  (1=good, 0=bad)
#    Format for PPO: {"prompt": str}
# ---------------------------------------------------------------------------
def load_rlhf_dataset(max_samples=300):
    """
    Returns (rm_data, ppo_prompts).
    rm_data: list of {"prompt", "response", "label"} for reward model training
    ppo_prompts: list of {"prompt"} for PPO rollouts
    """
    from datasets import load_dataset
    print("Loading phishing dataset (pirocheto/phishing-url)...")
    try:
        ds = load_dataset("pirocheto/phishing-url", split="train", trust_remote_code=True)
        urls = [(row["url"], int(row["status"] == "phishing")) for row in ds][:max_samples]
    except Exception as e:
        print(f"  Phishing dataset unavailable ({e}), using synthetic fallback...")
        urls = _synthetic_phishing_urls(max_samples)

    rm_data, ppo_prompts = [], []
    for url, is_phishing in urls:
        prompt = (
            "Analyze the following URL and explain whether it is likely phishing or legitimate. "
            "Provide specific indicators.\n\n"
            f"URL: {url}\n\nAnalysis:"
        )
        if is_phishing:
            good_resp = (
                f"This URL shows multiple phishing indicators: the domain appears to spoof a legitimate "
                f"brand, it uses an unusual TLD or subdomain structure, and the path contains urgency "
                f"keywords. Verdict: PHISHING. Recommend blocking and user awareness training."
            )
            bad_resp = "This looks suspicious I guess. Maybe don't click it."
        else:
            good_resp = (
                f"This URL appears legitimate: it uses a recognized top-level domain, the hostname "
                f"matches a known organization, and there are no unusual redirects or obfuscation "
                f"patterns. Verdict: LEGITIMATE. Standard security hygiene still applies."
            )
            bad_resp = "This URL is fine, it looks normal."

        rm_data.append({"prompt": prompt, "response": good_resp, "label": 1})
        rm_data.append({"prompt": prompt, "response": bad_resp,  "label": 0})
        ppo_prompts.append({"prompt": prompt})

    print(f"  RM training pairs: {len(rm_data)}, PPO prompts: {len(ppo_prompts)}")
    return rm_data, ppo_prompts


def _synthetic_phishing_urls(n=150):
    legit = [
        ("https://www.google.com/search?q=cybersecurity", 0),
        ("https://github.com/huggingface/transformers", 0),
        ("https://docs.python.org/3/library/os.html", 0),
        ("https://stackoverflow.com/questions/tagged/python", 0),
        ("https://en.wikipedia.org/wiki/Cybersecurity", 0),
    ]
    phishing = [
        ("http://paypa1-secure-login.xyz/verify?account=suspended", 1),
        ("https://amazon-account-alert.info/signin&redirect=billing", 1),
        ("http://192.168.1.1.malicious-domain.ru/bank/login.php", 1),
        ("https://microsoft-update-required.com/security/patch.exe", 1),
        ("http://secure-ebay-login.net/signin/confirm-identity", 1),
    ]
    combined = legit + phishing
    result = []
    for i in range(n):
        result.append(combined[i % len(combined)])
    return result


# ---------------------------------------------------------------------------
# Utility: train/val split
# ---------------------------------------------------------------------------
def train_val_split(records, val_ratio=0.1, seed=42):
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split], shuffled[split:]


if __name__ == "__main__":
    print("=== Testing all data loaders ===\n")

    sft = load_sft_dataset(max_samples=10)
    print(f"SFT sample:\n  prompt: {sft[0]['prompt'][:80]}...\n  response: {sft[0]['response'][:80]}...\n")

    dpo = load_dpo_dataset(max_samples=10)
    print(f"DPO sample:\n  prompt: {dpo[0]['prompt'][:80]}...\n  chosen: {dpo[0]['chosen'][:60]}...\n  rejected: {dpo[0]['rejected'][:60]}...\n")

    grpo = load_grpo_dataset(max_samples=10)
    print(f"GRPO sample:\n  prompt: {grpo[0]['prompt'][:80]}...\n  answer: {grpo[0]['answer']}\n")

    rm_data, ppo = load_rlhf_dataset(max_samples=10)
    print(f"RLHF RM sample:\n  prompt: {rm_data[0]['prompt'][:80]}...\n  label: {rm_data[0]['label']}\n")
    print("All loaders OK.")
