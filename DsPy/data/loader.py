"""
data/loader.py — Unified dataset loader for the DSPy SOC Assistant

Datasets:
  1. CICIDS 2017    — network intrusion (DDoS, PortScan, BruteForce, Normal)
                      Source: Canadian Institute for Cybersecurity
                      License: CC BY 4.0
                      Used for: Threat Classifier pipeline

  2. NVD CVE API    — real CVEs with CVSS scores from NIST
                      Source: https://nvd.nist.gov (public domain)
                      Used for: CVE Analyst pipeline

  3. UNSW-NB15      — network anomaly dataset from UNSW Canberra
                      Source: HuggingFace rdpahalavan/cyber-security-intrusion-detection
                      License: Open research use
                      Used for: supplementary threat classification data

  4. Synthetic SOC alerts — generated deterministically from templates
                            No external dependency
                            Used for: Alert Triage pipeline

All loaders return lists of dspy.Example objects with .with_inputs() set.
Every loader has a synthetic fallback — works fully offline.
"""

import sys
import json
import time
import random
import hashlib
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dspy
from configs.settings import (
    PROCESSED_DIR, RAW_DIR,
    CICIDS_URL, CICIDS_MAX_SAMPLES, CICIDS_VAL_RATIO,
    NVD_API_BASE, NVD_MAX_CVES, NVD_RESULTS_PER_PAGE, NVD_RATE_LIMIT_SLEEP,
    UNSW_HF_ID, UNSW_MAX_SAMPLES,
    ALERT_N_TRAIN, ALERT_N_TEST, ALERT_SEED,
    THREAT_CATEGORIES, SEVERITY_LEVELS,
)


# =============================================================================
# PIPELINE 1: Threat Classification — CICIDS 2017
# =============================================================================

def load_threat_dataset(
    max_samples_per_class: int = CICIDS_MAX_SAMPLES,
    val_ratio: float = CICIDS_VAL_RATIO,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Load CICIDS 2017 network intrusion data as DSPy Examples.

    Each example: log_entry (str) → threat_type (str)
    log_entry is formatted as a compact one-line network log summary.

    Returns (train_examples, test_examples).
    """
    print("Loading threat classification dataset (CICIDS 2017)...")

    # Try HuggingFace mirror first, then raw URL, then synthetic
    examples = (
        _load_cicids_huggingface(max_samples_per_class)
        or _load_cicids_url(max_samples_per_class)
        or _synthetic_threat_data(max_samples_per_class * len(THREAT_CATEGORIES))
    )

    random.seed(seed)
    random.shuffle(examples)
    split = int(len(examples) * (1 - val_ratio))
    train, test = examples[:split], examples[split:]

    _print_class_distribution("Threat", examples)
    print(f"  Train: {len(train)} | Test: {len(test)}\n")
    return train, test


def _load_cicids_huggingface(max_per_class: int) -> Optional[list[dspy.Example]]:
    try:
        from datasets import load_dataset
        print("  Trying HuggingFace: rdpahalavan/cyber-security-intrusion-detection...")
        ds = load_dataset(UNSW_HF_ID, split="train", trust_remote_code=True)

        label_col = next((c for c in ds.column_names if "label" in c.lower() or "attack" in c.lower()), None)
        if not label_col:
            return None

        examples = []
        seen = {}
        for row in ds:
            label_raw = str(row.get(label_col, "")).strip()
            label     = _normalize_threat_label(label_raw)
            if label not in THREAT_CATEGORIES:
                continue
            seen[label] = seen.get(label, 0)
            if seen[label] >= max_per_class:
                continue
            log_entry = _format_network_log(row)
            examples.append(
                dspy.Example(log_entry=log_entry, threat_type=label).with_inputs("log_entry")
            )
            seen[label] += 1

        if len(examples) >= 50:
            print(f"  Loaded {len(examples)} examples from HuggingFace.")
            return examples
    except Exception as e:
        print(f"  HuggingFace unavailable: {e}")
    return None


def _load_cicids_url(max_per_class: int) -> Optional[list[dspy.Example]]:
    try:
        import urllib.request, csv, io
        print(f"  Trying CICIDS 2017 URL...")
        with urllib.request.urlopen(CICIDS_URL, timeout=15) as r:
            content = r.read().decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(content))

        examples, seen = [], {}
        for row in reader:
            label_raw = row.get(" Label", row.get("Label", "")).strip()
            label     = _normalize_threat_label(label_raw)
            if label not in THREAT_CATEGORIES:
                continue
            seen[label] = seen.get(label, 0)
            if seen[label] >= max_per_class:
                continue
            log_entry = _format_cicids_row(row)
            examples.append(
                dspy.Example(log_entry=log_entry, threat_type=label).with_inputs("log_entry")
            )
            seen[label] += 1

        if len(examples) >= 20:
            print(f"  Loaded {len(examples)} examples from URL.")
            return examples
    except Exception as e:
        print(f"  URL unavailable: {e}")
    return None


def _synthetic_threat_data(n: int = 700) -> list[dspy.Example]:
    """Deterministic synthetic network log examples per threat type."""
    print("  Using synthetic threat dataset (no internet required).")
    random.seed(42)

    templates = {
        "DDoS": [
            "proto=UDP src=203.0.113.{r} dst=10.0.0.5 dport=80 pkts={p} bytes={b} dur=0.{d}s flags=- pkt_rate={pr}/s",
            "proto=TCP src=198.51.100.{r} dst=10.0.0.1 dport=443 pkts={p} bytes={b} dur=0.0{d}s SYN_FLOOD=True",
            "proto=ICMP src=192.0.2.{r} dst=10.0.0.{r2} pkts={p} bytes={b} dur=0.{d}s type=8 flood=True",
        ],
        "PortScan": [
            "proto=TCP src=45.33.32.{r} dst=10.0.0.{r2} dport={dp} pkts=1 bytes=40 dur=0.00{d}s flags=S",
            "proto=TCP src=104.236.{r}.{r2} dst=172.16.0.5 dport_range={dp}-{dp2} pkts={p} bytes=60 SYN_ONLY=True",
            "proto=TCP src=192.168.{r}.1 dst=10.10.0.{r2} scan_type=SYN dports={p} scanned dur={d}s",
        ],
        "BruteForce": [
            "proto=TCP src=185.220.{r}.{r2} dst=10.0.0.8 dport=22 pkts={p} bytes={b} dur={d}s service=SSH attempts={p}",
            "proto=TCP src=91.108.{r}.4 dst=10.0.0.3 dport=3389 pkts={p} bytes={b} dur={d}s service=RDP login_fails={p}",
            "proto=TCP src=178.62.{r}.{r2} dst=10.0.0.10 dport=21 pkts={p} bytes={b} dur={d}s service=FTP brute=True",
        ],
        "WebAttack": [
            "proto=TCP src=5.188.{r}.{r2} dst=10.0.0.7 dport=80 pkts={p} bytes={b} dur={d}s http_method=POST uri=/login payload=SQLi",
            "proto=TCP src=77.88.{r}.{r2} dst=10.0.0.7 dport=443 pkts=3 bytes=512 dur=0.{d}s http_method=GET uri=/<script> XSS=True",
            "proto=TCP src=95.142.{r}.{r2} dst=10.0.0.7 dport=80 pkts={p} bytes={b} dur={d}s uri=/../../../etc/passwd traversal=True",
        ],
        "Botnet": [
            "proto=TCP src=10.0.0.{r} dst=185.220.{r2}.4 dport=6667 pkts={p} bytes={b} dur={d}s c2=True irc_traffic=True",
            "proto=UDP src=10.0.0.{r} dst=8.8.8.8 dport=53 pkts={p} bytes={b} dur={d}s dns_tunnel=True query_entropy=7.{r}",
            "proto=TCP src=10.0.0.{r} dst=198.51.{r2}.1 dport=4444 pkts={p} bytes={b} dur={d}s beacon_interval=300s C2=True",
        ],
        "Infiltration": [
            "proto=TCP src=10.0.0.{r} dst=10.0.0.{r2} dport=445 pkts={p} bytes={b} dur={d}s lateral_movement=True smb_enum=True",
            "proto=TCP src=10.0.0.{r} dst=192.168.{r2}.5 dport=5985 pkts={p} bytes={b} dur={d}s winrm=True exfil_size={b}",
            "proto=TCP src=10.0.0.{r} dst=10.0.0.{r2} dport=135 pkts={p} bytes={b} dur={d}s rpc_enum=True priv_esc=True",
        ],
        "Normal": [
            "proto=TCP src=10.0.0.{r} dst=216.58.{r2}.14 dport=443 pkts=18 bytes={b} dur={d}s flags=ACK tls=True",
            "proto=UDP src=10.0.0.{r} dst=8.8.8.8 dport=53 pkts=2 bytes=120 dur=0.0{d}s dns_query=True response_ok=True",
            "proto=TCP src=10.0.0.{r} dst=151.101.{r2}.57 dport=80 pkts={p} bytes={b} dur={d}s http_200=True content=html",
        ],
    }

    examples = []
    per_class = n // len(THREAT_CATEGORIES)
    for label, tmpls in templates.items():
        for i in range(per_class):
            r   = random.randint(1, 254)
            r2  = random.randint(1, 254)
            p   = random.randint(10, 5000)
            b   = random.randint(500, 500000)
            d   = random.randint(1, 99)
            dp  = random.randint(1, 65535)
            dp2 = dp + random.randint(10, 100)
            pr  = random.randint(1000, 100000)
            log = tmpls[i % len(tmpls)].format(r=r, r2=r2, p=p, b=b, d=d, dp=dp, dp2=dp2, pr=pr)
            examples.append(
                dspy.Example(log_entry=log, threat_type=label).with_inputs("log_entry")
            )
    return examples


def _format_network_log(row: dict) -> str:
    """Format a HuggingFace row into a compact log string."""
    fields = {k.strip(): v for k, v in row.items()}
    parts = []
    for src in ["src_ip", "Source IP", "Src IP", "src"]:
        if src in fields and fields[src]:
            parts.append(f"src={fields[src]}")
            break
    for dst in ["dst_ip", "Destination IP", "Dst IP", "dst"]:
        if dst in fields and fields[dst]:
            parts.append(f"dst={fields[dst]}")
            break
    for p in ["Protocol", "proto", "protocol"]:
        if p in fields:
            parts.append(f"proto={fields[p]}")
            break
    for pk in ["Total Fwd Packets", "pkts", "Fwd Packets"]:
        if pk in fields:
            parts.append(f"pkts={fields[pk]}")
            break
    for bk in ["Total Length of Fwd Packets", "bytes", "Flow Bytes/s"]:
        if bk in fields:
            parts.append(f"bytes={fields[bk]}")
            break
    return " ".join(parts) if parts else str(row)[:200]


def _format_cicids_row(row: dict) -> str:
    """Format CICIDS CSV row into a compact log string."""
    def g(keys):
        for k in keys:
            v = row.get(k, row.get(k.strip(), ""))
            if v and str(v).strip() not in ("", "nan"):
                return str(v).strip()
        return "?"
    return (
        f"proto={g([' Protocol','Protocol'])} "
        f"src={g([' Source IP','Source IP'])} "
        f"sport={g([' Source Port','Source Port'])} "
        f"dst={g([' Destination IP','Destination IP'])} "
        f"dport={g([' Destination Port','Destination Port'])} "
        f"pkts={g([' Total Fwd Packets','Total Fwd Packets'])} "
        f"bytes={g([' Total Length of Fwd Packets'])} "
        f"dur={g([' Flow Duration','Flow Duration'])}us "
        f"flags={g([' Fwd PSH Flags','Fwd PSH Flags'])}"
    )


def _normalize_threat_label(raw: str) -> str:
    raw = raw.upper().replace("-", "").replace("_", "").replace(" ", "")
    if "DDOS" in raw or "DOS" in raw:           return "DDoS"
    if "PORTSCAN" in raw or "SCAN" in raw:      return "PortScan"
    if "BRUTEFORCE" in raw or "BRUTE" in raw:   return "BruteForce"
    if "WEB" in raw or "SQL" in raw or "XSS" in raw: return "WebAttack"
    if "BOT" in raw:                            return "Botnet"
    if "INFILTRAT" in raw or "LATERAL" in raw:  return "Infiltration"
    if "BENIGN" in raw or "NORMAL" in raw:      return "Normal"
    return raw


# =============================================================================
# PIPELINE 2: CVE Analysis — NIST NVD API
# =============================================================================

def load_cve_dataset(
    max_cves: int = NVD_MAX_CVES,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Load real CVEs from NIST NVD.

    Each example: cve_description + cvss_score → severity + impact_summary
    Returns (train_examples, test_examples).
    """
    print("Loading CVE dataset (NIST NVD API)...")
    examples = _fetch_nvd_cves(max_cves) or _synthetic_cve_data(max_cves)

    random.seed(seed)
    random.shuffle(examples)
    split = int(len(examples) * 0.8)
    train, test = examples[:split], examples[split:]
    _print_class_distribution("Severity", examples, field="severity")
    print(f"  Train: {len(train)} | Test: {len(test)}\n")
    return train, test


def _fetch_nvd_cves(n: int) -> Optional[list[dspy.Example]]:
    try:
        import urllib.request
        examples, start = [], 0
        print(f"  Fetching CVEs from NVD API (up to {n})...")
        while len(examples) < n:
            batch = min(NVD_RESULTS_PER_PAGE, n - len(examples))
            url   = f"{NVD_API_BASE}?resultsPerPage={batch}&startIndex={start}"
            req   = urllib.request.urlopen(url, timeout=15)
            data  = json.loads(req.read())
            vulns = data.get("vulnerabilities", [])
            if not vulns:
                break
            for item in vulns:
                cve = item.get("cve", {})
                desc_list = cve.get("descriptions", [])
                desc = next((d["value"] for d in desc_list if d["lang"] == "en"), None)
                if not desc:
                    continue
                metrics  = cve.get("metrics", {})
                severity, score = None, None
                for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                    entries = metrics.get(key, [])
                    if entries:
                        data_block = entries[0].get("cvssData", entries[0])
                        severity = data_block.get("baseSeverity") or entries[0].get("baseSeverity")
                        score    = data_block.get("baseScore")
                        break
                if not severity or severity.upper() not in ["CRITICAL","HIGH","MEDIUM","LOW"]:
                    continue
                cve_id = cve.get("id", "CVE-UNKNOWN")
                examples.append(dspy.Example(
                    cve_id=cve_id,
                    cve_description=desc[:600],
                    cvss_score=str(score or "N/A"),
                    severity=severity.upper(),
                ).with_inputs("cve_id", "cve_description", "cvss_score"))
                if len(examples) >= n:
                    break
            start += batch
            time.sleep(NVD_RATE_LIMIT_SLEEP)
        print(f"  Loaded {len(examples)} CVEs from NVD.")
        return examples if examples else None
    except Exception as e:
        print(f"  NVD API unavailable: {e}")
        return None


def _synthetic_cve_data(n: int = 120) -> list[dspy.Example]:
    print("  Using synthetic CVE dataset.")
    templates = [
        {"cve_id":"CVE-2021-44228","desc":"A remote code execution vulnerability in Apache Log4j 2.x via JNDI lookup allows unauthenticated remote attackers to execute arbitrary code via crafted log messages.","score":"10.0","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-26084","desc":"Confluence Server and Data Center have an OGNL injection vulnerability that allows unauthenticated users to execute arbitrary code on a Confluence instance.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2022-30190","desc":"A remote code execution vulnerability (Follina) exists when MSDT is called using the URL protocol from a calling application such as Microsoft Word.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-23397","desc":"Microsoft Outlook elevation of privilege vulnerability allows a remote attacker to steal NTLM hashes by sending a specially crafted email without user interaction.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-21985","desc":"The vSphere Client has a remote code execution vulnerability due to lack of input validation in the Virtual SAN Health Check plugin.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2022-22965","desc":"Spring Framework RCE via Data Binding on JDK 9+ (Spring4Shell) allows remote code execution through a ClassLoader parameter injection.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-44487","desc":"HTTP/2 Rapid Reset attack causes denial of service by exploiting the stream cancellation feature to overwhelm servers with requests.","score":"7.5","severity":"HIGH"},
        {"cve_id":"CVE-2022-1388","desc":"F5 BIG-IP iControl REST authentication bypass allows unauthenticated attackers to execute arbitrary system commands via the management interface.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-34527","desc":"Windows Print Spooler remote code execution vulnerability (PrintNightmare) allows privilege escalation and remote code execution by authenticated users.","score":"8.8","severity":"HIGH"},
        {"cve_id":"CVE-2022-26134","desc":"Atlassian Confluence Server OGNL injection pre-authentication remote code execution via a crafted HTTP request to the server.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2022-3786","desc":"X.509 Email Address Buffer Overflow in OpenSSL 3.0.x allows a malicious email address in a certificate to cause denial of service via buffer overflow.","score":"7.5","severity":"HIGH"},
        {"cve_id":"CVE-2021-44142","desc":"Samba vfs_fruit module out-of-bounds heap read/write allows unauthenticated remote code execution as root when the VFS module is configured.","score":"9.9","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-29357","desc":"Microsoft SharePoint Server elevation of privilege vulnerability allows an attacker with network access to bypass authentication and gain administrator access.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2022-47966","desc":"ManageEngine multiple products pre-authentication remote code execution via SAML in products using an outdated version of Apache Santuario.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-22986","desc":"F5 BIG-IP/BIG-IQ iControl REST unauthenticated RCE allows attackers with network access to the management port to execute arbitrary system commands.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-21716","desc":"Microsoft Word remote code execution vulnerability via a crafted RTF document allows arbitrary code execution in the context of the current user.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-3156","desc":"Sudo heap-based buffer overflow (Baron Samedit) allows local users to gain root privileges by exploiting an off-by-one error in argument parsing.","score":"7.8","severity":"HIGH"},
        {"cve_id":"CVE-2022-22954","desc":"VMware Workspace ONE Access server-side template injection allows unauthenticated RCE via the freemarker template engine in the UI.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2020-1472","desc":"Netlogon elevation of privilege (Zerologon) allows unauthenticated attacker to establish a Netlogon session to a domain controller and gain domain admin.","score":"10.0","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-34362","desc":"MOVEit Transfer SQL injection allows unauthenticated remote access to the database, enabling data exfiltration from MOVEit Transfer instances.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2022-30525","desc":"Zyxel firewall OS command injection via the HTTP interface allows unauthenticated attackers to inject commands via the setWanPortSt function.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-20198","desc":"Cisco IOS XE web UI privilege escalation allows unauthenticated remote attackers to create an account with privilege level 15.","score":"10.0","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-27065","desc":"Microsoft Exchange Server remote code execution via post-authentication write-what-where, part of the ProxyLogon chain.","score":"7.8","severity":"HIGH"},
        {"cve_id":"CVE-2023-36884","desc":"Windows Search remote code execution allows specially crafted Microsoft Office documents to execute arbitrary code via search protocol abuse.","score":"8.8","severity":"HIGH"},
        {"cve_id":"CVE-2022-40684","desc":"Fortinet FortiOS, FortiProxy authentication bypass on administrative interface allows an unauthenticated attacker to perform operations on the admin interface.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-26855","desc":"Microsoft Exchange Server SSRF vulnerability (ProxyLogon) allows unauthenticated attackers to authenticate as any Exchange user.","score":"9.1","severity":"CRITICAL"},
        {"cve_id":"CVE-2022-1292","desc":"OpenSSL c_rehash script command injection allows local attackers to inject OS commands via a crafted filename in the certificate hash directories.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-27997","desc":"Fortinet FortiGate SSL-VPN heap buffer overflow allows unauthenticated RCE via a crafted request to the VPN service, potentially pre-authentication.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-31207","desc":"Microsoft Exchange Server security feature bypass via Outlook Web App allows an authenticated attacker to upload an arbitrary file.","score":"6.6","severity":"MEDIUM"},
        {"cve_id":"CVE-2022-37434","desc":"zlib heap buffer over-read or buffer overflow via a large gzip header extra field in inflate.c, potentially causing application crash.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-42793","desc":"JetBrains TeamCity authentication bypass allows unauthenticated remote code execution on the CI/CD server via the REST API.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-4966","desc":"Citrix Bleed — NetScaler ADC and Gateway sensitive information disclosure allows session token theft without authentication.","score":"9.4","severity":"CRITICAL"},
        {"cve_id":"CVE-2020-14882","desc":"Oracle WebLogic Server remote code execution via HTTP allows unauthenticated attackers to take over the server without authentication.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-20038","desc":"SonicWall SMA100 Stack-based buffer overflow in the SMA100 SSL-VPN allows unauthenticated RCE via a crafted HTTP request to the management interface.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2022-29464","desc":"WSO2 unrestricted file upload via the management console allows unauthenticated remote code execution by uploading a malicious JSP file.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-22205","desc":"GitLab CE/EE remote code execution via image upload allows unauthenticated attackers to execute arbitrary code via ExifTool processing of uploaded images.","score":"10.0","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-38408","desc":"OpenSSH ssh-agent remote code execution when ssh-agent is forwarded to a compromised host, allowing the host to load malicious PKCS11 libraries.","score":"9.8","severity":"CRITICAL"},
        {"cve_id":"CVE-2022-26143","desc":"TP240PhoneHome amplification DDoS vulnerability allows attackers to generate 4,293,967,295:1 amplification in reflection-based DDoS attacks.","score":"9.1","severity":"CRITICAL"},
        {"cve_id":"CVE-2021-22893","desc":"Pulse Connect Secure authentication bypass allows unauthenticated attackers to perform file read and file write via the admin web interface.","score":"10.0","severity":"CRITICAL"},
        {"cve_id":"CVE-2023-35078","desc":"Ivanti EPMM (MobileIron) authentication bypass allows unauthenticated access to specific API paths, enabling disclosure of PII and limited device management.","score":"10.0","severity":"CRITICAL"},
    ]
    examples = []
    for i in range(n):
        t = templates[i % len(templates)]
        examples.append(dspy.Example(
            cve_id=t["cve_id"],
            cve_description=t["desc"],
            cvss_score=t["score"],
            severity=t["severity"],
        ).with_inputs("cve_id", "cve_description", "cvss_score"))
    return examples


# =============================================================================
# PIPELINE 3: Alert Triage — Synthetic SOC Alerts
# =============================================================================

def load_alert_dataset(
    n_train: int = ALERT_N_TRAIN,
    n_test:  int = ALERT_N_TEST,
    seed:    int = ALERT_SEED,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Generate synthetic SOC alert examples for triage training/evaluation.

    Each example: alert_queue (str) → prioritized_alerts (str) + analyst_notes (str)
    alert_queue is JSON of 3-5 alerts with metadata.
    Returns (train_examples, test_examples).
    """
    print("Generating synthetic SOC alert triage dataset...")
    random.seed(seed)
    train = [_generate_alert_example(seed + i) for i in range(n_train)]
    test  = [_generate_alert_example(seed + n_train + i) for i in range(n_test)]
    print(f"  Generated {n_train} train + {n_test} test alert triage examples.\n")
    return train, test


def _generate_alert_example(seed: int) -> dspy.Example:
    """Generate one alert triage example: a queue of alerts with expected priority."""
    random.seed(seed)
    n_alerts = random.randint(3, 6)
    alerts   = [_random_alert(seed * 10 + i) for i in range(n_alerts)]

    # Sort by expected priority for the ground truth
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFORMATIONAL": 4}
    sorted_alerts  = sorted(alerts, key=lambda a: priority_order.get(a["severity"], 5))
    top_priority   = sorted_alerts[0]["severity"]

    alert_queue = json.dumps(alerts, indent=2)
    expected_order = json.dumps([a["id"] for a in sorted_alerts])
    analyst_notes  = f"Highest priority: {sorted_alerts[0]['title']} ({top_priority}). Address first."

    return dspy.Example(
        alert_queue=alert_queue,
        expected_priority_order=expected_order,
        top_severity=top_priority,
        analyst_notes=analyst_notes,
    ).with_inputs("alert_queue")


ALERT_TEMPLATES = [
    {"title":"Ransomware process detected","severity":"CRITICAL","desc":"EDR flagged process matching WannaCry signature on WORKSTATION-{n}.","source":"EDR"},
    {"title":"Domain admin account compromise","severity":"CRITICAL","desc":"Impossible travel: admin account logged in from {country1} and {country2} within 4 minutes.","source":"IAM"},
    {"title":"Data exfiltration via DNS","severity":"HIGH","desc":"Host {ip} sent {mb}MB of DNS queries to external resolver in 10 minutes. DNS tunneling suspected.","source":"NDR"},
    {"title":"Lateral movement detected","severity":"HIGH","desc":"SMB authentication spray from {ip} to 47 internal hosts in 90 seconds.","source":"SIEM"},
    {"title":"Privilege escalation","severity":"HIGH","desc":"LSASS memory dump detected on {host}. Credential theft likely.","source":"EDR"},
    {"title":"SQL injection attempt","severity":"MEDIUM","desc":"WAF blocked 143 SQLi payloads targeting /api/users endpoint from {ip}.","source":"WAF"},
    {"title":"Failed MFA for admin","severity":"MEDIUM","desc":"Admin account {user}@corp.com failed MFA 12 times in 5 minutes.","source":"IAM"},
    {"title":"Outbound connection to known C2","severity":"HIGH","desc":"Host {ip} made connection to {c2ip}, listed in ThreatFox as Cobalt Strike C2.","source":"TI"},
    {"title":"Vulnerability scanner detected","severity":"LOW","desc":"Internal IP {ip} running nmap scan against 10.0.0.0/24.","source":"IDS"},
    {"title":"SSL certificate expiring","severity":"INFORMATIONAL","desc":"Certificate for {domain} expires in {days} days. Renew before expiration.","source":"Monitoring"},
    {"title":"Phishing email delivered","severity":"HIGH","desc":"{n} employees received phishing email with malicious attachment. {n2} opened it.","source":"Email"},
    {"title":"RDP exposed to internet","severity":"MEDIUM","desc":"Port scan shows RDP (3389) open on {ip} directly accessible from internet.","source":"EASM"},
    {"title":"Malicious macro execution","severity":"HIGH","desc":"Word document opened by {user}@corp.com spawned PowerShell process. Macro execution.","source":"EDR"},
    {"title":"Suspicious PowerShell","severity":"MEDIUM","desc":"Encoded PowerShell execution detected on {host}: base64 payload decoded to download cradle.","source":"EDR"},
    {"title":"Unpatched critical CVE","severity":"MEDIUM","desc":"{host} running {software} with {cve} unpatched. Public exploit available.","source":"Vuln"},
]

def _random_alert(seed: int) -> dict:
    random.seed(seed)
    tmpl = random.choice(ALERT_TEMPLATES)
    alert_id = f"ALT-{seed:05d}"
    desc = tmpl["desc"].format(
        n=random.randint(1,99), n2=random.randint(1,10),
        ip=f"10.0.{random.randint(0,255)}.{random.randint(1,254)}",
        c2ip=f"185.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}",
        host=f"WIN-{random.randint(1000,9999)}",
        user=f"user{random.randint(1,500)}",
        domain=f"app{random.randint(1,10)}.corp.com",
        days=random.randint(1,30),
        mb=random.randint(50,500),
        software=random.choice(["Apache 2.4.49","OpenSSL 3.0.0","Log4j 2.14"]),
        cve=f"CVE-2023-{random.randint(10000,50000)}",
        country1=random.choice(["India","Brazil","UK"]),
        country2=random.choice(["Russia","China","Nigeria"]),
    )
    return {
        "id":       alert_id,
        "title":    tmpl["title"],
        "severity": tmpl["severity"],
        "source":   tmpl["source"],
        "desc":     desc,
        "timestamp": f"2024-01-15T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00Z",
    }


# =============================================================================
# Utility helpers
# =============================================================================

def _print_class_distribution(name: str, examples: list, field: str = "threat_type"):
    from collections import Counter
    dist = Counter(getattr(e, field, "?") for e in examples)
    print(f"  {name} distribution: {dict(dist)}")


def train_val_split(
    examples: list,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split], shuffled[split:]


if __name__ == "__main__":
    print("=" * 60)
    print("Testing all dataset loaders")
    print("=" * 60)

    tr, te = load_threat_dataset(max_samples_per_class=20)
    print(f"Sample: {tr[0].log_entry[:80]} → {tr[0].threat_type}\n")

    cr, ce = load_cve_dataset(max_cves=20)
    print(f"Sample: {cr[0].cve_id} ({cr[0].cvss_score}) → {cr[0].severity}\n")

    ar, ae = load_alert_dataset(n_train=5, n_test=2)
    print(f"Sample alert queue (first 100 chars): {ar[0].alert_queue[:100]}...\n")

    print("All loaders OK.")
