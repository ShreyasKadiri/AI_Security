"""
src/signatures/threat.py — Signatures for Pipeline 1: Threat Classification

A DSPy Signature is a TYPED CONTRACT for a language model call.
It answers: "what goes in, what comes out, and what is the task?"

Key rule: signatures are NOT prompts. DSPy compiles them into prompts.
This means you can swap models, run optimizers, and version-control your
intent separately from the prompt text that implements it.
"""

import dspy


class ThreatClassifier(dspy.Signature):
    """
    You are a SOC analyst reviewing network traffic logs.
    Classify the given network log entry into the appropriate threat category.
    Analyze protocol, ports, packet counts, byte volumes, timing, and flag patterns.
    """

    log_entry: str = dspy.InputField(
        desc=(
            "A single-line network log summarizing one flow: "
            "protocol, source IP/port, destination IP/port, packet count, "
            "byte volume, duration, and TCP flags. "
            "Example: proto=TCP src=185.220.4.2 dst=10.0.0.8 dport=22 "
            "pkts=3200 bytes=98000 dur=45s flags=S service=SSH"
        )
    )

    threat_type: str = dspy.OutputField(
        desc=(
            "Exactly one of: DDoS, PortScan, BruteForce, WebAttack, "
            "Botnet, Infiltration, Normal. "
            "Output the category name only."
        )
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0. Output the numeric value only."
    )
    key_indicators: str = dspy.OutputField(
        desc=(
            "2-3 specific indicators from the log that led to this classification. "
            "E.g. 'High packet rate (3200 pkts) to SSH port 22, sustained 45s — brute force pattern.'"
        )
    )


class ThreatClassifierWithReasoning(dspy.Signature):
    """
    You are a senior SOC analyst reviewing network traffic logs.
    Systematically analyze the log entry by examining:
    1. Protocol and port — does the destination port match the suspected service?
    2. Traffic volume — packet rate and byte volume consistent with the threat?
    3. Duration — sustained vs burst traffic?
    4. Source patterns — single IP or distributed?
    5. Flag patterns — SYN flood, ACK only, mixed?
    Then classify the threat and explain your reasoning.
    """

    log_entry: str = dspy.InputField(
        desc="Raw network flow log entry with protocol, IPs, ports, metrics."
    )

    threat_type: str = dspy.OutputField(
        desc=(
            "Exactly one of: DDoS, PortScan, BruteForce, WebAttack, "
            "Botnet, Infiltration, Normal."
        )
    )
    confidence: float = dspy.OutputField(
        desc="Classification confidence from 0.0 to 1.0."
    )
    key_indicators: str = dspy.OutputField(
        desc="The 2-3 most decisive indicators from the log entry."
    )
    mitre_tactic: str = dspy.OutputField(
        desc=(
            "Most relevant MITRE ATT&CK tactic, e.g. "
            "'TA0001 - Initial Access', 'TA0011 - Command and Control'. "
            "Output 'N/A' for Normal traffic."
        )
    )
    recommended_action: str = dspy.OutputField(
        desc=(
            "Immediate recommended action: e.g. 'Block source IP at perimeter firewall', "
            "'Isolate host and initiate IR playbook', 'Monitor and alert'. "
            "One sentence."
        )
    )
