# Cybersecurity LLM Post-Training — Hands-On Project

Fine-tune **Qwen2.5-0.5B-Instruct** using four post-training methods on
cybersecurity tasks. Runs entirely on CPU (Windows/Mac/Linux).

## Methods covered

| Notebook | Method | Task | Dataset |
|----------|--------|------|---------|
| `01_sft.ipynb` | Supervised Fine-Tuning | Threat Q&A | SecQA (HuggingFace) |
| `02_dpo.ipynb` | Direct Preference Optimization | Response quality ranking | CyberGuard |
| `03_grpo.ipynb` | Group Relative Policy Optimization | CVE severity reasoning | NIST NVD API |
| `04_rlhf.ipynb` | RLHF (Reward Model + PPO) | Phishing URL explanation | pirocheto/phishing-url |

## Setup

```bash
# 1. Clone / download this project
cd cybersec_llm

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install base deps (each notebook also self-installs)
pip install transformers>=4.40 trl>=0.8 datasets accelerate torch

# 4. Run notebooks in order
jupyter notebook notebooks/01_sft.ipynb
```

## Project structure

```
cybersec_llm/
├── data/
│   └── loader.py          # All dataset loaders with fallbacks
├── shared/
│   └── model_utils.py     # Model loading, inference, eval utilities
├── notebooks/
│   ├── 01_sft.ipynb       # SFT
│   ├── 02_dpo.ipynb       # DPO
│   ├── 03_grpo.ipynb      # GRPO
│   └── 04_rlhf.ipynb      # RLHF
├── models/                # Saved checkpoints (created at runtime)
└── README.md
```

## CPU performance expectations

| Notebook | Samples | Est. time (CPU) |
|----------|---------|-----------------|
| SFT      | 200     | 30–60 min       |
| DPO      | 150     | 45–90 min       |
| GRPO     | 100     | 60–120 min      |
| RLHF     | 120     | 90–180 min      |

Reduce `MAX_SAMPLES` in each notebook to train faster for experimentation.

## Key learning goals

- **SFT**: understand cross-entropy loss masking on response tokens only
- **DPO**: see how log-ratio margins shift across training; check win rate on val set
- **GRPO**: inspect the group-relative advantage normalization; watch format compliance rise
- **RLHF**: verify RM correctly orders good > bad; watch KL divergence during PPO

## Notes

- All datasets have built-in synthetic fallbacks — notebooks work with no internet
- First run downloads Qwen2.5-0.5B (~1GB); subsequent runs use HuggingFace cache
- Run notebooks in order: SFT → DPO → GRPO → RLHF (each builds on the SFT checkpoint)
