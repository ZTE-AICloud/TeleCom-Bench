# TeleCom-Bench

## 📖 Introduction

While Large Language Models (LLMs) have achieved remarkable integration in various vertical scenarios, their deployment in the telecommunications domain remains exploratory due to the lack of a standardized evaluation framework. Current telecom benchmarks primarily focus on static, foundational knowledge and isolated "atomic" skills, neglecting the equipment-specific documentation and end-to-end industrial workflows essential for real-world production systems.

**TeleCom-Bench** bridges this gap. It is a comprehensive benchmark comprising **12 evaluation sets** with **22,678 curated samples**. It evaluates LLMs across a synergistic hierarchy:
1.  **Multi-dimensional Knowledge Comprehension:** Integrates telecommunication fundamentals, 3GPP protocols, and 5G network architecture with proprietary product knowledge.
2.  **End-to-End Knowledge Application:** Formalizes six core tasks on authentic trajectories from live network agent workflows (e.g., root cause analysis, solution generation).

## 🚀 Key Highlights

*   **Hierarchical Evaluation:** Covers both theoretical comprehension and practical application.
*   **Real-World Workflows:** Based on authentic trajectories from live network agent workflows, including network optimization and fault maintenance.
*   **The "Execution Wall":** Our evaluation of 8 state-of-the-art LLMs reveals a critical capability gap. While models achieve **>90% accuracy** in linguistic interface tasks (intent recognition, entity extraction), performance collapses to **~30%** in procedural execution tasks (solution generation).
*   **Actionable Diagnostics:** Provides standardized diagnostics to pinpoint deficits, offering guidance for domain-specific alignment toward production-ready telecom agents.

## 📊 Dataset Statistics

TeleCom-Bench consists of 12 tasks categorized into two main levels.

| Level 1 | Level 2 | Task Type | Task Name | Count |
| :--- | :--- | :--- | :--- | :--- |
| **Knowledge Comprehension** | Basic Theory | Multiple-Select Questions | Basic Knowledge | 2,662 |
| | | | 5G Network | 2,564 |
| | | | 3GPP Protocols | 4,043 |
| | Product Knowledge | Subjective QA | Wireless Network | 3,725 |
| | | | Wired Network | 3,488 |
| | | | Core Network | 960 |
| **Knowledge Application** | Network Optimization & Fault Maintenance | Structured QA | Intent Recognition | 2,174 |
| | | | Entity Extraction | 365 |
| | | | Tool Invocation | 585 |
| | | | Event Verification | 146 |
| | | | Root Cause Diagnosis | 983 |
| | | | Solution Generation | 983 |
| **Total** | | | | **22,678** |

> **Note on Data Availability:** To prevent evaluation dataset leakage and ensure benchmark integrity, this repository releases **all evaluation code** but only **a subset of the benchmark datasets** as examples.


## Environment Setup

### System Requirements

Python version >= 3.8

### Dependency Installation

Project dependencies are managed via `setup.py`, with the dependency list located in `requirements/requirements.txt`.

```bash
# 1. Clone the repository
git clone <repo_url>
cd git

# 2. (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the project (automatically installs dependencies from requirements.txt)
pip install -e .
```

### Verify Installation

```bash
python -c "from opencompass import __version__; print(__version__)"
# Output: 0.2.0
```

---

## Quick Start

### Step 1: Prepare Configuration File

Copy the example configuration file and modify the model API endpoint and key:

```bash
cp examples/CoreNetwork.py my_eval.py
```

Edit `my_eval.py` and set the following key parameters:

```python
# Model name (used for identification and output directory naming)
model = "your_model_name"

# Model API endpoint
api_url = "https://your-api-endpoint/v1/chat/completions"

# API authentication (if required)
api_headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-api-key",
}

# Judge model (for LLM-as-Judge scoring)
judge_model_cfg = dict(
    base_url="https://your-judge-api-url",
    model="judge_model_name",
    api_key="your-judge-api-key",
)
```

### Step 2: Run Evaluation

```bash
python run.py my_eval.py
```

This command automatically completes the full pipeline: **inference → evaluation → result aggregation**.

### Step 3: View Results

Evaluation results are saved by default to the `eval_result/<model_name>/` directory. You can customize the output location by setting `work_dir` in the configuration.

> **Note:** The public repo includes only example data subsets. To reproduce paper results, obtain the full dataset from the maintainers (see [MAINTAINER.md](./MAINTAINER.md)).

## Experiments and Analysis

We present a systematic evaluation of representative Large Language Models on TeleCom-Bench. Our analysis quantifies the discrepancy between theoretical proficiency and operational viability in telecommunications engineering, with particular focus on the **Execution Wall** — defined operationally as the performance gap exceeding 50 percentage points between diagnostic reasoning and executable solution generation within the same fault-handling workflow.

### Main Results

| Category | Task | Qwen3-32B | Qwen3-235B | DeepSeek-V3.2 | Gemini 2.5 | Grok 4.1 | GLM-4.7 | Doubao-pro | Kimi K2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Comprehension** | Basic Theory | 61.01 | 69.04 | 67.12 | 63.88 | **70.97** | 51.82 | 70.14 | 52.75 |
| | Wired Network | 59.41 | 60.88 | 57.34 | 54.68 | **62.60** | 60.90 | 58.78 | 60.99 |
| | Wireless Network | **73.04** | 70.80 | 45.89 | 63.64 | 52.58 | 22.63 | 61.99 | 64.81 |
| | Core Network | 60.22 | 62.13 | 62.34 | 59.28 | **66.79** | 47.68 | 64.46 | 56.79 |
| **Application** | Intent Recognition | 93.13 | **94.52** | 92.85 | 92.94 | 93.85 | 92.69 | 93.43 | 93.49 |
| | Entity Extraction | 95.74 | **99.72** | **99.72** | **99.72** | **99.72** | 81.25 | **99.72** | **99.72** |
| | Event Verification | 59.00 | 72.72 | 67.35 | 71.92 | **81.85** | 12.95 | 80.48 | 52.92 |
| | Tool Invocation | 84.71 | 45.54 | **94.06** | 90.80 | 93.20 | 56.50 | 87.30 | 84.50 |
| | Root Cause Diagnosis | 60.92 | **71.49** | 63.00 | 49.28 | 48.60 | 26.63 | 61.33 | 57.85 |
| | Solution Generation | 15.02 | 4.67 | 5.61 | 22.42 | 14.58 | 9.64 | **30.72** | 8.45 |

### Experimental Setup

Eight models (Qwen3-32B/235B, DeepSeek-V3.2, Gemini 2.5, Grok 4.1, GLM-4.7, Doubao-pro, Kimi K2) were evaluated with temperature=0.7, reasoning mode enabled, and majority voting over 3 trials. Metrics: Macro-F1 (multi-label), Exact Match (structured QA), LLM-as-Judge on 5-point scale (open-ended, α=0.82).

### Analysis

**Knowledge Comprehension.** Models score 60%–70% on theory but vary widely on product knowledge — Qwen3-32B beats 235B on Wireless (73% vs 71%), while GLM-4.7 drops to 23%, showing data distribution matters more than parameter count in vertical domains.

**The Execution Wall.** Application tasks reveal a systemic gap:
- **Interface is saturated**: Intent/Entity extraction top out at >92%, but this creates a false sense of reliability.
- **MoE advantage in tooling**: DeepSeek-V3.2 hits 94% in Tool Invocation vs Qwen3-235B's 46% (r=0.87 with MoE).
- **Diagnosis-Action Paradox**: Qwen3-235B scores 71% in Root Cause Diagnosis but collapses to 5% in Solution Generation — a 66-point gap. Even the best (Doubao-pro, 31%) is far from the 95% needed for autonomous deployment.

Current LLMs function as competent *Diagnosticians* but fail as *Field Engineers*.

## How to Start Contributing

Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md), which includes:

- Environment setup and dependency installation
- Code structure overview
- Dataset format details
- Step-by-step evaluation process explanation
- **Complete tutorial for adding new evaluation tasks** (with code examples)
- How to extend datasets
- PR process and code standards

## License

- Code: [Apache License 2.0](./CODE_LICENSE.md)
- Data: [Community Data License Agreement - Permissive v2.0](./DATASET_LICENSE.md)

## Maintainers

See [MAINTAINER.md](./MAINTAINER.md) for details.