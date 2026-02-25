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

### 📊 Dataset Statistics

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
