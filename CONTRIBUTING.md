# Contributing to TeleCom-Bench

Thank you for your interest in TeleCom-Bench! We welcome all forms of contributions, including but not limited to: adding new evaluation tasks, improving evaluation code, supplementing datasets, fixing bugs, and enhancing documentation. This document guides you through the contribution workflow based on the current code layout.

For a quick-start guide to running an existing evaluation, see [`code/docs/get_started.md`](./code/docs/get_started.md).

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-org/TeleCom-Bench.git
cd TeleCom-Bench
```

This project is developed based on the [OpenCompass](https://github.com/open-compass/opencompass) evaluation framework. Dependencies are managed via `code/setup.py` and listed in `code/requirements/requirements.txt`.

```bash
# (Recommended) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install project + dependencies in editable mode
pip install -e code/

# Verify
python -c "from opencompass import __version__; print(__version__)"
# Output: 0.2.0
```

Core dependencies include: `datasets`, `numpy`, `openai`, `transformers`, `httpx`, `jieba`, `rouge_score`, `sacrebleu`, `scipy`, `mmengine`, `nltk==3.8`, etc. (full list in `code/requirements/requirements.txt`).

### 2. Run an Example Evaluation

Two ready-to-run templates are provided under `code/examples/`:

```bash
cd code
python run.py examples/CoreNetwork.py        # subjective QA on core-network products
python run.py examples/BasicKnowledge.py     # multi-choice basic theory
```

You can also pass dataset/model overrides on the command line (see `run.py --help`).

### 3. Code Style

This project uses Flake8 for code style checking. The configuration is in the root `.flake8` file (line width 120):

```bash
flake8 --config .flake8 code/
```

---

## Code Structure Overview

```
TeleCom-Bench-master/
├── code/                                # evaluation framework (opencompass fork)
│   ├── setup.py                         # package definition (entry: `opencompass`)
│   ├── requirements/
│   │   └── requirements.txt             # pinned dependency list
│   ├── run.py                           # CLI entry: infer → eval → summarize
│   ├── opencompass/                     # core library (installed as `opencompass`)
│   │   ├── datasets/                    # dataset loaders + custom evaluators
│   │   ├── openicl/                     # inference / evaluation engine
│   │   │   ├── icl_inferencer/          # GenInferencer, etc.
│   │   │   ├── icl_evaluator/           # BaseEvaluator, AccEvaluator, ...
│   │   │   ├── icl_retriever/           # ZeroRetriever, FixKRetriever
│   │   │   ├── icl_prompt_template.py
│   │   │   └── icl_dataset_reader.py
│   │   ├── models/                      # API & local model wrappers
│   │   │   ├── reasoning_api.py         # ReasoningAPI (Qwen3, etc.)
│   │   │   ├── qwen3_api.py / qwen35_api.py
│   │   │   ├── non_reasoning_api.py
│   │   │   ├── general_api.py
│   │   │   └── huggingface.py
│   │   ├── judge_models/                # LLM-as-Judge models
│   │   │   └── judge_llama.py
│   │   ├── tasks/                       # OpenICLInferTask / OpenICLEvalTask
│   │   ├── partitioners/                # NaivePartitioner / SizePartitioner
│   │   ├── runners/                     # LocalRunner / SlurmRunner / DLCRunner
│   │   ├── summarizers/                 # DefaultSummarizer
│   │   ├── utils/                       # prompt helpers, postprocessors, etc.
│   │   └── registry.py                  # mmengine-based registries
│   ├── configs/                         # reusable dataset/model configs
│   │   └── datasets/                    # one sub-folder per task
│   ├── datasets/                        # raw JSON evaluation data (subsets only)
│   │   ├── Knowledge_Comprehension/
│   │   │   ├── Basic Theory/            # Basic_Knowledge / 5G_Network / 3GPP_Protocols
│   │   │   └── Product Knowledge/       # Core_Network / Wireless_Network / Wired_Network
│   │   └── Knowledge_Application/
│   │       ├── Intent_Recognition/
│   │       ├── Entity_Extraction/
│   │       ├── Event_Verification/
│   │       ├── Root_Cause_Diagnosis/
│   │       ├── Solution_Generation/
│   │       └── Tool_Invocation/
│   ├── docs/
│   │   └── get_started.md               # quick-start (Chinese)
│   └── examples/                        # runnable config templates
│       ├── BasicKnowledge.py
│       └── CoreNetwork.py
├── datasets/                            # top-level mirror of `code/datasets/`
├── .flake8                              # max-line-length = 120
├── README.md / CONTRIBUTING.md          # you are here
├── CODE_LICENSE.md / DATASET_LICENSE.md
├── CODE_OF_CONDUCT.md
└── MAINTAINER.md
```

### Core Directories

| Path | Purpose |
|------|---------|
| `code/opencompass/datasets/` | All dataset loaders (`@LOAD_DATASET`) and custom evaluators (`@ICL_EVALUATORS`). One file per task. |
| `code/opencompass/judge_models/` | LLM-as-Judge model wrappers (used for subjective scoring). |
| `code/opencompass/models/` | API & local model wrappers (ReasoningAPI, Qwen3API, GeneralApi, HuggingFace, ...). |
| `code/opencompass/openicl/` | Inference & evaluation engine: inferencers, retrievers, prompt templates, evaluators. |
| `code/configs/datasets/` | Reusable per-task configs that wire together dataset loader, prompt, inferencer, retriever, and evaluator. |
| `code/examples/` | Runnable end-to-end templates — copy and edit to start a new run. |
| `code/datasets/` | Raw JSON samples (subsets only — see "Obtaining the Complete Dataset" below). |
| `code/run.py` | CLI entry. Runs the full pipeline: **infer → eval → summarize**. |

---

## Dataset Format

All evaluation sets live under `code/datasets/`, organized by Knowledge Comprehension vs Knowledge Application.

### Knowledge Comprehension

**Single-/multi-choice (Basic Knowledge / 5G / 3GPP / Fault Maintenance / Network Optimization / Wired Network)**

Each file is a JSON object with a `questions` list. A sample from `code/datasets/Knowledge_Comprehension/Basic Theory/Basic_Knowledge/basic_knowledge.json`:

```json
{
  "id": 0,
  "question": "LTE系统容量评估指标，不包括（　）。",
  "A": "激活用户数",
  "B": "非激活用户数",
  "C": "非IP用户数",
  "D": "最大并发用户数",
  "answer": "C",
  "difficulty": "medium",
  "tag1": "传输与接入（无线）"
}
```

`question` + option fields are the model input; `answer` is the reference. `tag1` is used by the loader to slice subsets.

**Subjective QA (Core Network)**

```json
{
  "题目": "Ga接口主要连接哪些网元？",
  "答案": "SMF网元和AMF网元"
}
```

The loader maps `题目` → `question` and `答案` → `answer`.

### Knowledge Application

Each file is a JSON list. Two common shapes:

**Intent Recognition / Tool Invocation / Event Verification**

```json
{
  "id": "q_0000",
  "summary": "请改善黄家庄村的高负荷问题",
  "input": "## 背景\n...## 输入：请改善黄家庄村的高负荷问题\n## 输出：",
  "output": "DONE",
  "type": "网优专家_用户输入_意图分类"
}
```

The loader filters by `type`. For entity-extraction tasks the entire object (including the gold dict) is the reference; for classifier tasks the model is expected to emit a label string or a JSON object.

---

## Evaluation Pipeline

A run executes three steps driven by `code/run.py`:

```
Load Config → Model Inference (OpenICLInferTask) → Evaluation (OpenICLEvalTask) → Summarizer
```

### Step 1: Compose a Config

`code/examples/CoreNetwork.py` shows the canonical pattern (using `mmengine.read_base`):

```python
from mmengine import read_base
from opencompass.models import ReasoningAPI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from ..configs.datasets.core_network.core_network_gen import core_network_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets")), [])

# LLM-as-Judge config (used by CoreNetworkEvaluator etc.)
judge_model_cfg = dict(
    base_url="https://your-judge-api-url",
    model="judge_model_name",
    api_key="your-judge-api-key",
)
for _ds in datasets:
    if _ds.get("eval_cfg", {}).get("evaluator"):
        _ds["eval_cfg"]["evaluator"]["judge_model"] = judge_model_cfg

models = [
    dict(abbr="ReasoningAPI_your_model",
         type=ReasoningAPI,
         path="ReasoningAPI_your_model",
         api_url="https://your-api-endpoint/v1/chat/completions",
         api_headers={"Content-Type": "application/json", "Authorization": "Bearer ..."},
         api_data=dict(model="your_model_name", max_tokens=8192, temperature=0.7),
         meta_template=api_meta_template,
         enable_thinking=True,
         batch_size=8),
]

infer = dict(
    partitioner=dict(type=SizePartitioner, strategy="split",
                     gen_task_coef=1, max_task_size=256),
    runner=dict(type=LocalRunner, max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=64,
                task=dict(type=OpenICLEvalTask)),
)
work_dir = f"eval_result/your_model/"
```

To run only inference, only evaluation, or only visualization, pass `-m infer|eval|viz` (with `-r latest` for the last two).

### Step 2: Inference

`OpenICLInferTask` reads the dataset loader, builds prompts from the configured `PromptTemplate` and `ZeroRetriever` (or `FixKRetriever`), and queries the model. Predictions are written to `<work_dir>/predictions/`.

### Step 3: Evaluation

`OpenICLEvalTask` runs the registered evaluator against the predictions. The evaluator base classes are:

| Base class | Use case | Example |
|------------|----------|---------|
| `opencompass.openicl.icl_evaluator.AccEvaluator` | Rule matching on choice letters / exact match | `BasicKnowledgeDataset`, `Zte5gDataset`, `WiredNetworkDataset`, `FaultMaintenanceDataset`, `NetOptmDataset`, `Protocol3GPPDataset` |
| `opencompass.openicl.BaseEvaluator` | Custom structured matching (JSON / action strings / substring) | `IntentRecognitionEvaluator1..5`, `EntityExtractionEvaluator`, `EventVerificationEvaluator` / `Wangyou5GEvaluator` |
| `opencompass.datasets.BaseJudgeACCEvaluator` | LLM-as-Judge on open-ended answers | `CoreNetworkEvaluator` (Qwen3-style 0/1 judge) |

The summarizer (`opencompass.summarizers.DefaultSummarizer`) aggregates the per-task results.

---

## How to Add a New Evaluation Task

The following walks through adding a hypothetical "Wireless Parameter Compliance Check" task end-to-end.

### 1. Prepare Data

Create directories under `code/datasets/`:

```
code/datasets/Knowledge_Application/Parameter_Check/parameter_check.json
```

### 2. Implement the Dataset Loader & Evaluator

Add a single file `code/opencompass/datasets/param_check.py`:

```python
import json
import os
from datasets import Dataset
from opencompass.datasets import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET


@LOAD_DATASET.register_module()
class ParamCheckDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_list(data)


@ICL_EVALUATORS.register_module()
class ParamCheckEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        is_correct = [p.strip() == r.strip()
                      for p, r in zip(predictions, references)]
        accuracy = sum(is_correct) / len(is_correct) if is_correct else 0.0
        return {
            "accuracy": accuracy * 100,
            "detail_dict": {
                "processed_pred": list(predictions),
                "processed_gold": list(references),
                "is_correct": is_correct,
            },
        }
```

Then re-export the new symbols in `code/opencompass/datasets/__init__.py`:

```python
from .param_check import ParamCheckDataset, ParamCheckEvaluator  # noqa: F401
```

(If you need LLM-as-Judge, inherit from `BaseJudgeACCEvaluator` and implement `_get_prompt`, `_get_judge_model`, `_extract_judge` — see `code/opencompass/datasets/core_network.py`.)

### 3. Add a Reusable Config

Create `code/configs/datasets/param_check/param_check_gen.py`:

```python
from opencompass.datasets import ParamCheckDataset, ParamCheckEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

PROMPT = "{input}\n"

param_check_datasets = dict(
    abbr="ParamCheck",
    type=ParamCheckDataset,
    path="datasets/Knowledge_Application/Parameter_Check/parameter_check.json",
    reader_cfg=dict(input_columns=["input"], output_column="output"),
    infer_cfg=dict(
        prompt_template=dict(type=PromptTemplate, template=PROMPT),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    ),
    eval_cfg=dict(evaluator=dict(type=ParamCheckEvaluator)),
)
```

### 4. Compose an Example Runner

Drop a new file in `code/examples/`, e.g. `code/examples/ParamCheck.py`:

```python
from mmengine import read_base
from opencompass.models import ReasoningAPI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from ..configs.datasets.param_check.param_check_gen import param_check_datasets

datasets = [param_check_datasets]

# ... models, infer, eval, work_dir — see code/examples/CoreNetwork.py
```

Then run:

```bash
cd code
python run.py examples/ParamCheck.py
```

---

## How to Extend the Dataset

### Add New Samples to an Existing Task

1. Open the JSON file under `code/datasets/Knowledge_Comprehension/...` or `code/datasets/Knowledge_Application/...`.
2. Append a new object following the same schema as the existing entries.
3. Keep `id` unique and ensure input/output formats match what the loader and evaluator expect.

### Add a Brand-New Evaluation Dimension

1. Add the raw data under `code/datasets/` (mirror the existing `Knowledge_Comprehension` / `Knowledge_Application` split).
2. Add a loader + evaluator in `code/opencompass/datasets/<task>.py` and re-export it in `__init__.py`.
3. Add a reusable config in `code/configs/datasets/<task>/`.
4. Add a runner in `code/examples/` (or compose it into an existing one).
5. Run `flake8 --config .flake8 code/` and a smoke test before opening a PR.

---

## Available Model Wrappers

`opencompass.models` exposes:

| Wrapper | When to use |
|---------|-------------|
| `ReasoningAPI` | Reasoning models (e.g. Qwen3) that need a `enable_thinking` flag. |
| `NonReasoningAPI` | OpenAI-compatible chat APIs without reasoning mode. |
| `Qwen3API` / `Qwen35API` | Specialized adapters for Qwen families. |
| `GeneralApi` | Generic OpenAI-compatible endpoint. |
| `LightllmAPI` | LightLLM-served endpoints. |
| `HuggingFace` / `HuggingFaceCausalLM` / `HuggingFaceChatGLM3` | Local HF models. |

For a custom endpoint, subclass `BaseAPIModel` in `code/opencompass/models/`.

---

## Obtaining the Complete Dataset

To prevent evaluation set leakage, this repository only ships a **subset** of the data as examples. To access the full dataset (required to reproduce paper numbers), contact the maintainers listed in [`MAINTAINER.md`](./MAINTAINER.md).

---

## Pull Request Process

1. Fork this repository and create a feature branch on your fork.
2. Install the project in editable mode (`pip install -e code/`) and run `flake8 --config .flake8 code/`.
3. Make sure your changes do not break the example runs in `code/examples/`. A small smoke test (`python run.py examples/BasicKnowledge.py -m infer --debug --dry-run`) is a quick sanity check.
4. Update documentation (`README.md` / `CONTRIBUTING.md` / `code/docs/`) if you change public APIs, add a new task, or modify the dataset layout.
5. Describe the change and its motivation clearly in the PR description.
6. Wait for a maintainer review.

## Code of Conduct

This project follows the LF AI & Data Projects Code of Conduct. See [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) for details.

## Questions and Discussions

- Bugs & feature requests: please open a GitHub Issue with reproduction steps.
- Other questions: contact the maintainers via [`MAINTAINER.md`](./MAINTAINER.md).
