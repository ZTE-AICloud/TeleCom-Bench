# 网络通信大模型评估框架 — 入门指南

## 1. 项目概述

本项目是基于 [OpenCompass](https://github.com/open-compass/opencompass) 二次开发的**网络通信领域大模型评估框架**，用于对大语言模型在通信领域的知识理解与业务能力进行系统化评测。

### 核心能力

- **知识理解评测（Knowledge Comprehension）**：考察模型对通信基础知识、3GPP 协议、5G 网络、核心网、有线/无线网络等产品知识的掌握程度。
- **知识应用评测（Knowledge Application）**：考察模型在实际运维场景中的能力，包括实体提取、意图识别、事件核查、根因诊断、方案生成、工具调用等。
- **灵活的模型接入**：支持通过 API 方式接入各类大模型（ReasoningAPI、NonReasoningAPI、Qwen3API、GeneralApi 等），也支持 HuggingFace 本地模型。
- **自动化评估流水线**：推理（Infer）→ 评估（Eval）→ 汇总（Summarize）全流程自动化，支持本地、Slurm、DLC 多种运行方式。

---

## 2. 环境准备

### 2.1 系统要求

Python 版本 >= 3.8 

### 2.2 依赖安装

项目依赖通过 `setup.py` 管理，依赖列表位于 `requirements/requirements.txt`。

```bash
# 1. 克隆项目
git clone <repo_url>
cd git

# 2.（推荐）创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装项目（会自动安装 requirements.txt 中的依赖）
pip install -e .
```

### 2.3 验证安装

```bash
python -c "from opencompass import __version__; print(__version__)"
# 输出: 0.2.0
```

---

## 3. 快速开始

### 3.1 第一步：准备配置文件

从示例文件复制一份配置，并修改其中的模型 API 地址和密钥：

```bash
cp examples/CoreNetwork.py my_eval.py
```

编辑 `my_eval.py`，需要修改以下关键参数：

```python
# 模型名称（用于标识和输出目录命名）
model = "your_model_name"

# 模型 API 地址
api_url = "https://your-api-endpoint/v1/chat/completions"

# API 认证信息（如需）
api_headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-api-key",
}

# 裁判模型（用于 LLM-as-Judge 评分方式）
judge_model_cfg = dict(
    base_url="https://your-judge-api-url",
    model="judge_model_name",
    api_key="your-judge-api-key",
)
```

### 3.2 第二步：运行评估

```bash
python run.py my_eval.py
```

该命令会自动完成 **推理 → 评估 → 结果汇总** 的完整流程。

### 3.3 第三步：查看结果

评估结果默认输出到 `eval_result/<model_name>/` 目录，也可通过配置中的 `work_dir` 自定义。

---
