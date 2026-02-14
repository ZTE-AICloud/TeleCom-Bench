import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from opencompass.datasets import BaseJudgeScoreEvaluator
from opencompass.registry import ICL_EVALUATORS


@ICL_EVALUATORS.register_module()
class TeleYamlEvaluator(BaseJudgeScoreEvaluator):
    def __init__(self, prompt=None, judge_model=None):
        super().__init__(
            prompt=prompt,
            judge_model=judge_model,
            score_levels=[0, 1])

    def _get_prompt(self) -> str:
        prompt = """
You are an impartial judge model evaluating whether the assistant's YAML/config answer MATCHES the SME reference answer.

You will be given:
1. A telecom-related question.
2. The assistant's answer (prediction).
3. The reference answer provided by an SME (reference).

IMPORTANT EVALUATION PRINCIPLE (STRICT):
- Treat the reference answer as the ONLY ground truth for this task.
- Do NOT use your own general knowledge to “forgive” deviations: this is a reference-alignment task, not a free-form best-practice review.
- Judge primarily by correctness and alignment with the reference's configuration SCHEMA, required blocks, and field placement, not by writing quality.
- If the assistant invents a different schema/keys, omits required blocks present in the reference, changes the nesting/placement of key fields, or produces invalid YAML, the scores MUST be low even if the text is fluent.

How to compare (do NOT execute code; do a careful textual/structural comparison):
- Step 1: Identify the required values from the question (e.g., PLMN MCC/MNC, TAC, cluster/region string).
- Step 2: Extract the required STRUCTURE from the reference answer: top-level keys, section names, nesting, list vs object shape, and where key fields live.
- Step 3: Compare the assistant answer to the reference:
  - Schema match: Does it use the SAME section names and nesting as the reference? (Do NOT accept invented alternatives.)
  - Block coverage: If a block exists in the reference, it is REQUIRED in the assistant answer (this dataset expects a canonical full config, not a tiny snippet).
  - Key placement: TAC/PLMN must appear under the SAME keys as the reference (e.g., TAC under 'amf.tai' with 'plmn_id', not under 'access-control', 'allowed-tac-list', 'tac-list', 'tac_ranges', 'regions', etc.).
  - Value match: MCC/MNC/TAC must match the reference (treat type/format differences that change meaning as wrong; e.g., hex strings when reference uses integer).
  - YAML/config validity: the answer must be a coherent configuration block. Obvious indentation/nesting errors count as invalid.

For this dataset, the reference answers typically contain these core AMF configuration blocks. When they appear in the reference, treat them as REQUIRED:
- 'amf.sbi.server' (address/port)
- 'amf.ngap.server' (address/port)
- 'amf.guami' with 'plmn_id' and 'amf_id'
- 'amf.tai' with 'plmn_id' and 'tac' (this is where TAC should be set)
- 'amf.plmn_support' with 'plmn_id'
- 'amf.security' (integrity_order, ciphering_order)
- 'nrf.sbi.server'

Hard scoring rules (use these as gates to avoid inflated scores):
- If the assistant answer does NOT provide a YAML/config block at all: instruction_following_score <= 1, technical_accuracy_score <= 1, linguistic_quality_score <= 2.
- If the YAML/config is syntactically invalid (obvious indentation/nesting errors): technical_accuracy_score <= 1 and linguistic_quality_score <= 4.
- If MCC/MNC/TAC values are wrong OR TAC is placed under the wrong key compared to the reference (e.g., not under 'tai' with 'plmn_id'): technical_accuracy_score <= 2 and instruction_following_score <= 2.
- If the assistant uses a different schema (invented top-level keys, renamed sections, or wrong nesting/list shape) OR is missing ANY of the core required blocks that appear in the reference: technical_accuracy_score <= 2 and instruction_following_score <= 3.
- Linguistic score must NOT inflate a wrong answer:
  - If technical_accuracy_score <= 2, then linguistic_quality_score MUST be <= 4.
  - If technical_accuracy_score <= 1, then linguistic_quality_score MUST be <= 3.
- Only give technical_accuracy_score >= 7 when the assistant answer is strongly consistent with the reference structure AND includes the required blocks/fields with correct values (minor harmless formatting differences are allowed).

Score definitions:

### **Instruction Following** (0-10)
How well the answer satisfies the user's request AND matches the reference-required structure.
- 9-10: Provides YAML/config in the expected schema; includes the required sections/blocks present in the reference; places TAC/PLMN exactly as in the reference.
- 5-8: Mostly matches the reference schema; at most small omissions of non-core blocks; no key-placement errors.
- 1-4: Partial or wrong-schema config; missing required blocks; or places TAC/PLMN under wrong keys.
- 0: Irrelevant or misleading.

### **Technical Accuracy & Reference Alignment** (0-10)
Whether the configuration is correct, usable, and aligned to the reference.
- 9-10: Correct values AND correct schema/placement; would work as the intended AMF config per reference style.
- 5-8: Mostly correct but with limited missing non-critical details; schema largely consistent.
- 1-4: Wrong schema/placement, missing essential blocks, or would not be usable; even if values are mentioned.
- 0: Dangerous/misleading or completely wrong.

### **Linguistic Quality & Configuration Clarity** (0-10)
Clarity and readability ONLY after considering correctness. Fluency must NOT inflate a wrong answer.
- 8-10: Clear and concise; YAML is readable and consistent.
- 4-7: Understandable but verbose or slightly confusing.
- 0-3: Unclear, contradictory, or not a coherent config (or invalid YAML).

Provide your evaluation strictly in JSON format as follows (no extra text):

{{
  "instruction_following_score": <int 0-10>,
  "linguistic_quality_score": <int 0-10>,
  "technical_accuracy_score": <int 0-10>,
  "reasoning": "<brief explanation: highlight mismatches vs reference (missing blocks, wrong keys, wrong TAC placement/value, YAML validity)>"
}}

**Question**:
{question}

**Assistant's Answer**:
{prediction}

**Reference Answer**:
{reference}
"""
        return prompt

    def _get_judge_model(self):
        if self._judge_model is not None:
            return self._judge_model
        from opencompass.judge_models.judge_qwen235b import Qwen235B
        return Qwen235B()

    def _extract_judge(self, judge_message: str):
        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', judge_message, re.DOTALL)
            if not json_match:
                return None
            
            json_data = json.loads(json_match.group(0))
            if not isinstance(json_data, dict):
                return None
            
            score_keys = ["instruction_following_score", "linguistic_quality_score", "technical_accuracy_score"]
            scores = []
            for key in score_keys:
                if key in json_data:
                    score = int(json_data[key])
                    if 0 <= score <= 10:
                        scores.append(score)
            
            if not scores:
                return None
            
            return sum(scores) / len(scores) / 10.0
        except Exception:
            return None

    def score(self, predictions: List, references: List, questions: List) -> dict:
        if len(predictions) != len(references):
            raise ValueError('Predictions and references have different lengths')

        valid_count = 0
        total_score = 0.0
        results_order = [None] * len(questions)
        detail_dict = {}

        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = {
                executor.submit(self._process_sample, q, p, r): idx
                for idx, (q, p, r) in enumerate(zip(questions, predictions, references))
            }

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results_order[idx] = result
                score = result['judge_result']
                if score is not None:
                    total_score += score
                    valid_count += 1

        seen_keys = set()
        for r in results_order:
            if r is not None:
                seen_keys.update(r.keys())
        for k in seen_keys:
            detail_dict[k] = []
        for r in results_order:
            for k in seen_keys:
                if r is not None and k in r:
                    detail_dict[k].append(r[k])
                else:
                    detail_dict[k].append(None)

        avg_score = (total_score / valid_count) if valid_count > 0 else 0.0
        result = {
            'score': avg_score * 100,
            'detail_dict': detail_dict
        }

        post_result = self._postprocess(result)
        return post_result
