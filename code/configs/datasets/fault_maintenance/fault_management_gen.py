from opencompass.datasets import FaultMaintenanceDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import latex_last_mcq, extract_specified_options

PROMPT = """
<题目 begin>
{question}
{A}
{B}
{C}
{D}
{E}
<题目 end>

请认真阅读上述题目和选项，给出你认为最合适的选项答案。请将你的最终答案用\\boxed{}标记，例如：\\boxed{{A}}。
"""

_reader_cfg = dict(
    input_columns=["question", "options"],
    output_column="answer")

_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=PROMPT,
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=latex_last_mcq),
    data_postprocessor=dict(type=extract_specified_options))

fault_maintenance_datasets = dict(
    abbr="FaultMaintenance",
    type=FaultMaintenanceDataset,
    path="datasets/Knowledge_Comprehension/Product Knowledge/Wireless_Network/fault_maintenance.json",
    reader_cfg=_reader_cfg,
    infer_cfg=_infer_cfg,
    eval_cfg=_eval_cfg,
)
