from opencompass.datasets import Protocol3GPPDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import extract_specified_options

_reader_cfg = dict(
    input_columns=["prompt"],
    output_column="answer")

_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{prompt}",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=extract_specified_options),
    data_postprocessor=dict(type=extract_specified_options))

NUM_RUNS = 3

protocol_3gpp_datasets = [
    dict(
        abbr=f"protocol_3gpp_{i + 1}",
        type=Protocol3GPPDataset,
        path="datasets/Knowledge_Comprehension/Basic Theory/3GPP_Protocols/3GPP_protocols.json",
        reader_cfg=_reader_cfg,
        infer_cfg=_infer_cfg,
        eval_cfg=_eval_cfg,
    )
    for i in range(NUM_RUNS)
]
