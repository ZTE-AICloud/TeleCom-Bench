from opencompass.datasets import EventVerificationDataset, EventVerificationEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

_reader_cfg = dict(
    input_columns=["question"],
    output_column="best answer",
)

_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{question}",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8192),
)

_eval_cfg = dict(
    evaluator=dict(type=EventVerificationEvaluator),
    pred_role="BOT",
)

event_verification_datasets = dict(
    abbr="EventVerification",
    type=EventVerificationDataset,
    path="datasets/Knowledge_Application/Event_Verification/event_verification.json",
    reader_cfg=_reader_cfg,
    infer_cfg=_infer_cfg,
    eval_cfg=_eval_cfg,
)
