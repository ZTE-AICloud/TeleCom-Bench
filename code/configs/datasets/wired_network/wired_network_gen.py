from opencompass.datasets import WiredNetworkDataset
from opencompass.openicl import ZeroRetriever
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.utils import eoa_tag_postprocessor, extract_specified_options

NUM_RUNS = 3

PROMPT = (
    "假设你是中兴通讯的研发和运维助手，回答下列问题，并将正确的答案写在[正确答案]和<eoa>之间。例如[正确答案]C,D<eoa>。请你严格按照这个格式回答。\n"
    "### 问题\n"
    "{question}\n"
    "### 选项\n"
    "{options}\n"
)

_reader_cfg = dict(
    input_columns=["question", "options"],
    output_column="answer",
)

_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt=PROMPT),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    dataset_postprocessor=dict(type=extract_specified_options),
    pred_postprocessor=dict(type=eoa_tag_postprocessor),
)

wired_network_datasets = [
    dict(
        abbr=f"wired_network_{i + 1}",
        type=WiredNetworkDataset,
        path="datasets/Knowledge_Comprehension/Product Knowledge/Wired_Nerwork/wired_network.json",
        reader_cfg=_reader_cfg,
        infer_cfg=_infer_cfg,
        eval_cfg=_eval_cfg,
    )
    for i in range(NUM_RUNS)
]
