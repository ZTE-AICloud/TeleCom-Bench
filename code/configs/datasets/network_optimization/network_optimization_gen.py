from opencompass.datasets import NetOptmDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import eoa_tag_postprocessor, extract_specified_options

PROMPT = """
假设你是中兴通讯的研发和运维助手，擅长通信网络场景优化的问题，请选出其中正确的答案，请将正确的答案写在[正确答案]和<eoa>之间，例如：[正确答案]C,D<eoa>，请你严格按照以上格式回答。
本题是{question_type}。
{stem}
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}
"""

_reader_cfg = dict(
    input_columns=["stem", "question_type", "A", "B", "C", "D", "E"],
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
    pred_postprocessor=dict(type=eoa_tag_postprocessor),
    data_postprocessor=dict(type=extract_specified_options))

network_optimization_datasets = dict(
    abbr="NetworkOptimization",
    type=NetOptmDataset,
    path="datasets/Knowledge_Comprehension/Product Knowledge/Wireless_Network/network_optimization.json",
    reader_cfg=_reader_cfg,
    infer_cfg=_infer_cfg,
    eval_cfg=_eval_cfg,
)
