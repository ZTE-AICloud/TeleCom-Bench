from opencompass.datasets import BasicKnowledgeDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import latex_last_en

PROMPT = """以下是中国关于通信工程师考试的选择题，只有一个选项是正确的，请选出其中的正确的选项。
{question}
A. {A}
B. {B}
C. {C}
D. {D}
答案："""

_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer',
)

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
    pred_postprocessor=dict(type=latex_last_en),
)

_basic_knowledge_types = [
    '传输与接入（无线）',
    '传输与接入（有线）',
    '互联网技术',
    '设备环境',
    '终端与业务',
    '通信专业综合能力',
]

basic_knowledge_datasets = []

for _type in _basic_knowledge_types:
    basic_knowledge_datasets.append(
        dict(
            type=BasicKnowledgeDataset,
            path='datasets/Knowledge_Comprehension/Basic Theory/Basic_Knowledge/basic_knowledge.json',
            name=_type,
            abbr=_type,
            reader_cfg=_reader_cfg,
            infer_cfg=_infer_cfg,
            eval_cfg=_eval_cfg,
        ))
