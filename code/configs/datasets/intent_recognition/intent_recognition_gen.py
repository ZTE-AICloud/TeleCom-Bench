from opencompass.datasets import IntentRecognitionDataset, IntentRecognitionEvaluator1, IntentRecognitionEvaluator2, \
    pass_postprocessor1, IntentRecognitionEvaluator3, IntentRecognitionEvaluator4, IntentRecognitionEvaluator5
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils import json_str

_reader_cfg = dict(
    input_columns=['input'],
    output_column='output',
    test_split='test',
)

_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{input}",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

intent_recognition_configs = {
    '网优专家_覆盖agent场景分类': (pass_postprocessor1, IntentRecognitionEvaluator2),
    '网优专家_容量agent场景分类': (pass_postprocessor1, IntentRecognitionEvaluator2),
    '网优专家_用户输入_实体提取': (json_str, IntentRecognitionEvaluator4),
    '网优专家_网优总控agent分类': (pass_postprocessor1, IntentRecognitionEvaluator5),
    '网优专家_用户输入_意图分类': (pass_postprocessor1, IntentRecognitionEvaluator2),
    '网优专家_覆盖agent技能': (pass_postprocessor1, IntentRecognitionEvaluator3),
    '网优专家_工单列头识别': (json_str, IntentRecognitionEvaluator1),
    '网优专家_容量agent技能': (pass_postprocessor1, IntentRecognitionEvaluator3),
}

intent_recognition_datasets = []

for _type, (_processor, _evaluator) in intent_recognition_configs.items():
    _eval_cfg = dict(
        evaluator=dict(type=_evaluator),
        pred_postprocessor=dict(type=_processor),
    )

    intent_recognition_datasets.append(
        dict(
            type=IntentRecognitionDataset,
            path='datasets/Knowledge_Application/Intent_Recognition/intent_recognition.json',
            name=_type,
            abbr=_type,
            reader_cfg=_reader_cfg,
            infer_cfg=_infer_cfg,
            eval_cfg=_eval_cfg,
        ))
