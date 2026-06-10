from opencompass.datasets import (EntityExtractionDataset,
                                  EntityExtractionEvaluator)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

PROMPT_DIR = 'configs/datasets/entity_extraction/prompt'

_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
    test_split='test',
)

_key_list = {
    '实体提取-定界定位': ['rootCauseType', 'rootNetwork', 'rootNe'],
    '实体提取-查看拓扑': ['device', 'city', 'network'],
    '实体提取-查询事件关联告警': ['专业', '告警级别'],
    '实体提取-查询告警': ['时间范围', '告警级别', '专业', '地市', '网元', '告警标题'],
    '实体提取-查询告警是否恢复': ['title', 'device'],
    '实体提取-查询当前事件': ['时间范围', '事件级别', '地市', '网元'],
    '实体提取-查询操作日志': ['时间范围', '专业', '网元名称'],
    '实体提取-查询故障链路': ['本端设备', '对端设备', '地市'],
    '实体提取-查询机房设备': ['机房名称', '专业'],
    '实体提取-查询网元用户数量': ['网元'],
    '实体提取-查询网元相关': ['网元'],
    '实体提取-查询资源池': ['room', 'resource_type'],
    '实体提取-查询链路性能': ['时间范围', '网元', '端口', '方向', '指标'],
    '实体提取_核心网性能数据查询': ['时间范围', '网元'],
}

entity_extraction_datasets = []

for _type in _key_list:
    prompt_path = f'{PROMPT_DIR}/{_type}.txt'
    with open(prompt_path, 'r') as f:
        _prompt = f.read()

    _infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[dict(role='HUMAN', prompt=_prompt)]),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    _eval_cfg = dict(
        evaluator=dict(type=EntityExtractionEvaluator, key_list=_key_list[_type]),
    )

    entity_extraction_datasets.append(
        dict(
            type=EntityExtractionDataset,
            path='datasets/Knowledge_Application/Entity_Extraction/entity_extraction.json',
            name=_type,
            abbr=_type,
            reader_cfg=_reader_cfg,
            infer_cfg=_infer_cfg,
            eval_cfg=_eval_cfg,
        ))
