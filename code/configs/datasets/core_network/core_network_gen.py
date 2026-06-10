from opencompass.datasets import CoreNetwork, CoreNetworkEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer',
)

PROMPT = """
背景：
你是一个中兴通讯CCN产品知识问答的机器人，首先会识别出输入（用户问题）的类别，其次会基于问题类别给出对应的答案。
我们现在有中兴通讯CCN相关的问题，问题类别包括三类：概念问答、流程描述问答、运维方案问答，当提供给你问题后，你会先识别问题类别，并基于问题类别给出对应完整准确的答案

用户问题：{question}

任务要求：
1、**请从用户提交的问题中首先识别问题的类别**
  1）基于概念问答（是什么/作用/有哪些），比如：
    虚拟化是一种什么技术？
    TECS CloveStorage的产品特点有哪些？
    VNFM是否支持管理第三方厂商设备？
  2）流程描述问答（怎么做/流程/步骤/配置）：
    说明虚机救援的流程。
    CG如何配置NTP服务器？
    Elasticnet UME产品如何安装？
  3）运维方案问答（排查/故障/告警/恢复/应急）：
    告警无法处理或无法恢复一般该怎么办？
    检查EMSPlus系统能否将告警正常上报给ElasticNet UME系统的步骤是什么？
    冷迁移有哪些系统影响？
2、**其次根据不同类别的问题给出对应的答案，答案中需要包含如下的必要要点**
  1）对于概念问答类问题：答案要包含必含要点（核心属性 / 功能）以及可选要点（补充说明）
  2）流程描述问答类问题：答案要包含必含步骤（顺序不可错）、关键网元 / 接口、逻辑节点
  3）运维方案问答类问题：答案要包含排查维度、核心步骤、可能原因 / 解决方案


请根据上述要求，输出答案。
"""

_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=PROMPT,
                ),
            ],
        ),
        ice_token="</E>",
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

_eval_cfg = dict(
    evaluator=dict(type=CoreNetworkEvaluator),
)

core_network_datasets = dict(
    abbr="CoreNetwork",
    type=CoreNetwork,
    path="datasets/Knowledge_Comprehension/Product Knowledge/Core_Network/core_network.json",
    reader_cfg=_reader_cfg,
    infer_cfg=_infer_cfg,
    eval_cfg=_eval_cfg,
)
