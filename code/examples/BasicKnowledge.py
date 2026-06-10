from mmengine import read_base

from opencompass.models import ReasoningAPI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask


with read_base():
    from ..configs.datasets.basic_knowledge.basic_knowledge_gen import basic_knowledge_datasets

datasets = [basic_knowledge_datasets]

rest_type = "ReasoningAPI"
model = "model_name"
api_url = "api_url"
temperature = "0.7"
max_tokens = 2048
enable_thinking = True

api_headers = {
    "Content-Type": "application/json",
    "skip_special_tokens": "false",
    "Authorization": "None",
}

api_data = dict(
    model=model,
    max_tokens=4096,
    temperature=float(temperature),
)

api_meta_template = dict(
    round=[
        dict(role="SYSTEM", api_role="SYSTEM"),
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)


models = [
    dict(abbr=rest_type+"_"+model,
         type=rest_type,
         path=rest_type+"_"+model,
         api_url=api_url,
         api_headers=api_headers,
         api_data=api_data,
         meta_template=api_meta_template,
         enable_thinking=enable_thinking,
         batch_size=16
         ),
]

infer = dict(
    partitioner=dict(type=SizePartitioner,strategy="split", gen_task_coef=1, max_task_size=256),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner,
                max_num_workers=64,
                task=dict(type=OpenICLEvalTask)),
)

work_dir = "eval_result/ops-rl-full-data-0528-mix-transformer-step30/CCN横向根因推理/"