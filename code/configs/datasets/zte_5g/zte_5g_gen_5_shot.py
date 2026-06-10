from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import Zte5gDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess, extract_specified_options

single_choice_template = "下面是一道有多个选项的选择题，但仅有一个是正确答案，请选择正确答案；仅用A、B、C或者D作答，不要回答额外信息。\n问题：{question}\nA.{A}\nB.{B}\nC.{C}\nD.{D}\n答案："
multi_choice_template = "下面是一道有多个选项的选择题，并且有多个选项是正确的，请选择正确答案；仅用A、B、C或者D作答，不要回答额外信息。\n问题：{question}\nA.{A}\nB.{B}\nC.{C}\nD.{D}\n答案："
torf_template = "请判断下面说法是否正确，正确使用T回答，错误使用F回答，不要回答额外信息。\n问题：{question}\n答案："

zte_5g_area_sets = ['5G基础', '5G无线接入网', '5G核心网', '安全与规范']

zte_5g_templates = {
    "单选题": single_choice_template,
    "多选题": multi_choice_template,
    "判断题": torf_template,
}

zte_5g_postprocess = {
    "单选题": first_capital_postprocess,
    "多选题": extract_specified_options,
    "判断题": first_capital_postprocess,
}

_reader_cfg = dict(
    input_columns=["question", "A", "B", "C", "D"],
    output_column="answer",
)

zte_5g_datasets = []
for _area in zte_5g_area_sets:
    for _question_type in zte_5g_templates.keys():
        _name = _area + "_" + _question_type

        _infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=zte_5g_templates[_question_type],
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            inferencer=dict(type=GenInferencer),
        )

        _eval_cfg = dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=zte_5g_postprocess[_question_type]),
        )

        zte_5g_datasets.append(
            dict(
                abbr=_name,
                type=Zte5gDataset,
                path="datasets/Knowledge_Comprehension/Basic Theory/5G_Network/5G_network.json",
                name=_name,
                reader_cfg=_reader_cfg,
                infer_cfg=_infer_cfg,
                eval_cfg=_eval_cfg,
            )
        )
