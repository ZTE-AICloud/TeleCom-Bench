import json, re, os
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset
from .ragbench_eva import RagBenchEvaluator, RagBench_inscit_Evaluator, RagBench_quac_doqa_Evaluator


@LOAD_DATASET.register_module()
class RAGBenchDataset(BaseDataset):
    @staticmethod
    def load(path: str):
        with open(path + '/test.json', 'r') as f:
            data_list = json.load(f)
            dataset_name = os.path.basename(path)
            dataset = ragbench_get_dataset(data_list, dataset_name)
        return dataset


def ragbench_get_dataset(data_list, dataset_name, num_ctx=2):
    # 系统信息描述，前置条件prompt
    # 这是一个用户和人工智能助手之间的对话。该助手根据上下文，为用户提供有用、详细且礼貌的回答。当在上下文中找不到答案时，助手也应指出这一点。
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."

    rows = []
    for item in data_list:
        turn_list = item['messages']  # 用户和助手的对话
        question_formatted = ragbench_reformat_question(turn_list, dataset_name)

        # 此处修改，部分doc2dial qrecc quac topiocqa数据集context太多，如果有ground_truth_ctx的话就取ground_truth_ctx，没有就还是原样
        if dataset_name == 'inscit':
            ctx_list = [ctx['ctx'] for ctx in item['ground_truth_ctx']]
            context = "\n".join(ctx_list)
        elif 'ground_truth_ctx' in item:
            context = item['ground_truth_ctx']['ctx']
        else:
            ctx_list = ["title: " + ctx["title"] + ", source: " + ctx["text"] for ctx in item['ctxs'][:num_ctx]]
            context = "\n".join(ctx_list)

        # 将系统信息、上下文和格式化后的问题组合成一个字符串
        # model_input = system + "\n" + context + "\n" + question_formatted

        if "answers" in item:
            answers = item["answers"]
        elif "answer" in item:
            if type(item["answer"]) is str:
                answers = [item["answer"]]
            elif type(item["answer"]) is list:
                answers = item["answer"]
            else:
                answers = [str(item["answer"])]
        else:
            raise ValueError("need to have answer or answers")

        # convfinqa由于涉及数学计算，单独处理
        if dataset_name == 'convfinqa':
            answers = item['answers'][:1]
            answers.append(str(item["exe_answer"]))
            answers.append(item['messages'][-1]['content'])

        rows.append({
            'query': question_formatted,
            'docs': context,
            'answer': answers,
        })

    dataset = Dataset.from_dict({
        'query': [row['query'] for row in rows],
        'docs': [row['docs'] for row in rows],
        'answer': [row['answer'] for row in rows]
    })
    return dataset


# 输入：对话记录列表和数据集名称。
# 输出：问题字符串。
# 功能：根据数据集名称对用户提问进行特定的修改，并将对话记录格式化为字符串。
def ragbench_reformat_question(turn_list, dataset_name):
    # 只截取最近的7轮对话
    turn_list = turn_list[-7:]
    assert turn_list[-1]['role'] == 'user'

    # 定义数据集类别列表：
    long_answer_dataset_list = ["doc2dial", "quac", "qrecc", "inscit", "doqa_movies", "doqa_travel", "doqa_cooking",
                                "hybridial"]
    long_and_short_dataset_list = ["topiocqa"]
    entity_dataset_list = ["sqa"]
    short_dataset_list = ["coqa"]
    math_dataset_list = ["convfinqa"]

    # 长答案数据集：找到第一轮用户提问并在其内容前添加
    # 请对这个问题给出一个完整且详尽的回答。
    if dataset_name in long_answer_dataset_list:
        for item in turn_list:
            if item['role'] == 'user':
                ## only needs to add it on the first user turn
                item['content'] = 'Please give a complete answer for the question. ' + item['content']
                break

    # 在最后一轮用户提问内容前添加
    # 长短答案数据集：请用简短的回答，或一个完整且详尽的回答来回答以下问题。
    # 实体数据集：请用一个或一列表item来回答以下问题。
    # 短答案数据集：请用简短的回答来回答以下问题。回答需要用几句话简洁明了地表达。
    elif dataset_name in long_and_short_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with a short span, or a full and complete answer. " + \
                                   turn_list[-1]['content']
    elif dataset_name in entity_dataset_list:
        turn_list[-1]['content'] = "Answer the following question with one or a list of items. " + turn_list[-1][
            'content']
    elif dataset_name in short_dataset_list:
        turn_list[-1][
            'content'] = "Answer the following question with a short span. The answer needs to be just in a few words. " + \
                         turn_list[-1]['content']
    elif dataset_name in math_dataset_list:
        turn_list[-1][
            'content'] = "Please respond to the following questions with direct calculation results or concise statements." + \
                         turn_list[-1]['content']
    else:
        raise Exception("please input a correct dataset name!")

    # 构建格式化后的对话字符串：
    # 遍历 turn_list，将每一轮对话内容按角色添加到字符串 question 中。
    # 最后在字符串末尾添加 "Assistant:"。
    question = ""
    for item in turn_list:
        if item["role"] == "user":
            question += "User: " + item["content"] + "\n"
        else:
            assert item["role"] == "assistant"
            question += "Assistant: " + item["content"] + "\n"
    question += "Assistant:"
    return question
