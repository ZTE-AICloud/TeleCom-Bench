import math, random, json
from datasets import Dataset, DatasetDict
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET, TEXT_POSTPROCESSORS
from .base import BaseDataset

# def flatten_list(nested_list):
#     # 递归展平嵌套列表nested_list
#     flat_list = []
#     for item in nested_list:
#         if isinstance(item, list):
#             flat_list.extend(flatten_list(item))
#         else:
#             flat_list.append(item)
#     return flat_list

# 鲁棒性、拒答能力、信息整合能力参数调整
rob_noise_rate, rob_passage_num = 0.3, 10
rej_noise_rate, rej_passage_num = 1, 10
int_noise_rate, int_passage_num = 0.1, 10


def rgb_processdata(instance, filename, noise_rate, passage_num, correct_rate=0):
    # 返回处理后的数据
    query = instance['query']  # 查询问题字符串
    answer = instance['answer']  # 格式不统一的列表，期望输出元素格式统一的列表（比如都是字符串或都是列表）
    for i in range(len(answer)):
        if not isinstance(answer[i], list):
            answer[i] = [answer[i]]

    neg_num = math.ceil(passage_num * noise_rate)
    pos_num = passage_num - neg_num
    docs = []

    if '_refine' in filename:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num
        positive = instance['positive'][:pos_num]
        negative = instance['negative'][:neg_num]
        docs = positive + negative

    elif '_int' in filename:
        for positive_docs in instance['positive']:
            random.shuffle(positive_docs)
        docs = [i[0] for i in instance['positive']]
        if len(docs) < pos_num:
            maxnum = max([len(i) for i in instance['positive']])
            for i in range(1, maxnum):
                for positive_docs in instance['positive']:
                    if len(positive_docs) > i:
                        docs.append(positive_docs[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            negative = instance['negative'][:neg_num]
            docs += negative
        # answer = []
        # if len(instance['asnwer1']) > 0:
        #     answer.append(flatten_list(instance['asnwer1']))
        # if len(instance['answer2']) > 0:
        #     answer.append(flatten_list(instance['answer2']))

    # print(answer)
    random.shuffle(docs)
    docs = '\n'.join(docs)
    return query, docs, answer


def load_rgbdataset(path, noise_rate, passage_num):
    with open(path, 'r', encoding='utf-8') as f:
        rows = []
        for line in f:
            instance = json.loads(line)
            query, docs, answer = rgb_processdata(instance, path, noise_rate, passage_num)
            rows.append({
                'query': query,
                'docs': docs,
                'answer': answer,
            })

        dataset = Dataset.from_dict({
            'query': [row['query'] for row in rows],
            'docs': [row['docs'] for row in rows],
            'answer': [row['answer'] for row in rows]
        })
    return dataset


@LOAD_DATASET.register_module()
class RgbRobZhDataset(BaseDataset):
    # RAG噪声鲁棒性测试数据集
    @staticmethod
    def load(path: str):
        return load_rgbdataset(path, noise_rate=rob_noise_rate, passage_num=rob_passage_num)


@LOAD_DATASET.register_module()
class RgbRobEnDataset(BaseDataset):
    # RAG噪声鲁棒性测试数据集
    @staticmethod
    def load(path: str):
        return load_rgbdataset(path, noise_rate=rob_noise_rate, passage_num=rob_passage_num)


@LOAD_DATASET.register_module()
class RgbRejZhDataset(BaseDataset):
    # RAG拒答能力测试数据集
    @staticmethod
    def load(path: str):
        return load_rgbdataset(path, noise_rate=rej_noise_rate, passage_num=rej_passage_num)


@LOAD_DATASET.register_module()
class RgbRejEnDataset(BaseDataset):
    # RAG拒答能力测试数据集
    @staticmethod
    def load(path: str):
        return load_rgbdataset(path, noise_rate=rej_noise_rate, passage_num=rej_passage_num)


@LOAD_DATASET.register_module()
class RgbIntZhDataset(BaseDataset):
    # RAG信息集成能力测试数据集
    @staticmethod
    def load(path: str):
        return load_rgbdataset(path, noise_rate=int_noise_rate, passage_num=int_passage_num)


@LOAD_DATASET.register_module()
class RgbIntEnDataset(BaseDataset):
    # RAG信息集成能力测试数据集
    @staticmethod
    def load(path: str):
        return load_rgbdataset(path, noise_rate=int_noise_rate, passage_num=int_passage_num)


@ICL_EVALUATORS.register_module()
class RgbRobEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        details = []
        cnt = 0
        for prediction, reference in zip(predictions, references):
            labels = rgb_checkanswer(prediction, reference)
            detail = {'pred': prediction, 'answer': reference, 'correct': False}
            if 0 not in labels and 1 in labels:
                cnt += 1
                detail['correct'] = True
            details.append(detail)

        score = cnt / len(predictions) * 100
        return {'accuracy_rob': score, 'details': details}


@ICL_EVALUATORS.register_module()
class RgbRejEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        details = []
        cnt = 0
        for prediction, reference in zip(predictions, references):
            detail = {'pred': prediction, 'answer': reference, 'correct': False}
            if '信息不足' in prediction or 'insufficient information' in prediction:
                cnt += 1
                detail['correct'] = True
            details.append(detail)

        score = cnt / len(predictions) * 100
        return {'accuracy_rej': score, 'details': details}


@ICL_EVALUATORS.register_module()
class RgbIntEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }
        details = []
        cnt = 0
        for prediction, reference in zip(predictions, references):
            labels = rgb_checkanswer(prediction, reference)
            detail = {'pred': prediction, 'answer': reference, 'correct': False}
            if 0 not in labels and 1 in labels:
                cnt += 1
                detail['correct'] = True
            details.append(detail)

        score = cnt / len(predictions) * 100
        return {'accuracy_int': score, 'details': details}


def rgb_checkanswer(prediction, ground_truth):
    prediction = prediction.lower()
    labels = []
    for instance in ground_truth:
        flag = True
        if type(instance) == list:
            flag = False
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels

# @TEXT_POSTPROCESSORS.register_module('rgb_zh')
# def RgbZh_postprocess(text):
#     text = text.lower().replace(" ", "")
#     return text
