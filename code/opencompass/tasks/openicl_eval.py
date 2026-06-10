import argparse
import copy
import fnmatch
import math
import os
import os.path as osp
import re
import statistics
import sys
import time
from collections import Counter
from inspect import signature
from typing import List, Optional

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from opencompass.registry import (ICL_EVALUATORS, MODELS, TASKS,
                                  TEXT_POSTPROCESSORS)
from opencompass.tasks.base import BaseTask
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger,
                               task_abbr_from_cfg, ResultsUpdate)


def extract_role_pred(s: str, begin_str: Optional[str],
                      end_str: Optional[str]) -> str:
    """Extract the role prediction from the full prediction string. The role
    prediction may be the substring between the begin and end string.

    Args:
        s (str): Full prediction string.
        begin_str (str): The beginning string of the role
        end_str (str): The ending string of the role.

    Returns:
        str: The extracted role prediction.
    """
    start = 0
    end = len(s)

    if begin_str and re.match(r'\s*', begin_str) is None:
        begin_idx = s.find(begin_str)
        if begin_idx != -1:
            start = begin_idx + len(begin_str)

    if end_str and re.match(r'\s*', end_str) is None:
        # TODO: Support calling tokenizer for the accurate eos token
        # and avoid such hardcode
        end_idx = s.find(end_str, start)
        if end_idx != -1:
            end = end_idx

    return s[start:end]


@TASKS.register_module(force=(__name__ == '__main__'))  # A hack for script run
class OpenICLEvalTask(BaseTask):
    """OpenICL Evaluation Task.

    This task is used to evaluate the metric between predictions and
    references.
    """

    name_prefix = 'OpenICLEval'
    log_subdir = 'logs/eval'
    output_subdir = 'results'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.logger = get_logger()
        self.num_gpus = max(
            c.get('eval_cfg', {}).get('num_gpus', 0)
            for c in sum(self.dataset_cfgs, []))
        self.dump_details = cfg.get('eval', {}).get('runner', {}).get(
            'task', {}).get('dump_details', False)

    def get_command(self, cfg_path, template):
        sys.path.append(os.getcwd())
        script_path = __file__
        python = sys.executable
        command = f'{python} {script_path} {cfg_path}'
        return template.format(task_cmd=command)

    def run(self):
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            for dataset_cfg in dataset_cfgs:
                self.model_cfg = model_cfg
                self.dataset_cfg = dataset_cfg

                # Load Dataset
                self.eval_cfg = self.dataset_cfg.get('eval_cfg')
                self.output_column = dataset_cfg['reader_cfg']['output_column']

                # overwrite postprocessor if the model has specified one
                ds_abbr = dataset_abbr_from_cfg(self.dataset_cfg)
                model_postprocessors = self.model_cfg.get(
                    'pred_postprocessor', {})
                for pattern in model_postprocessors.keys():
                    if fnmatch.fnmatch(ds_abbr, pattern):
                        self.eval_cfg[
                            'pred_postprocessor'] = model_postprocessors[
                            pattern]  # noqa
                        break

                out_path = get_infer_output_path(
                    self.model_cfg, self.dataset_cfg,
                    osp.join(self.work_dir, 'results'))
                if osp.exists(out_path):
                    self.logger.info(f"存在 {out_path}, 跳过Eval")
                    continue
                self._score()

    def _merge_prediction_shards_before_eval(self):
        """在构建数据集与 ICL 评估器之前合并 prediction 分片并落盘。

        避免数据集构建或后续步骤失败时，分片已生成但从未合并为单一文件。
        """
        filename = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext
        pred_shard_files_loaded = []

        if osp.exists(osp.realpath(filename)):
            return filename, pred_shard_files_loaded

        if not osp.exists(osp.realpath(partial_filename)):
            return filename, pred_shard_files_loaded

        origin_preds = {}
        offset = 0
        shard_file = partial_filename
        shard_idx = 0
        while osp.exists(osp.realpath(shard_file)):
            try:
                sub_preds = mmengine.load(shard_file)
                pred_shard_files_loaded.append(shard_file)
                for j in range(len(sub_preds)):
                    origin_preds[str(offset + j)] = sub_preds[str(j)]
                offset += len(sub_preds)
                shard_file = root + f'_{shard_idx + 1}' + ext
                shard_idx += 1
            except Exception as e:
                self.logger.error(
                    f'Error loading prediction file {shard_file}: {e}')
                break

        merged_path = root + ext
        if origin_preds:
            mkdir_or_exist(osp.split(merged_path)[0])
            mmengine.dump(origin_preds, merged_path, ensure_ascii=False,
                          indent=4)
            self.logger.info(
                f'Merged prediction shards into {merged_path} before '
                'dataset/evaluator.')
        return merged_path, pred_shard_files_loaded

    def _score(self):
        filename, pred_shard_files_loaded = (
            self._merge_prediction_shards_before_eval())
        root, ext = osp.splitext(filename)

        if not osp.exists(osp.realpath(filename)):
            self.logger.error(
                f'Task {task_abbr_from_cfg(self.cfg)}: No predictions found.')
            return

        test_set = build_dataset_from_cfg(self.dataset_cfg).test
        # Postprocess dataset if necessary
        if 'dataset_postprocessor' in self.eval_cfg:
            proc = self.eval_cfg['dataset_postprocessor']['type']
            if isinstance(proc, str):
                proc = TEXT_POSTPROCESSORS.get(proc)

            def postprocess(sample):
                s = sample[self.output_column]
                sample[self.output_column] = proc(s)
                return sample

            test_set = test_set.map(postprocess)

        # Load predictions（分片已在上方合并为单一文件）
        # Get sc_size if use Self-Consistency
        sc_size = self.eval_cfg.get('sc_size')
        origin_preds = mmengine.load(filename)
        preds = [origin_preds[str(i)] for i in range(len(origin_preds))]
        if not preds:
            self.logger.error(
                f'Task {task_abbr_from_cfg(self.cfg)}: Empty predictions.')
            return

        pred_dicts = copy.deepcopy(preds)
        preds = {k: [pred.get(k) for pred in preds] for k in preds[0]}

        pred_strs = preds.pop('prediction', None)

        pred_list_flag = pred_strs is not None and isinstance(
            pred_strs[0], list)
        if ('pred_role' in self.eval_cfg
                and 'meta_template' in self.model_cfg
                and not MODELS.get(self.model_cfg['type']).is_api):
            # Create a prompt template for role config parsing
            from opencompass.models.base import LMTemplateParser
            parser = LMTemplateParser(self.model_cfg['meta_template'])
            role = parser.roles[self.eval_cfg['pred_role']]
            if sc_size is not None:
                assert pred_list_flag, (
                    'The prediction for Self-Consistency'
                    'must be list.')
            if pred_list_flag:
                pred_strs = [[
                    extract_role_pred(_pred, role.get('begin', None),
                                      role.get('end', None))
                    for _pred in pred
                ] for pred in pred_strs]
            else:
                pred_strs = [
                    extract_role_pred(pred, role.get('begin', None),
                                      role.get('end', None))
                    for pred in pred_strs
                ]

        # Postprocess predictions if necessary
        if 'pred_postprocessor' in self.eval_cfg:
            kwargs = self.eval_cfg['pred_postprocessor']
            proc = kwargs.pop('type')
            if isinstance(proc, str):
                proc = TEXT_POSTPROCESSORS.get(proc)
            if pred_list_flag:
                pred_strs = [[proc(s, **kwargs) for s in preds]
                             for preds in pred_strs]
            else:
                pred_strs = [proc(s, **kwargs) for s in pred_strs]

                # Get majority voting predictions if use self-consistency
        if sc_size is not None:
            pred_strs = [
                Counter(s).most_common(1)[0][0] for s in pred_strs
            ]
        
        # print(self.eval_cfg)
        # print(self.eval_cfg['evaluator'])
        icl_evaluator = ICL_EVALUATORS.build(self.eval_cfg['evaluator'])
        # need results dir to save other files
        out_path = get_infer_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'results'))
        icl_evaluator._out_dir = osp.splitext(out_path)[
            0]  # strip extension

        preds['predictions'] = pred_strs
        preds['references'] = (test_set[self.output_column]
                               if self.output_column else None)
        preds['test_set'] = test_set
        input_columns = [col for col in test_set.features if col != self.output_column]
        preds['questions'] = [
            "\n".join([str(test_set[col][i]) for col in input_columns])
            for i in range(len(test_set[input_columns[0]]))
        ]

        for feature in test_set.features:
            if feature not in preds:
                preds[feature] = list(test_set[feature])

        handler = ResultsUpdate.get_handler(icl_evaluator)
        if handler:
            preds.update(handler.get_extra_preds(test_set))

        preds = {
            k: preds[k]
            for k in signature(icl_evaluator.score).parameters
            if k in preds
        }

        if handler:
            result = handler.process(
                evaluator=icl_evaluator, preds=preds,
                origin_preds=origin_preds, filename=filename)
        else:
            result = icl_evaluator.score(**preds)
            if 'detail_dict' in result:
                detail_dict = result['detail_dict']
                for key, values in detail_dict.items():
                    for i, value in enumerate(values):
                        origin_preds[str(i)][key] = value
                mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)
        
        if origin_preds and 'info' in test_set.features:
            for idx in range(len(test_set)):
                if str(idx) in origin_preds:
                    origin_preds[str(idx)]['info'] = test_set['info'][idx]
            # 统一保存一次，确保info字段被写入文件
            mmengine.dump(origin_preds, filename, ensure_ascii=False, indent=4)


        # print(f"打分:{result}")

        if self.dump_details:
            try:
                details = result.pop('details', None)
                result['details'] = self.format_details(
                    pred_strs, test_set[self.output_column], details,
                    pred_dicts)
                result['type'] = result['details'].pop('type', None)

                if 'PPL' in str(
                        self.dataset_cfg.infer_cfg.inferencer.type):
                    result['correct_bpb'], result[
                        'incorrect_bpb'] = self.calculate_bpb(pred_dicts)
                else:
                    result['incorrect_bpb'] = result['correct_bpb'] = -1
            except Exception:
                result['incorrect_bpb'] = result['correct_bpb'] = -1
        else:
            if isinstance(result, tuple):
                merged = {}
                for d in result:
                    merged.update(d)
                result = merged
            else:
                result.pop('details', None)

        if 'error' in result:
            self.logger.error(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result["error"]}')
            return
        else:
            result_wo_details = {
                i: result[i]
                for i in result if i != 'details'
            }
            self.logger.info(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result_wo_details}')

        # Save result
        out_path = get_infer_output_path(self.model_cfg, self.dataset_cfg,
                                         osp.join(self.work_dir, 'results'))
        mkdir_or_exist(osp.split(out_path)[0])
        mmengine.dump(result, out_path, ensure_ascii=False, indent=4)

        # 分片合并并打分成功后，删除推理阶段的分片文件；并清理历史误写入的 root_{N}.json
        if pred_shard_files_loaded:
            for p in pred_shard_files_loaded:
                try:
                    if osp.exists(p):
                        os.remove(p)
                except OSError as e:
                    self.logger.warning(f'Failed to remove prediction shard {p}: {e}')
            n = len(pred_shard_files_loaded)
            while osp.exists(root + f'_{n}' + ext):
                stale = root + f'_{n}' + ext
                try:
                    os.remove(stale)
                except OSError as e:
                    self.logger.warning(
                        f'Failed to remove stale prediction file {stale}: {e}')
                    break
                n += 1

    def format_details(self, predictions, references, details, pred_dicts):
        """This function is responsible for formatting prediction details.

        Args:
            predictions (list): The prediction list.
            references (list): The reference list.
            details (list): Contains the 'pred' 'answer' and 'correct' for each
                sample. Such as `[{'pred': '光荣和ωforce',
                'answers': ['光荣和ω-force', '光荣和ωforce'], 'correct': True}]`
            pred_dicts (list): Contains a list of samples with the original
                prompts. Such as
                `[{'origin_prompt': '根据文章回答问题。你的答案应该尽可能3》…………',
                'prediction': ' 光荣和ω-force\n', 'gold': ['光荣和ω-force']}]`

        Returns:
            list: The formatted prediction details.
        """
        results = {}
        for i in range(len(predictions)):
            ppl_flag = False
            result = {}
            origin_prediction = copy.deepcopy(pred_dicts[i])
            origin_prediction.pop('in-context examples', None)
            origin_prediction.pop('prediction', None)
            keys = copy.deepcopy(list(origin_prediction.keys()))
            for key in keys:
                if key.startswith('label:'):
                    ppl_flag = True
                    origin_prediction[key].pop('testing input', None)
                    new_key = key.replace('label: ', '')
                    origin_prediction[new_key] = origin_prediction.pop(key)
            if ppl_flag:
                results['type'] = 'PPL'
                result['origin_prediction'] = origin_prediction
                result['predictions'] = str(predictions[i])
                result['references'] = str(references[i])
                result['correct'] = str(predictions[i]) == str(references[i])
            elif details is not None:
                results['type'] = 'GEN'
                result['prompt'] = origin_prediction['origin_prompt']
                result['origin_prediction'] = pred_dicts[i]['prediction']
                result['predictions'] = details[i]['pred']
                result['references'] = details[i]['answer']
                result['correct'] = details[i]['correct']
            else:
                results['type'] = 'GEN'
                result['prompt'] = origin_prediction['origin_prompt']
                result['origin_prediction'] = pred_dicts[i]['prediction']
                result['predictions'] = str(predictions[i])
                result['references'] = str(references[i])
            results[str(i)] = result
        return results

    def calculate_bpb(self, pred_dicts: List):
        """This function is used to calculate the BPB (Bits Per Byte) for the
        data. The correct BPB is obtained directly from the values in the
        'predictions' file. The incorrect BPB is the average of the remaining
        BPB values for each sample under different labels after subtracting the
        correct BPB. The calculation of BPB (Bits Per Byte) is similar to PPL,
        with the difference that it computes the additional bits needed on
        average, in terms of character length, to encode the true sequence
        based on the predictions. This calculation involves applying a
        weighting factor based on the ratio of words to characters.

        Args:
            pred_dicts (list): Contains a list of samples with each options
                and BPB scores.

        Returns:
            dict: Contains correct and incorrect bpb.
        """
        incorrect_bpb_list = []
        bpb_list = []
        for pred_dict in pred_dicts:
            preds = {
                key: value
                for key, value in pred_dict.items()
                if key.startswith('label: ')
            }
            values = []
            for item in preds.items():
                values.append(item[1])
            bpbs = [value['BPB'] for value in values]
            incorrect_bpb_list.append(
                (sum(bpbs) - min(bpbs)) / (len(bpbs) - 1))
            bpb_list.append(min(bpbs))

        def filters(origins):
            targets = [target for target in origins if not math.isnan(target)]
            return targets

        mean_incorrect = statistics.mean(filters(incorrect_bpb_list))
        mean_correct = statistics.mean(filters(bpb_list))
        return 100 * mean_correct, 100 * mean_incorrect


def parse_args():
    parser = argparse.ArgumentParser(description='Score Calculator')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = OpenICLEvalTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
