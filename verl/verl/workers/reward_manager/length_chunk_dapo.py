# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from functools import cache
from pickle import FALSE

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from recipe.reward_ours.semantic_repetition import calculate_semantic_repetition
from recipe.reward_ours.repetition import detect_repetition_with_hash
from recipe.reward_ours.math_verify_reward_new import  reward_fn_math_verify_no_think
from verl.workers.reward_manager import register


def get_delta_linear_remove_upper_refined(num_tokens, used_tokens, alpha = 1/14650):
    z_max = num_tokens + 500
    z_min = num_tokens - 500
    if used_tokens > z_max:
        z_score = 1.0
    elif used_tokens < z_min:
        z_score = 1.0 - (z_min - used_tokens) * alpha
    else:
        z_score = 1.0
    z_score = max(0, min(z_score, 1))
    return z_score

def get_delta_linear_remove_upper_refined_additive(num_tokens, used_tokens, alpha = 0.3/9000):
    z_max = num_tokens + 500
    z_min = num_tokens - 500
    if used_tokens > z_max:
        z_score = 0
    elif used_tokens < z_min:
        z_score = (z_min - used_tokens) * alpha
    else:
        z_score = 0
    z_score = max(0, min(z_score, 0.3))
    return z_score

@register("length_chunk_dapo")
class LengthChunkDAPORewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False, reward_config: dict = {"length_penalty_type": None,
        "alpha": 1/14650,
        "skip_length_penalty_for_low_acc_group": False,
        "skip_length_penalty_for_high_acc_group": False,
        "threshold":0.5,
        "skip_right_sample": False,
        "target_length_type": "offline",
        "repetition_penalty": False,
        "repetition_penalty_type": "ngram",
        "truncated_bonus": False,
        "repetition_penalty_max_repetitions_limit": 10
        }
    ):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        if reward_config["length_penalty_type"] != None or data.meta_info.get("validate", False) == False:
            length_penalty_list = []

        # 先统一计算原始score
       
        uid_to_scores = defaultdict(list)
        uid_to_lengths = defaultdict(list)
        cached_results = {}
        all_responses_str = []
        original_score_list = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            all_responses_str.append(response_str)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
            
            cached_results[i] = {
                'score': score,
                'response_str': response_str,
                'valid_response_length': valid_response_length,
                'valid_response_ids': valid_response_ids
            }
            original_score_list.append(score)

            if data.meta_info.get("validate", False) == False:
                uid = data_item.non_tensor_batch['uid']
                uid_to_lengths[uid].append(valid_response_length)
                uid_to_scores[uid].append(score)

        hard_sample_count = 0
        medium_sample_count = 0
        easy_sample_count = 0
        low_05_sample_count = 0
        if data.meta_info.get("validate", False) == False:
            # 计算每个uid组的统计量
            uid_stats = {}
           
            for uid, scores in uid_to_scores.items():
                assert len(scores) == 8
                avg_score = sum(scores) / len(scores)
                avg_length = sum(uid_to_lengths[uid]) / len(uid_to_lengths[uid])
                uid_stats[uid] = {
                    'avg_score': avg_score,
                    'avg_length': avg_length  
                }
                # 统计难度分布 (基于UID/问题维度)
                if avg_score <= 0.25:
                    hard_sample_count += 1
                elif avg_score <= 0.75:
                    medium_sample_count += 1
                else:
                    easy_sample_count += 1
                if avg_score <= 0.5:
                    low_05_sample_count += 1


        # 计算semantic metric
        if data.meta_info.get("validate", False) == False and reward_config['repetition_penalty_type'] == 'semantic':
            _, _, _, high_similarity_flags = calculate_semantic_repetition(all_responses_str, chunk_size=200)

        repetition_penalty_list = []
        # 为每个数据agg reward
        for i in range(len(data)):
            data_item = data[i]

            score = cached_results[i]['score']
            response_str = cached_results[i]['response_str']
            valid_response_length = cached_results[i]['valid_response_length']
            valid_response_ids = cached_results[i]['valid_response_ids']
            
            
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)


            if data.meta_info.get("validate", False) == True:
                val_type = data.meta_info.get("val_type", None)
                budget = data.meta_info.get("budget", None)
                if val_type == "truncated" and budget is not None:
                    truncated_response_length = min(valid_response_length, budget)
                    truncated_response_ids = valid_response_ids[:truncated_response_length]
                    truncated_response_str = self.tokenizer.decode(truncated_response_ids, skip_special_tokens=True)
                    
                    truncated_result = self.compute_score(
                        data_source=data_source,
                        solution_str=truncated_response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
                    score = truncated_result["score"] if isinstance(truncated_result, dict) else truncated_result
            else:
                uid = data_item.non_tensor_batch["uid"]
                stats = uid_stats[uid]
                if reward_config['skip_length_penalty_for_low_acc_group'] and stats['avg_score'] <= reward_config['threshold']:
                    skip_length_penalty = True
                elif reward_config['skip_length_penalty_for_high_acc_group'] and stats['avg_score'] >= reward_config['threshold']:
                    skip_length_penalty = True
                else:
                    skip_length_penalty = False
                
                
                if reward_config['length_penalty_type'] == 'remove_upper_refined':
                    num_tokens = abs(data.non_tensor_batch["reward_model"][i]['num_tokens'])
                    delta_score = get_delta_linear_remove_upper_refined(num_tokens, valid_response_length, alpha=reward_config["alpha"])
                    length_penalty_list.append(delta_score)
                    if not skip_length_penalty:
                        score = score * delta_score
                elif reward_config['length_penalty_type'] == 'remove_upper_refined_plus':
                    if reward_config["target_length_type"] == 'offline':
                        num_tokens = abs(data.non_tensor_batch["reward_model"][i]['num_tokens'])
                    elif reward_config["target_length_type"] == 'max':
                        num_tokens = 8192
                        print(f"[INFO] MAX: use default max value {num_tokens} as target length")
                    delta_score = get_delta_linear_remove_upper_refined_additive(num_tokens, valid_response_length, alpha=reward_config["alpha"])
                    length_penalty_list.append(delta_score)
                    if reward_config["skip_right_sample"]:
                        if score == 0:
                            if not skip_length_penalty:
                                score = score - delta_score
                            
                            slopes = data_item.non_tensor_batch["chunk_gt_slopes"]
                            if slopes > 0:
                                score += 0.1
                            print("[INFO] slopes: ", slopes, "final_score: ", score)

                        elif reward_config["truncated_bonus"] == True:
                            if valid_response_length > 4000:
                                
                                truncated_response_ids = valid_response_ids[:4000]
                                truncated_response_str = self.tokenizer.decode(truncated_response_ids, skip_special_tokens=True)
                                
                                truncated_result, predict_answers = reward_fn_math_verify_no_think(truncated_response_str, ground_truth)
                                print(f"[INFO] truncated_bonus: {truncated_result}, {predict_answers}, score + 0.5")
                                if truncated_result == False and predict_answers[0] is not None:
                                    score = score + 0.5
                        

                    elif not skip_length_penalty:
                        score = score - delta_score
                else:
                    assert False, f"no application for {reward_config['length_penalty_type']}"
                

                # 计算repetition
                ngram_repetition_penalty = detect_repetition_with_hash(response_str, window_size=10, max_repetitions_limit=reward_config["repetition_penalty_max_repetitions_limit"])
                repetition_penalty_list.append(ngram_repetition_penalty)

                if reward_config['repetition_penalty_type']:
                    repetition_penalty_value = ngram_repetition_penalty if reward_config["repetition_penalty_type"] == 'ngram' else - high_similarity_flags[i]

                    if reward_config['length_penalty_type'] == 'remove_upper_refined':
                        if score > 0: # 对于acc>0的情况重复时
                            if score == 1: #长度达标（score=1表示答案正确且长度达标）
                                repetition_penalty_value = repetition_penalty_value
                            else: #长度没达标，给0.1的缓冲（确保score不会降到0.1以下）
                                # repetition_penalty是负数（-1），-score+0.1也是负数
                                # 使用max取较大的值（即较小的惩罚），确保score不会降到0.1以下
                                repetition_penalty_value = max(repetition_penalty_value, -score)
                        elif score == 0: #错误的例子不给repetition penalty
                            repetition_penalty_value = 0
                
                
                    elif reward_config['length_penalty_type'] == 'remove_upper_refined_plus':
                        if reward_config["repetition_penalty_type"] == 'ngram':
                            repetition_penalty_value *= 0.6
                        elif reward_config["repetition_penalty_type"] == 'semantic':
                            repetition_penalty_value *= 0.1
                        else:
                            assert False, reward_config["repetition_penalty_type"]
                    else:
                        assert False, f"No repetition application for {reward_config['length_penalty_type']}"
                    score += repetition_penalty_value
                      
                print(f"[INFO] repetition_penalty: {repetition_penalty_value}, length_penalty: {delta_score}, original_score: {original_score_list[i]}, final_score: {score}")


            reward = score

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        
        reward_extra_info.update({"original_scores": torch.tensor(original_score_list, dtype=torch.float32)})

        
        if len(repetition_penalty_list) > 0:
            repetition_penalty_rate = sum([1 for i in range(len(repetition_penalty_list)) if repetition_penalty_list[i] != 0]) / len(repetition_penalty_list)
        else:
            repetition_penalty_rate = 0.0
        
        reward_extra_info.update({"repetition_penalty_rate" : repetition_penalty_rate})
        if reward_config['length_penalty_type'] != None:
            reward_extra_info.update({"length_penalty_list": torch.tensor(length_penalty_list, dtype=torch.float32)})
        
        reward_extra_info.update(
            {"hard_sample_count":  hard_sample_count, 
            "medium_sample_count": medium_sample_count,
            "easy_sample_count" :easy_sample_count,
            "low_05_sample_count": low_05_sample_count})

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
