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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    # compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
    _compute_response_info
)
from recipe.length_src.ray_trainer import AdvantageEstimator, RayPPOTrainer, apply_kl_penalty, compute_advantage, compute_response_mask
from verl.utils.debug import marked_timer
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import os
from typing import Dict, Any
from verl.trainer.ppo.metric_utils import process_validation_metrics


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Initialize best model tracking
        self.best_avg_score = float('-inf')
        self.best_four_sets = float('-inf')
        self.best_four_model_step = 0
        self.best_model_step = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._load_best_model_info()
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps,
                            initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in (
                    self.config.trainer.profile_steps or [])
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids",
                                    "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids",
                                    "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                print("original input",  self.tokenizer.decode(
                    gen_batch.batch["input_ids"][0], skip_special_tokens=True))

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    print("output shape: ", gen_batch_output.batch["responses"].shape, "output", self.tokenizer.decode(
                        gen_batch_output.batch["responses"][0], skip_special_tokens=True))
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(
                                gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(
                                dim=-1)

                            new_batch.pop(batch_keys=list(
                                gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(
                                new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            # main change
                            reward_config = {
                                "length_penalty_type": self.config.reward.get("length_penalty_type", None),
                                "alpha": self.config.reward.get("alpha", 1/14650),
                                "skip_length_penalty_for_low_acc_group": self.config.reward.get("skip_length_penalty_for_low_acc_group", False),
                                "skip_length_penalty_for_high_acc_group": self.config.reward.get("skip_length_penalty_for_high_acc_group", False),
                                "threshold_low": self.config.reward.get("threshold_low", 0.25),
                                "threshold_high": self.config.reward.get("threshold_high", 0.75),
                                "skip_right_sample": self.config.reward.get("skip_right_sample", False),
                                "target_length_type": self.config.reward.get("target_length_type", "offline"),
                                "repetition_penalty": self.config.reward.get("repetition_penalty", False),
                                "repetition_penalty_type": self.config.reward.get("repetition_penalty_type", "ngram"),
                                "truncated_bonus": self.config.reward.get("truncated_bonus", False),
                                "repetition_penalty_max_repetitions_limit": self.config.reward.get("repetition_penalty_max_repetitions_limit", 10),
                                "extra_tokens": self.config.reward.get("extra_tokens", 0)
                            }

                            reward_result = self.reward_fn(
                                new_batch, return_dict=True, reward_config=reward_config)
                            reward_tensor = reward_result["reward_tensor"]
                            length_tensor = reward_result["length_tensor"]
                            repetition_tensor = reward_result["repetition_tensor"]
                            correctness_tensor = reward_result["correctness_tensor"]
                            reward_extra_infos_dict = reward_result.get(
                                "reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor
                        new_batch.batch["token_level_scores_length"] = length_tensor
                        new_batch.batch["token_level_scores_repetition"] = repetition_tensor
                        new_batch.batch["token_level_scores_correctness"] = correctness_tensor

                        metrics.update(
                            {"critic/easy_sample_count (>=0.75)": reward_extra_infos_dict.get("easy_sample_count", 0)})
                        metrics.update(
                            {"critic/medium_sample_count (0.25-0.75)": reward_extra_infos_dict.get("medium_sample_count", 0)})
                        metrics.update(
                            {"critic/hard_sample_count (<=0.25)": reward_extra_infos_dict.get("hard_sample_count", 0)})
                        metrics.update(
                            {"critic/low_05_sample_count (<=0.5)": reward_extra_infos_dict.get("low_05_sample_count", 0)})
                        original_scores = reward_extra_infos_dict.get(
                            "original_scores", reward_tensor.sum(-1))
                        metrics.update(
                            {"critic/repetition_penalty_rate": reward_extra_infos_dict.get("repetition_penalty_rate", 0)})
                        # These are summary scalars (not per-trajectory arrays). If we keep them in
                        # `reward_extra_infos_dict`, the later `np.array(v)` will create 0-d arrays and
                        # crash when `DataProto.reorder()` tries to index them.
                        for _k in [
                            "easy_sample_count",
                            "medium_sample_count",
                            "hard_sample_count",
                            "low_05_sample_count",
                            "repetition_penalty_rate",
                        ]:
                            reward_extra_infos_dict.pop(_k, None)
                        new_batch.batch["original_scores"] = original_scores
                        lp = reward_extra_infos_dict.get(
                            "length_penalty_list", None)
                        if lp is not None:
                            new_batch.batch["length_penalty_list"] = lp

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            # TODO: This will be cleared if we use multiple genenration batches
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch["token_level_rewards"].sum(
                                dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch["token_level_scores"].sum(
                                dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(
                                metric_vals)

                        kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items(
                        ) if std > 0 or len(prompt_uid2metric_vals[uid]) == 1]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat(
                            [batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                continue
                            else:
                                raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." +
                                                 " You could also try set max_num_gen_batches=0 to enable endless trials.")
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * \
                                self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(
                            batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(
                                batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics

                            # Check and save best model
                            self._check_and_save_best_model(val_metrics)

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(
                    batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(
                    batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(
                    batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                if self.config.trainer.get("save_train_data", False):
                    # Save training data for each step
                    self._save_training_data(batch, self.global_steps)

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

    def _validate(self):
        """对于truncated类型：生成一次完整回答，然后用不同budget截断评估"""
        metric_dict = {}
        budgets = self.config.trainer.val_budgets

        # 生成一次完整的回答
        generated_batches = []
        sample_inputs = []
        sample_outputs = []

        for test_data in tqdm(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)
            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # 对于truncated类型，不需要添加budget prompt，因为我们会用不同budget截断
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(
                ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "max_tokens": budgets[-1],  # ,32000,  # 生成完整回答
                "validate": True,
                "val_type": "truncated",
            }

            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                    test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                    test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(
                ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            # test_batch.meta_info["budget"] = 32000
            test_batch.meta_info["val_type"] = "truncated"

            generated_batches.append(test_batch)

        # 对每个budget进行截断评估
        reward_config = {
            "length_penalty_type": self.config.reward.get("length_penalty_type", None),
            "alpha": self.config.reward.get("alpha", 1/14650),
            "skip_length_penalty_for_low_acc_group": self.config.reward.get("skip_length_penalty_for_low_acc_group", False),
            "skip_length_penalty_for_high_acc_group": self.config.reward.get("skip_length_penalty_for_high_acc_group", False),
            "threshold": self.config.reward.get("threshold", 0.5),
            "skip_right_sample": self.config.reward.get("skip_right_sample", False),
            "target_length_type": self.config.reward.get("target_length_type", "offline"),
            "repetition_penalty": self.config.reward.get("repetition_penalty", False),
            "repetition_penalty_type": self.config.reward.get("repetition_penalty_type", "ngram")
        }
        for budget in budgets:
            # 为每个budget创建独立的metric_dict
            budget_metric_dict = {}
            data_source_lst = []
            reward_extra_infos_dict: dict[str, list] = defaultdict(list)

            sample_scores = []
            sample_turns = []
            sample_response_lengths = []

            for test_batch in tqdm(generated_batches):
                # 为当前budget设置截断信息
                test_batch.meta_info["budget"] = budget

                # Safely update reward_model num_tokens
                if "reward_model" in test_batch.non_tensor_batch:
                    reward_model = test_batch.non_tensor_batch["reward_model"]
                    if isinstance(reward_model, dict):
                        reward_model['num_tokens'] = budget
                    elif hasattr(reward_model, '__iter__') and len(reward_model) > 0:
                        # If it's a numpy array of dictionaries, update all elements
                        if hasattr(reward_model, 'shape') and isinstance(reward_model[0], dict):
                            # It's a numpy array of dictionaries
                            for i in range(len(reward_model)):
                                reward_model[i]['num_tokens'] = budget
                        elif isinstance(reward_model[0], dict):
                            # It's a regular list of dictionaries
                            for item in reward_model:
                                item['num_tokens'] = budget
                        else:
                            # Create a new dict structure
                            test_batch.non_tensor_batch["reward_model"] = [
                                {'num_tokens': budget}]

                # evaluate using reward_function
                result = self.val_reward_fn(
                    test_batch, return_dict=True, reward_config=reward_config)

                response_lengths = result["reward_extra_info"]["response_lengths"]
                sample_response_lengths.extend(response_lengths.cpu().tolist())

                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                del result["reward_extra_info"]

                reward_extra_infos_dict["reward"].extend(scores)
                print(
                    f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
                if "reward_extra_info" in result:
                    for key, lst in result["reward_extra_info"].items():
                        reward_extra_infos_dict[key].extend(lst)
                        print(
                            f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

                # collect num_turns of each prompt
                if "__num_turns__" in test_batch.non_tensor_batch:
                    sample_turns.append(
                        test_batch.non_tensor_batch["__num_turns__"])

                data_source_lst.append(test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]))

            self._maybe_log_val_generations(
                inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

            # dump generations
            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir and budget == budgets[-1]:
                self._dump_generations(
                    inputs=sample_inputs,
                    outputs=sample_outputs,
                    scores=sample_scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=val_data_dir,
                    budget=budget,
                    sample_response_lengths=sample_response_lengths
                )

            for key_info, lst in reward_extra_infos_dict.items():
                assert len(lst) == 0 or len(lst) == len(
                    sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

            data_sources = np.concatenate(data_source_lst, axis=0)
            print("responses shape",
                  test_output_gen_batch.batch["responses"].shape, "\nbudget:", budget)

            data_src2var2metric2val = process_validation_metrics(
                data_sources, sample_inputs, reward_extra_infos_dict)

            for data_source, var2metric2val in data_src2var2metric2val.items():
                core_var = "acc" if "acc" in var2metric2val else "reward"
                for var_name, metric2val in var2metric2val.items():
                    n_max = max([int(name.split("@")[-1].split("/")[0])
                                for name in metric2val.keys()])
                    for metric_name, metric_val in metric2val.items():
                        if (
                            (var_name == core_var)
                            and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                            and (f"@{n_max}" in metric_name)
                        ):
                            metric_sec = "val-core"
                        else:
                            metric_sec = "val-aux"
                        pfx = f"{metric_sec}/{budget}/{data_source}/{var_name}/{metric_name}"
                        budget_metric_dict[pfx] = metric_val

            budget_metric_dict[f'avg_score/{budget}'] = np.mean(
                [budget_metric_dict[key] for key in budget_metric_dict if 'mean@' in key and str(budget) in key and 'polaris' not in key])
            budget_metric_dict[f'avg_score/{budget}/four_sets'] = np.mean([budget_metric_dict[key] for key in budget_metric_dict if 'mean@' in key and str(
                budget) in key and ('aime' in key or 'olympiad_bench' in key or 'math' in key or 'amc' in key)])

            if len(sample_turns) > 0:
                sample_turns = np.concatenate(sample_turns)
                budget_metric_dict["val-aux/num_turns/min"] = sample_turns.min()
                budget_metric_dict["val-aux/num_turns/max"] = sample_turns.max()
                budget_metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

            metric_dict.update(budget_metric_dict)

        return metric_dict

    # def _check_and_save_best_model(self, val_metrics):
    #     """
    #     Check if current validation metrics indicate a new best model and save if so.
    #     """
    #     # Get the score from validation metrics using the last validation budget
    #     avg_key = f"avg_score/{self.config.trainer.val_budgets[-1]}"
    #     four_sets_key = f"avg_score/{self.config.trainer.val_budgets[-1]}/four_sets"

    #     # Check avg_key
    #     if avg_key in val_metrics:
    #         current_avg = val_metrics[avg_key]
    #         # Check if this is a new best score
    #         if current_avg > self.best_avg_score:
    #             print(f"New best avg_{self.config.trainer.val_budgets[-1]} score: {current_avg:.4f} (previous best: {self.best_avg_score:.4f})")
    #             self.best_avg_score = current_avg
    #             self.best_model_step = self.global_steps

    #             # Save the best model
    #             self._save_best_checkpoint()
    #         else:
    #             print(f"Current avg_{self.config.trainer.val_budgets[-1]} score: {current_avg:.4f} (best: {self.best_avg_score:.4f} at step {self.best_model_step})")
    #     else:
    #         print(f"Warning: {avg_key} not found in validation metrics. Available keys: {list(val_metrics.keys())}")

    #     # Check four_sets_key
    #     if four_sets_key in val_metrics:
    #         current_four_sets = val_metrics[four_sets_key]
    #         if current_four_sets > self.best_four_sets:
    #             print(f"New best {four_sets_key} score: {current_four_sets:.4f} (previous best: {self.best_four_sets:.4f})")
    #             self.best_four_sets = current_four_sets
    #             self.best_four_model_step = self.global_steps

    #             self._save_best_checkpoint(suffix="_four_sets")
    #         else:
    #             print(f"Current {four_sets_key} score: {current_four_sets:.4f} (best: {self.best_four_sets:.4f} at step {self.best_four_model_step})")
    #     else:
    #         print(f"Warning: {four_sets_key} not found in validation metrics. Available keys: {list(val_metrics.keys())}")

    # def _load_best_model_info(self):
    #     """
    #     Load best model information from saved checkpoints.
    #     """
    #     import json
    #     import os

    #     # Load best_model info
    #     best_model_dir = os.path.join(self.config.trainer.default_local_dir, "best_model")
    #     best_info_path = os.path.join(best_model_dir, "best_model_info.json")
    #     if os.path.exists(best_info_path):
    #         try:
    #             with open(best_info_path, "r") as f:
    #                 best_model_info = json.load(f)
    #                 avg_key = f"best_avg_{self.config.trainer.val_budgets[-1]}_score"
    #                 if avg_key in best_model_info:
    #                     self.best_avg_score = best_model_info[avg_key]
    #                     self.best_model_step = best_model_info.get("best_model_step", 0)
    #                     print(f"Loaded best avg_{self.config.trainer.val_budgets[-1]} score: {self.best_avg_score:.4f} from step {self.best_model_step}")
    #         except Exception as e:
    #             print(f"Warning: Failed to load best_model info from {best_info_path}: {e}")

    #     # Load best_model_four_sets info
    #     best_model_four_sets_dir = os.path.join(self.config.trainer.default_local_dir, "best_model_four_sets")
    #     best_four_sets_info_path = os.path.join(best_model_four_sets_dir, "best_model_info.json")
    #     if os.path.exists(best_four_sets_info_path):
    #         try:
    #             with open(best_four_sets_info_path, "r") as f:
    #                 best_four_sets_info = json.load(f)
    #                 four_sets_key = "best_four_sets_score"
    #                 if four_sets_key in best_four_sets_info:
    #                     self.best_four_sets = best_four_sets_info[four_sets_key]
    #                     self.best_four_model_step = best_four_sets_info.get("best_four_model_step", 0)
    #                     print(f"Loaded best four_sets score: {self.best_four_sets:.4f} from step {self.best_four_model_step}")
    #         except Exception as e:
    #             print(f"Warning: Failed to load best_model_four_sets info from {best_four_sets_info_path}: {e}")

    # def _save_best_checkpoint(self, suffix=None):
    #     """
    #     Save the current model as the best model checkpoint.
    #     """
    #     from verl.utils.fs import local_mkdir_safe
    #     import os
    #     import shutil

    #     # Handle suffix: if None, use empty string to avoid "best_modelNone"
    #     suffix_str = suffix if suffix is not None else ""

    #     # Create best model directory
    #     best_model_dir = os.path.join(self.config.trainer.default_local_dir, f"best_model{suffix_str}")
    #     local_mkdir_safe(best_model_dir)

    #     print(f"Saving best model to: {best_model_dir}")

    #     # Save actor
    #     actor_best_path = os.path.join(best_model_dir, "actor")
    #     actor_remote_path = (
    #         None
    #         if self.config.trainer.default_hdfs_dir is None
    #         else os.path.join(self.config.trainer.default_hdfs_dir, f"best_model{suffix_str}", "actor")
    #     )

    #     # Save best model without max_ckpt_to_keep limit to avoid deletion
    #     self.actor_rollout_wg.save_checkpoint(
    #         actor_best_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=None
    #     )

    #     # Save critic if used
    #     if self.use_critic:
    #         critic_best_path = os.path.join(best_model_dir, "critic")
    #         critic_remote_path = (
    #             None
    #             if self.config.trainer.default_hdfs_dir is None
    #             else os.path.join(self.config.trainer.default_hdfs_dir, f"best_model{suffix_str}", "critic")
    #         )
    #         self.critic_wg.save_checkpoint(
    #             critic_best_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=None
    #         )

    #     # Save best model info
    #     if suffix is None:
    #         best_model_info = {
    #             f"best_avg_{self.config.trainer.val_budgets[-1]}_score": self.best_avg_score,
    #             "best_model_step": self.best_model_step,
    #             "global_steps": self.global_steps
    #         }
    #     else:
    #         best_model_info = {
    #             "best_four_sets_score": self.best_four_sets,
    #             "best_four_model_step": self.best_four_model_step,
    #             "global_steps": self.global_steps
    #         }

    #     best_info_path = os.path.join(best_model_dir, "best_model_info.json")
    #     import json
    #     with open(best_info_path, "w") as f:
    #         json.dump(best_model_info, f, indent=2)

    #     if suffix is None:
    #         print(f"Best model saved at step {self.global_steps} with avg_{self.config.trainer.val_budgets[-1]} score: {self.best_avg_score:.4f}")
    #     else:
    #         four_sets_key = f"avg_score/{self.config.trainer.val_budgets[-1]}/four_sets"
    #         print(f"Best model{suffix_str} saved at step {self.global_steps} with {four_sets_key} score: {self.best_four_sets:.4f}")

    #     # save dataloader
    #     dataloader_local_path = os.path.join(best_model_dir, "data.pt")
    #     dataloader_state_dict = self.train_dataloader.state_dict()
    #     torch.save(dataloader_state_dict, dataloader_local_path)
    #     print(f"Dataloader saved to: {dataloader_local_path}")

    def _save_training_data(self, batch, step):
        """
        Save training data for each step including prompts, responses, finish reasons, and accuracy.
        """
        if batch is None:
            return

        from verl.utils.fs import local_mkdir_safe
        import json

        # Create training data directory
        training_data_dir = os.path.join(
            self.config.trainer.default_local_dir, "training_data")
        local_mkdir_safe(training_data_dir)

        # Save data for this step
        step_data_file = os.path.join(
            training_data_dir, f"step_{step}_traindata.jsonl")

        # Extract data from batch
        prompts = self.tokenizer.batch_decode(
            batch.batch["prompts"], skip_special_tokens=True)
        responses = self.tokenizer.batch_decode(
            batch.batch["responses"], skip_special_tokens=True)
        original_scores = batch.batch["original_scores"].tolist()

        lp = batch.non_tensor_batch.get("length_penalty_list", None)
        if lp is not None:
            length_penalty = lp.tolist() if hasattr(lp, 'tolist') else list(lp)
        else:
            length_penalty = [None] * len(prompts)

        final_scores = batch.batch["token_level_scores"].sum(dim=-1).tolist()

        final_scores = batch.batch["token_level_scores"].sum(dim=-1).tolist()
        # Get finish reasons if available
        finish_reasons = []
        if "finish_reasons" in batch.non_tensor_batch:
            finish_reasons = batch.non_tensor_batch["finish_reasons"].tolist()
        else:
            # If finish_reasons not available, try to infer from overlong_mask
            if "overlong_mask" in batch.batch:
                overlong_mask = batch.batch["overlong_mask"].tolist()
                finish_reasons = [
                    "length" if not mask else "stop" for mask in overlong_mask]
            else:
                finish_reasons = ["unknown"] * len(prompts)

        # Get UIDs
        uids = batch.non_tensor_batch.get(
            "uid", [f"unknown_{i}" for i in range(len(prompts))])

        # Group data by UID
        uid_data = {}
        for i in range(len(prompts)):
            uid = str(uids[i]) if i < len(uids) else f"unknown_{i}"

            if uid not in uid_data:
                uid_data[uid] = {
                    "step": step,
                    "uid": uid,
                    "input": prompts[i],  # 单个值
                    # 单个值
                    "ground_truth": batch[i].non_tensor_batch["reward_model"]["ground_truth"],
                    "responses": [],  # 列表
                    "finish_reasons": [],  # 列表
                    "accuracies": [],  # 列表
                    "length_penalty": [],
                    "final_scores": [],
                }

            # 添加到列表中
            uid_data[uid]["responses"].append(responses[i])
            uid_data[uid]["finish_reasons"].append(finish_reasons[i])
            uid_data[uid]["accuracies"].append(original_scores[i])
            uid_data[uid]["length_penalty"].append(length_penalty[i])
            uid_data[uid]["final_scores"].append(final_scores[i])
        # Save grouped data
        with open(step_data_file, "w", encoding="utf-8") as f:
            for uid, data in uid_data.items():
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print(f"Training data saved for step {step}: {step_data_file}")


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    sequence_length = batch.batch["token_level_scores_length"].sum(-1)
    sequence_repetition = batch.batch["token_level_scores_repetition"].sum(-1)

    length_penalty = batch.non_tensor_batch.get("length_penalty_list", None)
    if length_penalty is not None:
        if isinstance(length_penalty, np.ndarray):
            length_penalty = torch.from_numpy(length_penalty)
        elif not isinstance(length_penalty, torch.Tensor):
            length_penalty = torch.tensor(length_penalty)

    # overlong_mask_expanded = batch.batch["overlong_mask"].unsqueeze(1).expand_as(batch.batch["token_level_scores"])
    # filter_score = batch.batch["token_level_scores"] * overlong_mask_expanded
    # filter_score = filter_score.sum(-1)
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:,
                                                :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # 计算正确case和错误case的平均长度
    original_scores = batch.batch["original_scores"]
    success_value = 1
    fail_value = 0

    # 创建正确和错误的mask
    correct_mask = (original_scores == success_value)
    incorrect_mask = (original_scores == fail_value)

    # 计算正确case的平均长度
    if correct_mask.any():
        correct_lengths = response_length[correct_mask]
        num_correct = correct_mask.sum().item()
        avg_length_correct = torch.sum(
            correct_lengths).detach().item() / num_correct

    else:
        avg_length_correct = 0.0
        num_correct = 0

    # 计算错误case的平均长度
    if incorrect_mask.any():
        incorrect_lengths = response_length[incorrect_mask]
        num_incorrect = incorrect_mask.sum().item()
        avg_length_incorrect = torch.sum(
            incorrect_lengths).detach().item() / num_incorrect

    else:
        avg_length_incorrect = 0.0
        num_incorrect = 0

    metrics = {
        # score
        # "critic/semantic_similarity/mean": torch.mean(batch.batch["semantic_similarities"]).detach().item(),
        # "critic/semantic_similarity/max": torch.max(batch.batch["semantic_similarities"]).detach().item(),
        # "critic/semantic_similarity/min": torch.min (batch.batch["semantic_similarities"]).detach().item(),
        "critic/acc/mean": torch.mean(batch.batch["original_scores"]).detach().item(),

        # "critic/filter_score/mean": torch.mean(filter_score).detach().item(),
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # length
        "critic/length/mean": torch.mean(sequence_length).detach().item(),
        "critic/length/max": torch.max(sequence_length).detach().item(),
        "critic/length/min": torch.min(sequence_length).detach().item(),
        # repetition
        "critic/repetition/mean": torch.mean(sequence_repetition).detach().item(),
        "critic/repetition/max": torch.max(sequence_repetition).detach().item(),
        "critic/repetition/min": torch.min(sequence_repetition).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),

        # # overlong
        # "critic/overlong/sum": torch.sum(~batch.batch["overlong_mask"]).detach().item(),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length by correctness
        "response_length/correct/mean": avg_length_correct,
        "response_length/correct/count": num_correct,
        "response_length/incorrect/mean": avg_length_incorrect,
        "response_length/incorrect/count": num_incorrect,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    if length_penalty is not None:
        metrics["critic/length_penalty/mean"] = torch.mean(
            length_penalty).detach().item()
        metrics["critic/length_penalty/max"] = torch.max(
            length_penalty).detach().item()
        metrics["critic/length_penalty/min"] = torch.min(
            length_penalty).detach().item()

    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    return metrics
