# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

    # """
    # This function calculates the advantage for Generalized Reward Policy Optimization (GRPO) by focusing solely on 
    # the outcome reward, which is represented as a single scalar reward for each response. The advantage is computed 
    # by normalizing the scores derived from the token-level rewards.

    # Args:
    #     token_level_rewards: `(torch.Tensor)`
    #         A tensor of shape (bs, response_length) representing the rewards at the token level for each response.
    #     eos_mask: `(torch.Tensor)`
    #         A tensor of shape (bs, response_length) that acts as a mask, indicating the presence of an End Of Sequence (EOS) token.
    #     index: `(torch.Tensor)`
    #         A tensor used to group scores by a specific index, typically representing different prompts or contexts.
    #     epsilon: `(float)`
    #         A small constant added for numerical stability during standard deviation calculation.

    # Returns:
    #     advantages: `(torch.Tensor)`
    #         A tensor of shape (bs, response_length) representing the normalized advantages for each token in the response.
    #     Returns: `(torch.Tensor)`
    #         A tensor of shape (bs, response_length) which is identical to the advantages tensor, as the function currently returns the same value twice.
    # """
    # response_length = token_level_rewards.shape[-1]
    # scores = token_level_rewards.sum(dim=-1)  # Sum rewards across the response length to get a single score per response.

    # id2score = defaultdict(list)  # Dictionary to store scores grouped by index.
    # id2mean = {}  # Dictionary to store mean scores for each index.
    # id2std = {}  # Dictionary to store standard deviation of scores for each index.

    # with torch.no_grad():  # Disable gradient tracking for efficiency.
    #     bsz = scores.shape[0]  # Batch size.
    #     for i in range(bsz):
    #         id2score[index[i]].append(scores[i])  # Group scores by index.
    #     for idx in id2score:
    #         if len(id2score[idx]) == 1:
    #             # If only one score is present, set mean to 0 and std to 1 to avoid division by zero.
    #             id2mean[idx] = torch.tensor(0.0)
    #             id2std[idx] = torch.tensor(1.0)
    #         elif len(id2score[idx]) > 1:
    #             # Calculate mean and standard deviation for indices with multiple scores.
    #             id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
    #             id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
    #         else:
    #             raise ValueError(f"no score in prompt index: {idx}")  # Error if no scores are found for an index.
    #     for i in range(bsz):
    #         # Normalize scores by subtracting the mean and dividing by the standard deviation.
    #         scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    #     scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask  # Expand scores to match response length and apply mask.

    # return scores, scores  # Return the normalized scores as both advantages and returns.


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl

    # """
    # This function calculates the policy loss for Proximal Policy Optimization (PPO), a popular reinforcement learning algorithm. 
    # It uses the clipped surrogate objective to ensure stable updates by limiting the change in policy.

    # Args:
    #     old_log_prob: `(torch.Tensor)`
    #         The log probabilities of actions taken by the old policy, with shape (batch_size, response_length).
    #     log_prob: `(torch.Tensor)`
    #         The log probabilities of actions taken by the current policy, with shape (batch_size, response_length).
    #     advantages: `(torch.Tensor)`
    #         The advantage estimates for each action, with shape (batch_size, response_length).
    #     eos_mask: `(torch.Tensor)`
    #         A mask tensor indicating valid positions in the sequence, with shape (batch_size, response_length).
    #     cliprange: (float)
    #         The range for clipping the policy ratio to prevent large updates, as described in the PPO paper.

    # Returns:
    #     pg_loss: `a scalar torch.Tensor`
    #         The computed policy gradient loss, which is a measure of how well the current policy performs compared to the old policy.
    #     pg_clipfrac: (float)
    #         The fraction of the policy gradient loss that was clipped, indicating how often the clipping was active.
    #     ppo_kl: `a scalar torch.Tensor`
    #         The average KL divergence between the old and new policy, providing a measure of how much the policy has changed.
    # """
    # # Calculate the negative approximate KL divergence between the old and new log probabilities.
    # negative_approx_kl = log_prob - old_log_prob
    # # Compute the ratio of the new and old policy probabilities.
    # ratio = torch.exp(negative_approx_kl)
    # # Calculate the average KL divergence using a masked mean to consider only valid positions.
    # ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    # # Compute the policy gradient losses using the advantage estimates and the probability ratio.
    # pg_losses = -advantages * ratio
    # # Apply clipping to the probability ratio to ensure stable updates.
    # pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # # Calculate the final policy gradient loss using the maximum of the clipped and unclipped losses.
    # pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    # # Determine the fraction of the loss that was clipped.
    # pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    # return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
