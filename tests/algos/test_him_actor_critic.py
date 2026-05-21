from __future__ import annotations

import torch
import torch.nn as nn

from unilab.algos.torch.him_ppo.actor_critic import HIMActorCritic


class _ZeroEstimator(nn.Module):
    num_latent = 0

    def forward(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros((obs_history.shape[0], 3), dtype=obs_history.dtype),
            torch.zeros((obs_history.shape[0], 0), dtype=obs_history.dtype),
        )


class _FirstInputActor(nn.Module):
    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        return actor_input[:, 0:1]


def test_him_actor_critic_uses_latest_history_frame_for_policy_input() -> None:
    actor_critic = HIMActorCritic(
        num_actor_obs=4,
        num_critic_obs=4,
        num_one_step_obs=2,
        num_actions=1,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        estimator={"enc_hidden_dims": [4], "tar_hidden_dims": [4]},
    )
    actor_critic.estimator = _ZeroEstimator()
    actor_critic.actor = _FirstInputActor()

    obs_history = torch.tensor([[1.0, 2.0, 9.0, 8.0]])

    action = actor_critic.act_inference(obs_history)

    torch.testing.assert_close(action, torch.tensor([[9.0]]))
