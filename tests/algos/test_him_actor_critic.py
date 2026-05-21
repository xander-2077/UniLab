from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from unilab.algos.torch.him_ppo.actor_critic import HIMActorCritic
from unilab.algos.torch.him_ppo.runner import HIMOnPolicyRunner, _HIMLogger


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


class _ScalarWriter:
    def __init__(self) -> None:
        self.scalars: dict[str, tuple[float, int]] = {}

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars[tag] = (float(value), int(step))

    def flush(self) -> None:
        pass


def test_him_runner_logs_rsl_rl_style_tensorboard_scalars() -> None:
    writer = _ScalarWriter()
    runner = object.__new__(HIMOnPolicyRunner)
    runner._writer = writer
    runner.device = "cpu"
    runner.logger = _HIMLogger()
    runner.logger.rewbuffer.extend([1.0, 3.0])
    runner.logger.lenbuffer.extend([10.0, 14.0])
    runner.logger.ep_extras.extend(
        [
            {"reward/height": torch.tensor(0.5), "success": 1.0},
            {"reward/height": torch.tensor(1.5), "success": 0.0},
        ]
    )
    runner.num_steps_per_env = 4
    runner.env = SimpleNamespace(num_envs=2)
    runner.alg = SimpleNamespace(learning_rate=1.0e-3)
    runner.actor_critic = SimpleNamespace(std=torch.tensor([0.2, 0.4]))

    runner._write_tensorboard_scalars(
        it=7,
        collect_time=0.25,
        learn_time=0.75,
        loss_dict={
            "value": 1.0,
            "surrogate": 2.0,
            "entropy": 3.0,
            "estimation": 4.0,
            "swap": 5.0,
        },
    )

    for tag in (
        "reward/height",
        "Episode/success",
        "Loss/value",
        "Loss/surrogate",
        "Loss/entropy",
        "Loss/estimation",
        "Loss/swap",
        "Loss/learning_rate",
        "Policy/mean_std",
        "Perf/total_fps",
        "Perf/collection_time",
        "Perf/learning_time",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Train/mean_reward/time",
        "Train/mean_episode_length/time",
    ):
        assert tag in writer.scalars

    assert writer.scalars["reward/height"] == (1.0, 7)
    assert writer.scalars["Episode/success"] == (0.5, 7)
    assert writer.scalars["Train/mean_reward"] == (2.0, 7)
    assert runner.logger.ep_extras == []
