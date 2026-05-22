from __future__ import annotations

import torch
from tensordict import TensorDict

from unilab.algos.torch.rsl_rl_ppo import FinalObservationAwarePPO
from unilab.training.rsl_rl import (
    HistoryObsDistillationWrapper,
    RslRlVecEnvWrapper,
)


class _FakeActor:
    def update_normalization(self, obs):
        return None

    def reset(self, dones):
        return None


class _FakeCritic:
    def __init__(self, values: torch.Tensor):
        self.values = values
        self.last_obs = None

    def update_normalization(self, obs):
        return None

    def reset(self, dones):
        return None

    def __call__(self, obs):
        self.last_obs = obs
        return self.values


class _FakeTransition:
    def __init__(self):
        self.values = torch.tensor([[10.0], [20.0]])
        self.rewards = None
        self.dones = None

    def clear(self):
        return None


class _FakeStorage:
    def __init__(self):
        self.saved_rewards = None

    def add_transition(self, transition):
        self.saved_rewards = transition.rewards.clone()


def test_final_observation_aware_ppo_bootstraps_from_final_observation():
    algo = object.__new__(FinalObservationAwarePPO)
    algo.actor = _FakeActor()
    algo.critic = _FakeCritic(torch.tensor([[3.0], [4.0]]))
    algo.rnd = None
    algo.gamma = 0.99
    algo.transition = _FakeTransition()
    algo.storage = _FakeStorage()
    algo.device = "cpu"

    obs = TensorDict({"policy": torch.zeros((2, 1))}, batch_size=[2])
    rewards = torch.tensor([1.0, 2.0])
    dones = torch.tensor([True, True])
    final_obs = TensorDict({"policy": torch.tensor([[30.0], [40.0]])}, batch_size=[2])

    algo.process_env_step(
        obs,
        rewards,
        dones,
        {
            "time_outs": torch.tensor([True, False]),
            "time_out_bootstrap_obs": final_obs,
        },
    )

    assert torch.allclose(algo.storage.saved_rewards, torch.tensor([1.0 + 0.99 * 3.0, 2.0]))
    assert torch.equal(algo.critic.last_obs["policy"], final_obs["policy"])


def test_rsl_rl_adapter_outputs_combined_dones_and_time_outs_alias():
    class FakeEnv:
        def __init__(self):
            self.num_envs = 3
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (2,)})()
            self.action_space = type("Space", (), {"shape": (1,)})()
            self.obs_groups_spec = {"obs": 2}
            self.state = type("State", (), {"obs": {"obs": torch.zeros(3, 2).numpy()}})()

        def init_state(self):
            pass

        def reset(self, env_indices):
            del env_indices
            return {"obs": torch.zeros(3, 2).numpy()}, {}

        def step(self, actions):
            del actions
            return type(
                "StepState",
                (),
                {
                    "obs": {"obs": torch.zeros(3, 2).numpy()},
                    "reward": torch.zeros(3).numpy(),
                    "terminated": torch.tensor([True, False, False]).numpy(),
                    "truncated": torch.tensor([False, True, False]).numpy(),
                    "info": {},
                    "final_observation": None,
                },
            )()

    wrapper = RslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="actor")

    _, _, dones, infos = wrapper.step(torch.zeros(3, 1))

    assert torch.equal(dones, torch.tensor([True, True, False]))
    assert torch.equal(infos["time_outs"], torch.tensor([False, True, False]))


def test_history_obs_distillation_wrapper_projects_student_obs_per_frame():
    class FakeEnv:
        def __init__(self):
            self.num_envs = 2
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (10,)})()
            self.action_space = type("Space", (), {"shape": (1,)})()
            self.obs_groups_spec = {"obs": 10}
            self.state = type("State", (), {"obs": {"obs": self._obs()}})()

        def _obs(self):
            return torch.tensor(
                [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                ],
                dtype=torch.float32,
            ).numpy()

        def init_state(self):
            pass

        def reset(self, env_indices):
            del env_indices
            return {"obs": self._obs()}, {}

    wrapper = HistoryObsDistillationWrapper(
        FakeEnv(),
        device="cpu",
        teacher_obs_group="obs",
        student_frame_dim=5,
        student_drop_start=0,
        student_drop_dim=1,
    )

    obs, _ = wrapper.reset()

    assert wrapper.num_obs == 8
    assert wrapper.num_privileged_obs == 10
    assert obs["teacher"].shape == (2, 10)
    assert obs["student"].shape == (2, 8)
    assert torch.equal(
        obs["student"],
        torch.tensor(
            [
                [1, 2, 3, 4, 6, 7, 8, 9],
                [11, 12, 13, 14, 16, 17, 18, 19],
            ],
            dtype=torch.float32,
        ),
    )
