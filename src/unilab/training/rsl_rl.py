"""RSL-RL-specific training helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from unilab.base.final_observation import resolve_terminal_observation_contract
from unilab.base.np_env import NpEnvState
from unilab.utils.tensor import to_numpy, to_torch


def get_policy_obs_dims(obs_groups_spec: dict[str, int]) -> tuple[int, int]:
    """Return ``(actor_obs_dim, flat_policy_obs_dim)`` for RSL-RL policies."""
    actor_obs_dim = int(obs_groups_spec.get("obs", 0))
    flat_policy_obs_dim = int(
        sum(dim for group_name, dim in obs_groups_spec.items() if group_name != "critic")
    )
    return actor_obs_dim, flat_policy_obs_dim or actor_obs_dim


def normalize_ppo_train_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Map UniLab PPO owner config to the current RSL-RL schema."""
    normalized = deepcopy(train_cfg)
    algorithm_cfg = normalized.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        for key in (
            "target_kl_stop",
            "adaptive_kl_beta",
            "adaptive_lr_growth",
            "adaptive_lr_decay",
            "adaptive_lr_update_interval",
            "metrics_interval",
            "finite_check_interval",
            "enable_compile",
            "warmup_strict_iters",
            "warmup_metrics_interval",
            "warmup_finite_check_interval",
            "disable_finite_checks",
        ):
            algorithm_cfg.pop(key, None)

    if "actor" in normalized and "critic" in normalized:
        return normalized

    policy_cfg = normalized.pop("policy", None)
    if not isinstance(policy_cfg, dict):
        return normalized

    actor_class_name = policy_cfg.get("actor_class_name", "rsl_rl.models.MLPModel")
    critic_class_name = policy_cfg.get("critic_class_name", "rsl_rl.models.MLPModel")
    actor_hidden_dims = policy_cfg.get("actor_hidden_dims", [512, 256, 128])
    critic_hidden_dims = policy_cfg.get("critic_hidden_dims", actor_hidden_dims)
    activation = policy_cfg.get("activation", "elu")
    init_noise_std = float(policy_cfg.get("init_noise_std", 1.0))
    obs_normalization = bool(normalized.get("empirical_normalization", False))

    normalized["actor"] = {
        "class_name": actor_class_name,
        "hidden_dims": actor_hidden_dims,
        "activation": activation,
        "obs_normalization": obs_normalization,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": init_noise_std,
            "std_type": "scalar",
        },
    }
    if "linvel_estimator" in policy_cfg:
        normalized["actor"]["linvel_estimator"] = deepcopy(policy_cfg["linvel_estimator"])

    normalized["critic"] = {
        "class_name": critic_class_name,
        "hidden_dims": critic_hidden_dims,
        "activation": activation,
        "obs_normalization": obs_normalization,
    }

    obs_groups = normalized.get("obs_groups")
    if isinstance(obs_groups, dict) and "actor" not in obs_groups and "default" in obs_groups:
        default_groups = obs_groups.pop("default")
        if isinstance(default_groups, list) and default_groups:
            obs_groups["actor"] = list(default_groups)

    return normalized


class RslRlVecEnvWrapper:
    """Adapter from UniLab's env contract to the RSL-RL VecEnv contract."""

    def __init__(
        self,
        env: Any,
        device: str = "cpu",
        policy_obs_mode: str = "flat",
    ) -> None:
        if policy_obs_mode == "auto":
            policy_obs_mode = "flat"
        if policy_obs_mode not in {"actor", "flat"}:
            raise ValueError(
                f"Unsupported policy_obs_mode={policy_obs_mode!r}; expected 'actor' or 'flat'."
            )

        self.env = env
        self.cfg = env.cfg
        self.device = device
        self.policy_obs_mode = policy_obs_mode
        self.num_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._actor_obs_dim, self._flat_obs_dim = get_policy_obs_dims(env.obs_groups_spec)
        self.num_obs = (
            self._actor_obs_dim if self.policy_obs_mode == "actor" else self._flat_obs_dim
        )
        self.num_privileged_obs = int(env.obs_groups_spec.get("critic", self.num_obs))
        action_shape = env.action_space.shape
        if action_shape is None:
            raise ValueError("env.action_space.shape must be defined")
        self.num_actions = int(action_shape[0])

        self.episode_returns = torch.zeros(self.num_envs, device=device)
        self.episode_lengths = torch.zeros(self.num_envs, device=device)
        self.episode_length_buf = self.episode_lengths
        self.max_episode_length = np.ceil(env.cfg.max_episode_seconds / env.cfg.ctrl_dt)
        self.reset()

    def _policy_obs(self, obs: dict[str, Any]) -> torch.Tensor:
        if self.policy_obs_mode == "actor":
            return to_torch(obs["obs"], self.device)

        policy_groups = [
            to_numpy(value) for group_name, value in obs.items() if group_name != "critic"
        ]
        if not policy_groups:
            raise KeyError("Observation dict must contain at least one non-critic group")
        if len(policy_groups) == 1:
            return to_torch(policy_groups[0], self.device)
        return to_torch(np.concatenate(policy_groups, axis=1), self.device)

    def _obs_to_tensordict(
        self,
        obs: dict[str, Any],
        info: dict[str, Any] | None = None,
    ) -> TensorDict:
        del info
        actor_obs = to_torch(obs["obs"], self.device)
        td_dict: dict[str, torch.Tensor] = {
            "actor": actor_obs,
            "policy": self._policy_obs(obs),
        }
        if "critic" in obs:
            td_dict["critic"] = to_torch(obs["critic"], self.device)
        return TensorDict(td_dict, batch_size=self.num_envs, device=self.device)

    def _resolve_final_observation(self, state: NpEnvState) -> dict[str, Any] | None:
        if isinstance(state.final_observation, dict):
            return state.final_observation
        if isinstance(state.info, dict):
            final_observation = state.info.get("final_observation")
            if isinstance(final_observation, dict):
                return final_observation
        return None

    def _resolve_done(self, state: NpEnvState) -> torch.Tensor:
        return to_torch(state.terminated | state.truncated, self.device).bool()

    def step(
        self, actions: torch.Tensor | np.ndarray
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions_np = to_numpy(actions)
        state = self.env.step(actions_np)
        rewards = to_torch(state.reward, self.device)
        dones = self._resolve_done(state)

        self.episode_returns += rewards
        self.episode_lengths += 1

        infos: dict[str, torch.Tensor | TensorDict | dict[str, Any]] = {}
        done_idx = torch.nonzero(dones).flatten()
        if len(done_idx) > 0:
            infos["time_outs"] = to_torch(state.truncated, self.device).bool()

            final_observation = self._resolve_final_observation(state)
            terminal_contract = resolve_terminal_observation_contract(
                next_obs_batch_size=self.num_envs,
                final_observation=final_observation,
                done=to_numpy(dones),
                info=state.info,
                truncated=to_numpy(infos["time_outs"]),
            )
            if np.any(terminal_contract.timeout_terminal_mask) and final_observation is not None:
                infos["time_out_bootstrap_obs"] = self._obs_to_tensordict(final_observation)

            self.episode_returns[done_idx] = 0
            self.episode_lengths[done_idx] = 0

        if "log" in state.info:
            infos["log"] = state.info["log"]

        return (
            self._obs_to_tensordict(state.obs, getattr(state, "info", None)),
            rewards,
            dones,
            infos,
        )

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        if self.env.state is None:
            self.env.init_state()

        env_indices = np.arange(self.num_envs, dtype=np.int32)
        obs_out, info = self.env.reset(env_indices)
        self.episode_returns[:] = 0
        self.episode_lengths[:] = 0
        return self._obs_to_tensordict(obs_out, info), info

    def get_observations(self) -> TensorDict:
        assert self.env.state is not None
        return self._obs_to_tensordict(self.env.state.obs, self.env.state.info)

    def get_privileged_observations(self) -> torch.Tensor:
        assert self.env.state is not None
        obs = self.env.state.obs
        return to_torch(obs.get("critic", obs["obs"]), self.device)

    def close(self) -> None:
        self.env.close()


class HistoryObsDistillationWrapper(RslRlVecEnvWrapper):
    """RSL-RL distillation wrapper that derives student obs by dropping per-frame fields."""

    def __init__(
        self,
        env: Any,
        device: str = "cpu",
        *,
        teacher_obs_group: str = "obs",
        student_frame_dim: int,
        student_drop_start: int = 0,
        student_drop_dim: int = 3,
    ) -> None:
        self.teacher_obs_group = teacher_obs_group
        self.student_frame_dim = int(student_frame_dim)
        self.student_drop_start = int(student_drop_start)
        self.student_drop_dim = int(student_drop_dim)
        self._validate_projection_config()
        super().__init__(env, device=device, policy_obs_mode="actor")

        teacher_dim = int(env.obs_groups_spec[self.teacher_obs_group])
        student_dim = self._projected_dim(teacher_dim)
        self._actor_obs_dim = student_dim
        self._flat_obs_dim = student_dim
        self.num_obs = student_dim
        self.num_privileged_obs = teacher_dim

    def _validate_projection_config(self) -> None:
        if self.student_frame_dim <= 0:
            raise ValueError("student_frame_dim must be positive")
        if self.student_drop_start < 0:
            raise ValueError("student_drop_start must be non-negative")
        if self.student_drop_dim <= 0:
            raise ValueError("student_drop_dim must be positive")
        if self.student_drop_start + self.student_drop_dim > self.student_frame_dim:
            raise ValueError(
                "student drop range must fit inside one frame: "
                f"start={self.student_drop_start} dim={self.student_drop_dim} "
                f"frame_dim={self.student_frame_dim}"
            )

    def _projected_dim(self, teacher_dim: int) -> int:
        if teacher_dim % self.student_frame_dim != 0:
            raise ValueError(
                f"teacher obs dim {teacher_dim} must be divisible by "
                f"student_frame_dim={self.student_frame_dim}"
            )
        history_len = teacher_dim // self.student_frame_dim
        return history_len * (self.student_frame_dim - self.student_drop_dim)

    def _student_obs_from_teacher(self, teacher_obs: torch.Tensor) -> torch.Tensor:
        teacher_dim = int(teacher_obs.shape[-1])
        self._projected_dim(teacher_dim)
        frames = teacher_obs.reshape(*teacher_obs.shape[:-1], -1, self.student_frame_dim)
        keep = torch.ones(self.student_frame_dim, dtype=torch.bool, device=teacher_obs.device)
        drop_end = self.student_drop_start + self.student_drop_dim
        keep[self.student_drop_start : drop_end] = False
        projected = frames[..., keep]
        return projected.reshape(*teacher_obs.shape[:-1], -1)

    def _obs_to_tensordict(
        self,
        obs: dict[str, Any],
        info: dict[str, Any] | None = None,
    ) -> TensorDict:
        del info
        if self.teacher_obs_group not in obs:
            raise KeyError(
                f"Observation dict does not contain teacher_obs_group={self.teacher_obs_group!r}"
            )

        teacher_obs = to_torch(obs[self.teacher_obs_group], self.device)
        student_obs = self._student_obs_from_teacher(teacher_obs)
        td_dict: dict[str, torch.Tensor] = {
            "student": student_obs,
            "teacher": teacher_obs,
            "actor": student_obs,
            "policy": student_obs,
        }
        if "critic" in obs:
            td_dict["critic"] = to_torch(obs["critic"], self.device)
        return TensorDict(td_dict, batch_size=self.num_envs, device=self.device)

    def get_privileged_observations(self) -> torch.Tensor:
        assert self.env.state is not None
        return to_torch(self.env.state.obs[self.teacher_obs_group], self.device)
