from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, resolve_optimizer, unpad_trajectories
from tensordict import TensorDict


class FinalObservationAwarePPO(PPO):
    """PPO variant that bootstraps time limits from env final_observation."""

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor | TensorDict],
    ) -> None:
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        timeouts = extras.get("time_outs")
        timeout_bootstrap_obs = extras.get("time_out_bootstrap_obs")
        if isinstance(timeouts, torch.Tensor):
            timeout_mask = timeouts.to(self.device).float()
            if timeout_bootstrap_obs is not None and torch.count_nonzero(timeout_mask) > 0:
                bootstrap_obs = timeout_bootstrap_obs.to(self.device)
                bootstrap_values = self.critic(bootstrap_obs).detach()
                self.transition.rewards += self.gamma * torch.squeeze(
                    bootstrap_values * timeout_mask.unsqueeze(1), 1
                )
            else:
                transition_values = self.transition.values
                assert transition_values is not None
                self.transition.rewards += self.gamma * torch.squeeze(
                    transition_values * timeout_mask.unsqueeze(1), 1
                )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)


class LinvelEstimatorActor(nn.Module):
    """RSL-RL actor that replaces unavailable base linvel with an MLP estimate."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        linvel_estimator: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = dict(linvel_estimator or {})

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.target_obs_group = str(cfg.get("target_obs_group", "critic"))
        self.target_start = int(cfg.get("target_start", 0))
        self.target_dim = int(cfg.get("target_dim", 3))
        self.detach_estimate_in_policy = bool(cfg.get("detach_in_policy", True))
        estimator_hidden_dims = cfg.get("hidden_dims", [128, 64])
        estimator_activation = str(cfg.get("activation", activation))

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.estimator_obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.policy_obs_normalizer = EmpiricalNormalization(self.obs_dim + self.target_dim)
        else:
            self.estimator_obs_normalizer = nn.Identity()
            self.policy_obs_normalizer = nn.Identity()

        self.linvel_estimator = MLP(
            self.obs_dim,
            self.target_dim,
            estimator_hidden_dims,
            estimator_activation,
        )

        distribution_cfg_copy = copy.deepcopy(distribution_cfg)
        if distribution_cfg_copy is not None:
            dist_class: type[Distribution] = resolve_callable(
                distribution_cfg_copy.pop("class_name")
            )  # type: ignore[assignment]
            self.distribution: Distribution | None = dist_class(
                output_dim, **distribution_cfg_copy
            )
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.mlp = MLP(self.obs_dim + self.target_dim, mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self._actor_obs(obs)
        estimated_linvel = self.predict_linvel(obs)
        if self.detach_estimate_in_policy:
            estimated_linvel = estimated_linvel.detach()
        policy_obs = torch.cat([estimated_linvel, actor_obs], dim=-1)
        return self.policy_obs_normalizer(policy_obs)

    def predict_linvel(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        actor_obs = self._actor_obs(obs)
        estimator_input = self.estimator_obs_normalizer(actor_obs)
        return self.linvel_estimator(estimator_input)

    def linvel_target(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        target_obs = obs[self.target_obs_group]
        end = self.target_start + self.target_dim
        return target_obs[..., self.target_start : end]

    def linvel_estimator_parameters(self) -> list[nn.Parameter]:
        return list(self.linvel_estimator.parameters())

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params  # type: ignore[union-attr]

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

    def as_jit(self) -> nn.Module:
        return _TorchLinvelEstimatorActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxLinvelEstimatorActor(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return

        actor_obs = self._actor_obs(obs)
        self.estimator_obs_normalizer.update(actor_obs)  # type: ignore[attr-defined]
        if self.target_obs_group in obs.keys():
            target_linvel = self.linvel_target(obs)
            policy_obs = torch.cat([target_linvel, actor_obs], dim=-1)
            self.policy_obs_normalizer.update(policy_obs)  # type: ignore[attr-defined]

    def _actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group_name] for group_name in self.obs_groups], dim=-1)

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    "LinvelEstimatorActor only supports 1D observation groups, "
                    f"got shape {obs[obs_group].shape} for {obs_group!r}."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim


class _TorchLinvelEstimatorActor(nn.Module):
    def __init__(self, model: LinvelEstimatorActor) -> None:
        super().__init__()
        self.estimator_obs_normalizer = copy.deepcopy(model.estimator_obs_normalizer)
        self.linvel_estimator = copy.deepcopy(model.linvel_estimator)
        self.policy_obs_normalizer = copy.deepcopy(model.policy_obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        estimated_linvel = self.linvel_estimator(self.estimator_obs_normalizer(x))
        latent = self.policy_obs_normalizer(torch.cat([estimated_linvel, x], dim=-1))
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxLinvelEstimatorActor(nn.Module):
    is_recurrent: bool = False

    def __init__(self, model: LinvelEstimatorActor, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.estimator_obs_normalizer = copy.deepcopy(model.estimator_obs_normalizer)
        self.linvel_estimator = copy.deepcopy(model.linvel_estimator)
        self.policy_obs_normalizer = copy.deepcopy(model.policy_obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        estimated_linvel = self.linvel_estimator(self.estimator_obs_normalizer(x))
        latent = self.policy_obs_normalizer(torch.cat([estimated_linvel, x], dim=-1))
        return self.deterministic_output(self.mlp(latent))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class LinvelEstimatorPPO(FinalObservationAwarePPO):
    """PPO with a supervised linvel estimator trained on the same rollouts."""

    def __init__(
        self,
        *args: Any,
        linvel_estimator: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        cfg = dict(linvel_estimator or {})
        self.linvel_estimator_enabled = bool(cfg.get("enabled", True)) and hasattr(
            self.actor, "linvel_estimator_parameters"
        )
        self.linvel_estimator_loss_coef = float(cfg.get("loss_coef", 1.0))
        self.linvel_estimator_max_grad_norm = float(
            cfg.get("max_grad_norm", self.max_grad_norm)
        )
        self.linvel_estimator_optimizer = None
        self._linvel_estimator_params: list[nn.Parameter] = []

        if self.linvel_estimator_enabled:
            self._linvel_estimator_params = list(self.actor.linvel_estimator_parameters())
            if len(self._linvel_estimator_params) == 0:
                raise ValueError("linvel estimator is enabled but has no parameters")
            optimizer_name = str(cfg.get("optimizer", "adam"))
            learning_rate = float(cfg.get("learning_rate", 1.0e-3))
            self.linvel_estimator_optimizer = resolve_optimizer(optimizer_name)(
                self._linvel_estimator_params,
                lr=learning_rate,
            )

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict.update(self._update_linvel_estimator())
        return loss_dict

    def save(self) -> dict:
        saved_dict = super().save()
        if self.linvel_estimator_optimizer is not None:
            saved_dict["linvel_estimator_optimizer_state_dict"] = (
                self.linvel_estimator_optimizer.state_dict()
            )
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        should_load_optimizer = load_cfg is None or bool(
            load_cfg.get("linvel_estimator_optimizer", load_cfg.get("optimizer", True))
        )
        if (
            should_load_optimizer
            and self.linvel_estimator_optimizer is not None
            and "linvel_estimator_optimizer_state_dict" in loaded_dict
        ):
            self.linvel_estimator_optimizer.load_state_dict(
                loaded_dict["linvel_estimator_optimizer_state_dict"]
            )
        return load_iteration

    def _update_linvel_estimator(self) -> dict[str, float]:
        if not self.linvel_estimator_enabled or self.linvel_estimator_optimizer is None:
            return {}

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )

        mean_mse = 0.0
        num_updates = 0
        for batch in generator:
            predicted_linvel = self.actor.predict_linvel(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
            )
            target_linvel = self.actor.linvel_target(batch.observations, masks=batch.masks).detach()
            mse = F.mse_loss(predicted_linvel, target_linvel)
            loss = self.linvel_estimator_loss_coef * mse

            self.linvel_estimator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self._linvel_estimator_params,
                self.linvel_estimator_max_grad_norm,
            )
            self.linvel_estimator_optimizer.step()

            mean_mse += mse.item()
            num_updates += 1

        if num_updates == 0:
            return {}

        mean_mse /= num_updates
        return {
            "estimation": mean_mse,
            "estimation_rmse": math.sqrt(mean_mse),
        }


class TeacherRegularizedPPO(FinalObservationAwarePPO):
    """PPO fine-tuning with a frozen teacher action/KL regularizer."""

    def __init__(
        self,
        *args: Any,
        teacher_regularization: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        cfg = dict(teacher_regularization or {})
        self.teacher_regularization_enabled = bool(cfg.get("enabled", True))
        self.teacher_action_loss_coef = float(cfg.get("action_loss_coef", 0.0))
        self.teacher_kl_loss_coef = float(cfg.get("kl_loss_coef", 0.0))
        self.teacher_policy: nn.Module | None = None

    def set_teacher_policy(self, teacher_policy: nn.Module) -> None:
        self.teacher_policy = teacher_policy.to(self.device).eval()
        for param in self.teacher_policy.parameters():
            param.requires_grad_(False)

    def _teacher_regularization_loss(
        self,
        observations: TensorDict,
        student_distribution_params: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        zero = student_distribution_params[0].new_zeros(())
        if (
            not self.teacher_regularization_enabled
            or self.teacher_policy is None
            or (self.teacher_action_loss_coef == 0.0 and self.teacher_kl_loss_coef == 0.0)
        ):
            return zero, {}

        with torch.no_grad():
            self.teacher_policy(observations, stochastic_output=True)
            teacher_distribution_params = tuple(
                param.detach() for param in self.teacher_policy.output_distribution_params
            )

        total = zero
        metrics: dict[str, float] = {}
        if self.teacher_action_loss_coef != 0.0:
            action_loss = F.mse_loss(student_distribution_params[0], teacher_distribution_params[0])
            total = total + self.teacher_action_loss_coef * action_loss
            metrics["teacher_action"] = action_loss.item()

        if self.teacher_kl_loss_coef != 0.0:
            kl_loss = self.actor.get_kl_divergence(
                teacher_distribution_params,
                student_distribution_params,
            ).mean()
            total = total + self.teacher_kl_loss_coef * kl_loss
            metrics["teacher_kl"] = kl_loss.item()

        return total, metrics

    def update(self) -> dict[str, float]:
        if not self.teacher_regularization_enabled or self.teacher_policy is None:
            return super().update()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_teacher_action_loss = 0.0
        mean_teacher_kl_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches,
                self.num_learning_epochs,
            )

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params,
                        distribution_params,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss
            loss = loss - self.entropy_coef * entropy.mean()

            teacher_loss, teacher_metrics = self._teacher_regularization_loss(
                batch.observations[:original_batch_size],
                distribution_params,
            )
            loss = loss + teacher_loss

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations,
                        actions=None,
                        env=self.symmetry["_env"],
                    )

                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None,
                    actions=action_mean_orig,
                    env=self.symmetry["_env"],
                )

                symmetry_loss = F.mse_loss(
                    mean_actions[original_batch_size:],
                    actions_mean_symm.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = F.mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_teacher_action_loss += teacher_metrics.get("teacher_action", 0.0)
            mean_teacher_kl_loss += teacher_metrics.get("teacher_kl", 0.0)
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_teacher_action_loss /= num_updates
        mean_teacher_kl_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "teacher_action": mean_teacher_action_loss,
            "teacher_kl": mean_teacher_kl_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict
