from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
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
