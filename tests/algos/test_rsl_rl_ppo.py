from __future__ import annotations

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from unilab.algos.torch.rsl_rl_ppo import (
    FinalObservationAwarePPO,
    LinvelEstimatorActor,
    LinvelEstimatorPPO,
    TeacherRegularizedPPO,
)
from unilab.training.rsl_rl import (
    HistoryObsDistillationWrapper,
    RslRlVecEnvWrapper,
    normalize_ppo_train_cfg,
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


def _linvel_estimator_obs(batch_size: int = 4) -> TensorDict:
    actor_obs = torch.randn(batch_size, 6)
    target_linvel = actor_obs[:, :3] * 0.5
    critic_obs = torch.cat([target_linvel, torch.zeros(batch_size, 5)], dim=-1)
    return TensorDict(
        {"actor": actor_obs, "critic": critic_obs},
        batch_size=[batch_size],
    )


def _distribution_cfg() -> dict:
    return {
        "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
        "init_std": 0.2,
        "std_type": "scalar",
    }


def test_teacher_regularized_ppo_computes_teacher_action_and_kl_losses():
    torch.manual_seed(0)
    obs = TensorDict(
        {
            "actor": torch.randn(4, 3),
            "teacher": torch.randn(4, 5),
        },
        batch_size=[4],
    )
    actor = MLPModel(
        obs,
        {"actor": ["actor"], "teacher": ["teacher"]},
        "actor",
        output_dim=2,
        hidden_dims=[8],
        distribution_cfg=_distribution_cfg(),
    )
    teacher = MLPModel(
        obs,
        {"actor": ["actor"], "teacher": ["teacher"]},
        "teacher",
        output_dim=2,
        hidden_dims=[8],
        distribution_cfg=_distribution_cfg(),
    )

    algo = object.__new__(TeacherRegularizedPPO)
    algo.actor = actor
    algo.device = "cpu"
    algo.teacher_regularization_enabled = True
    algo.teacher_action_loss_coef = 1.0
    algo.teacher_kl_loss_coef = 1.0
    algo.set_teacher_policy(teacher)

    actor(obs, stochastic_output=True)
    loss, metrics = algo._teacher_regularization_loss(obs, actor.output_distribution_params)

    assert loss.item() > 0.0
    assert metrics["teacher_action"] > 0.0
    assert metrics["teacher_kl"] > 0.0
    assert all(not param.requires_grad for param in teacher.parameters())


def test_normalize_ppo_train_cfg_preserves_custom_linvel_estimator_actor():
    normalized = normalize_ppo_train_cfg(
        {
            "empirical_normalization": True,
            "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
            "policy": {
                "actor_class_name": "unilab.algos.torch.rsl_rl_ppo:LinvelEstimatorActor",
                "actor_hidden_dims": [16],
                "critic_hidden_dims": [16],
                "activation": "elu",
                "init_noise_std": 0.2,
                "linvel_estimator": {
                    "hidden_dims": [8],
                    "target_obs_group": "critic",
                    "target_start": 0,
                    "target_dim": 3,
                },
            },
            "algorithm": {},
        }
    )

    assert normalized["actor"]["class_name"] == "unilab.algos.torch.rsl_rl_ppo:LinvelEstimatorActor"
    assert normalized["actor"]["obs_normalization"] is True
    assert normalized["actor"]["linvel_estimator"]["target_obs_group"] == "critic"
    assert normalized["critic"]["class_name"] == "rsl_rl.models.MLPModel"


def test_linvel_estimator_actor_predicts_target_and_exports_tensor_policy():
    obs = _linvel_estimator_obs()
    actor = LinvelEstimatorActor(
        obs,
        {"actor": ["actor"], "critic": ["critic"]},
        "actor",
        output_dim=2,
        hidden_dims=[8],
        distribution_cfg=_distribution_cfg(),
        linvel_estimator={
            "hidden_dims": [8],
            "target_obs_group": "critic",
            "target_start": 0,
            "target_dim": 3,
        },
    )

    actions = actor(obs)
    predicted_linvel = actor.predict_linvel(obs)
    target_linvel = actor.linvel_target(obs)
    onnx_policy = actor.as_onnx(verbose=False)

    assert actions.shape == (4, 2)
    assert predicted_linvel.shape == (4, 3)
    assert torch.equal(target_linvel, obs["critic"][:, :3])
    assert onnx_policy.get_dummy_inputs()[0].shape == (1, 6)
    assert onnx_policy(torch.zeros(1, 6)).shape == (1, 2)


def test_linvel_estimator_ppo_updates_estimator_and_saves_optimizer():
    torch.manual_seed(0)
    obs = _linvel_estimator_obs()
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}
    actor = LinvelEstimatorActor(
        obs,
        obs_groups,
        "actor",
        output_dim=1,
        hidden_dims=[8],
        distribution_cfg=_distribution_cfg(),
        linvel_estimator={
            "hidden_dims": [8],
            "target_obs_group": "critic",
            "target_start": 0,
            "target_dim": 3,
        },
    )
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[8])
    storage = RolloutStorage(
        "rl",
        num_envs=4,
        num_transitions_per_env=2,
        obs=obs,
        actions_shape=[1],
        device="cpu",
    )
    algo = LinvelEstimatorPPO(
        actor,
        critic,
        storage,
        num_learning_epochs=1,
        num_mini_batches=2,
        desired_kl=None,
        linvel_estimator={
            "enabled": True,
            "learning_rate": 1.0e-2,
            "loss_coef": 1.0,
        },
    )

    before = [p.detach().clone() for p in actor.linvel_estimator_parameters()]
    for _ in range(2):
        rollout_obs = _linvel_estimator_obs()
        algo.act(rollout_obs)
        algo.process_env_step(
            rollout_obs,
            rewards=torch.ones(4),
            dones=torch.zeros(4, dtype=torch.bool),
            extras={},
        )
    algo.compute_returns(_linvel_estimator_obs())

    loss_dict = algo.update()
    after = actor.linvel_estimator_parameters()
    changed = any(not torch.allclose(prev, curr) for prev, curr in zip(before, after))

    assert changed
    assert loss_dict["estimation"] >= 0.0
    assert loss_dict["estimation_rmse"] >= 0.0
    assert "linvel_estimator_optimizer_state_dict" in algo.save()
