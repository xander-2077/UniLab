import datetime
import statistics
import sys
import time
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.base.backend.mujoco.xml import materialize_scene_visual_override
from unilab.training import (
    BackendAdapter,
    ExperimentTracker,
    apply_configured_training_seed,
    create_env,
    ensure_registries,
    get_log_root,
    log_playback_plan,
    parse_checkpoint_path,
    resolve_task_checkpoint_path,
)
from unilab.training.experiment import patch_rsl_rl_wandb_writer
from unilab.training.rsl_rl import HistoryObsDistillationWrapper
from unilab.utils.device import get_default_device

try:
    from rsl_rl.runners import DistillationRunner
except ImportError:
    print("Could not import rsl_rl. Please ensure it is installed.")
    sys.exit(1)


def _backend_adapter(cfg: DictConfig) -> BackendAdapter:
    return BackendAdapter(
        cfg,
        root_dir=ROOT_DIR,
        algo_name="rsl_rl_distill",
        scene_materializer=materialize_scene_visual_override,
    )


def _algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    train_cfg_raw = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(train_cfg_raw, dict):
        raise TypeError("cfg.algo must resolve to a dict")
    return cast(dict[str, Any], train_cfg_raw)


def _build_env_cfg_override(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], _backend_adapter(cfg).build_task_env_cfg_override())


def _resolve_teacher_checkpoint(cfg: DictConfig) -> tuple[Path | None, Path | None]:
    selected_checkpoint = OmegaConf.select(cfg, "teacher.checkpoint", default=-1)
    return resolve_task_checkpoint_path(
        ROOT_DIR,
        task_name=str(OmegaConf.select(cfg, "teacher.task_name", default=cfg.training.task_name)),
        load_run=str(OmegaConf.select(cfg, "teacher.load_run", default="-1")),
        algo_log_name=str(OmegaConf.select(cfg, "teacher.algo_log_name", default="rsl_rl_ppo")),
        checkpoint=(
            None if selected_checkpoint in (None, "", -1, "-1") else str(selected_checkpoint)
        ),
        log_root=OmegaConf.select(cfg, "teacher.log_root", default=None),
    )


def _format_teacher_checkpoint_error(
    cfg: DictConfig,
    teacher_checkpoint: Path | None,
    teacher_run_dir: Path | None,
) -> str:
    return (
        "Could not resolve teacher checkpoint for RSL-RL distillation. "
        f"teacher.algo_log_name={OmegaConf.select(cfg, 'teacher.algo_log_name', default='rsl_rl_ppo')!r} "
        f"teacher.task_name={OmegaConf.select(cfg, 'teacher.task_name', default=cfg.training.task_name)!r} "
        f"teacher.load_run={OmegaConf.select(cfg, 'teacher.load_run', default='-1')!r} "
        f"teacher.checkpoint={OmegaConf.select(cfg, 'teacher.checkpoint', default=-1)!r} "
        f"resolved_checkpoint={teacher_checkpoint} resolved_run={teacher_run_dir}. "
        "Set teacher.load_run=<teacher-run-dir-or-name> and optionally teacher.checkpoint=<iter>."
    )


def _make_wrapper(env: Any, cfg: DictConfig, device: str) -> HistoryObsDistillationWrapper:
    distill_cfg = cfg.distillation
    return HistoryObsDistillationWrapper(
        env,
        device=device,
        teacher_obs_group=str(distill_cfg.teacher_obs_group),
        student_frame_dim=int(distill_cfg.student_frame_dim),
        student_drop_start=int(distill_cfg.student_drop_start),
        student_drop_dim=int(distill_cfg.student_drop_dim),
    )


def _format_play_checkpoint_error(
    cfg: DictConfig,
    *,
    load_path: Path | None,
    load_path_dir: Path | None,
) -> str:
    return (
        "Could not resolve a distilled checkpoint for eval. "
        f"task={cfg.training.task_name} algo.load_run={cfg.algo.load_run!r} "
        f"algo.checkpoint={cfg.algo.checkpoint!r} "
        f"resolved_checkpoint={load_path} resolved_run={load_path_dir}. "
        "Use algo.load_run=<distill-run-dir-or-name> and optionally "
        "algo.checkpoint=<iteration-or-filename>."
    )


def _resolve_play_num_steps(cfg: DictConfig) -> int | None:
    play_steps = OmegaConf.select(cfg, "training.play_steps", default=None)
    if play_steps is None:
        return None
    return int(play_steps)


def play_rsl_rl_distill(cfg: DictConfig, device: str) -> str | None:
    load_path, load_path_dir = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)
    if load_path is None or load_path_dir is None or not load_path.exists():
        print(
            _format_play_checkpoint_error(
                cfg,
                load_path=load_path,
                load_path_dir=load_path_dir,
            )
        )
        return None

    print(f"Loading distilled model: {load_path}")
    ckpt_keys = set(torch.load(load_path, map_location="cpu", weights_only=True).keys())
    if "student_state_dict" not in ckpt_keys:
        print(
            f"Checkpoint at {load_path} is not an RSL-RL distillation checkpoint "
            f"(found keys: {ckpt_keys}). Aborting eval."
        )
        return None

    env_cfg_override = cast(dict[str, Any], _backend_adapter(cfg).build_play_env_cfg_override())
    env = create_env(
        cfg,
        num_envs=int(cfg.training.play_env_num),
        env_cfg_override=env_cfg_override,
    )
    wrapped_env = _make_wrapper(env, cfg, device)
    train_cfg = _algo_config_dict(cfg)
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "tensorboard"
    train_cfg["logger"] = "tensorboard"

    runner = cast(
        Any,
        DistillationRunner(cast(Any, wrapped_env), train_cfg, log_dir=None, device=device),
    )
    runner.load(
        str(load_path),
        load_cfg={"student": True, "iteration": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    runner.export_policy_to_onnx(path=str(load_path_dir))
    runner.export_policy_to_jit(path=str(load_path_dir))

    num_steps = _resolve_play_num_steps(cfg)
    output_video = Path(load_path_dir) / "play_video.mp4"
    playback_mode: str | None = None

    def _log_plan(plan) -> None:
        nonlocal playback_mode
        playback_mode = plan.mode
        log_playback_plan(plan)

    with torch.inference_mode():
        play_video_path = env.run_playback_mode(
            play_render_mode=getattr(cfg.training, "play_render_mode", "auto"),
            play_steps=num_steps,
            output_video=output_video,
            render_spacing=float(
                getattr(cfg.training, "render_spacing", getattr(env.cfg, "render_spacing", 1.0))
            ),
            render_offset_mode=str(getattr(env.cfg, "render_offset_mode", "grid")),
            initialize=lambda: wrapped_env.reset()[0],
            step=lambda obs: wrapped_env.step(policy(obs))[0],
            camera_kwargs={
                "cam_distance": cfg.training.cam_distance,
                "cam_elevation": cfg.training.cam_elevation,
                "cam_azimuth": cfg.training.cam_azimuth,
                "cam_lookat": getattr(cfg.training, "cam_lookat", None),
                "cam_tracking": getattr(cfg.training, "cam_tracking", False),
                "cam_tracking_env_idx": getattr(cfg.training, "cam_tracking_env_idx", 0),
                "cam_tracking_extra_envs": getattr(cfg.training, "cam_tracking_extra_envs", 2),
            },
            extra_data_getter=(
                (lambda: getattr(env, "curr_ee_goal_world", None))
                if hasattr(env, "curr_ee_goal_world")
                else None
            ),
            on_plan=_log_plan,
        )
    env.close()
    if playback_mode != "none" and num_steps is not None:
        print("Done.")
    return play_video_path


@hydra.main(version_base="1.3", config_path="../conf/rsl_rl_distill", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()

    seed_info = apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    env_cfg_override = _build_env_cfg_override(cfg)

    device = get_default_device()
    print(f"Using device: {device}")

    if cfg.training.play_only:
        play_rsl_rl_distill(cfg, device)
        return

    max_iterations = int(cfg.algo.max_iterations)
    if cfg.training.num_timesteps:
        n_steps_per_iter = int(cfg.algo.num_steps_per_env) * int(cfg.algo.num_envs)
        max_iterations = max(1, int(cfg.training.num_timesteps / n_steps_per_iter))
        print(
            f"Overriding max_iterations to {max_iterations} based on "
            f"num_timesteps {cfg.training.num_timesteps}"
        )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = get_log_root(ROOT_DIR, cfg)
    log_dir = str(
        Path(log_root) / cfg.training.task_name / f"{timestamp}_{cfg.training.sim_backend}"
    )

    tracker = ExperimentTracker(
        root_dir=ROOT_DIR,
        log_dir=log_dir,
        algo_name="rsl_rl_distill",
        task_name=str(cfg.training.task_name),
        sim_backend=str(cfg.training.sim_backend),
        training_cfg=cfg.training,
        full_cfg=cfg,
        device=device,
        seed_info=seed_info,
    )
    tracker.start()

    try:
        env = create_env(
            cfg,
            num_envs=int(cfg.algo.num_envs),
            env_cfg_override=env_cfg_override,
        )
        wrapped_env = _make_wrapper(env, cfg, device)
        train_cfg = _algo_config_dict(cfg)
        if "runner" not in train_cfg:
            train_cfg["runner"] = {}

        logger_type = "wandb" if cfg.training.logger == "wandb" else "tensorboard"
        train_cfg["runner"]["logger"] = logger_type
        train_cfg["logger"] = logger_type

        if logger_type == "wandb":
            patch_rsl_rl_wandb_writer()
            wandb_settings = tracker.wandb_settings
            train_cfg["wandb_project"] = wandb_settings["project"]
            train_cfg["wandb_entity"] = wandb_settings["entity"]
            train_cfg["wandb_group"] = wandb_settings["group"]
            train_cfg["wandb_job_type"] = wandb_settings["job_type"]
            train_cfg["wandb_tags"] = wandb_settings["tags"]
            train_cfg["wandb_notes"] = wandb_settings["notes"]
            train_cfg["wandb_mode"] = wandb_settings["mode"]

        runner = cast(
            Any,
            DistillationRunner(cast(Any, wrapped_env), train_cfg, log_dir=log_dir, device=device),
        )

        resume_path: Path | None = None
        if cfg.algo.load_run != "-1":
            resume_path, _ = parse_checkpoint_path(cfg, root_dir=ROOT_DIR)

        if resume_path is not None:
            print(f"Resuming distillation from {resume_path}")
            runner.load(str(resume_path), map_location=device)
        else:
            teacher_checkpoint, teacher_run_dir = _resolve_teacher_checkpoint(cfg)
            if teacher_checkpoint is None or not teacher_checkpoint.exists():
                raise FileNotFoundError(
                    _format_teacher_checkpoint_error(cfg, teacher_checkpoint, teacher_run_dir)
                )
            print(f"Loading teacher model: {teacher_checkpoint}")
            runner.load(
                str(teacher_checkpoint),
                load_cfg={"teacher": True, "iteration": False},
                strict=bool(cfg.teacher.strict),
                map_location=device,
            )

        train_start_wall = time.time()
        runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
        runner.export_policy_to_onnx(path=log_dir)
        runner.export_policy_to_jit(path=log_dir)

        train_summary = {
            "status": "completed",
            "completed_iterations": int(runner.current_learning_iteration),
            "total_env_steps": int(getattr(runner.logger, "tot_timesteps", 0)),
            "final_mean_reward": (
                float(statistics.mean(runner.logger.rewbuffer))
                if len(getattr(runner.logger, "rewbuffer", [])) > 0
                else None
            ),
            "best_mean_reward": (
                float(max(runner.logger.rewbuffer))
                if len(getattr(runner.logger, "rewbuffer", [])) > 0
                else None
            ),
            "mean_episode_length": (
                float(statistics.mean(runner.logger.lenbuffer))
                if len(getattr(runner.logger, "lenbuffer", [])) > 0
                else None
            ),
            "last_checkpoint": str(
                Path(log_dir) / f"model_{int(runner.current_learning_iteration)}.pt"
            ),
            "training_wall_time_sec": time.time() - train_start_wall,
        }
        tracker.update_summary(train_summary)
        env.close()
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
