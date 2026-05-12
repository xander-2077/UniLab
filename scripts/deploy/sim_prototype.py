#!/usr/bin/env python3
"""Python prototype of State_WBT — drives the ONNX policy in MuJoCo using the
exact obs assembly the C++ State_WBT will use, so we can validate the
deploy_config.yaml + dance1.bin against training-side expectations BEFORE
writing C++.

Inputs (defaults match the artifacts produced by export_*.py):
  ~/deploy_ws/assets/policy.onnx
  ~/deploy_ws/assets/deploy_config.yaml
  ~/deploy_ws/assets/dance1.bin

What this verifies:
  1. obs assembly matches training (160 dim, 8 segments in order).
  2. ONNX inference + apply_action(action*2.0 + default_angles + clip + EMA)
     produces motion that visually tracks dance1 in MuJoCo.
  3. linvel_strategy="zero" works on this dance.

Differences from training-side eval (deliberate):
  - obs construction is reimplemented in pure numpy here, NOT routed through
    the env class — this is the same code path State_WBT will reproduce in C++.
  - No noise injected on obs (matches deploy convention).
  - Robot anchor pos in world is locked to the first motion frame's torso pos
    (since real robot has no GPS/SLAM).
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from unilab.envs.common.rotation import (  # noqa: E402
    np_matrix_from_quat,
    np_subtract_frame_transforms,
)

DEFAULT_ONNX = Path.home() / "deploy_ws/assets/policy.onnx"
DEFAULT_CFG = Path.home() / "deploy_ws/assets/deploy_config.yaml"
DEFAULT_BIN = Path.home() / "deploy_ws/assets/dance1.bin"
DEFAULT_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"


def load_motion_bin(path: Path) -> dict:
    with open(path, "rb") as f:
        fps, nf, nj, nb = struct.unpack("<iiii", f.read(16))
        jp = np.frombuffer(f.read(nf * nj * 4), dtype="<f4").reshape(nf, nj).copy()
        jv = np.frombuffer(f.read(nf * nj * 4), dtype="<f4").reshape(nf, nj).copy()
        bp = np.frombuffer(f.read(nf * nb * 3 * 4), dtype="<f4").reshape(nf, nb, 3).copy()
        bq = np.frombuffer(f.read(nf * nb * 4 * 4), dtype="<f4").reshape(nf, nb, 4).copy()
        blv = np.frombuffer(f.read(nf * nb * 3 * 4), dtype="<f4").reshape(nf, nb, 3).copy()
        bav = np.frombuffer(f.read(nf * nb * 3 * 4), dtype="<f4").reshape(nf, nb, 3).copy()
    return {
        "fps": fps,
        "num_frames": nf,
        "num_joints": nj,
        "num_bodies": nb,
        "joint_pos": jp,
        "joint_vel": jv,
        "body_pos_w": bp,
        "body_quat_w": bq,
        "body_lin_vel_w": blv,
        "body_ang_vel_w": bav,
    }


def assemble_obs(
    cfg: dict,
    motion_frame: dict,
    *,
    robot_torso_pos_w: np.ndarray,
    robot_torso_quat_w: np.ndarray,
    gyro: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    last_actions: np.ndarray,
) -> np.ndarray:
    """Build the 160-dim actor obs in the exact training order.

    All inputs are batch-less (1D arrays). Output is (160,) float32.
    """
    default_angles = np.asarray(cfg["default_angles"], dtype=np.float32)
    anchor_idx = int(cfg["anchor_body_idx_in_tracked"])

    ref_joint_pos = motion_frame["joint_pos"]  # (29,)
    ref_joint_vel = motion_frame["joint_vel"]  # (29,)
    ref_torso_pos_w = motion_frame["body_pos_w"][anchor_idx]  # (3,)
    ref_torso_quat_w = motion_frame["body_quat_w"][anchor_idx]  # (4,) wxyz

    command = np.concatenate([ref_joint_pos, ref_joint_vel]).astype(np.float32)  # (58,)

    pos_b, ori_q = np_subtract_frame_transforms(
        robot_torso_pos_w[None, :],
        robot_torso_quat_w[None, :],
        ref_torso_pos_w[None, :],
        ref_torso_quat_w[None, :],
    )
    motion_anchor_pos_b = pos_b[0].astype(np.float32)  # (3,)
    ori_R = np_matrix_from_quat(ori_q)[0]  # (3, 3)
    motion_anchor_ori_b = ori_R[:, :2].reshape(6).astype(np.float32)  # (6,)

    # linvel: deploy strategy = zero (no real-robot sensor)
    linvel = np.zeros(3, dtype=np.float32) if cfg.get("linvel_strategy", "zero") == "zero" \
        else gyro * 0.0  # extension hook
    gyro = gyro.astype(np.float32)
    joint_pos_rel = (dof_pos - default_angles).astype(np.float32)
    dof_vel = dof_vel.astype(np.float32)
    last_actions = last_actions.astype(np.float32)

    obs = np.concatenate([
        command, motion_anchor_pos_b, motion_anchor_ori_b,
        linvel, gyro, joint_pos_rel, dof_vel, last_actions,
    ]).astype(np.float32)
    assert obs.shape == (160,), f"obs shape {obs.shape} != (160,)"
    return obs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument("--motion", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--render", action="store_true",
                    help="Open MuJoCo passive viewer (requires display).")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 = play to end of clip then loop once.")
    ap.add_argument("--cheat-anchor", action="store_true",
                    help="Use sim-true robot torso pos for anchor (debug; "
                         "would not be available on real robot).")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    motion = load_motion_bin(args.motion)
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"ONNX: input={inp_name} {sess.get_inputs()[0].shape}, "
          f"output={out_name} {sess.get_outputs()[0].shape}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    ctrl_dt = float(cfg["ctrl_dt"])
    sim_dt = float(model.opt.timestep)
    substeps = max(1, int(round(ctrl_dt / sim_dt)))
    print(f"sim_dt={sim_dt:.5f}, ctrl_dt={ctrl_dt:.3f}, substeps/ctrl={substeps}")

    # Init pose: Reference State Initialization (RSI) — start robot at motion
    # frame 0's pose, matching what the training env does on reset. NOTE: this
    # is sim-only; on the real robot we'll need a separate strategy
    # (e.g. crouch in stand keyframe and only switch to State_WBT once the
    # operator has moved the robot near the dance start pose).
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        raise SystemExit("'stand' keyframe not found")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Override base + dof from motion frame 0 (pelvis is body[0] in tracked
    # list, which matches MuJoCo body_id=1 — the root body of the kinematic
    # tree under the free joint).
    pelvis_id_in_tracked = 0  # 'pelvis' is first in TRACKED_BODY_NAMES
    data.qpos[0:3] = motion["body_pos_w"][0, pelvis_id_in_tracked]
    data.qpos[3:7] = motion["body_quat_w"][0, pelvis_id_in_tracked]  # wxyz
    data.qpos[7:] = motion["joint_pos"][0]
    data.qvel[0:3] = motion["body_lin_vel_w"][0, pelvis_id_in_tracked]
    data.qvel[3:6] = motion["body_ang_vel_w"][0, pelvis_id_in_tracked]
    data.qvel[6:] = motion["joint_vel"][0]
    mujoco.mj_forward(model, data)
    print(f"RSI: robot init from motion frame 0 — base xyz={data.qpos[:3]}, "
          f"base quat={data.qpos[3:7]}")

    default_angles = np.asarray(cfg["default_angles"], dtype=np.float32)
    action_scale = float(cfg["action_scale"])
    ema_alpha = float(cfg["ema_alpha"])
    joint_lower = np.asarray(cfg["joint_lower"], dtype=np.float32)
    joint_upper = np.asarray(cfg["joint_upper"], dtype=np.float32)
    anchor_idx = int(cfg["anchor_body_idx_in_tracked"])
    anchor_body_name = cfg["anchor_body_name"]
    anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name)

    # Resolve gyro sensor address (matches training-side `Sensor.gyro = "gyro"`).
    gyro_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
    if gyro_sid < 0:
        raise SystemExit("'gyro' sensor not found in model")
    gyro_adr = int(model.sensor_adr[gyro_sid])
    gyro_dim = int(model.sensor_dim[gyro_sid])
    if gyro_dim != 3:
        raise SystemExit(f"gyro sensor has dim {gyro_dim}, expected 3")

    # Lock robot anchor world-position to the first motion frame's torso pos
    # (proxy for "no GPS/SLAM" — real robot has no absolute world XY/Z).
    robot_anchor_pos_w_locked = motion["body_pos_w"][0, anchor_idx].astype(np.float32)
    print(f"robot_anchor_pos_w (locked) = {robot_anchor_pos_w_locked}")

    last_actions = np.zeros(29, dtype=np.float32)
    q_target_smoothed = default_angles.copy()
    n_frames = motion["num_frames"]
    total_steps = args.max_steps if args.max_steps > 0 else n_frames

    # Stats for sanity
    obs_norms = []
    action_amplitudes = []
    z_errors = []

    viewer = None
    if args.render:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    t_wall = time.time()
    for step in range(total_steps):
        frame_idx = step % n_frames

        # ----- read robot state (pretend this came from IMU + encoders) -----
        # IMU quat = torso body quat. In MuJoCo qpos[3:7] is wxyz of free joint.
        robot_torso_quat_w = data.xquat[anchor_body_id].astype(np.float32)  # wxyz
        # gyro: read from MuJoCo sensor (matches training-side path
        # backend.get_sensor_data("gyro") → values of <gyro> on imu_in_torso).
        gyro = data.sensordata[gyro_adr:gyro_adr + gyro_dim].astype(np.float32)

        dof_pos = data.qpos[7:].astype(np.float32)
        dof_vel = data.qvel[6:].astype(np.float32)

        # ----- assemble obs -----
        motion_frame = {
            "joint_pos": motion["joint_pos"][frame_idx],
            "joint_vel": motion["joint_vel"][frame_idx],
            "body_pos_w": motion["body_pos_w"][frame_idx],
            "body_quat_w": motion["body_quat_w"][frame_idx],
        }
        if args.cheat_anchor:
            robot_torso_pos_w_used = data.xpos[anchor_body_id].astype(np.float32)
        else:
            robot_torso_pos_w_used = robot_anchor_pos_w_locked
        obs = assemble_obs(
            cfg, motion_frame,
            robot_torso_pos_w=robot_torso_pos_w_used,
            robot_torso_quat_w=robot_torso_quat_w,
            gyro=gyro, dof_pos=dof_pos, dof_vel=dof_vel,
            last_actions=last_actions,
        )

        # ----- ONNX inference -----
        action = sess.run([out_name], {inp_name: obs[None, :].astype(np.float32)})[0][0]
        action = action.astype(np.float32)
        last_actions = action.copy()

        # ----- apply_action: q* = action*scale + default; clip; EMA -----
        q_target = action * action_scale + default_angles
        q_target = np.clip(q_target, joint_lower, joint_upper)
        q_target_smoothed = ema_alpha * q_target + (1.0 - ema_alpha) * q_target_smoothed

        # MuJoCo position actuators take target qpos directly.
        data.ctrl[:] = q_target_smoothed

        # ----- step physics for one ctrl_dt -----
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        # ----- diagnostics -----
        obs_norms.append(float(np.linalg.norm(obs)))
        action_amplitudes.append(float(np.max(np.abs(action))))
        # Z error between robot torso and ref torso
        robot_z = float(data.xpos[anchor_body_id, 2])
        ref_z = float(motion["body_pos_w"][frame_idx, anchor_idx, 2])
        z_errors.append(abs(robot_z - ref_z))

        if viewer is not None:
            viewer.sync()
            elapsed = time.time() - t_wall
            target = (step + 1) * ctrl_dt
            if elapsed < target:
                time.sleep(target - elapsed)

        if step % 50 == 0:
            print(f"step {step:4d} frame={frame_idx:3d}  "
                  f"obs_norm={obs_norms[-1]:7.2f}  "
                  f"|action|={action_amplitudes[-1]:5.2f}  "
                  f"z_err={z_errors[-1]:.3f}m  "
                  f"q_target[:3]={q_target_smoothed[:3]}")

        # bail out on physics divergence
        if not np.all(np.isfinite(data.qpos)):
            print(f"!! NaN at step {step}, aborting")
            break
        if z_errors[-1] > 0.6:
            print(f"!! z_err {z_errors[-1]:.2f}m exceeds 0.6m, robot likely fell at step {step}")
            break

    if viewer is not None:
        viewer.close()

    n = len(obs_norms)
    print()
    print(f"Ran {n} ctrl steps ({n*ctrl_dt:.2f}s of motion).")
    print(f"obs_norm:        mean={np.mean(obs_norms):.3f}  max={np.max(obs_norms):.3f}")
    print(f"|action| max:    mean={np.mean(action_amplitudes):.3f}  max={np.max(action_amplitudes):.3f}")
    print(f"|torso z err|:   mean={np.mean(z_errors):.4f}m  max={np.max(z_errors):.4f}m")
    if np.max(z_errors) < 0.2 and n == total_steps:
        print("PROTOTYPE OK — obs assembly + ONNX inference produces tracking behavior.")
    elif n < total_steps:
        print("WARNING: prototype aborted before clip end (see message above).")
    else:
        print("WARNING: large torso z error — obs assembly may be off, or policy weak.")


if __name__ == "__main__":
    main()
