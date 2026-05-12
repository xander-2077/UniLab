#!/usr/bin/env python3
"""Export deploy_config.yaml for the C++ G1-29DOF WBT deployment side.

Reads g1.xml + scene_flat.xml + tracking.py defaults to emit a single yaml
that the deploy framework (~/deploy_ws/unitree_rl_lab/.../State_WBT) can load
to drive the actor at runtime.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"
DEFAULT_OUT = REPO_ROOT / "logs/deploy/deploy_config.yaml"

TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
ANCHOR_BODY_NAME = "torso_link"

ACTION_SCALE = 2.0
# EMA alpha for q_target smoothing on the deploy side. Training applies q_target
# directly with no smoothing; alpha=1.0 means deploy also applies directly (best
# for sim2sim correctness). Lower it (~0.5–0.8) only on real hardware if jitter
# requires smoothing — but every step of lag pushes obs out of training
# distribution, so verify sim2sim impact at the chosen alpha first.
EMA_ALPHA = 1.0
CTRL_DT = 0.02
KEYFRAME_NAME = "stand"
ROOT_QPOS_DIM = 7  # free joint: xyz + quat(wxyz)


def _round_list(arr, ndigits=6):
    return [round(float(v), ndigits) for v in arr]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE,
                    help="MuJoCo scene file containing the 'stand' keyframe.")
    ap.add_argument("--output", "-o", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.scene.exists():
        raise SystemExit(f"Scene not found: {args.scene}")

    model = mujoco.MjModel.from_xml_path(str(args.scene))

    if model.nu != 29:
        raise SystemExit(f"Expected 29 actuators, got {model.nu}")

    # Joint names in actuator order (action[i] drives actuator i).
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    ]

    # kp/kv from XML <position> actuators.
    kp = model.actuator_gainprm[:, 0].copy()  # gainprm[0] is kp for position actuator
    kd = model.actuator_biasprm[:, 2].copy() * -1.0  # biasprm[2]=-kv for position actuator
    # Sanity-check: actuator_biasprm sign convention varies; cross-check with
    # explicit reading of XML attributes via the position-actuator mapping.
    # Use mujoco's stored gainprm/biasprm but verify magnitudes look reasonable.
    if not (kp > 0).all():
        raise SystemExit(f"kp parsing produced non-positive values: {kp}")
    # kv recovery: for position actuator, biasprm = [0, -kp, -kv]
    kv = -model.actuator_biasprm[:, 2].copy()
    if not (kv > 0).all():
        raise SystemExit(f"kv parsing produced non-positive values: {kv}")

    # Joint limits: skip the floating-root joint (jnt 0).
    if model.njnt < 1 + model.nu:
        raise SystemExit(f"Insufficient joints: njnt={model.njnt}")
    jnt_range = model.jnt_range[1:1 + model.nu].copy()
    joint_lower = jnt_range[:, 0]
    joint_upper = jnt_range[:, 1]

    # Force range from actuator forcerange.
    force_range = model.actuator_forcerange.copy()
    force_lower = force_range[:, 0]
    force_upper = force_range[:, 1]

    # default_angles = stand keyframe qpos[7:36].
    if model.nkey == 0:
        raise SystemExit(f"No keyframes in {args.scene}; expected '{KEYFRAME_NAME}'.")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
    if key_id < 0:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
        raise SystemExit(f"Keyframe '{KEYFRAME_NAME}' not found; have {names}")
    stand_qpos = model.key_qpos[key_id]
    if len(stand_qpos) != ROOT_QPOS_DIM + model.nu:
        raise SystemExit(
            f"stand qpos len {len(stand_qpos)} != {ROOT_QPOS_DIM}+{model.nu}"
        )
    default_angles = stand_qpos[ROOT_QPOS_DIM:].copy()

    # 14 tracked body indices (in MuJoCo body-id space; deploy side won't use
    # these directly but they are useful for debugging / cross-checks).
    tracked_body_ids = []
    for nm in TRACKED_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid < 0:
            raise SystemExit(f"Tracked body '{nm}' missing from model")
        tracked_body_ids.append(int(bid))
    anchor_body_idx_in_tracked = TRACKED_BODY_NAMES.index(ANCHOR_BODY_NAME)

    cfg = {
        # ---- meta ----
        # 154 = command(58) + motion_anchor_ori_b(6) + base_ang_vel(3)
        #       + joint_pos_rel(29) + joint_vel(29) + last_actions(29).
        # Aligns with the deploy-profile training run (mujoco_deploy.yaml)
        # which drops motion_anchor_pos_b and base_lin_vel from actor obs
        # to match Unitree's verified mjlab "No-State-Estimation" deploy yaml.
        "obs_dim": 154,
        "action_dim": 29,
        "ctrl_dt": CTRL_DT,
        "action_scale": ACTION_SCALE,
        "ema_alpha": EMA_ALPHA,

        # ---- joint config (in MuJoCo actuator order; deploy side assumes
        # this matches the SDK motor index 1:1 for G1-29DOF — verify per-motor
        # before real-robot run) ----
        "joint_names": list(joint_names),
        # Identity SDK mapping; replace with measured order if 1:1 assumption fails.
        "joint_ids_map": list(range(model.nu)),
        "default_angles": _round_list(default_angles),
        "kp": _round_list(kp),
        "kd": _round_list(kv),
        "joint_lower": _round_list(joint_lower),
        "joint_upper": _round_list(joint_upper),
        "force_lower": _round_list(force_lower, 3),
        "force_upper": _round_list(force_upper, 3),

        # ---- motion / anchor ----
        "tracked_body_names": list(TRACKED_BODY_NAMES),
        "tracked_body_mujoco_ids": tracked_body_ids,
        "anchor_body_name": ANCHOR_BODY_NAME,
        "anchor_body_idx_in_tracked": int(anchor_body_idx_in_tracked),

        # ---- noise (NOT applied at deploy; documentation only — per-step
        # uniform noise scales used during training, plus persistent encoder
        # bias absorbed into joint_pos_rel) ----
        "training_noise_scales": {
            "joint_angle": 0.01,
            "joint_vel": 0.5,
            "gyro": 0.2,
            "anchor_ori": 0.05,
            "joint_pos_encoder_bias_per_episode": 0.01,
        },

        # ---- obs layout (for State_WBT to assemble correctly).
        # NOTE: motion_anchor_pos_b and base_lin_vel are intentionally absent
        # because the deploy-profile training run masks them out — there is
        # no torso-pose estimator and no base-linvel sensor on G1. ----
        "obs_layout": [
            {"name": "command_joint_pos", "dim": 29, "source": "motion_ref_frame.joint_pos"},
            {"name": "command_joint_vel", "dim": 29, "source": "motion_ref_frame.joint_vel"},
            {"name": "motion_anchor_ori_b", "dim": 6, "source": "rotation_matrix(subtract_frame(...).quat)[:, :2].flatten()"},
            {"name": "gyro", "dim": 3, "source": "imu.gyroscope"},
            {"name": "joint_pos_rel", "dim": 29, "source": "dof_pos - default_angles"},
            {"name": "dof_vel", "dim": 29, "source": "dof_vel"},
            {"name": "last_actions", "dim": 29, "source": "previous raw actor output"},
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None, width=120)

    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"  joints: {model.nu}, tracked bodies: {len(TRACKED_BODY_NAMES)}, "
          f"anchor='{ANCHOR_BODY_NAME}' (idx_in_tracked={anchor_body_idx_in_tracked})")
    print(f"  default_angles[:6] = {_round_list(default_angles[:6], 3)}")
    print(f"  kp[:3] = {_round_list(kp[:3], 3)}, kd[:3] = {_round_list(kv[:3], 3)}")


if __name__ == "__main__":
    main()
