"""
Spawn csento-4ac with the same default pose as locomotion training.

Use this to check URDF, base height, and default_joint_angles before training.

  cd ~/Genesis
  export DISPLAY=:0   # WSL GUI
  uv run python examples/locomotion/csento_4ac_spawn.py

  # load pose from a training run
  uv run python examples/locomotion/csento_4ac_spawn.py --cfgs logs/csento-4ac/cfgs.pkl

  # tweak base height
  uv run python examples/locomotion/csento_4ac_spawn.py --z 0.18
"""

import argparse
import os
import pickle

import genesis as gs
import torch

from csento_4ac_env import Csento4acEnv
from csento_4ac_pd_test import setup_pd_gains


def default_spawn_cfg() -> dict:
    """Keep in sync with get_cfgs() in csento_4ac_train.py."""
    return {
        "urdf_path": Csento4acEnv.URDF_PATH,
        "joint_names": [
            "left_hip_joint",
            "left_knee_joint",
            "left_wheel_joint",
            "right_hip_joint",
            "right_knee_joint",
            "right_wheel_joint",
        ],
        "default_joint_angles": {
            "left_hip_joint": 1.0,
            "left_knee_joint": 1.6,
            "left_wheel_joint": 0.0,
            "right_hip_joint": 1.0,
            "right_knee_joint": 1.6,
            "right_wheel_joint": 0.0,
        },
        "base_init_pos": [0.0, 0.0, 0.16],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "kp": 15.0,
        "kd": 0.4,
    }


def load_spawn_cfg(cfgs_path: str | None) -> dict:
    cfg = default_spawn_cfg()
    if cfgs_path is None:
        return cfg
    with open(cfgs_path, "rb") as f:
        env_cfg, *_ = pickle.load(f)
    for key in (
        "urdf_path",
        "joint_names",
        "default_joint_angles",
        "base_init_pos",
        "base_init_quat",
        "kp",
        "kd",
        "position_joint_names",
        "torque_joint_names",
        "wheel_kp",
        "wheel_kd",
    ):
        if key in env_cfg:
            cfg[key] = env_cfg[key]
    return cfg


def apply_spawn_pose(robot, cfg: dict, motors_dof_idx: torch.Tensor) -> torch.Tensor:
    base_pos = torch.tensor(cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
    base_quat = torch.tensor(cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
    robot.set_qpos(
        torch.cat((base_pos, base_quat)),
        qs_idx_local=list(range(7)),
        zero_velocity=True,
        skip_forward=True,
    )

    dof_pos = torch.tensor(
        [cfg["default_joint_angles"][name] for name in cfg["joint_names"]],
        dtype=gs.tc_float,
        device=gs.device,
    )
    robot.set_dofs_position(dof_pos.unsqueeze(0), motors_dof_idx, zero_velocity=True, skip_forward=True)
    joint_names = cfg["joint_names"]
    pos_names = cfg.get(
        "position_joint_names",
        [n for n in joint_names if n not in cfg.get("torque_joint_names", [])],
    )
    pos_idx = torch.tensor(
        [robot.get_joint(n).dof_start for n in pos_names],
        dtype=gs.tc_int,
        device=gs.device,
    )
    pos_dof = torch.tensor(
        [cfg["default_joint_angles"][n] for n in pos_names],
        dtype=gs.tc_float,
        device=gs.device,
    )
    robot.control_dofs_position(pos_dof.unsqueeze(0), pos_idx)
    return dof_pos


def main():
    parser = argparse.ArgumentParser(description="Spawn csento-4ac with train default pose.")
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis")
    parser.add_argument("--cfgs", type=str, default=None, help="Path to logs/.../cfgs.pkl from training")
    parser.add_argument("--z", type=float, default=None, help="Override base_init_pos z [m]")
    parser.add_argument("--steps", type=int, default=0, help="Simulation steps (0 = run until viewer closes)")
    parser.add_argument("--no-pd", action="store_true", help="Do not hold pose with PD each step")
    args = parser.parse_args()

    cfg = load_spawn_cfg(args.cfgs)
    if args.z is not None:
        cfg["base_init_pos"] = [cfg["base_init_pos"][0], cfg["base_init_pos"][1], args.z]

    gs.init(backend=gs.gpu if args.vis else gs.cpu, logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 0.0, 1.2),
            camera_lookat=(0.0, 0.0, cfg["base_init_pos"][2]),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=cfg["urdf_path"],
            pos=cfg["base_init_pos"],
            quat=cfg["base_init_quat"],
        ),
    )
    scene.build()

    joint_names = cfg["joint_names"]
    motors_dof_idx = torch.tensor(
        [robot.get_joint(name).dof_start for name in joint_names],
        dtype=gs.tc_int,
        device=gs.device,
    )
    pos_names, pos_idx = setup_pd_gains(robot, cfg, joint_names, motors_dof_idx)

    dof_pos = apply_spawn_pose(robot, cfg, motors_dof_idx)
    sim_dof_pos = robot.get_dofs_position(motors_dof_idx)[0]

    print("Spawn configuration:")
    print(f"  base_init_pos = {cfg['base_init_pos']}")
    for name, target, actual in zip(joint_names, dof_pos.tolist(), sim_dof_pos.tolist(), strict=True):
        print(f"  {name:20s}  target={target:+.3f} rad  sim={actual:+.3f} rad")

    max_steps = args.steps if args.steps > 0 else (5 if "PYTEST_VERSION" in os.environ else None)
    step = 0
    while max_steps is None or step < max_steps:
        if not args.no_pd:
            pos_default = torch.tensor(
                [cfg["default_joint_angles"][n] for n in pos_names],
                dtype=gs.tc_float,
                device=gs.device,
            )
            robot.control_dofs_position(pos_default.unsqueeze(0), pos_idx)
        scene.step()
        step += 1


if __name__ == "__main__":
    main()
