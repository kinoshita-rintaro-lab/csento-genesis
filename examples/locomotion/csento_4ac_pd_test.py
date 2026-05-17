"""
Check PD position control on csento-4ac (same kp/kd as train).

  cd ~/Genesis
  export DISPLAY=:0

  # 1) Hold default pose 5 s — should stay upright with small joint error
  uv run python examples/locomotion/csento_4ac_pd_test.py --mode hold --seconds 5

  # 2) Track sinusoid on left hip — PD should follow the moving target
  uv run python examples/locomotion/csento_4ac_pd_test.py --mode track --joint left_hip_joint --seconds 5

  # Use training cfgs (kp/kd, default angles, base height)
  uv run python examples/locomotion/csento_4ac_pd_test.py --cfgs logs/csento-4ac/cfgs.pkl --mode hold
"""

import argparse
import math
import os
import pickle

import genesis as gs
import torch

from csento_4ac_spawn import apply_spawn_pose, load_spawn_cfg


def build_robot_scene(cfg: dict, show_viewer: bool):
    dt = 0.02
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2),
        rigid_options=gs.options.RigidOptions(enable_self_collision=False),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 0.0, 1.2),
            camera_lookat=(0.0, 0.0, cfg["base_init_pos"][2]),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
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
    return scene, robot, dt


def setup_pd_gains(robot, cfg: dict, joint_names: list[str], motors_dof_idx: torch.Tensor):
    pos_names = cfg.get(
        "position_joint_names",
        [n for n in joint_names if n not in cfg.get("torque_joint_names", [])],
    )
    pos_idx = torch.tensor(
        [robot.get_joint(n).dof_start for n in pos_names],
        dtype=gs.tc_int,
        device=gs.device,
    )
    kp = cfg.get("kp", 15.0)
    kd = cfg.get("kd", 0.4)
    robot.set_dofs_kp([kp] * len(pos_names), pos_idx)
    robot.set_dofs_kv([kd] * len(pos_names), pos_idx)

    torque_names = cfg.get("torque_joint_names", [])
    if torque_names:
        t_idx = torch.tensor(
            [robot.get_joint(n).dof_start for n in torque_names],
            dtype=gs.tc_int,
            device=gs.device,
        )
        robot.set_dofs_kp([cfg.get("wheel_kp", 0.0)] * len(torque_names), t_idx)
        robot.set_dofs_kv([cfg.get("wheel_kd", 0.0)] * len(torque_names), t_idx)

    return pos_names, pos_idx


def run_hold(scene, robot, pos_names, pos_idx, pos_default, cfg, dt, seconds, verbose):
    target = pos_default.clone()
    n_steps = int(seconds / dt)
    max_err = torch.zeros(len(pos_names), device=gs.device)
    base_z_min = 1e9
    base_z_max = -1e9

    for _ in range(n_steps):
        robot.control_dofs_position(target.unsqueeze(0), pos_idx)
        scene.step()

        actual = robot.get_dofs_position(pos_idx)[0]
        max_err = torch.maximum(max_err, torch.abs(actual - target))
        z = robot.get_pos()[0, 2].item()
        base_z_min = min(base_z_min, z)
        base_z_max = max(base_z_max, z)

    if verbose:
        print("\n[hold] PD keeps default pose")
        print(f"  duration={seconds}s  kp={cfg.get('kp')}  kd={cfg.get('kd')}")
        print(f"  base z range: {base_z_min:.3f} .. {base_z_max:.3f} m")
        for name, err in zip(pos_names, max_err.tolist(), strict=True):
            status = "OK" if err < 0.05 else ("WARN" if err < 0.15 else "BAD")
            print(f"  {name:20s}  max|q-q*|={err:.4f} rad  [{status}]")
        ok = max_err.max().item() < 0.15 and (base_z_max - base_z_min) < 0.05
        print(f"\n  => {'PD looks OK' if ok else 'PD may be weak or pose unstable — tune kp/kd or default angles'}")


def run_track(scene, robot, pos_names, pos_idx, pos_default, dt, seconds, joint_name, amp, freq, verbose):
    if joint_name not in pos_names:
        raise ValueError(f"Joint {joint_name!r} is not position-controlled. Choose from {pos_names}")

    j = pos_names.index(joint_name)
    q0 = pos_default[j].item()
    n_steps = int(seconds / dt)
    max_err = 0.0

    for step in range(n_steps):
        t = step * dt
        target = pos_default.clone()
        target[j] = q0 + amp * math.sin(2.0 * math.pi * freq * t)
        robot.control_dofs_position(target.unsqueeze(0), pos_idx)
        scene.step()

        actual = robot.get_dofs_position(pos_idx)[0, j].item()
        max_err = max(max_err, abs(actual - target[j].item()))

    if verbose:
        print(f"\n[track] sinusoid on {joint_name}")
        print(f"  q*(t) = {q0:.3f} + {amp:.3f}*sin(2*pi*{freq}*t)")
        print(f"  max|q-q*| = {max_err:.4f} rad")
        print(f"  => {'tracking OK' if max_err < 0.1 else 'poor tracking — increase kp or check joint index'}")


def main():
    parser = argparse.ArgumentParser(description="Test PD position control for csento-4ac.")
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    parser.add_argument("-nv", "--no-vis", action="store_false", dest="vis")
    parser.add_argument("--cfgs", type=str, default=None)
    parser.add_argument("--mode", choices=["hold", "track"], default="hold")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--joint", type=str, default="left_hip_joint")
    parser.add_argument("--amp", type=float, default=0.2, help="Sinusoid amplitude [rad] for track mode")
    parser.add_argument("--freq", type=float, default=0.5, help="Sinusoid frequency [Hz] for track mode")
    args = parser.parse_args()

    cfg = load_spawn_cfg(args.cfgs)
    gs.init(backend=gs.gpu if args.vis else gs.cpu, logging_level="warning")

    scene, robot, dt = build_robot_scene(cfg, args.vis)
    joint_names = cfg["joint_names"]
    motors_dof_idx = torch.tensor(
        [robot.get_joint(n).dof_start for n in joint_names],
        dtype=gs.tc_int,
        device=gs.device,
    )
    pos_names, pos_idx = setup_pd_gains(robot, cfg, joint_names, motors_dof_idx)
    default_dof_pos = apply_spawn_pose(robot, cfg, motors_dof_idx)

    print("PD test setup:")
    print(f"  position joints ({len(pos_names)}): {pos_names}")
    print(f"  torque joints: {cfg.get('torque_joint_names', [])}")
    print(f"  kp={cfg.get('kp')}  kd={cfg.get('kd')}")

    pos_default = torch.tensor(
        [cfg["default_joint_angles"][n] for n in pos_names],
        dtype=gs.tc_float,
        device=gs.device,
    )

    if args.mode == "hold":
        run_hold(scene, robot, pos_names, pos_idx, pos_default, cfg, dt, args.seconds, verbose=True)
    else:
        run_track(
            scene, robot, pos_names, pos_idx, pos_default, dt, args.seconds,
            args.joint,
            args.amp,
            args.freq,
            verbose=True,
        )

    if args.vis and "PYTEST_VERSION" not in os.environ:
        print("\nViewer open — close window to exit.")
        while True:
            robot.control_dofs_position(pos_default.unsqueeze(0), pos_idx)
            scene.step()


if __name__ == "__main__":
    main()
