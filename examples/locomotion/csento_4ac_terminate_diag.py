"""Diagnose why episodes terminate early (roll/pitch/sim_error/timeout)."""

import math
import pickle
from collections import defaultdict

import genesis as gs
import torch

from csento_4ac_train import get_cfgs


def build_init_qpos_broken(robot, env_cfg):
    """Current env.py logic (joints[1:] order)."""
    default_angles = env_cfg["default_joint_angles"]
    init_base_pos = torch.tensor(env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
    init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
    init_dof = torch.tensor(
        [default_angles.get(j.name, 0.0) for j in robot.joints[1:]],
        dtype=gs.tc_float,
        device=gs.device,
    )
    return torch.cat((init_base_pos, init_base_quat, init_dof)), init_dof


def build_init_qpos_fixed(robot, env_cfg):
    """Per-joint q_start indexing."""
    default_angles = env_cfg["default_joint_angles"]
    qpos = torch.zeros(robot.n_qs, dtype=gs.tc_float, device=gs.device)
    qpos[0:3] = torch.tensor(env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
    qpos[3:7] = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
    q_off = robot.q_start
    for name, angle in default_angles.items():
        j = robot.get_joint(name)
        if j.n_qs == 1:
            qpos[j.q_start - q_off] = angle
    return qpos, torch.tensor(
        [default_angles[n] for n in env_cfg["joint_names"]],
        dtype=gs.tc_float,
        device=gs.device,
    )


def run_episode_diag(env_cfg, use_fixed_reset, num_steps=500, zero_action=True):
    gs.init(backend=gs.cpu, logging_level="warning")
    dt = 0.02

    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=2))
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(file=env_cfg["urdf_path"], pos=env_cfg["base_init_pos"], quat=env_cfg["base_init_quat"])
    )
    scene.build(n_envs=1)

    names = env_cfg["joint_names"]
    motors = torch.tensor([robot.get_joint(n).dof_start for n in names], dtype=gs.tc_int, device=gs.device)
    robot.set_dofs_kp([env_cfg["kp"]] * len(names), motors)
    robot.set_dofs_kv([env_cfg["kd"]] * len(names), motors)

    default_dof = torch.tensor([env_cfg["default_joint_angles"][n] for n in names], dtype=gs.tc_float, device=gs.device)

    if use_fixed_reset:
        init_qpos, _ = build_init_qpos_fixed(robot, env_cfg)
    else:
        init_qpos, _ = build_init_qpos_broken(robot, env_cfg)

    robot.set_qpos(init_qpos, zero_velocity=True, skip_forward=True)
    robot.set_dofs_position(default_dof.unsqueeze(0), motors, zero_velocity=True, skip_forward=True)

    from genesis.utils.geom import inv_quat, quat_to_xyz, transform_quat_by_quat

    inv_init = inv_quat(torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device))

    counts = defaultdict(int)
    first_term = None

    for step in range(num_steps):
        if zero_action:
            target = default_dof.clone()
        else:
            target = default_dof + 0.5 * torch.sin(torch.tensor(step * 0.1))
        robot.control_dofs_position(target.unsqueeze(0), motors)
        scene.step()

        quat = robot.get_quat()[0]
        euler = quat_to_xyz(transform_quat_by_quat(inv_init, quat), rpy=True, degrees=True)
        roll, pitch = euler[0].item(), euler[1].item()
        z = robot.get_pos()[0, 2].item()
        sim_err = scene.rigid_solver.get_error_envs_mask()[0].item()

        term_timeout = step + 1 > math.ceil(env_cfg["episode_length_s"] / dt)
        term_pitch = abs(pitch) > env_cfg["termination_if_pitch_greater_than"]
        term_roll = abs(roll) > env_cfg["termination_if_roll_greater_than"]
        term_sim = bool(sim_err)

        if term_pitch or term_roll or term_sim or term_timeout:
            if first_term is None:
                reason = []
                if term_roll:
                    reason.append(f"roll={roll:.1f}")
                if term_pitch:
                    reason.append(f"pitch={pitch:.1f}")
                if term_sim:
                    reason.append("sim_error")
                if term_timeout:
                    reason.append("timeout")
                first_term = (step, ", ".join(reason), z)
            if term_roll:
                counts["roll"] += 1
            if term_pitch:
                counts["pitch"] += 1
            if term_sim:
                counts["sim_error"] += 1
            if term_timeout:
                counts["timeout"] += 1
            break

    if first_term is None:
        first_term = (num_steps, "none", robot.get_pos()[0, 2].item())
    return first_term, counts


def compare_joint_mapping(robot, env_cfg):
    print("\n=== Joint / qpos mapping ===")
    print(f"robot.n_qs={robot.n_qs}  n_dofs={robot.n_dofs}")
    for i, j in enumerate(robot.joints):
        print(f"  joints[{i}] {j.name:22s} n_qs={j.n_qs} q_start={j.q_start}")

    broken_q, broken_dof = build_init_qpos_broken(robot, env_cfg)
    fixed_q, fixed_dof = build_init_qpos_fixed(robot, env_cfg)
    names = env_cfg["joint_names"]
    motors = torch.tensor([robot.get_joint(n).dof_start for n in names], dtype=gs.tc_int, device=gs.device)

    robot.set_qpos(broken_q, zero_velocity=True, skip_forward=True)
    after_broken = robot.get_dofs_position(motors)[0].tolist()

    robot.set_qpos(fixed_q, zero_velocity=True, skip_forward=True)
    after_fixed = robot.get_dofs_position(motors)[0].tolist()

    target = [env_cfg["default_joint_angles"][n] for n in names]
    print("\n  joint_names order     ", [f"{t:+.2f}" for t in target])
    print("  after set_qpos BROKEN ", [f"{v:+.2f}" for v in after_broken])
    print("  after set_qpos FIXED  ", [f"{v:+.2f}" for v in after_fixed])
    mismatch = any(abs(a - t) > 0.01 for a, t in zip(after_broken, target, strict=True))
    print(f"  BROKEN init_qpos mismatch: {mismatch}")


def main():
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    print("=== Config ===")
    print(f"  base_init_pos z = {env_cfg['base_init_pos'][2]}")
    print(f"  base_height_target = {reward_cfg['base_height_target']}")
    print(f"  termination roll/pitch = {env_cfg['termination_if_roll_greater_than']} deg")
    print(f"  kp={env_cfg['kp']} kd={env_cfg['kd']}")
    print(f"  reward_scales keys = {list(reward_cfg['reward_scales'].keys())}")

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene()
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(gs.morphs.URDF(file=env_cfg["urdf_path"]))
    scene.build(n_envs=1)
    compare_joint_mapping(robot, env_cfg)

    print("\n=== Episode until terminate (action=0, PD holds default) ===")
    for label, fixed in [("BROKEN reset (env.py)", False), ("FIXED reset (q_start)", True)]:
        step, reason, z = run_episode_diag(env_cfg, fixed, num_steps=1000, zero_action=True)[0]
        t_sec = (step + 1) * 0.02
        print(f"  {label}: terminates at step {step} ({t_sec:.2f}s) — {reason}  z={z:.3f}m")

    try:
        with open("logs/csento-4ac/cfgs.pkl", "rb") as f:
            pkl_cfg, *_ = pickle.load(f)
        if pkl_cfg.get("base_init_pos") != env_cfg["base_init_pos"]:
            print("\n=== cfgs.pkl vs train.py ===")
            print(f"  pkl base_init_pos = {pkl_cfg.get('base_init_pos')}")
            print(f"  train base_init_pos = {env_cfg['base_init_pos']}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
