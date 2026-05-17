import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from csento_4ac_env import Csento4acEnv


def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,
        "urdf_path": Csento4acEnv.URDF_PATH,
        "default_joint_angles": {
            "left_hip_joint": 1.0,
            "left_knee_joint": 1.6,
            "left_wheel_joint": 0.0,
            "right_hip_joint": 1.0,
            "right_knee_joint": 1.6,
            "right_wheel_joint": 0.0,
        },
        "joint_names": [
            "left_hip_joint",
            "left_knee_joint",
            "left_wheel_joint",
            "right_hip_joint",
            "right_knee_joint",
            "right_wheel_joint",
        ],
        "position_joint_names": [
            "left_hip_joint",
            "left_knee_joint",
            "right_hip_joint",
            "right_knee_joint",
        ],
        "torque_joint_names": [
            "left_wheel_joint",
            "right_wheel_joint",
        ],
        "kp": 15.0,
        "kd": 20.0,
        "wheel_kp": 0.0,
        "wheel_kd": 0.0,
        # wheel action in [-1, 1] -> torque [Nm]; raise if wheels barely spin in eval
        "wheel_torque_scale": 15.0,
        "clip_torques": 30.0,
        "termination_if_roll_greater_than": 45,
        "termination_if_pitch_greater_than": 45,
        "base_init_pos": [0.0, 0.0, 0.25],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 60.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.8,
        "lin_vel_z_sigma": 0.8,
        "orientation_sigma": 0.5,
        "ang_vel_xy_sigma": 0.5,
        "action_rate_sigma": 0.01,
        "similar_to_default_sigma": 0.5,
        "base_height_sigma": 0.5,
        "base_height_target": 0.14,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 1.0,
            "lin_vel_z": 0.5,
            "orientation": 1.0,
            "ang_vel_xy": 1.0,
            "action_rate": 0.005,
            "similar_to_default": 2.0,
            # "tracking_lin_vel": 1.0,
            # "tracking_ang_vel": 1.0,
            # "lin_vel_z": 1.0,
            # "orientation": 1.0,
            # "ang_vel_xy": 1.0,
            # "action_rate": 1.0,
            # "similar_to_default": 1.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 0.0],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="csento-4ac")
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        help="Open Genesis viewer during training (shows env 0). Use a small -B for smooth playback.",
    )
    parser.add_argument("-B", "--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=101)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.num_envs is None:
        args.num_envs = 16 if args.vis else 512

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=not args.vis,
    )

    if args.vis:
        print(f"[vis] Viewer on (env 0 of {args.num_envs}). WSL: export DISPLAY=:0 before running.")

    env = Csento4acEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# headless training (from repo root)
uv run python examples/locomotion/csento_4ac_train.py -B 512 --max_iterations 20

# watch robot while learning (GUI; default -B 16)
export DISPLAY=:0   # WSL: run in Windows Terminal if needed
uv run python examples/locomotion/csento_4ac_train.py -v --max_iterations 500

# learning curves (separate terminal)
uv run tensorboard --logdir logs/csento-4ac

# longer headless run
uv run python examples/locomotion/csento_4ac_train.py -B 4096 --max_iterations 500
"""
