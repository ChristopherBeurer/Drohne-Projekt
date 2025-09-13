import os
import torch
import numpy as np
import argparse

from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from envs.drone_wrapper import MultiAgentDroneEnv
from onpolicy.runner.separated.mpe_runner import MPERunner 
from pathlib import Path


def make_env():
    """Erzeuge eine Trainings- oder Evaluationsumgebung."""
    return MultiAgentDroneEnv(
        num_drones=10,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        reward_type="formation"
    )

def main():
    # Argument-Setup
    all_args = argparse.Namespace()
    all_args.cuda = torch.cuda.is_available()
    all_args.use_eval = True
    all_args.use_recurrent_policy = False
    all_args.use_obs_instead_of_state = False
    all_args.use_naive_recurrent_policy = False
    all_args.use_centralized_V = True
    all_args.algorithm_name = "mappo"
    all_args.experiment_name = "drone_formation"
    all_args.env_name = "drone_env"
    all_args.scenario_name = "drone_formation"
    all_args.model_dir = None
    all_args.use_wandb = False
    all_args.use_render = False

    # Hardware & Logging
    all_args.cuda = torch.cuda.is_available()
    all_args.use_wandb = False
    all_args.use_render = False
    all_args.log_interval = 10
    all_args.save_interval = 1000
    all_args.eval_interval = 1000
    all_args.model_dir = None
    all_args.save_dir = "./results/mappo_drones/"

    # MAPPO & Umgebung
    all_args.algorithm_name = "mappo"
    all_args.env_name = "drone_env"
    all_args.experiment_name = "drone_formation"
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1
    all_args.num_env_steps = int(1e6)
    all_args.episode_length = 250
    all_args.use_eval = True
    all_args.seed = 4

    # Netzstruktur
    all_args.hidden_size = 64
    all_args.layer_N = 1
    all_args.use_ReLU = True
    all_args.use_orthogonal = True
    all_args.gain = 0.01

    # Optimierung
    all_args.lr = 5e-4
    all_args.critic_lr = 5e-4
    all_args.opti_eps = 1e-5
    all_args.weight_decay = 0.0
    all_args.ppo_epoch = 5
    all_args.clip_param = 0.2
    all_args.num_mini_batch = 1
    all_args.data_chunk_length = 10
    all_args.max_grad_norm = 0.5

    # Verluste
    all_args.value_loss_coef = 0.5
    all_args.entropy_coef = 0.01
    all_args.use_huber_loss = True
    all_args.huber_delta = 10.0
    all_args.use_clipped_value_loss = True

    # Normalisierung & Stabilität
    all_args.use_feature_normalization = True
    all_args.use_popart = False
    all_args.use_tanh = False

    # Policy
    all_args.use_centralized_V = True
    all_args.use_naive_recurrent_policy = False
    all_args.use_recurrent_policy = False
    all_args.recurrent_N = 1
    all_args.use_obs_instead_of_state = False
    all_args.use_policy_active_masks = False
    all_args.use_value_active_masks = False
    all_args.share_policy = False

    # Sonstiges
    all_args.gamma = 0.99
    all_args.gae_lambda = 0.95
    all_args.use_linear_lr_decay = False
    all_args.stacked_frames = 1
    all_args.use_proper_time_limits = False


    all_args.num_env_steps = int(1e6)
    all_args.episode_length = 250
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1
    all_args.num_mini_batch = 1
    all_args.data_chunk_length = 10
    all_args.recurrent_N = 1
    all_args.hidden_size = 64
    all_args.layer_N = 1
    all_args.use_popart = False
    all_args.ppo_epoch = 5
    all_args.ppo_epoch = 5
    all_args.use_huber_loss = True
    all_args.huber_delta = 10.0
    all_args.use_proper_time_limits = False
    all_args.use_policy_active_masks = False
    all_args.use_value_active_masks = False
    all_args.use_tanh = False
    all_args.share_policy = False
    all_args.entropy_coeff = 0.01
    all_args.lr = 5e-4
    all_args.critic_lr = 5e-4 
    all_args.opti_eps = 1e-5
    all_args.weight_decay = 0.0
    all_args.gain = 0.01
    all_args.gamma = 0.99
    all_args.gae_lambda = 0.95
    all_args.use_clipped_value_loss = True
    all_args.use_orthogonal = True
    all_args.use_policy_active_masks = False
    all_args.use_feature_normalization = True
    all_args.use_valuenorm = False
    all_args.use_gae = True
    all_args.use_ReLU = True
    all_args.stacked_frames = 1
    all_args.clip_param = 0.2
    all_args.entropy_coeff = 0.01
    all_args.value_loss_coef = 0.5
    all_args.max_grad_norm = 0.5
    all_args.use_max_grad_norm = True
    all_args.use_value_active_masks = False
    all_args.use_linear_lr_decay = False
    all_args.seed = 4
    all_args.log_interval = 10
    all_args.save_interval = 1000
    all_args.eval_interval = 1000
    all_args.save_dir = "./results/mappo_drones/"

    train_env = make_env()
    eval_env = make_env()

    n_agents = train_env.n_agents
    obs_shape = train_env.observation_space[0].shape[-1]
    act_shape = train_env.action_space[0].shape[-1]

    run_dir = Path(all_args.save_dir) / all_args.env_name / all_args.experiment_name / f"seed{all_args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "all_args": all_args,
        "envs": train_env,       
        "eval_envs": eval_env,    
        "num_agents": n_agents,
        "n_agents": n_agents,
        "obs_shape": obs_shape,
        "act_shape": act_shape,
        "episode_length": all_args.episode_length,
        "device": torch.device("cuda" if all_args.cuda else "cpu"),
        "run_dir": run_dir        
    }

    print(f"\n[INFO] Starte MAPPO-Training auf {n_agents} Drohnen...\n")
    runner = MPERunner(config)
    runner.run()
    print("\n[INFO] Training abgeschlossen. Ergebnisse gespeichert unter:", run_dir)

    print("[DEBUG] Alle Argumente:")
    for k, v in vars(all_args).items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
