import gym
from envs.drone_wrapper import MultiAgentDroneEnv

gym.register(
    id='DroneFormation-v0',
    entry_point='envs.drone_wrapper:MultiAgentDroneEnv',
    kwargs={'num_drones': 10, 'obs': 'kin', 'act': 'one_d_rpm', 'reward_type': "formation"}
)
