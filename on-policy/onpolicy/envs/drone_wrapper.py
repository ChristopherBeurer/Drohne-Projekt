import gym
import numpy as np
from gymnasium import spaces
from envs.DecentralizedCommMultiHoverAviary import DecentralizedCommMultiHoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

class MultiAgentDroneEnv(gym.Env):
    def __init__(self, num_drones=10, obs=None, act=None, reward_type="formation"):
        self.env = DecentralizedCommMultiHoverAviary(
            num_drones=num_drones,
            obs=obs,
            act=act,
            reward_type=reward_type,
        )
        self.n_agents = num_drones

        single_obs_box = spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype
        )
        self.observation_space = [single_obs_box for _ in range(self.n_agents)]
        self.share_observation_space = [single_obs_box for _ in range(self.n_agents)]

        single_act_box = spaces.Box(
            low=self.env.action_space.low[0],
            high=self.env.action_space.high[0],
            dtype=self.env.action_space.dtype
        )
        self.action_space = [single_act_box for _ in range(self.n_agents)]

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return [obs[i] for i in range(self.n_agents)]


    def step(self, actions):
        if isinstance(actions, list):
            if len(actions) == 1 and (isinstance(actions[0], list) or isinstance(actions[0], np.ndarray)):
                actions = actions[0]
            actions = np.array(actions)

        res = self.env.step(actions)
        if len(res) == 5:
  
            obs, reward, terminated, truncated, info = res
            done = np.logical_or(terminated, truncated)
        elif len(res) == 4:
         
            obs, reward, done, info = res
        else:
            raise ValueError(f"env.step returned {len(res)} values! Expected 4 or 5.")

      
        if isinstance(reward, (int, float)):
            reward = [reward] * self.n_agents

        obs_list = [obs[i] for i in range(self.n_agents)]
        if isinstance(done, (list, np.ndarray)):
        
            done_array = np.array(done, dtype=bool)[None, ...] 
        else:
        
            done_array = np.array([done]*self.n_agents, dtype=bool)[None, ...]

        return obs_list, reward, done_array, info
