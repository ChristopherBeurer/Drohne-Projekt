import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary

class DecentralizedCommMultiHoverAviary(MultiHoverAviary):
    def __init__(self, *args, reward_type="hover", **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_type = reward_type
        own_state_len = len(self._getDroneStateVector(0)) 
        comm_len = 3 
        obs_len = own_state_len + (self.NUM_DRONES - 1) * comm_len
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.NUM_DRONES, obs_len),
            dtype=np.float32
        )

    def _getFormationTargets(self):
    
        base_pos = np.array([0, 0, 1])
        spacing = 1.0
        return [base_pos + np.array([i * spacing, 0, 0]) for i in range(self.NUM_DRONES)]

    def _computeObs(self):
        obs = []
        for i in range(self.NUM_DRONES):
            own_state = self._getDroneStateVector(i)
            others_state = []
            for j in range(self.NUM_DRONES):
                if i != j:
                    other = self._getDroneStateVector(j)
                    others_state.extend(other[0:3]) 
            obs.append(np.concatenate([own_state, np.array(others_state)]))
        return np.array(obs)

    def _computeReward(self):

        if self.reward_type == "formation":
            return self._formation_reward()
        elif self.reward_type == "hover":
            return self._hover_reward()
        elif self.reward_type == "collision_avoidance":
            return self._collision_avoidance_reward()
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

    def _formation_reward(self):
        formation_targets = self._getFormationTargets()
        reward = 0
        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[0:3]
            target_pos = formation_targets[i]
            reward -= np.linalg.norm(drone_pos - target_pos)
        return reward

    def _hover_reward(self):
    
        target_z = 1.0
        reward = 0
        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[0:3]
            reward -= abs(drone_pos[2] - target_z)
        return reward

    def _collision_avoidance_reward(self):
    
        min_dist = 0.5
        penalty = 0
        for i in range(self.NUM_DRONES):
            for j in range(i+1, self.NUM_DRONES):
                pos_i = self._getDroneStateVector(i)[0:3]
                pos_j = self._getDroneStateVector(j)[0:3]
                dist = np.linalg.norm(pos_i - pos_j)
                if dist < min_dist:
                    penalty -= 5.0 
        return self._hover_reward() + penalty

