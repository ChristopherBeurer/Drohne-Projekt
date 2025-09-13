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

      
        self._last_dists = np.array([1e6] * self.NUM_DRONES)  

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._last_dists = np.array([1e6] * self.NUM_DRONES)
        return obs, info


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
        curr_dists = np.zeros(self.NUM_DRONES)
        progress_reward = 0
        in_formation_count = 0

        baseline = 0.1 * self.NUM_DRONES 

        for i in range(self.NUM_DRONES):
            drone_pos = self._getDroneStateVector(i)[0:3]
            target_pos = formation_targets[i]
            dist = np.linalg.norm(drone_pos - target_pos)
            curr_dists[i] = dist
            if np.isfinite(self._last_dists[i]):
                improvement = max(0, self._last_dists[i] - dist)
                improvement = np.clip(improvement, 0.0, 10.0)
            else:
                improvement = 0.0
            progress_reward += improvement
            if dist < 0.2:
                progress_reward += 1.0
                in_formation_count += 1
        if in_formation_count >= self.NUM_DRONES * 0.8:
            progress_reward += 5.0

        self._last_dists = curr_dists

        return min(progress_reward + baseline, 20.0)



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

