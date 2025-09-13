from gymnasium import spaces
import numpy as np
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary

class DecentralizedCommMultiHoverAviary(MultiHoverAviary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        own_state_len = len(self._getDroneStateVector(0))
        comm_len = 3  # Nur Position (x, y, z) pro anderer Drohne
        obs_len = own_state_len + (self.NUM_DRONES - 1) * comm_len
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.NUM_DRONES, obs_len),
            dtype=np.float32
        )

    def _computeObs(self):
        obs = []
        for i in range(self.NUM_DRONES):
            own_state = self._getDroneStateVector(i)
            others_state = []
            for j in range(self.NUM_DRONES):
                if i != j:
                    other = self._getDroneStateVector(j)
                    others_state.extend(other[0:3])  # Nur x, y, z
            obs.append(np.concatenate([own_state, np.array(others_state)]))
        return np.array(obs)
