import time
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary

MODEL_PATH = "../files/logs/ppo_multiagent.zip"  

NUM_DRONES = 3  

env = MultiHoverAviary(
    num_drones=NUM_DRONES,
    gui=True,           
    record=False,       
    obs=None,            
    act=None             
)

# === Modell laden ===
model = PPO.load(MODEL_PATH)

obs = env.reset()
done = False
step = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    step += 1
    time.sleep(1.0 / 60.0) 

env.close()
print(f"Simulation beendet nach {step} Schritten.")
