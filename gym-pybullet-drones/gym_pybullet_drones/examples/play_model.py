from stable_baselines3 import PPO
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

MODEL_PATH = "results/save-07.09.2025_13.15.43/final_model.zip"  
NUM_DRONES = 2  


OBS_TYPE = ObservationType('kin')
ACT_TYPE = ActionType('one_d_rpm')
SEED = 42 

env = MultiHoverAviary(
    num_drones=NUM_DRONES,
    gui=True,
    obs=OBS_TYPE,
    act=ACT_TYPE
)
model = PPO.load(MODEL_PATH)

num_episodes = 5 

for episode in range(num_episodes):
    print(f"Starte Episode {episode+1}/{num_episodes}")
    obs, info = env.reset(seed=SEED, options={})
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
env.close()
