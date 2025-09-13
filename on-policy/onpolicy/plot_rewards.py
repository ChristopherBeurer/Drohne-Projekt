import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/mappo_drones/drone_env/drone_formation/seed1/episode_rewards_per_agent.csv', index_col=0)

plt.figure(figsize=(12, 6))
for agent_id in df.columns:
    plt.plot(df.index, df[agent_id], label=f'Agent {agent_id}')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Episoden-Gesamtreward pro Agent')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
