import numpy as np
import matplotlib.pyplot as plt

runs = [
    "results/save-07.09.2025_13.15.43/evaluations.npz",
    "results/save-07.09.2025_13.51.27/evaluations.npz"
]

plt.figure(figsize=(10, 5))

for run in runs:
    data = np.load(run)
    timesteps = data["timesteps"]
    rewards = data["results"].squeeze()
    plt.plot(timesteps, rewards, marker='o', label=run.split('/')[1]) 

plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Vergleich der Lernkurven verschiedener Trainingsläufe")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
