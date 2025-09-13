import numpy as np
import matplotlib.pyplot as plt

data = np.load("results/MAPPO_Test1.csv")
timesteps = data["timesteps"]
rewards = data["results"].squeeze()

plt.figure(figsize=(8,4))
plt.plot(timesteps, rewards, marker='o')
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Lernfortschritt (Reward über Zeit)")
plt.grid(True)
plt.show()
