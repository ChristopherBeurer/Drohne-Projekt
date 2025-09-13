import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/MAPPO_Test1.csv")


plt.figure(figsize=(10, 5))
plt.plot(df["Step"], df["Value"], color='mediumorchid', marker='o', linestyle='-',
         markersize=4, linewidth=1.5, alpha=0.7)

plt.title("MAPPO: Evaluierter Reward über Timesteps")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.grid(True)
plt.tight_layout()

plt.savefig("plot_mappo_test1.png", dpi=150)
plt.show()
