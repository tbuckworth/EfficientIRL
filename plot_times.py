import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/EIRL_times.csv")

# Create the plot
plt.figure(figsize=(10, 6))

# Group by algorithm
for algo, group in df.groupby("algo"):
    plt.plot(group["total_time"], group["mean_rewards"], label=algo)
    plt.fill_between(
        group["total_time"], 
        group["mean_rewards"] - group["std_rewards"], 
        group["mean_rewards"] + group["std_rewards"], 
        alpha=0.2
    )

# Labels and legend
plt.xlabel("Total Time")
plt.ylabel("Mean Rewards")
plt.legend(title="Algorithm")
plt.title("Mean Rewards Over Time")
plt.grid(True)

# Show plot
plt.show()
