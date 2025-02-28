import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def plot(csv_file="data/EIRL_times2.csv", output_file="data/times.png"):
    # Load data
    df = pd.read_csv(csv_file)

    df = df.rename(columns={"mean_reards": "mean_rewards"})

    max_time = df.total_time.max()
    # TODO: add the final value of each algo to the end.
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Group by algorithm
    for algo, group in df.groupby("algo"):
        plt.plot(group["total_time"], group["mean_rewards"], label=algo)
        plt.fill_between(
            group["total_time"],
            group["mean_rewards"] - group["std_error"],
            group["mean_rewards"] + group["std_error"],
            alpha=0.2
        )

    # Labels and legend
    plt.xlabel("Total Time")
    plt.ylabel("Mean Rewards")
    plt.legend(title="Algorithm")
    plt.title("Mean Rewards Over Time")
    plt.grid(True)
    plt.savefig(output_file)
    # Show plot
    plt.show()

if __name__ == "__main__":
    plot("data/EIRL_timing_suboptimal.csv")
