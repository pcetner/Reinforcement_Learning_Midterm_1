import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
import seaborn as sns
from scipy.stats import norm
import time

# Number of arms
n_arms = 5

# True means for each arm
true_means = [.1, .3, .5, .7, .9]

# Set the standard deviation for all arms
std_dev = 1

# Parameters for ε-Greedy
epsilon = 0.1
n_iterations = 1000  # Total number of pulls
plot_delay = 0.000001  # Delay in seconds for each update

# Initialize arrays to track the estimated means and counts of pulls for each arm
estimated_means = np.zeros(n_arms)
n_pulls = np.zeros(n_arms)

# Prepare to collect samples for plotting
all_samples = [[] for _ in range(n_arms)]

# Create a figure for the plot
plt.style.use("cyberpunk")
fig, axs = plt.subplots(2, 1, figsize=(18, 9))  # Increased width for better legend display

# Adjust the positions of the subplots
axs[0].set_position([0.05, 0.55, 0.8, 0.4])  # [left, bottom, width, height]
axs[1].set_position([0.05, 0.05, 0.8, 0.4])  # Adjusting the bottom subplot

# Defining the X limits for plotting
xlow = -3
xhigh = 4

# Plotting the true distributions in the first subplot
x = np.linspace(xlow,xhigh, 500)  # X values for the probability density function



palette = sns.color_palette("flare", n_arms + 1)  # Generate a color palette for plots

# Top plot: Actual normal distributions
for i in range(n_arms):
    axs[0].plot(x, norm.pdf(x, true_means[i], std_dev), color=palette[i], label=f'Arm {i+1} (True μ={true_means[i]:.2f})')
axs[0].set_title('True Distributions of Each Arm')
axs[0].set_ylabel('Probability Density')
# Adjust the legend position to the right of the plot
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), title="Legends", fontsize='medium')
axs[0].set_xlim(xlow, xhigh)  # Apply x-axis bounds

# Bottom plot: Initial setup for estimated distributions based on sampled points
for i in range(n_arms):
    axs[1].plot(x, norm.pdf(x, estimated_means[i], std_dev), color=palette[i], label=f'Arm {i+1} (Estimated μ={estimated_means[i]:.2f})')

# Main loop for the ε-Greedy algorithm
for _ in range(n_iterations):
    # Choose arm based on ε-Greedy strategy
    if np.random.rand() < epsilon:  # Explore
        arm = np.random.choice(n_arms)
    else:  # Exploit
        arm = np.argmax(estimated_means)

    # Draw a sample from the selected arm's true distribution
    reward = np.random.normal(loc=true_means[arm], scale=std_dev)
    all_samples[arm].append(reward)  # Store the sample for plotting

    # Update counts and estimated means for the selected arm
    n_pulls[arm] += 1
    estimated_means[arm] += (reward - estimated_means[arm]) / n_pulls[arm]  # Incremental update

    # Clear the current axes to redraw
    axs[1].cla()  # Clear the second plot
    
    # Plot the estimated distributions again after updating
    for i in range(n_arms):
        axs[1].plot(x, norm.pdf(x, estimated_means[i], std_dev), color=palette[i], label=f'Arm {i+1} (Estimated μ={estimated_means[i]:.2f})')
        # Plot small sampled points for the arm
        y_offsets = .1 * i  # Small offsets for clarity
        axs[1].scatter(all_samples[i], np.zeros_like(all_samples[i]) + y_offsets, marker='o', color=palette[i], s=30, alpha=0.6)

        # Plot larger glowing dots at the mean position
        if n_pulls[i] > 0:
            axs[1].scatter(estimated_means[i], 0.02 + y_offsets, color=palette[i], s=200, edgecolor='black', alpha=0.8, linewidth=2, label=f'Mean Arm {i+1}')

    axs[1].set_title('Estimated Distributions Based on Samples')
    axs[1].set_ylabel('Probability Density')
    axs[1].set_xlabel('Reward')
    # Adjust the legend position to the right of the plot
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), title="Legends", fontsize='medium')
    axs[1].set_xlim(xlow, xhigh)  # Apply x-axis bounds

    # Pause for a moment to visualize updates
    plt.pause(plot_delay)

plt.show()
