import numpy as np
import matplotlib.pyplot as plt
import mplcyberpunk
import seaborn as sns
from scipy.stats import norm
from matplotlib.animation import FuncAnimation

# Number of arms
n_arms = 5

# True means for each arm
true_means = [.1, .3, .5, .7, .9]

# Set initial standard deviation for all arms
initial_std_dev = 1

# Number of iterations (total number of pulls)
n_iterations = 1000
animation_duration = 20  # Duration of the animation in seconds
fps = 30  # Frames per second
n_frames = animation_duration * fps  # Total number of frames
plot_delay = 1 / fps  # Delay in seconds for each update

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
x = np.linspace(xlow, xhigh, 500)  # X values for the probability density function

palette = sns.color_palette("flare", n_arms + 1)  # Generate a color palette for plots

# Top plot: Actual normal distributions
for i in range(n_arms):
    axs[0].plot(x, norm.pdf(x, true_means[i], initial_std_dev), color=palette[i], label=f'Arm {i+1} (True μ={true_means[i]:.2f})')
axs[0].set_title('True Distributions of Each Arm')
axs[0].set_ylabel('Probability Density')
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), title="Legends", fontsize='medium')
axs[0].set_xlim(xlow, xhigh)  # Apply x-axis bounds

# Bottom plot: Initial setup for estimated distributions based on sampled points
for i in range(n_arms):
    axs[1].plot(x, norm.pdf(x, estimated_means[i], initial_std_dev), color=palette[i], label=f'Arm {i+1} (Estimated μ={estimated_means[i]:.2f})')

# Animation function
def update(t):
    global estimated_means, n_pulls

    # Upper Confidence Bound (UCB) strategy
    ucb_values = np.zeros(n_arms)
    for i in range(n_arms):
        if n_pulls[i] > 0:
            ucb_values[i] = estimated_means[i] + np.sqrt(2 * np.log(t + 1) / n_pulls[i])
        else:
            ucb_values[i] = float('inf')  # Pull each arm at least once
    
    # Select the arm with the highest UCB value
    arm = np.argmax(ucb_values)

    # Draw a sample from the selected arm's true distribution
    reward = np.random.normal(loc=true_means[arm], scale=initial_std_dev)
    all_samples[arm].append(reward)  # Store the sample for plotting

    # Update counts and estimated means for the selected arm
    n_pulls[arm] += 1
    estimated_means[arm] += (reward - estimated_means[arm]) / n_pulls[arm]  # Incremental update

    # Update the standard deviation based on the number of pulls
    current_std_dev = np.array([1/np.sqrt(n) if n > 0 else initial_std_dev for n in n_pulls])

    # Clear the current axes to redraw
    axs[1].cla()  # Clear the second plot

    # Plot the estimated distributions again after updating
    y_max_vals = []
    for i in range(n_arms):
        y_vals = norm.pdf(x, estimated_means[i], current_std_dev[i])
        axs[1].plot(x, y_vals, color=palette[i], label=f'Arm {i+1} (Estimated μ={estimated_means[i]:.2f})')
        y_max_vals.append(np.max(y_vals))

    # Get the maximum y-value across all arms to dynamically adjust y offsets
    max_y_value = max(y_max_vals)

    for i in range(n_arms):
        # Use the maximum y value to adjust the offsets
        y_offsets = 0.18 * max_y_value * i  # Scale based on max y-value

        # Plot small sampled points for the arm
        axs[1].scatter(all_samples[i], np.zeros_like(all_samples[i]) + y_offsets, marker='o', color=palette[i], s=30, alpha=0.6)

        # Plot larger glowing dots at the mean position
        if n_pulls[i] > 0:
            axs[1].scatter(estimated_means[i], 0.02 * max_y_value + y_offsets, color=palette[i], s=200, edgecolor='black', alpha=0.8, linewidth=2, label=f'Mean Arm {i+1}')

            # Confidence interval for each arm
            z = 1.96  # For a 95% confidence interval
            conf_interval = z * (1 / np.sqrt(n_pulls[i]))
            axs[1].vlines(estimated_means[i], 0.02 * max_y_value + y_offsets - conf_interval, 
                          0.02 * max_y_value + y_offsets + conf_interval, color=palette[i], linewidth=3, alpha=0.8)

    axs[1].set_title('Estimated Distributions Based on Samples (UCB) with Confidence Intervals')
    axs[1].set_ylabel('Probability Density')
    axs[1].set_xlabel('Reward')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), title="Legends", fontsize='medium')
    axs[1].set_xlim(xlow, xhigh)  # Apply x-axis bounds

# Create animation
anim = FuncAnimation(fig, update, frames=n_frames, interval=plot_delay * 1000, repeat=False)

# Save the animation as a GIF file using Pillow
anim.save('ucb_algorithm_animation.mp4', writer='ffmpeg', fps=fps)

# Show the final plot (optional)
plt.show()
