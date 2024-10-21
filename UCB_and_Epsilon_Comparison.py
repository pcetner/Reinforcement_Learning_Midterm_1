import numpy as np
import mplcyberpunk
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn Flare color palette for the line colors
colors = sns.color_palette("flare", as_cmap=False)

# Parameters for the bandit problem
n_arms = 5
true_means = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # True means for each arm
std_dev = 1  # Standard deviation for the rewards
n_iterations = 1000  # Total time steps
n_experiments = 100  # Number of experiments
best_arm_mean = true_means[-1]  # The 5th arm has the highest expected reward (0.9)

# ε-Greedy algorithm function
def epsilon_greedy(epsilon=0.1):
    estimated_means = np.zeros(n_arms)
    n_pulls = np.zeros(n_arms)
    rewards = np.zeros(n_iterations)
    
    for t in range(n_iterations):
        # Choose arm using ε-Greedy strategy
        if np.random.rand() < epsilon:
            arm = np.random.choice(n_arms)  # Explore
        else:
            arm = np.argmax(estimated_means)  # Exploit
        
        # Draw reward from the selected arm
        reward = np.random.normal(loc=true_means[arm], scale=std_dev)
        rewards[t] = reward
        
        # Update estimated mean of the selected arm
        n_pulls[arm] += 1
        estimated_means[arm] += (reward - estimated_means[arm]) / n_pulls[arm]
    
    return np.cumsum(rewards), best_arm_mean * np.arange(1, n_iterations + 1) - np.cumsum(rewards)

# ε-Greedy with Decay algorithm function
def epsilon_greedy_decay(initial_epsilon=1.0, epsilon_decay=0.01):
    estimated_means = np.zeros(n_arms)
    n_pulls = np.zeros(n_arms)
    rewards = np.zeros(n_iterations)
    epsilon = initial_epsilon  # Start with initial epsilon

    for t in range(n_iterations):
        # Choose arm using ε-Greedy strategy with decay
        if np.random.rand() < epsilon:
            arm = np.random.choice(n_arms)  # Explore
        else:
            arm = np.argmax(estimated_means)  # Exploit
        
        # Draw reward from the selected arm
        reward = np.random.normal(loc=true_means[arm], scale=std_dev)
        rewards[t] = reward
        
        # Update estimated mean of the selected arm
        n_pulls[arm] += 1
        estimated_means[arm] += (reward - estimated_means[arm]) / n_pulls[arm]
        
        # Decay epsilon, ensuring it doesn't drop below 0
        epsilon = max(0, epsilon - epsilon_decay)
    
    return np.cumsum(rewards), best_arm_mean * np.arange(1, n_iterations + 1) - np.cumsum(rewards)

# UCB algorithm function
def ucb():
    estimated_means = np.zeros(n_arms)
    n_pulls = np.zeros(n_arms)
    rewards = np.zeros(n_iterations)
    
    for t in range(1, n_iterations + 1):
        # Calculate UCB values for each arm
        ucb_values = np.zeros(n_arms)
        for i in range(n_arms):
            if n_pulls[i] > 0:
                ucb_values[i] = estimated_means[i] + np.sqrt(2 * np.log(t) / n_pulls[i])
            else:
                ucb_values[i] = float('inf')  # Force each arm to be selected at least once
        
        # Select arm with the highest UCB value
        arm = np.argmax(ucb_values)
        
        # Draw reward from the selected arm
        reward = np.random.normal(loc=true_means[arm], scale=std_dev)
        rewards[t - 1] = reward
        
        # Update estimated mean of the selected arm
        n_pulls[arm] += 1
        estimated_means[arm] += (reward - estimated_means[arm]) / n_pulls[arm]
    
    return np.cumsum(rewards), best_arm_mean * np.arange(1, n_iterations + 1) - np.cumsum(rewards)

# Initialize arrays to store cumulative rewards and regrets
epsilon_greedy_rewards = np.zeros(n_iterations)
epsilon_greedy_regret = np.zeros(n_iterations)
epsilon_greedy_decay_rewards = np.zeros(n_iterations)
epsilon_greedy_decay_regret = np.zeros(n_iterations)
ucb_rewards = np.zeros(n_iterations)
ucb_regret = np.zeros(n_iterations)

# Run experiments
for _ in range(n_experiments):
    eg_rewards, eg_regret = epsilon_greedy()
    epsilon_greedy_rewards += eg_rewards
    epsilon_greedy_regret += eg_regret
    
    eg_decay_rewards, eg_decay_regret = epsilon_greedy_decay()
    epsilon_greedy_decay_rewards += eg_decay_rewards
    epsilon_greedy_decay_regret += eg_decay_regret
    
    ucb_experiment_rewards, ucb_experiment_regret = ucb()
    ucb_rewards += ucb_experiment_rewards
    ucb_regret += ucb_experiment_regret

# Average the results over all experiments
epsilon_greedy_rewards /= n_experiments
epsilon_greedy_regret /= n_experiments
epsilon_greedy_decay_rewards /= n_experiments
epsilon_greedy_decay_regret /= n_experiments
ucb_rewards /= n_experiments
ucb_regret /= n_experiments

# Calculate the average reward per timestep (by dividing cumulative rewards by timestep)
epsilon_greedy_avg_reward = epsilon_greedy_rewards / np.arange(1, n_iterations + 1)
epsilon_greedy_decay_avg_reward = epsilon_greedy_decay_rewards / np.arange(1, n_iterations + 1)
ucb_avg_reward = ucb_rewards / np.arange(1, n_iterations + 1)

# Plot average reward per timestep and regret in subplots
plt.style.use("cyberpunk")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Average Reward per Timestep comparison plot
ax1.plot(epsilon_greedy_avg_reward, label='ε-Greedy Avg Reward', color=colors[2])
ax1.plot(epsilon_greedy_decay_avg_reward, label='ε-Greedy with Decay Avg Reward', color=colors[1])
ax1.plot(ucb_avg_reward, label='UCB Avg Reward', color=colors[5])
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Average Reward per Timestep')
ax1.set_title('Average Reward per Timestep: ε-Greedy vs UCB vs ε-Greedy with Decay')
ax1.legend()
ax1.set_ylim(0, 1)  # Set y-axis limits to 0 and 1

# Regret comparison plot
ax2.plot(epsilon_greedy_regret, label='ε-Greedy Regret', color=colors[2])
ax2.plot(epsilon_greedy_decay_regret, label='ε-Greedy with Decay Regret', color=colors[1])
ax2.plot(ucb_regret, label='UCB Regret', color=colors[5])
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Regret')
ax2.set_title('Regret Comparison: ε-Greedy vs UCB vs ε-Greedy with Decay')
ax2.legend()

plt.show()

# Print final regret after 1000 iterations
print(f"Average regret after {n_iterations} time steps:")
print(f"ε-Greedy: {epsilon_greedy_regret[-1]:.2f}")
print(f"ε-Greedy with Decay: {epsilon_greedy_decay_regret[-1]:.2f}")
print(f"UCB: {ucb_regret[-1]:.2f}")
