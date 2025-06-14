import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from config import param
from eNB import eNB
from UE import UE
from utils.Ticker import Ticker
import random
from dqn_simulation import DQNAgent
from Simulate_UE import Simulate_UE
import pandas as pd

from config.param import ParamConfig
param = ParamConfig({})


from itertools import product
from utils.misc import gigahertz_to_megahertz, kmph_to_mpms

import datetime


def plot_single_segment_from_df(df_overall_simulation, rlf_threshold, episode_number=1, extra_steps=4, save_path="."):
    """
    Plot only one segment, up to the point where the serving BS changes + extra_steps.
    Args:
        df_overall_simulation: DataFrame with columns ['t', 'serv_power', 'serv_id', 'target_power', 'target_id']
        rlf_threshold: Threshold line to plot on graphs
        episode_number: For naming the output image
        extra_steps: Additional timesteps to include after serving BS change
        save_path: Base path to save plot
        by removing 500 as the taget network update time
    """
    if df_overall_simulation.empty:
        print(f"Warning: Episode {episode_number} produced an empty simulation result. Cannot plot.")
        return

    segment = {
        "times": [],
        "serving_powers": [],
        "target_powers": [],
        "serving_eNB_id": df_overall_simulation.iloc[0]['serv_id'],
        "target_eNB_ids": []
    }

    i = 0
    bs_changed = False
    while i < len(df_overall_simulation):
        row = df_overall_simulation.iloc[i]
        t = row['t']
        serv_power = row['serv_power']
        targ_power = row['target_power']
        serv_id = row['serv_id']
        targ_id = row['target_id']

        segment["times"].append(t)
        segment["serving_powers"].append(serv_power)
        segment["target_powers"].append(targ_power)
        segment["target_eNB_ids"].append(targ_id)

        if not bs_changed and serv_id != segment["serving_eNB_id"]:
            bs_changed = True
            # Include a few more steps after change
            for j in range(1, extra_steps + 1):
                if i + j >= len(df_overall_simulation):
                    break
                future_row = df_overall_simulation.iloc[i + j]
                segment["times"].append(future_row['t'])
                segment["serving_powers"].append(future_row['serv_power'])
                segment["target_powers"].append(future_row['target_power'])
                segment["target_eNB_ids"].append(future_row['target_id'])
            break

        i += 1

    # Create output directory
    output_dir = os.path.join(save_path, "results_simulation")
    os.makedirs(output_dir, exist_ok=True)

    # Plot the single segment
    plt.figure(figsize=(10, 6))
    plt.plot(segment["times"], segment["serving_powers"], label="Serving BS Power", color="blue")
    plt.plot(segment["times"], segment["target_powers"], label="Target BS Power", color="red")
    plt.axhline(y=rlf_threshold, color='gray', linestyle='--', label="RLF Threshold")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Received Power (dBm)")
    serving_id = segment["serving_eNB_id"]
    target_id = segment["target_eNB_ids"][-1] if segment["target_eNB_ids"] else "N/A"
    plt.title(f"Serving BS: {serving_id} | Target BS: {target_id}")
    plt.legend()
    plt.grid(True)

    image_path = os.path.join(output_dir, f"episode_{episode_number}_single_segment.png")
    plt.savefig(image_path)
    plt.close()

def plot_reward_summary(reward_order, save_path="."):
    episode_ids = reward_order['episode_id']
    rewards = reward_order['reward']

    # Compute moving average (over last 100 episodes)
    avg_rewards = []
    for i in range(len(rewards)):
        start_idx = max(0, i - 99)
        avg_rewards.append(np.mean(rewards[start_idx:i+1]))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(episode_ids, rewards, label="Episode Reward", color="green")
    plt.plot(episode_ids, avg_rewards, label="Average Reward (last 100)", color="orange")
    plt.xlabel("Episode ID")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend()
    plt.grid(True)

    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    image_path = os.path.join(save_path, "reward_summary.png")
    plt.savefig(image_path)
    plt.close()
    print(f"Reward summary plot saved to: {image_path}")

def main():

    run_info = """
Training Details :
- Multi-parameter parameters
- Using regular DQN algorithm dated today
- Fading enabled
- Retraining with different RL thresholds
- SINR multiplied by 300
-  Tuned parameters to:  def __init__(self, state_size, action_size, hidden_size=64, 
                 lr=1e-3, gamma=0.95, buffer_size=1000, batch_size=64, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=5000,
                 learning_starts=50, target_update_interval=100, max_grad_norm=10):
- Running for more epochs as the previous continued to learn more
- Hopes:
    - The model will try to explore more than converging early
- Future Experiments:
"""

    state_size = 10
    action_size = 2
    random.seed(42)

    agent = DQNAgent(state_size, action_size)

    num_episodes = 2
    # num_episodes = 70
    # num_episodes = 10

    # Define lists of possible values for each parameter
    param_options = {
        "TRAIN": [True],
        "USE_AGENT": [True],
        "AGENT": [4],
        "INCREASE_OHO": [True],
        "T_prep": [80,100],  # in ms
        "T_exec": [60,80],  # in ms
        "O_prep": [1,4,6],  # in dBm
        "O_exec": [4,6,8],  # in dB
        # "RLF_THRESHOLD": [-35, -23, -30, -27, -24,  -32],  # -65 dBm
        "RLF_THRESHOLD": [-82 , -75 , -72, -70, -78],  # dBm
        # "RLF_THRESHOLD": [-23, -35, -30, -27, -24,  -32],  # -65 dBm
        # "RLF_THRESHOLD": [-45 , -50.5, -55, -60],  # dBm
        "T310": [500,800,1000],  # in ms
        "N310_THRESHOLD": [4,5,6],  # in count
        "MIN_SPEED": [kmph_to_mpms(30)],  # in m/s
        "MAX_SPEED": [kmph_to_mpms(50)],  # in m/s
        "INCLUDE_SINR": [True],  # Include SINR in reward calculation
        "FADING": [True]  # Fading enabled/disabled
    }
    # param_options = {
    #     "TRAIN": [True],
    #     "USE_AGENT": [True],
    #     "AGENT": [4],
    #     "INCREASE_OHO": [False],
    #     "T_prep": [100],  # in ms
    #     "T_exec": [80],  # in ms
    #     "O_prep": [10],  # in dBm
    #     "O_exec": [8],  # in dBm
    #     "RLF_THRESHOLD": [-35, -23, -30, -27, -24,  -32],  # -65 dBm
    #     # "RLF_THRESHOLD": [-23, -35, -30, -27, -24,  -32],  # -65 dBm
    #     # "RLF_THRESHOLD": [-45 , -70 , -65, -50.5, -45, -60],  # dBm
    #     # "RLF_THRESHOLD": [-55 , -70 , -65, -55, -62, -73],  # dBm
    #     "T310": [40,60],  # in ms
    #     "N310_THRESHOLD": [2,3],  # in count
    #     "MIN_SPEED": [kmph_to_mpms(30)],  # in m/s
    #     "MAX_SPEED": [kmph_to_mpms(50)],  # in m/s
    #     "INCLUDE_SINR": [True],  # Include SINR in reward calculation
    #     "FADING": [False]  # Fading enabled/disabled
    # }


    final_path = 'train_logs/' + str(datetime.datetime.now()).replace(':', '-')
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    # Generate all possible combinations of parameter values
    param_combinations = list(product(*param_options.values()))

    # Convert combinations into list of dictionaries
    configs = [
        {key: value for key, value in zip(param_options.keys(), combination)}
        for combination in param_combinations
    ]

    # Create a list of ParamConfig objects
    param_config_list = [ParamConfig(config) for config in configs]

    episode_id = 0
    param_config_episode_order = {'episode_id': [], 'param_config': []}
    reward_order = {'episode_id': [], 'reward': []}

    reward_final_path = os.path.join(final_path, "episodic_reward_log")
    os.makedirs(reward_final_path, exist_ok=True)

    for episode in range(num_episodes):
        for param in param_config_list:
            episode_id+=1
            #save episode id and param config list to save into csv later
            param_config_episode_order['episode_id'].append(episode_id)
            param_config_episode_order['param_config'].append(param.__dict__)

            ue = UE(1, 0, param)
            enbs = [eNB(1, (random.randint(2, 100), random.randint(230, 240)), param), eNB(2, (random.randint(200, 350), random.randint(260, 270)), param)]
            ticker = Ticker()
            S = Simulate_UE(ue, enbs, param, model = agent)
            (df, df_overall_simulation, total_reward) = S.run(ticker ,total_time=10000000)
            print(df)

            reward_order['episode_id'].append(episode_id)
            reward_order['reward'].append(total_reward)


            df.to_csv(f"{reward_final_path}/reward_episode_{episode_id}.csv", index=False)
            plot_single_segment_from_df(df_overall_simulation, param.RLF_THRESHOLD, episode_number=episode_id, extra_steps=4, save_path=final_path)
    
    plot_reward_summary(reward_order, final_path)
    DQNAgent.save_model(agent, f"{final_path}/dqn_agent.pth")
    with open(f"{final_path}/run_info.txt", "w") as f:
        f.write(run_info)
    pd.DataFrame(param_config_episode_order).to_csv(f"{final_path}/param_config_episode_order.csv", index=False)



if __name__ == "__main__":
    main()