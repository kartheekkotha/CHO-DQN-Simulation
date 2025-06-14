import os
import random
import datetime
import matplotlib.pyplot as plt
from utils.misc import kmph_to_mpms
from config.param import ParamConfig
from Simulate_UE import Simulate_UE
from utils.Ticker import Ticker
import UE
from config import eNB_param
import csv


# Pre-generate multi-UE positions if SINR is enabled
param_global = ParamConfig({})
multi_ue_position_list = []
if param_global.CALC_SINR:
    def create_multi_users(count):
        positions = []
        for i in range(count):
            x = random.randint(0, param_global.TOTAL_DISTANCE)
            temp = random.random()
            if temp < 0.3:
                y = random.randint(125, 150)
            elif temp < 0.6:
                y = random.randint(242, 258)
            else:
                y = random.randint(350, 375)
            while (0 <= x <= 50) or (950 <= x <= 1000):
                x = random.randint(0, param_global.TOTAL_DISTANCE)
            positions.append([i, (x, y)])
        return positions
    multi_ue_position_list = create_multi_users(param_global.SINR_COUNT)

# Parameters to test
test_params = {
    # 'Oprep': [4, 6],
    'Oprep': [2, 4, 6, 8, 10, 12, 14, 15, 18, 20],
    'Oexec': [2, 4, 6, 8, 10, 12, 14, 15, 18, 20],
    'Tprep': [20, 40, 60, 80, 100, 120, 140, 160],
    'Texec': [20, 40, 60, 80, 100, 120, 140, 160],
    # 'speed': [20, 30, 40, 50, 60],
    # 'RLF_THRESHOLD': [-20, -23, -26, -30, -33, -35]
    # 'RLF_THRESHOLD': [-65, -67, -70, -72, -75, -78, -80, -82, -85],
    'N310_THRESHOLD' : [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'T310' : [500, 700, 900, 1000, 1200, 1400, 1600, 1800, 2000]
}

# Single function to run all configs for one parameter
def run_configs_for_param(param_name, values, base_path):
    # Setup output folders
    # cfgs = [
    #     ('before_training', False, 0, None),
    #     ('after_training',  True,  4,   'E:/3_OUR/CHO_RL/Intellectual_6G/train_logs/2025-05-12 23-34-48.275229/dqn_agent.pth'),
    #     ('default',         True, 0,   None)
    # ]
    cfgs = [
        ('before_training', False, 0, None),
        ('after_training',  True,  4,   'E:/3_OUR/CHO_RL/Intellectual_6G/train_logs/2025-05-12 23-34-48.275229/dqn_agent.pth')
    ]    
    
    for cfg_type, _, _, _ in cfgs:
        os.makedirs(os.path.join(base_path, cfg_type), exist_ok=True)

    # Collect metrics per config
    metrics = {cfg_type: {'prep': [], 'exec': [], 'fp': [], 'fe': [], 'rlf': [], 'sinr': []}
               for cfg_type, _, _, _ in cfgs}

    # Loop over each config first, then values
    for cfg_type, use_agent, agent_idx, agent_path in cfgs:
        folder = os.path.join(base_path, cfg_type)
        for val in values:
            run_path = os.path.join(folder, str(val))
            os.makedirs(run_path, exist_ok=True)

            # Build config dict
            cfg = {
                'FADING': True,
                'TEST_OUTPUT_FOLDER': run_path,
                'TRAIN': False,
                'USE_AGENT': use_agent,
                'INCREASE_OHO': True
            }
            if param_name == 'speed':
                cfg['MIN_SPEED'] = kmph_to_mpms(val)
                cfg['MAX_SPEED'] = kmph_to_mpms(val + 5)
                cfg['RLF_THRESHOLD'] = -67.5
            elif param_name == 'RLF_THRESHOLD':
                cfg['RLF_THRESHOLD'] = val
            else:
                cfg[param_name] = val
                cfg['RLF_THRESHOLD'] = -67.5
            if use_agent:
                cfg['AGENT'] = agent_idx
                cfg['AGENT_PATH'] = agent_path
            else:
                cfg['AGENT'] = 0
                cfg['AGENT_PATH'] = None

            param_cfg = ParamConfig(cfg)

            # Run simulation
            ue = UE.UE(1, 0, param_cfg)
            enbs = eNB_param.eNB_2_list
            ticker = Ticker()
            S = Simulate_UE(ue, enbs, param_cfg, multi_ue_position_list=multi_ue_position_list)
            (
                df, total_HO, total_RLF, total_attempts,
                total_prep, total_exec, fail_prep,
                fail_exec, rlf_times, sinr_avg
            ) = S.run(ticker, total_time=10000000)

            # Save per-run results
            with open(os.path.join(run_path, 'results.txt'), 'w') as f:
                f.write(f"Param {param_name}={val}, Config={cfg_type}\n")
                f.write(f"Total HO: {total_HO}, RLF: {total_RLF}, Attempts: {total_attempts}\n")
                f.write(f"Prep: {total_prep}, Exec: {total_exec}, PrepFail: {fail_prep}, ExecFail: {fail_exec}\n")
                f.write(f"RLF times: {rlf_times}, SINR avg: {sinr_avg}\nDataframe: {df}\n")

            # Append metrics
            m = metrics[cfg_type]
            m['prep'].append(total_prep)
            m['exec'].append(total_exec)
            m['fp'].append(fail_prep)
            m['fe'].append(fail_exec)
            m['rlf'].append(total_RLF)
            m['sinr'].append(sinr_avg)

    # Plot all metrics for this parameter
    _plot_all(param_name, values, metrics, base_path)

# Plotting helper
# Plotting helper
def _plot_all(param_name, values, metrics, base_path):
    # Failure rates
    for metric, ylabel, func in [
        ('prep', 'Prep Failure Rate', lambda m: [fp/hp if hp else 0 for fp,hp in zip(m['fp'], m['prep'])]),
        ('exec', 'Exec Failure Rate', lambda m: [fe/he if he else 0 for fe,he in zip(m['fe'], m['exec'])]),
        ('overall', 'Overall Fail Rate', lambda m: [(fp+fe)/hp if hp else 0 for fp,fe,hp in zip(m['fp'], m['fe'], m['prep'])]),
        ('rlf', 'RLF Count', lambda m: m['rlf']),
        ('sinr', 'SINR Average', lambda m: m['sinr']),
        ('total_Ho_count', 'Total HO Count', lambda m: [(fp+fe) for fp,fe in zip(m['fp'], m['fe'])])
    ]:
        plt.figure(figsize=(10,6))

        for cfg_type in ['before_training', 'after_training']:
            vals = func(metrics[cfg_type])
            label = cfg_type.replace('_', ' ').title()

            # Plot the values
            plt.plot(values, vals, marker='o', label=label)

            # Save x and y values to CSV
            csv_path = os.path.join(base_path, f"{param_name}_{metric}_{cfg_type}.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([param_name, metric])
                writer.writerows(zip(values, vals))

        plt.title(f"{ylabel} vs {param_name}")
        plt.xlabel(param_name)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(base_path, f"{param_name}_{metric}.png"))
        plt.close()



# Main
if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root = os.path.join('test_logs_overall', timestamp)
    os.makedirs(root, exist_ok=True)
    for param_name, values in test_params.items():
        folder = os.path.join(root, param_name)
        os.makedirs(folder, exist_ok=True)
        run_configs_for_param(param_name, values, folder)
