import os
import sys
import random
import datetime
import matplotlib.pyplot as plt
from utils.misc import kmph_to_mpms
from config.param import ParamConfig
from Simulate_UE import Simulate_UE
from utils.Ticker import Ticker
import UE
from config import eNB_param

param = ParamConfig({})
if param.CALC_SINR:
    multi_ue_position_list = []
    # def create_multi_users(self , count):
    for i in range(param.SINR_COUNT):
        x = random.randint(0, param.TOTAL_DISTANCE)
        temp = random.random()
        if temp < 0.3:
            y = random.randint(125, 150)
        elif temp < 0.6:
            y = random.randint(242, 258)
        else:
            y = random.randint(350, 375)
        while (x >= 0 and x <= 50) or (x >= 950 and x <= 1000):
            x = random.randint(0, param.TOTAL_DISTANCE)
        # how to check whether the elements are unique or not
        multi_ue_position_list.append([i, (x,y)])


# Configuration
# parameter_name = 'Oexec'  # or 'speed'
parameter_name = 'Texec'  # or 'speed'
paramter_list = [20,40,60,80,100,120,140]
# paramter_list = [4,6,8,10,12,14,15,18,20,22]
# paramter_list = [4,6]


# Metrics containers
ho_prep_total_before = []
ho_exec_total_before = []
ho_fail_prep_before = []
ho_fail_exec_before = []
rlf_count_before_training = []
sinr_avg_before = []


ho_prep_total_after = []
ho_exec_total_after = []
ho_fail_prep_after = []
ho_fail_exec_after = []
rlf_count_after_training = []
sinr_avg_after = []


ho_prep_total_default = []
ho_exec_total_default = []
ho_fail_prep_default = []
ho_fail_exec_default = []
rlf_count_default = []
sinr_avg_default = []



def run_loop(final_path_cfg, use_agent, agent_idx, agent_path,
             ho_prep_list, ho_exec_list, ho_fail_prep_list,
             ho_fail_exec_list, rlf_list, sinr_list):
    os.makedirs(final_path_cfg, exist_ok=True)
    count = 0
    for val in paramter_list:
        count += 1
        output_path = os.path.join(final_path_cfg, f'test_{count}')
        os.makedirs(output_path, exist_ok=True)
        # Build ParamConfig
        if parameter_name == 'speed':
            cfg = {
                'MIN_SPEED': kmph_to_mpms(val),
                'MAX_SPEED': kmph_to_mpms(val + 5),
                'FADING': False,
                'TEST_OUTPUT_FOLDER': output_path,
                'TRAIN': False,
                'USE_AGENT': use_agent,
                'RLF_THRESHOLD': -26,
                'INCREASE_OHO': False
            }
        else:
            cfg = {
                'FADING': False,
                parameter_name: val,
                'TEST_OUTPUT_FOLDER': output_path,
                'TRAIN': False,
                'USE_AGENT': use_agent,
                'RLF_THRESHOLD': -26,
                'INCREASE_OHO': False
            }
        # Agent settings
        if use_agent:
            cfg['AGENT'] = agent_idx
            cfg['AGENT_PATH'] = agent_path
        else:
            cfg['AGENT'] = 0
            cfg['AGENT_PATH'] = None
        param_cfg = ParamConfig(cfg)

        # Run simulation
        ue = UE.UE(1, 0, param_cfg)
        enbs = eNB_param.eNB_1_list
        ticker = Ticker()
        S = Simulate_UE(ue, enbs, param_cfg, multi_ue_position_list = multi_ue_position_list)
        df, total_HO, total_RLF, total_attempts, total_prep, total_exec, fail_prep, fail_exec, rlf_times, sinr_avg = S.run(ticker, total_time=10000000)
        print(df)

        # Save results file
        res_file = os.path.join(output_path, f'{val}.txt')
        with open(res_file, 'w') as f:
            f.write(f"Total Handovers: {total_HO}\n")
            f.write(f"Total RLFs: {total_RLF}\n")
            f.write(f"Total Handover Attempts: {total_attempts}\n")
            f.write(f"Total Handover Preparations: {total_prep}\n")
            f.write(f"Total Handover Executions: {total_exec}\n")
            f.write(f"Total Handover Failures (Preparation): {fail_prep}\n")
            f.write(f"Total Handover Failures (Execution): {fail_exec}\n")
            f.write(f"RLF Times: {rlf_times}\n")
            f.write(f"SINR Average: {sinr_avg}\n")
            f.write(f"Dataframe: {df}\n")
            f.write("Random Seed: 42\n")

        # Aggregate metrics
        ho_prep_list.append(total_prep)
        ho_exec_list.append(total_exec)
        ho_fail_prep_list.append(fail_prep)
        ho_fail_exec_list.append(fail_exec)
        rlf_list.append(total_RLF)
        sinr_list.append(sinr_avg)


def plot_graphs(output_path):
    def rate(fails, totals):
        return [f / t if t > 0 else 0 for f, t in zip(fails, totals)]

    def overall(fail_p, fail_e, total_p):
        return [(p + e) / tp if tp > 0 else 0 for p, e, tp in zip(fail_p, fail_e, total_p)]

    # Overall HO Failure Rate
    plt.figure(figsize=(10, 6))
    plt.plot(paramter_list, overall(ho_fail_prep_before, ho_fail_exec_before, ho_prep_total_before), label='Before Training', marker='o')
    plt.plot(paramter_list, overall(ho_fail_prep_after, ho_fail_exec_after, ho_prep_total_after), label='After Training', marker='o')
    plt.plot(paramter_list, overall(ho_fail_prep_default, ho_fail_exec_default, ho_prep_total_default), label='Default', marker='o')
    plt.title('Overall Handover Failure Rate vs Parameter')
    plt.xlabel('Parameter Value')
    plt.ylabel('Failure Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_path, 'ho_overall_failure_rate.png'))
    plt.close()

    # Preparation Failure Rate
    plt.figure(figsize=(10, 6))
    plt.plot(paramter_list, rate(ho_fail_prep_before, ho_prep_total_before), label='Before Training', marker='o')
    plt.plot(paramter_list, rate(ho_fail_prep_after, ho_prep_total_after), label='After Training', marker='o')
    plt.plot(paramter_list, rate(ho_fail_prep_default, ho_prep_total_default), label='Default', marker='o')
    plt.title('Handover Preparation Failure Rate vs Parameter')
    plt.xlabel('Parameter Value')
    plt.ylabel('Failure Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_path, 'ho_prep_failure_rate.png'))
    plt.close()

    # Execution Failure Rate
    plt.figure(figsize=(10, 6))
    plt.plot(paramter_list, rate(ho_fail_exec_before, ho_exec_total_before), label='Before Training', marker='o')
    plt.plot(paramter_list, rate(ho_fail_exec_after, ho_exec_total_after), label='After Training', marker='o')
    plt.plot(paramter_list, rate(ho_fail_exec_default, ho_exec_total_default), label='Default', marker='o')
    plt.title('Handover Execution Failure Rate vs Parameter')
    plt.xlabel('Parameter Value')
    plt.ylabel('Failure Rate')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_path, 'ho_exec_failure_rate.png'))
    plt.close()

    # RLF Count
    plt.figure(figsize=(10, 6))
    plt.plot(paramter_list, rlf_count_before_training, label='Before Training', marker='o')
    plt.plot(paramter_list, rlf_count_after_training, label='After Training', marker='o')
    plt.plot(paramter_list, rlf_count_default, label='Default', marker='o')
    plt.title('RLF Count vs Parameter')
    plt.xlabel('Parameter Value')
    plt.ylabel('RLF Count')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_path, 'rlf_count.png'))
    plt.close()

    # SINR Average
    plt.figure(figsize=(10, 6))
    plt.plot(paramter_list, sinr_avg_before, label='Before Training', marker='o')
    plt.plot(paramter_list, sinr_avg_after, label='After Training', marker='o')
    plt.plot(paramter_list, sinr_avg_default, label='Default', marker='o')
    plt.title('Average SINR vs Parameter')
    plt.xlabel('Parameter Value')
    plt.ylabel('SINR Average')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_path, 'sinr_average.png'))
    plt.close()



def main():
    # Base output directory with timestamp
    final_path = os.path.join('test_logs', str(datetime.datetime.now()).replace(':', '-'))
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    # Run before-training (no agent)
    run_loop(
        os.path.join(final_path, 'before_training'),
        use_agent=False,
        agent_idx=0,
        agent_path=None,
        ho_prep_list=ho_prep_total_before,
        ho_exec_list=ho_exec_total_before,
        ho_fail_prep_list=ho_fail_prep_before,
        ho_fail_exec_list=ho_fail_exec_before,
        rlf_list=rlf_count_before_training,
        sinr_list=sinr_avg_before
    )

    # Run after-training (with agent)
    run_loop(
        os.path.join(final_path, 'after_training'),
        use_agent=True,
        agent_idx=4,
        agent_path='E:/3_OUR/CHO_RL/Intellectual_6G/train_logs/2025-05-02 13-48-17.948026/dqn_agent.pth',
        ho_prep_list=ho_prep_total_after,
        ho_exec_list=ho_exec_total_after,
        ho_fail_prep_list=ho_fail_prep_after,
        ho_fail_exec_list=ho_fail_exec_after,
        rlf_list=rlf_count_after_training,
        sinr_list=sinr_avg_after
    )

    # Run default configuration (no agent)
    run_loop(
        os.path.join(final_path, 'default'),
        use_agent=True,
        agent_idx=0,
        agent_path=None,
        ho_prep_list=ho_prep_total_default,
        ho_exec_list=ho_exec_total_default,
        ho_fail_prep_list=ho_fail_prep_default,
        ho_fail_exec_list=ho_fail_exec_default,
        rlf_list=rlf_count_default,
        sinr_list=sinr_avg_default
    )

    # Plot and save graphs
    plot_graphs(final_path)

    # Save summary
    summary_file = os.path.join(final_path, 'results.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Parameter Name: {parameter_name}\n")
        f.write(f"Parameter Values: {paramter_list}\n")
        f.write(f"Prep Failures Before: {ho_fail_prep_before}\n")
        f.write(f"Prep Counts Before: {ho_prep_total_before}\n")
        f.write(f"Exec Failures Before: {ho_fail_exec_before}\n")
        f.write(f"Exec Counts Before: {ho_exec_total_before}\n")
        f.write(f"RLF Before: {rlf_count_before_training}\n")
        f.write(f"SINR Before: {sinr_avg_before}\n")
        f.write(f"Prep Failures After: {ho_fail_prep_after}\n")
        f.write(f"Prep Counts After: {ho_prep_total_after}\n")
        f.write(f"Exec Failures After: {ho_fail_exec_after}\n")
        f.write(f"Exec Counts After: {ho_exec_total_after}\n")
        f.write(f"RLF After: {rlf_count_after_training}\n")
        f.write(f"SINR After: {sinr_avg_after}\n")
        f.write(f"Prep Failures Default: {ho_fail_prep_default}\n")
        f.write(f"Prep Counts Default: {ho_prep_total_default}\n")
        f.write(f"Exec Failures Default: {ho_fail_exec_default}\n")
        f.write(f"Exec Counts Default: {ho_exec_total_default}\n")
        f.write(f"RLF Default: {rlf_count_default}\n")
        f.write(f"SINR Default: {sinr_avg_default}\n")

        f.write("Random Seed: 42\n")

if __name__ == '__main__':
    main()
