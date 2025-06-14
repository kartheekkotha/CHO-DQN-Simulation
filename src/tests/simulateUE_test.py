import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import random
from config import eNB_param
import UE
from utils.Ticker import Ticker
from utils.grapher import calc_RSRP_2D
from Simulate_UE import Simulate_UE
from config.param import ParamConfig
param = ParamConfig({})

ue = UE.UE(1, 0)
enbs = eNB_param.eNB_1_list
ticker = Ticker()
random.seed(42)
# calc_RSRP_2D(enbs)
S = Simulate_UE(ue, enbs)
# return df, self.total_HO, self.total_RLF, self.total_HO_prep, self.total_HO_exec, self.total_HO_fail_prep, self.total_HO_fail_exec, self.rlf_times

df, total_HO, total_RLF, total_HO_attempts, total_HO_prep, total_HO_exec, total_HO_fail_prep, total_HO_fail_exec, rlf_times, sinr_avg = S.run(ticker ,total_time=10000000)

# open file in results/param.SAVE_PATH and save all the above details
if not os.path.exists("results"):
    os.makedirs("results")

with open(f"results/new/{param.SAVE_PATH}.txt", "w") as f:
    f.write(f"Total Handovers: {total_HO}\n")
    f.write(f"Total RLFs: {total_RLF}\n")
    f.write(f"Total Handover Attempts: {total_HO_attempts}\n")
    f.write(f"Total Handover Preparations: {total_HO_prep}\n")
    f.write(f"Total Handover Executions: {total_HO_exec}\n")
    f.write(f"Total Handover Failures (Preparation): {total_HO_fail_prep}\n")
    f.write(f"Total Handover Failures (Execution): {total_HO_fail_exec}\n")
    f.write(f"RLF Times: {rlf_times}\n")
    f.write(f"SINR Average: {sinr_avg}\n")
    f.write(f"Dataframe: {df}\n")
    f.write(f"Random Seed: 42\n")
