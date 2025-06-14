import os
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import utils.grapher as grapher
# from config import param
from config.eNB_param import eNB_2_list
from config.param import ParamConfig
param = ParamConfig({})

eNB_list = eNB_2_list

# Compute pairwise distances
print("\nPairwise Distances Between Base Stations:\n")
for i in range(len(eNB_list) -1):
    for j in range(i + 1, i+2):  # Avoid duplicate calculations
        eNB1 = eNB_list[i]
        eNB2 = eNB_list[j]
        
        x1, y1 = eNB1.location
        x2, y2 = eNB2.location
        
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        print(f"Distance between eNB {eNB1.get_id()} and eNB {eNB2.get_id()}: {distance:.2f} meters")

# Call grapher after printing distances
grapher.calc_RSRP_2D(eNB_list, grid_size_x=param.TOTAL_DISTANCE//10, grid_size_y=50, spacing=10)
