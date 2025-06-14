import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from config import param
from config.param import ParamConfig
param = ParamConfig({})


import UE
from utils.Ticker import Ticker

t = Ticker()
ue = UE.UE(1, 999)
ue.set_direction(-1)
path = []
while t.time < 100000:
    if(ue.get_position()[0] >=param.TOTAL_DISTANCE or ue.get_position()[0] <= 0):
        print("tour completed")
        break
    ue.move(t)
    path.append([ue.get_position(), ue.get_velocity()])

for i in path:
    print(i)
