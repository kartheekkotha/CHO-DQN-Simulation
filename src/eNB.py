import math
import random

import numpy.random

# from config import self.param
import utils
import utils.misc

from config.param import ParamConfig
# self.param = self.paramConfig({})

# random.seed(42)
numpy.random.seed(42)


class eNB:
    def __init__(self, id, location,param = None):
        if param != None:
            self.param = param
        else:   
            self.param = ParamConfig({})

        self.id = id
        self.location = location
        self.wavelength = utils.misc.freq_to_wavelength(self.param.FREQ_NR)
        self.fixed_power = self.param.PTX
        self.power = self.fixed_power
    
    def __str__(self):
        return "eNB located at %s with power %s mW" % (self.location, self.power)
    
    def get_id(self):
        return self.id
    
    def get_location(self):
        return self.location
    
    def get_power(self):
        return self.power
    
    def get_wavelength(self):
        return self.wavelength
        
    def reset_power(self):
        self.power = self.fixed_power
    
    def increase_power(self, power):
        if self.power + power > self.param.MAX_PTX:
            raise Exception(f"Cannot increase power by {power} mW. Maximum power limit of {self.param.MAX_PTX} mW would be exceeded.")
        self.power += power
    
    def calculate_distance(self, ue_position):
        return math.sqrt((ue_position[0] - self.location[0])**2 + (ue_position[1] - self.location[
            1])**2)

    def calculate_received_power(self , ue_position):
        pt = utils.misc.calc_power_in_dbm(self.power)
        if self.location != ue_position:
            distance = math.sqrt((ue_position[0] - self.location[0])**2 + (ue_position[1] - self.location[1])**2)
            d0 = self.param.REFERENCE_DISTANCE
            alpha = self.param.PATH_LOSS_EXPONENT
            path_loss = (d0 / distance)
            path_loss_dB = 10 * alpha * math.log10(distance / d0)
            # path_loss  = ((4 * math.pi * distance) / self.wavelength)
            # path_loss_dB = 20 * math.log10(path_loss)
            if self.param.FADING:
                x = numpy.random.normal(0, 1)
                y = numpy.random.normal(0, 1)
                z = complex(x, y)
                fading = path_loss * z
                fading = abs(fading) **2
                fading_dB = 10 * math.log10(fading)
            else:
                fading_dB = 0
            # print(f'power {pt} path_loss {path_loss_dB} fading {fading_dB}')
            received_power_dBm = pt - path_loss_dB + fading_dB
            # Apply dynamic obstacles model if enabled
            if self.param.OBSTACLES:
                # Calculate probability of line of sight using the formula
                # pLoS = min(20/d, 1)(1-ε^39) + ε^(-d/39)
                epsilon = self.param.EPSILON  # Add this to your param config
                exponent = distance / 39.0
                p_los = min(20 / distance, 1.0) * (1 - epsilon**exponent) + epsilon**exponent
                if p_los > 0:
                    los_attenuation_dB = 10 * math.log10(p_los)
                    received_power_dBm+= los_attenuation_dB
        else:
            received_power_dBm = pt
        return received_power_dBm

    
# if __name__ == "__main__":
#     eNB_1 = eNB(1, (20, 230))
    
#     print(eNB_1.calculate_received_power((20, 230)))
#     print(eNB_1.calculate_received_power((100, 1000)))
#     print(eNB_1.calculate_received_power((1000, 1000)))



