import math
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch

# from config import self.param
import utils.misc
from utils.Ticker import Ticker
# from UE import UE
# from eNB import eNB

from dqn_simulation import DQN

from config.param import ParamConfig

class Simulate_UE:
    def __init__(self, ue, eNBs, param= None, model=None, multi_ue_position_list = []):
        if param != None:
            self.param = param
        else:   
            self.param = ParamConfig({})

        self.ticker = Ticker()  # Ticker instance for time tracking
        self.ue = ue
        self.enbs = eNBs
        self.total_HO = 0
        self.total_RLF = 0
        self.total_HO_prep = 0
        self.total_HO_exec = 0
        self.total_HO_fail_prep = 0
        self.total_HO_fail_exec = 0
        self.total_HO_attempts = 0

        #for power_increase 
        self.is_power_increased = False
        self.cool_down_timer = 0
        
        # for fading scenario
        self.serving_power_counting = 0
        self.target_power_counting = 0
        self.serving_power_list = []
        self.target_power_list = []
        self.past_target_id = None

        # Record handover completion time steps
        self.handover_times = []
        # Record radio link failure (RLF) events time steps
        self.rlf_times = []

        # Time series data for plotting (serving BS data)
        self.time_series = []         # Simulation time (from Ticker)
        self.assoc_power_series = []  # Received power (dBm) from associated (serving) BS
        self.assoc_eNB_series = []    # Associated BS id at each time step

        # Target BS (candidate) data series
        self.target_power_series = []   # Received power (dBm) from target BS
        self.target_eNB_series = []     # Target BS id at each time step

        # Handover state variables and timers
        self.preparation_phase = False
        self.execution_phase = False
        self.waiting_exec = False
        self.tprep = 0
        self.texec = 0

        #
        self.preparation_attempts = []
        self.execution_attempts = []
        self.completed_handover = False

        self.rlf_induced_handover_count = 0

        # for SINR calculation
        self.sinr_record_time  = []
        self.sinr_list = []
        self.sinr_individual_list = {}
        for i in range(self.param.SINR_COUNT):
            self.sinr_individual_list[i] = []
        self.multi_ue_position_list = multi_ue_position_list

        # RLF
        self.n310_counter = 0
        self.t310_start_time = None

        #model
        if self.param.TRAIN and model != None:
            self.agent = model

        if self.param.USE_AGENT and not self.param.TRAIN:
            if self.param.AGENT == 0:
                self.agent = None
            if self.param.AGENT == 1:
                self.agent = DQN(5,2,64)
                self.agent.load_state_dict(torch.load('model_weights/dqn_agent.pth'))
            elif self.param.AGENT == 2:
                self.agent = DQN(7,2,64)
                self.agent.load_state_dict(torch.load('model_weights/dqn_agent_2_morePenalize.pth'))
            elif self.param.AGENT == 3:
                self.agent = DQN(8,2,64)
                self.agent.load_state_dict(torch.load('model_weights/dqn_agent_3.pth'))
            elif self.param.AGENT == 4:
                self.agent = DQN(10,2,64)
                self.agent.load_state_dict(torch.load(self.param.AGENT_PATH))

        # for training paarameters
        self.time_steps_rewards = []
        self.rewards = []
        self.information_reward = []

        # for power increasse
        self.power_increase_times = []

    def run(self, t, total_time):
        """
        Run the simulation until the ticker's time reaches total_time.
        """
        self.ticker = t
        self.discover_bs()
        self.associate_ue_with_bs()
        self.simulate_motion(total_time)
        if not self.param.TRAIN:
            self.plot_segments(episode_number=1)
            self.plot_sinr()
        if self.param.TRAIN:
            return self.calculate_rewards()
        return self.return_results()

    def return_results(self):
        #merge time_series, assoc_power_series, assoc_eNB_series, target_power_series, target_eNB_series
        # data = {
        #     'time': self.time_series,
        #     'serving_power': self.assoc_power_series,
        #     'serving_eNB': self.assoc_eNB_series,
        #     'target_power': self.target_power_series,
        #     'target_eNB': self.target_eNB_series
        # }
        # df = pd.DataFrame(data)
        count = 0
        sinr_avg = 0
        for i in range(len(self.sinr_list)):
            count+=1
            sinr_avg += self.sinr_list[i]
        if count != 0:
            sinr_avg = sinr_avg/count
        return "None", self.total_HO, self.total_RLF, self.total_HO_attempts, self.total_HO_prep, self.total_HO_exec, self.total_HO_fail_prep, self.total_HO_fail_exec, self.rlf_times, sinr_avg
# Add this method to Simulate_UE class
    def normalize_state(self, state):
        """Normalize state components to suitable ranges for NN training"""
        normalized = []
        # Power values (typically -100 to -20 dBm) -> [-1, 1]
        normalized.append((state[0] + 60) / 40)  # Serving power
        normalized.append((state[1] + 60) / 40)  # Target power
        # Velocity (0 to 50 m/s) -> [0, 1]
        normalized.append(state[2] / 50)  
        # Parameters normalization
        normalized.append(state[3] / 200)  # Texec
        normalized.append(state[4] / 200)  # Tprep
        normalized.append(state[5] / 20)   # Oexec
        normalized.append(state[6] / 20)   # Oprep
        normalized.append((state[7] + 100) / 100)  # T310
        normalized.append(state[8] / 10)   # N310_THRESHOLD
        normalized.append((state[9] + 60) / 40)  # RLF_THRESHOLD
        
        return normalized
    # for agent 
    def get_state(self, serving_power, target_power, serving_distance, target_distance):
        if self.param.AGENT == 1:
            input_state = [
                serving_power,
                target_power,
                self.ue.get_velocity(),
                # current_assoc_bs.calculate_distance(current_position),
                # best_bs_temp.calculate_distance(current_position)
                serving_distance,
                target_distance
            ]
        elif self.param.AGENT == 2:
            input_state = [
                serving_power,
                target_power,
                self.ue.get_velocity(),
                self.param.Texec,
                self.param.Tprep,
                self.param.Oexec,
                self.param.Oprep
            ]
        elif self.param.AGENT == 3:
            input_state = [
                serving_power,
                target_power,
                self.ue.get_velocity(),
                self.param.Texec,
                self.param.Tprep,
                self.param.Oexec,
                self.param.Oprep,
                self.param.RLF_THRESHOLD
            ]
        elif self.param.AGENT == 4:
            input_state = [
                serving_power,
                target_power,
                self.ue.get_velocity(),
                self.param.Texec,
                self.param.Tprep,
                self.param.Oexec,
                self.param.Oprep,
                self.param.T310,
                self.param.N310_THRESHOLD,
                self.param.RLF_THRESHOLD
            ]
            # input_state = self.normalize_state(input_state)
        # input_state = torch.FloatTensor(input_state).unsqueeze(0)
        return input_state
    # for sinr calculation : 

    def create_multi_users(self, count):
        if self.param.DYNAMIC:
            pass
        else:
            for i in range(count):
                #create a different random points in the field, which are not at positions of bs and other ue's added
                x = random.randint(0, self.param.TOTAL_DISTANCE)
                temp = random.random()
                if temp < 0.3:
                    y = random.randint(125, 150)
                elif temp < 0.6:
                    y = random.randint(242, 258)
                else:
                    y = random.randint(350, 375)
                while (x >= 0 and x <= 50) or (x >= 950 and x <= 1000):
                    x = random.randint(0, self.param.TOTAL_DISTANCE)
                # how to check whether the elements are unique or not
                self.multi_ue_position_list.append([i, (x,y)])


    def calculate_servingPower_inference(self, base_stations, position):
        """
        Calculate the received power from each base station at the given position.
        """
        received_power = []
        sorted_base_stations = sorted(base_stations, key=lambda bs: bs.calculate_distance(position))
        serving_power = sorted_base_stations[0].calculate_received_power(position)
        for bs in sorted_base_stations[1:]:
            received_power.append(bs.calculate_received_power(position))
        total_inference = 0
        for power in received_power:
            total_inference += self.param.LOAD_FACTOR * power
        return serving_power, self.param.LOAD_FACTOR * total_inference

    
    def simulate_motion(self, total_time):
        """
        Simulate UE motion until simulation time (tracked by the Ticker) reaches total_time.
        This function moves the UE (using the ticker), records the received power from the current
        associated BS (serving) and the best candidate BS (target), and implements conditional
        handover logic as well as radio link failure (RLF) detection.
        """

        total_ticks = 0

        if self.param.CALC_SINR and len(self.multi_ue_position_list) == 0:
            self.create_multi_users(self.param.SINR_COUNT)

        #for training
        if self.param.TRAIN:
            reward = 0
            information = ''
            input_state = None
            action = None
            next_state = None
            final_bs_id = self.enbs[-1].get_id()
            is_agent_used = False
            increased_power_his = False

        while self.ticker.get_time() < total_time:
            #for training
            # print(self.ticker.get_time() ,self.ue.get_associated_eNB().get_id() , self.enbs[-1].get_id())
            if self.param.TRAIN:
                # self.time_steps_rewards.append(self.ticker.get_time())
                # self.rewards.append(reward)
                # self.information_reward.append(information)
                if self.ue.get_associated_eNB() is not None:
                    if self.ue.get_associated_eNB().get_id() == final_bs_id:
                        print(self.ticker.get_time() ,self.ue.get_associated_eNB().get_id() , self.enbs[-1].get_id())
                        print("UE is connected to the BS 2")
                        break
            current_time = self.ticker.get_time()
            self.ue.move(self.ticker)
            total_ticks += 1

            if self.param.COOL_DOWN and self.param.USE_AGENT and self.is_power_increased:
                if self.cool_down_timer >= self.param.COOL_DOWN_PERIOD:
                    self.ue.get_associated_eNB().reset_power()
                    print(f"[t={current_time:.2f}] Power reset to default value.")
                    self.is_power_increased = False
                    self.cool_down_timer = 0
                else:
                    self.cool_down_timer += self.param.TICKER_INTERVAL
            
            current_position = self.ue.get_position()
            # Stop simulation if UE reaches simulation area edge
            if current_position[0] >= self.param.TOTAL_DISTANCE or current_position[0] <= 0:
                print(f"[t={current_time:.2f}] UE reached the edge of the simulation area. Stopping simulation.")
                break

            # Get current associated BS and its received power
            current_assoc_bs = self.ue.get_associated_eNB()
            if current_assoc_bs is None:
                self.associate_ue_with_bs()
                current_assoc_bs = self.ue.get_associated_eNB()
            if current_assoc_bs is None:
                print(f"[t={current_time:.2f}] UE is not associated with any base station.")
                continue
            serving_power = current_assoc_bs.calculate_received_power(current_position)

            # Record simulation time, serving power and associated BS id
            self.time_series.append(current_time)
            self.assoc_power_series.append(serving_power)
            self.assoc_eNB_series.append(current_assoc_bs.get_id())
            candidates = [bs for bs in self.enbs if bs.get_id() != current_assoc_bs.get_id()]
            if candidates:
                best_bs = max(candidates, key=lambda bs: bs.calculate_received_power(current_position))
                best_bs_power = best_bs.calculate_received_power(current_position)
            else:
                best_bs_power = 0
                best_bs = 0
            self.target_power_series.append(best_bs_power)
            self.target_eNB_series.append(best_bs.get_id())

            if self.param.CALC_SINR:
                self.sinr_record_time.append(current_time)
                sinr = 0
                for user in self.multi_ue_position_list:
                    sp , interference = self.calculate_servingPower_inference(self.enbs, user[1])
                    sinr_i = sp / (self.param.NOISE + interference)
                    self.sinr_individual_list[user[0]].append(sinr_i)
                    sinr += sinr_i
                self.sinr_list.append(sinr/10)
                # print(f"[t={current_time:.2f}] SINR: {sinr:.2f} dBm")  
    
            if (len(self.power_increase_times) >0 and (current_time-self.power_increase_times[-1]) <=40 ) and self.param.TRAIN and self.param.INCLUDE_SINR:
                reward-=((self.sinr_list[-1] - self.sinr_list[-2]) * 300.0)
                information+= f" SINR change: {-(self.sinr_list[-1] - self.sinr_list[-2])*300} , "

            # --- Radio Link Failure (RLF) Check ---
            # print(f"Time {current_time:.2f} - Serving Power: {serving_power} dBm and Threshold: {self.param.RLF_THRESHOLD} dBm")
            if serving_power < self.param.RLF_THRESHOLD:
                # Increment N310 counter for each out-of-sync event
                self.n310_counter += 1
                print(f"[t={current_time:.2f}] Out-of-sync detected. N310 counter: {self.n310_counter}, power {serving_power} dBm (threshold: {self.param.RLF_THRESHOLD} dBm)")
                if self.n310_counter <= self.param.N310_THRESHOLD:
                    continue
                elif self.t310_start_time is None:
                    self.t310_start_time = current_time
                else:
                    best_bs_temp = None
                    # Check T310 timer
                    if current_time - self.t310_start_time >= self.param.T310:
                        # RLF detected
                        print(f"[t={current_time:.2f}] RLF detected. Serving power: {serving_power:.2e} dBm (threshold: {self.param.RLF_THRESHOLD} dBm)")

                        if self.preparation_phase or self.waiting_exec or self.execution_phase:
                            if self.preparation_phase:
                                self.total_HO_fail_prep += 1
                                self.preparation_attempts.append((current_time, current_position, 'failed'))
                                print(f"[t={current_time:.2f}] Handover preparation failed after {self.tprep} timesteps.")
                                self.preparation_phase = False
                                self.waiting_exec = False
                                self.execution_phase = False
                            elif self.waiting_exec or self.execution_phase:
                                self.total_HO_fail_exec += 1
                                self.execution_attempts.append((current_time, current_position, 'failed'))
                                print(f"[t={current_time:.2f}] Handover execution failed after {self.texec} timesteps.")
                                self.execution_phase = False
                                self.waiting_exec = False
                                self.preparation_phase = False
                        if not self.param.INCREASE_OHO or ((self.param.INCREASE_OHO) and (self.preparation_phase or self.waiting_exec or self.execution_phase)):
                            if self.param.TRAIN:
                                if self.is_power_increased or increased_power_his:
                                    reward-=15
                                    information+= "Power increased and RLF so -5 , "
                                else:
                                    reward-=5
                                    information+= "Power not increased and RLF so -2 , "
                                if input_state != None:
                                    self.agent.push_memory(input_state, action, reward, next_state, True)
                                    self.agent.update()
                                self.time_steps_rewards.append(self.ticker.get_time())
                                self.rewards.append(reward)
                                self.information_reward.append(information)
                                reward = 0
                                information = ''
                                input_state = None
                                action = None
                                next_state = None
                                
                        self.total_RLF += 1
                        self.rlf_times.append(current_time)

                        if self.param.USE_AGENT:
                            # Reset power to default value
                            self.ue.get_associated_eNB().reset_power()
                            self.is_power_increased = False
                            self.cool_down_timer = 0
                        self.associate_ue_with_bs()
                        self.n310_counter = 0
                        self.t310_start_time = None
                        if self.ue.get_associated_eNB() is None:
                            print(f"[t={current_time:.2f}] UE is not associated with any base station.")
                            continue
                        # self.target_power_series.append(serving_power)
                        # self.target_eNB_series.append(current_assoc_bs.get_id())
                        # Reset N310/T310
                        continue
                    else :
                        print(f"[t={current_time:.2f}] T310 timer not expired. Waiting...")
                        if self.param.USE_AGENT:
                            if self.param.TRAIN and  is_agent_used and input_state is not None :
                                self.agent.push_memory(input_state, action, reward, next_state, False)
                                self.agent.update()
                            input_state = None
                            action = None
                            next_state = None
                            # Check if power increase is needed
                            if not self.param.INCREASE_OHO or ((self.param.INCREASE_OHO) and (self.preparation_phase or self.waiting_exec or self.execution_phase)):
                                candidates_temp = [bs for bs in self.enbs if bs.get_id() != current_assoc_bs.get_id()]
                                best_bs_temp = max(candidates_temp, key=lambda bs: bs.calculate_received_power(current_position))
                                best_bs_power_temp = best_bs_temp.calculate_received_power(current_position)
                                if not self.param.TRAIN :
                                    if self.param.AGENT == 0:
                                        action = 0
                                    else :
                                        input_state = self.get_state(serving_power, best_bs_power_temp, current_assoc_bs.calculate_distance(current_position), best_bs_temp.calculate_distance(current_position))
                                        input_state = torch.FloatTensor(input_state).unsqueeze(0)
                                        q_values = self.agent.forward(input_state)
                                        action = q_values.max(1)[1].item()
                                elif self.param.TRAIN:
                                    input_state = self.get_state(serving_power, best_bs_power_temp, current_assoc_bs.calculate_distance(current_position), best_bs_temp.calculate_distance(current_position))
                                    action = self.agent.select_action(input_state)
                                    is_agent_used = True
                                temp_flag = False
                                if action == 0 and (len(self.power_increase_times) >= self.param.POWER_INCREASE_COUNT_THRESHOLD and current_time - self.power_increase_times[-self.param.POWER_INCREASE_COUNT_THRESHOLD] <= self.param.POWER_COUNT_RESET_TIMER):
                                    temp_flag = True
                                    print(f"[t={current_time:.2f}] Power increase action not taken due to cooldown period.")
                                if action == 0 and temp_flag == False:
                                    try:
                                        self.ue.get_associated_eNB().increase_power(self.param.POWER_INCREASE_QUANTITY)
                                        self.is_power_increased = True
                                        increased_power_his = True
                                        self.cool_down_timer = 0
                                        print(f"[t={current_time:.2f}] Power increased by {self.param.POWER_INCREASE_QUANTITY} mW.")
                                        self.power_increase_times.append(current_time)
                                    except Exception as e:
                                        print(f"[t={current_time:.2f}] Power increase failed due to the power Max limit: {str(e)}")
                                        self.is_power_increased = False
                                        if self.param.TRAIN:
                                            reward -= 2
                                            information += "Power increase failed due to MAX_PTX limit, -1 , "
                                elif action == 1:
                                    print("Model decided not to increase the power")
                                    self.is_power_increased = False
                                elif action == 0 and temp_flag == True:
                                    print(f"[t={current_time:.2f}] Power increase action not taken due to cooldown period.")
                                    self.is_power_increased = False

                                if self.param.TRAIN:
                                    updated_serving_power = self.ue.get_associated_eNB().calculate_received_power(current_position)
                                    if action == 0 and temp_flag == True:
                                        reward-=2
                                        information+= "Power not increased and cooldown so -2 , "
                                        
                                    if updated_serving_power > self.param.RLF_THRESHOLD:
                                        reward+=5
                                        information+= "Power recovered so +5 , "
                                        next_state = self.get_state(updated_serving_power, best_bs_power_temp, current_assoc_bs.calculate_distance(current_position), best_bs_temp.calculate_distance(current_position))
                                        # self.agent.push_memory(input_state, action, reward, next_state, False)
                                        # self.agent.update()
                                    else:
                                        reward-=2
                                        information+= "Power not recovered so -2 , "
                                        next_state = self.get_state(updated_serving_power, best_bs_power_temp, current_assoc_bs.calculate_distance(current_position), best_bs_temp.calculate_distance(current_position))
                                        # self.agent.push_memory(input_state, action, reward, next_state, False)
                                        # self.agent.update()
                continue
            else:
                # If serving power recovers above threshold, reset N310 and T310.
                if self.n310_counter > 0:
                    print(f"[t={current_time:.2f}] Serving power recovered. Resetting N310 and T310.")
                self.n310_counter = 0
                self.t310_start_time = None            # Update the target base station 
            
            if not self.preparation_phase and not self.waiting_exec and not self.execution_phase:
                self.discover_bs()
                # Determine best candidate BS (target) among those different from the current associated BS
                candidates = [bs for bs in self.enbs if bs.get_id() != current_assoc_bs.get_id()]
                if candidates:
                    best_bs = max(candidates, key=lambda bs: bs.calculate_received_power(current_position))
                    best_bs_power = best_bs.calculate_received_power(current_position)
                else:
                    # No candidate BS available
                    # best_bs = current_assoc_bs
                    # best_bs_power = serving_power
                    print(f"No best candidate BS available at t={current_time:.2f}.")
                    continue

            # Record target BS data
            if candidates:
                # self.target_power_series.append(best_bs_power)
                # self.target_eNB_series.append(best_bs.get_id())
                if self.param.FADING == True and self.param.POW_AVG == True:
                    self.target_power_list.append(best_bs_power)
                    if self.past_target_id != best_bs.get_id():
                        self.target_power_counting = 0
                        self.target_power_list = []
                        self.past_target_id = best_bs.get_id()
                    self.target_power_counting += 1
                    if self.target_power_counting >= self.param.AVG_COUNT:
                        best_bs_power = sum(self.target_power_list[-min(5, self.serving_power_counting):]) / min(5, self.serving_power_counting)
            if self.param.FADING == True and self.param.POW_AVG == True:
                self.serving_power_list.append(serving_power)
                self.serving_power_counting += 1
                if self.serving_power_counting >= self.param.AVG_COUNT:
                        serving_power = sum(self.serving_power_list[-min(5, self.serving_power_counting):]) / min(5, self.serving_power_counting)

            if self.param.FADING == True and self.param.POW_AVG == True:
                if self.serving_power_counting <self.param.AVG_COUNT or self.target_power_counting <self.param.AVG_COUNT:
                    # print(f"[t={current_time:.2f}] and location ue : {self.ue.get_position()} Not enough power values for averaging values {self.serving_power_counting} {self.target_power_counting}")
                    continue
            # --- Conditional Handover Logic ---
            if not self.preparation_phase and not self.waiting_exec and not self.execution_phase:
                if best_bs_power > serving_power + self.param.Oprep:
                    self.preparation_phase = True
                    self.tprep = 1
                    self.total_HO_prep += 1
                    self.total_HO_attempts += 1
                    # self.total_HO+=1
                    self.preparation_attempts.append((current_time, current_position, 'started'))
                    print(f"[t={current_time:.2f}] and location ue : {self.ue.get_position()} Handover preparation started. Serving: {serving_power} dBm, Target: {best_bs_power} dBm")
            elif self.preparation_phase:
                self.tprep += self.param.TICKER_INTERVAL
                if self.tprep >= self.param.Tprep:
                    # if random.random() > self.param.HANDOVER_FAILURE_RATE:
                    #     self.waiting_exec = True
                    #     self.preparation_phase = False
                    #     self.preparation_attempts.append((current_time, current_position, 'success'))
                    #     print(f"[t={current_time:.2f}] Handover preparation successful after {self.tprep} timesteps.")
                    # else:
                    #     self.total_HO_fail_prep += 1
                    #     self.preparation_attempts.append((current_time, current_position, 'failed'))
                    #     print(f"[t={current_time:.2f}] Handover preparation failed after {self.tprep} timesteps. Retrying...")
                    #     self.tprep = 0
                    #     self.preparation_phase = False
                    self.waiting_exec = True
                    self.preparation_phase = False
                    self.tprep = 0
                    self.preparation_attempts.append((current_time, current_position, 'success'))
                    print(f"[t={current_time:.2f}] and location ue : {self.ue.get_position()} Handover preparation successful after {self.tprep} timesteps.")

            elif self.waiting_exec and not self.execution_phase:
                if best_bs_power > serving_power + self.param.Oexec:
                    self.execution_phase = True
                    self.waiting_exec = False
                    self.texec = 1
                    self.total_HO_exec += 1
                    self.execution_attempts.append((current_time, current_position, 'started'))
                    print(f"[t={current_time:.2f}] Handover execution started.")
                else:
                    print(f"[t={current_time:.2f}] and location ue : {self.ue.get_position()} Waiting for execution condition. Target: {best_bs_power} dBm, Serving: {serving_power} dBm")
           
            elif self.execution_phase:
                self.texec += self.param.TICKER_INTERVAL
                if self.texec >= self.param.Texec:
                    # if random.random() > self.param.HANDOVER_FAILURE_RATE:
                    #     self.execution_attempts.append((current_time, current_position, 'success'))
                    #     self.total_HO += 1
                    #     self.handover_times.append(current_time)
                    #     print(f"[t={current_time:.2f}] Handover execution successful after {self.texec} timesteps. Switching from {current_assoc_bs.get_id()} to {best_bs.get_id()}.")
                    #     self.ue.set_associated_eNB(best_bs)
                    #     self.execution_phase = False
                    #     self.waiting_exec = False
                    #     self.tprep = 0
                    #     self.texec = 0
                    # else:
                    #     self.total_HO_fail_exec += 1
                    #     self.execution_attempts.append((current_time, current_position, 'failed'))
                    #     print(f"[t={current_time:.2f}] Handover execution failed after {self.texec} timesteps. Retrying execution...")
                    #     self.execution_phase = False
                    #     self.texec = 0
                    #     self.waiting_exec = True
                    self.execution_attempts.append((current_time, current_position, 'success'))
                    self.total_HO += 1
                    self.handover_times.append(current_time)
                    print(f"[t={current_time:.2f}] and location ue : {self.ue.get_position()} Handover execution successful after {self.texec} timesteps. Switching from {current_assoc_bs.get_id()} to {best_bs.get_id()}.")
                    if self.param.USE_AGENT:
                        # Reset power to default value
                        # self.ue.get_associated_eNB().reset_power()
                        if self.param.TRAIN:
                            if is_agent_used and input_state != None:
                                reward+=15
                                information+= "Sucessful HO so +15"
                                self.agent.push_memory(input_state, action, reward, input_state, True)            
                                self.agent.update()
                                self.time_steps_rewards.append(self.ticker.get_time())
                                self.rewards.append(reward)
                                self.information_reward.append(information)
                            input_state = None
                            action = None
                            next_state = None
                            reward = 0
                            information = ''


                    self.ue.set_associated_eNB(best_bs)
                    if self.param.FADING == True and self.param.POW_AVG == True:
                        self.serving_power_counting = 0
                        self.target_power_counting = 0
                        self.serving_power_list = []
                        self.target_power_list = []
                    self.execution_phase = False
                    self.waiting_exec = False
                    self.tprep = 0
                    self.texec = 0
        # End simulation loop when simulation time >= total_time

    def discover_bs(self):
        """
        Sort available base stations by distance from the UE.
        """
        self.enbs.sort(key=lambda bs: math.sqrt(
            (bs.get_location()[0] - self.ue.get_position()[0])**2 +
            (bs.get_location()[1] - self.ue.get_position()[1])**2))

    def search_for_bs(self):
        """
        Return a list of base stations within 2000 meters of the UE.
        """
        nearby_bs = []
        for bs in self.enbs:
            dist = math.sqrt(
                (self.ue.get_position()[0] - bs.get_location()[0])**2 +
                (self.ue.get_position()[1] - bs.get_location()[1])**2)
            if dist <= 2000:
                nearby_bs.append(bs)
        return nearby_bs

    def associate_ue_with_bs(self):
        """
        Associate the UE with the base station providing the strongest received power.
        """
        nearby_bs = self.search_for_bs()
        if not nearby_bs:
            print(f"UE at position {self.ue.get_position()} is out of range")
            return
            # raise Exception("UE is out of range")
        sorted_bs = sorted(nearby_bs,
                           key=lambda bs: bs.calculate_received_power(self.ue.get_position()),
                           reverse=True)
        if(sorted_bs[0].calculate_received_power(self.ue.get_position()) <= self.param.RLF_THRESHOLD):
            self.ue.set_associated_eNB(None)
            self.ue.set_eNBs([])
        else:
            self.ue.set_associated_eNB(sorted_bs[0])
            self.ue.set_eNBs(sorted_bs)
            if self.param.FADING == True and self.param.POW_AVG == True:
                self.serving_power_counting = 0
                self.target_power_counting = 0
                self.serving_power_list = []
                self.target_power_list = []
        
    def plot_segments(self, episode_number=1):
        """
        Segment the simulation data by serving base station changes and save separate plots for each segment.
        Each segment is plotted only if a serving BS change occurred (i.e. more than one segment exists).
        Each plot displays two curves:
          - The serving BS power (blue).
          - The target BS power (red).
        The plot title clearly indicates which BS was serving and which was the target during that segment.
        The plots are saved in a folder named 'results_simulation'.
        """
        # Segment the data based on the serving BS id from assoc_eNB_series.
        segments = []
        current_segment = {
            "times": [],
            "serving_powers": [],
            "target_powers": [],
            "serving_eNB_id": None,
            "target_eNB_ids": []
        }
        for t, serv_power, targ_power, serv_id, targ_id in zip(
                self.time_series,
                self.assoc_power_series,
                self.target_power_series,
                self.assoc_eNB_series,
                self.target_eNB_series):
            if current_segment["serving_eNB_id"] is None:
                current_segment["serving_eNB_id"] = serv_id
            if serv_id == current_segment["serving_eNB_id"]:
                current_segment["times"].append(t)
                current_segment["serving_powers"].append(serv_power)
                current_segment["target_powers"].append(targ_power)
                current_segment["target_eNB_ids"].append(targ_id)
            else:
                segments.append(current_segment)
                current_segment = {
                    "times": [t],
                    "serving_powers": [serv_power],
                    "target_powers": [targ_power],
                    "serving_eNB_id": serv_id,
                    "target_eNB_ids": [targ_id]
                }
        if current_segment["times"]:
            segments.append(current_segment)
        
        # Only plot if there is more than one segment (i.e. a change in serving BS occurred)
        if len(segments) <= 1:
            print("No serving BS change detected. No segmented plots to save.")
            return
        
        # Create output folder
        output_dir = self.param.TEST_OUTPUT_FOLDER
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot and save each segment
        for idx, seg in enumerate(segments):
            plt.figure(figsize=(10, 6))
            # Plot serving BS power (blue) and target BS power (red)
            plt.plot(seg["times"], seg["serving_powers"], label="Serving BS Power", color="blue")
            plt.plot(seg["times"], seg["target_powers"], label="Target BS Power", color="red")
            plt.axhline(y=self.param.RLF_THRESHOLD, color='gray', linestyle='--', label="RLF Threshold")
            plt.xlabel("Simulation Time (s)")
            plt.ylabel("Received Power (dBm)")
            serving_id = seg["serving_eNB_id"]
            target_id = seg["target_eNB_ids"][-1] if seg["target_eNB_ids"] else "N/A"
            plt.title(f"Segment {idx+1}: Serving BS: {serving_id} | Target BS: {target_id}")
            plt.legend()
            plt.grid(True)
            
            # image_path = os.path.join(output_dir, f"episode_{episode_number}_segment_{idx+1}.png")
            image_path  = f'{output_dir}/episode_{episode_number}_segment_{idx+1}.png'
            plt.savefig(image_path)
            plt.close()

    def plot_sinr(self):
        output_dir = self.param.TEST_OUTPUT_FOLDER
        # individual_dir = os.path.join(output_dir, "individual_sinr")
        individual_dir = f'{output_dir}/individual_sinr'

        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(individual_dir, exist_ok=True)

        # Plot overall SINR
        plt.figure(figsize=(8, 5))
        plt.plot(self.sinr_record_time, self.sinr_list, marker='o', linestyle='-', color='b', label='SINR')
        plt.xlabel("Recorded Time")
        plt.ylabel("SINR (dB)")
        plt.title("Overall SINR vs. Recorded Time")
        plt.legend()
        plt.grid(True)
        
        # overall_path = os.path.join(output_dir, "sinr_time.png")
        overall_path = f'{output_dir}/sinr_time.png'
        plt.savefig(overall_path, dpi=300)
        # plt.show()

        # Plot individual SINR for each user
        for user_id, sinr_values in self.sinr_individual_list.items():
            plt.figure(figsize=(8, 5))
            plt.plot(self.sinr_record_time, sinr_values, marker='o', linestyle='-', label=f'User {user_id}')
            plt.xlabel("Recorded Time")
            plt.ylabel("SINR (dB)")
            plt.title(f"SINR vs. Recorded Time for User {user_id} with location {self.multi_ue_position_list[user_id][1]}")
            plt.legend()
            plt.grid(True)

            # Save each user's SINR plot
            user_image_path = os.path.join(individual_dir, f"sinr_user_{user_id}.png")
            plt.savefig(user_image_path, dpi=300)
            plt.close()  # Close the plot to save memory

    def calculate_rewards(self):
        total_reward = 0
        for i in self.rewards:
            total_reward += i
        # make a df with the time_steps, rewards and information_reward
        data = {
            'time_steps': self.time_steps_rewards,
            'rewards': self.rewards,
            'information_reward': self.information_reward
        }
        df = pd.DataFrame(data)
        data_overall_simulation = {
            't' : self.time_series,
            'serv_power' : self.assoc_power_series,
            'serv_id' : self.assoc_eNB_series,
            'target_power' : self.target_power_series,
            'target_id' : self.target_eNB_series
        }
        df_overall_simulation = pd.DataFrame(data_overall_simulation)
        return (df, df_overall_simulation, total_reward)
