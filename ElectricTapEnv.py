import gymnasium
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

import random
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import math

from TapSimulator import TapSimulator

P_ACTION_INDEX = 0
I_ACTION_INDEX = 1
D_ACTION_INDEX = 2

MIN_SETPOINT = 0
MAX_SETPOINT = 3.023-2.353 # max tensao lido - min tensao sensor
AVG_SETPOINT = MAX_SETPOINT/2

MIN_ERROR = 0.001
MAX_ERROR = (MAX_SETPOINT - MIN_SETPOINT)/2

MIN_OVERALL_REWARD = -50
MAX_OVERALL_REWARD = 10

MIN_CONTROL_ACTION = 0
MAX_CONTROL_ACTION = 10*10/math.pi

SIMULATION_STEP_PERIOD_SEC = 0.25

MIN_KP = -15
MAX_KP = 0

MIN_KI = -15
MAX_KI = 0

MIN_KD = -15
MAX_KD = 0
 
# TODO? VOLTAR VALOR ORIGINAL
# NOISE_GAMMA = 0.0099
NOISE_GAMMA = 0.0

class ElectricTapEnv(Env):
    def __init__(self):
        super().__init__()
        self.simulator = TapSimulator()

        # actions can be decrement, maintain or decrement gains by a pre-defined value
        # action is to choose the pid gains
        self.action_space = Box(np.array([MIN_KP, MIN_KI, MIN_KD]), np.array([MAX_KP, MAX_KI, MAX_KD]), shape=(3,), dtype=np.float32)
        # observation space has the form: [pv, mv, error, error_integral, error_derivative, kp, ki, kd]
        lower_limits = np.array([-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf])
        upper_limits = np.array([math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf])
        self.observation_space = Box(np.array(lower_limits), np.array(upper_limits), shape=(8,), dtype=np.float32)
        self.reward_range = (-50,25)

        # initialize internal state
        self.reset()

        # get random initial state
        self.__set_initial_control_action()
        self.__set_initial_pv()
        self.__set_setpoint()
    
    def step(self, action):
        self.__increment_counter()
        self.__take_action(action)
        self.__simulate_plant()
        
        reward = self.__get_reward()
        done = self.__get_is_done()

        truncated = self.internal_state["control_action_arr"][-1] < MIN_CONTROL_ACTION or self.internal_state["control_action_arr"][-1] > MAX_CONTROL_ACTION
        info = {}

        return self.__serialize_state(), reward, done, truncated, info

    def render(self):
        pass
    
    def reset(self, seed=None, options=None):
        self.internal_state = {
            "KP": -1,
            "KI": 0,
            "KD": 0,
            "integral_error": 0,
            "pv_arr": [], 
            "control_action_arr": [],
            # "setpoint": random.uniform(MIN_SETPOINT, MAX_SETPOINT),
            "setpoint": AVG_SETPOINT,
            "noise": 0,
            "iterations_counter": 0,
            "max_iterations": 240,
            "x1": 0,
            "x1_ponto": 0,
            "forced_stop": False
        }

        self.__set_initial_control_action()
        self.__set_initial_pv()
        self.__set_setpoint()

        return self.__serialize_state(), {}

    def __get_reward(self):
        error = self.internal_state["setpoint"] - self.internal_state["pv_arr"][-1]

        # if abs(error) < MIN_ERROR:
        #     return MAX_OVERALL_REWARD  # maximum reward when minimum error

        reward = max(MIN_OVERALL_REWARD, -(error * error) + NOISE_GAMMA * self.internal_state["noise"] * self.internal_state["noise"])

        if np.isnan(reward) or not np.isfinite(reward):
            print(f"Invalid reward value: {reward}. Returning 0")
            return 0

        return reward * SIMULATION_STEP_PERIOD_SEC


    def __set_setpoint(self):
        self.internal_state["setpoint"] = random.uniform(MIN_SETPOINT, MAX_SETPOINT)

    def __set_initial_pv(self):
        # random initial reading value (between lower and upper bound temperature voltages)
        initial_pv = random.uniform(MIN_SETPOINT, MAX_SETPOINT)
        self.internal_state["pv_arr"].append(initial_pv)
        self.internal_state["pv_arr"].append(initial_pv)
        
        self.internal_state["x1"] = initial_pv
        self.internal_state["x1_ponto"] = 0

    def __set_initial_control_action(self):
        # random initial control action
        control_action = random.uniform(0,10)
        self.internal_state["control_action_arr"].append(control_action)
        self.internal_state["control_action_arr"].append(control_action)

    def __increment_counter(self):
        self.internal_state["iterations_counter"] += 1

    def __get_is_done(self):
        if(self.internal_state["forced_stop"]):
            return True

        ran_out_of_time = self.internal_state["iterations_counter"] >= self.internal_state["max_iterations"]
        
        abs_error_arr = list(map(lambda pv: abs(self.internal_state["setpoint"]-pv), self.internal_state["pv_arr"]))
        abs_error_window = abs_error_arr[-10:] # last 10 elements of the list

        stabilized = all(error < MIN_ERROR for error in abs_error_window) # is stable if the window has a small error

        return ran_out_of_time or stabilized

    def __take_action(self, action):
        action = np.clip(action, [MIN_KP, MIN_KI, MIN_KD], [MAX_KP, MAX_KI, MAX_KD])
        self.internal_state["KP"] = action[0]
        self.internal_state["KI"] = action[1]
        self.internal_state["KD"] = action[2]

    def __calculate_control_action(self):
        error = list(map(lambda pv: self.internal_state["setpoint"]-pv, self.internal_state["pv_arr"]))
        P = self.internal_state["KP"] * error[-1]
        
        self.internal_state["integral_error"] += error[-1] * SIMULATION_STEP_PERIOD_SEC
        I = self.internal_state["integral_error"] * self.internal_state["KI"] 

        D = (error[-1]-error[-2]) * self.internal_state["KD"]/SIMULATION_STEP_PERIOD_SEC

        return P + I + D
    
    def __simulate_plant(self):
        control_action = self.__calculate_control_action()

        y0 = [self.internal_state["x1"], self.internal_state["x1_ponto"]]

        num_of_points =  math.floor((SIMULATION_STEP_PERIOD_SEC*1000)/5)
        standart_deviation = .0149
        simulation_noise = list(map(lambda x: x*standart_deviation, self.simulator.generate_noise(num_of_points)))
        simulation_result = self.simulator.simulate(y0, control_action, SIMULATION_STEP_PERIOD_SEC, simulation_noise[-1])
        
        
        pv = simulation_result[-1,0]

        self.internal_state["pv_arr"].append(pv)
        self.internal_state["noise"] = simulation_noise[-1]
        self.internal_state["x1"] = pv
        self.internal_state["x1_ponto"] = simulation_result[-1,1]

        return pv
    
    def __serialize_state(self):
        # [pv, mv, error, error_integral, error_derivative, kp, ki, kd]
        all_error_arr = list(map(lambda pv: self.internal_state["setpoint"] - pv, self.internal_state["pv_arr"]))

        pv = self.internal_state["pv_arr"][-1]
        mv = self.internal_state["control_action_arr"][-1]
        error = all_error_arr[-1]
        error_integral = reduce(lambda acc, err: acc + err*SIMULATION_STEP_PERIOD_SEC, all_error_arr)
        error_derivative = (all_error_arr[-1] - all_error_arr[-2]) / SIMULATION_STEP_PERIOD_SEC if len(all_error_arr) > 1 else 0
        kp = self.internal_state["KP"]
        ki = self.internal_state["KI"]
        kd = self.internal_state["KD"]

        try:
            raw_output = np.array([pv, mv, error, error_integral, error_derivative, kp, ki, kd]).astype(np.float32)
            output, replaced = self.__replace_if_invalid(raw_output)

            if replaced:
                print(f"Replacement needed for output: {raw_output}")
                self.__force_stop_episode()

            return output
        except Exception as e:
            print(f"Overflow occurred with values: {[pv, mv, error, error_integral, error_derivative, kp, ki, kd]}")
            print(f"Exception: {e}")
            self.__force_stop_episode()
            return np.array([0, 0, MAX_ERROR, 0, 0, 0, 0, 0]).astype(np.float32)
        
    def __force_stop_episode(self):
        self.reset()
        self.internal_state["iterations_counter"] = 500
        self.internal_state["foced_stop"] = True

    def __replace_if_invalid(self, input_values):
        output_values = []
        replaced = False
        for val in input_values:
            if not isinstance(val, np.float32):
                val = np.float32(val)

            if not np.isfinite(val):
                val = np.float32(0.0)
                replaced = True

            output_values.append(val)

        return np.array(output_values), replaced