import gymnasium
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

import random
import numpy as np
from functools import reduce
import math

from TapSimulator import TapSimulator

P_ACTION_INDEX = 0
I_ACTION_INDEX = 1
D_ACTION_INDEX = 2

MIN_SETPOINT = 0
MAX_SETPOINT = 0

MIN_ERROR = 0.001
MAX_ERROR = (MAX_SETPOINT - MIN_SETPOINT)/2

MIN_CALCULATED_REWARD = -25
MAX_CALCULATED_REWARD = 25

MIN_OVERALL_REWARD = -100
MAX_OVERALL_REWARD = 100

MIN_CONTROL_ACTION = 0
MAX_CONTROL_ACTION = 10*10/math.pi

class ElectricTapEnv(Env):
    def __init__(self):
        super().__init__()
        self.simulator = TapSimulator()

        # actions can be decrement, maintain or decrement gains by a pre-defined value
        self.action_space = Discrete(7)
        # observation space has the form: [pv, mv, error, error_integral, error_derivative, kp, ki, kd]
        lower_limits = np.array([-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf])
        upper_limits = np.array([math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf])
        self.observation_space = Box(np.array(lower_limits), np.array(upper_limits), shape=(8,), dtype=np.float32)

        
        self.internal_state = {
            "KP": 1,
            "KI": 0,
            "KD": 0,
            "integral_error": 0,
            "pv_arr": [], 
            "control_action_arr": [],
            "setpoint": random.uniform(MIN_SETPOINT, MAX_SETPOINT),
            "noise_power": 1,
            "iterations_counter": 0,
            "max_iterations": 500
        }

        # get random initial state
        self.__set_initial_control_action()
        self.__set_initial_pv()
        self.__set_setpoint()
    
    def step(self, action):
        self.__increment_counter()
        self.__take_action(action)
        self.__simulate_plant()
        
        # simulator.simulate()
        reward = self.__get_reward()
        done = self.__get_is_done()

        truncated = self.internal_state["control_action_arr"][-1] < MIN_CONTROL_ACTION or self.internal_state["control_action_arr"][-1] > MAX_CONTROL_ACTION
        info = {}

        return self.__serialize_state(), reward, done, truncated, info

    def render(self):
        pass
    
    def reset(self, seed=None, options=None):
        self.internal_state = {
            "KP": 1,
            "KI": 0,
            "KD": 0,
            "integral_error": 0,
            "pv_arr": [], 
            "control_action_arr": [],
            "setpoint": random.uniform(MIN_SETPOINT, MAX_SETPOINT),
            "noise_power": 1,
            "iterations_counter": 0,
            "max_iterations": 500
        }

        self.__set_initial_control_action()
        self.__set_initial_pv()
        self.__set_setpoint()

        return self.__serialize_state(), {}

    def __get_reward(self):
        error = self.internal_state["setpoint"] - self.internal_state["pv_arr"][-1]

        # error off thresholds (insignificant or giant error)
        if(error<MIN_ERROR): return MAX_OVERALL_REWARD # maximum reward when minimum error
        if(error>MAX_ERROR): return MIN_OVERALL_REWARD # minimum reward when maximum error

        delta_reward = MAX_CALCULATED_REWARD-MIN_CALCULATED_REWARD
        delta_error = (MAX_ERROR*MAX_ERROR - MIN_ERROR*MIN_ERROR)/self.internal_state["noise_power"]

        quadratic_error_param = (error*error)/self.internal_state["noise_power"]
        reward = MIN_CALCULATED_REWARD + delta_reward/delta_error * quadratic_error_param

        return reward

    def __set_setpoint(self):
        self.internal_state["setpoint"] = random.uniform(MIN_SETPOINT, MAX_SETPOINT)

    def __set_initial_pv(self):
        # random initial reading value (between lower and upper bound temperature voltages)
        initial_pv = random.uniform(MIN_SETPOINT, MAX_SETPOINT)
        self.internal_state["pv_arr"].append(initial_pv)
        self.internal_state["pv_arr"].append(initial_pv)

    def __set_initial_control_action(self):
        # random initial control action
        control_action = random.uniform(0,10*self.simulator.r)
        self.internal_state["control_action_arr"].append(control_action)
        self.internal_state["control_action_arr"].append(control_action)

    def __increment_counter(self):
        self.internal_state["iterations_counter"] += 1

    def __get_is_done(self):
        ran_out_of_time = self.internal_state["iterations_counter"] >= self.internal_state["max_iterations"]
        
        abs_error_arr = list(map(lambda pv: abs(self.internal_state["setpoint"]-pv), self.internal_state["pv_arr"]))
        abs_error_window = abs_error_arr[-10:] # last 10 elements of the list

        stabilized = all(error < MIN_ERROR for error in abs_error_window) # is stable if the window has a small error

        return ran_out_of_time or stabilized

    def __take_action(self, action):
        # increase, decrement or maintain each of the PID parameters
        pid_increment = 0.01
        action_map = [
            [0,0,0], # maintain
            [pid_increment, 0, 0], [-pid_increment, 0, 0], # change P
            [0, pid_increment, 0], [0, -pid_increment, 0], # change I
            [0, 0, pid_increment], [0, 0, -pid_increment], # change D
        ]
        
        self.internal_state["KP"] += action_map[action][P_ACTION_INDEX]
        self.internal_state["KI"] += action_map[action][I_ACTION_INDEX]
        self.internal_state["KD"] += action_map[action][D_ACTION_INDEX]

    def __calculate_control_action(self):
        error = list(map(lambda pv: self.internal_state["setpoint"]-pv, self.internal_state["pv_arr"]))
        P = self.internal_state["KP"] * error[-1]
        
        self.internal_state["integral_error"] += error[-1]
        I = self.internal_state["integral_error"] * self.internal_state["KI"]

        D = (error[-1]-error[-2]) * self.internal_state["KD"]

        return P + I + D
    
    def __simulate_plant(self):
        control_action = self.__calculate_control_action()

        x1 = self.internal_state["pv_arr"][-1]
        x1_ponto = (x1 - self.internal_state["pv_arr"][-2])/self.simulator.Ts
        y0 = [x1, x1_ponto]
        simulation_result = self.simulator.simulate(y0, self.internal_state["control_action_arr"][-1], control_action)

        pv = simulation_result[-1,0]

        self.internal_state["pv_arr"].append(pv)

        return pv
    
    def __serialize_state(self):
        # [pv, mv, error, error_integral, error_derivative, kp, ki, kd]
        all_error_arr = list(map(lambda pv: self.internal_state["setpoint"]-pv, self.internal_state["pv_arr"]))

        pv = self.internal_state["pv_arr"][-1]
        mv = self.internal_state["control_action_arr"][-1]
        error = all_error_arr[-1]
        error_integral = reduce(lambda acc, err: acc+err, all_error_arr)
        error_derivative = (all_error_arr[-1] - all_error_arr[-1])/self.simulator.Ts
        kp = self.internal_state["KP"]
        ki = self.internal_state["KI"]
        kd = self.internal_state["KD"]

        return np.array([pv, mv, error, error_integral, error_derivative, kp, ki, kd]).astype(np.float32)