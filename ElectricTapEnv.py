import gymnasium
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

import random
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import math

from TapSimulator import TapSimulator

MIN_SETPOINT = 2.353 # min tensao sensor
MAX_SETPOINT = 3.023 # max tensao lido

MAX_ERROR = MAX_SETPOINT-MIN_SETPOINT

MAX_STEP_SIZE_PERCENTAGE = 20/100

VOLTS_TO_RADS = 10/math.pi
RADS_TO_VOLTS = 1/VOLTS_TO_RADS

MIN_CONTROL_ACTION = 0.0
MAX_CONTROL_ACTION = 10*VOLTS_TO_RADS

MIN_REWARD = -2
MAX_REWARD = 2

# amount of seconds of each step simulation
SIMULATION_STEP_PERIOD_SEC = 0.25
# total seconds of simulation
SIMULATION_TOTAL_TIME_SEC = 180 # value chosen after some manual simulations on closed loop
# number of simulations needed to achieve defined simulation time
SIMULATION_MAX_ITERATIONS = math.ceil(SIMULATION_TOTAL_TIME_SEC/SIMULATION_STEP_PERIOD_SEC)

MIN_KP = -30
MAX_KP = 0

MIN_KI = -30
MAX_KI = 0


NOISE_GAMMA = 10

class ElectricTapEnv(Env):
    def __init__(self, plot_results = False):
        super().__init__()
        self.simulator = TapSimulator()

        # action is to choose the pid gains
        self.action_space = Box(np.array([MIN_KP, MIN_KI]), np.array([MAX_KP, MAX_KI]), shape=(2,), dtype=np.float16)

        # observation space has the form: [pv, mv, error, error_integral, error_derivative, kp, ki]
        lower_limits = np.array([MIN_SETPOINT, MIN_CONTROL_ACTION, -MAX_ERROR, 0, 0, MIN_KP, MIN_KI])
        upper_limits = np.array([MAX_SETPOINT, MAX_CONTROL_ACTION, MAX_SETPOINT - MIN_SETPOINT, math.inf, math.inf, MAX_KP, MAX_KI])
        self.observation_space = Box(np.array(lower_limits), np.array(upper_limits), shape=(7,), dtype=np.float16)
                
        self.reward_range = (MIN_REWARD,MAX_REWARD)

        # initialize internal state
        self.reset()

        self.plot_results = plot_results
    
    def step(self, action):
        self.__increment_counter()
        self.__take_action(action)
        self.__simulate_plant()
        
        reward = self.__get_reward()
        done = self.__get_is_done()

        if(done and self.plot_results):
            self.__plot()

        truncated = False
        info = {}
        return self.__serialize_state(), reward, done, truncated, info

    def render(self):
        pass
    
    def reset(self, seed=None, options=None):
        self.internal_state = {
            "KP": np.random.uniform(MIN_KP, MAX_KP),
            "KI": np.random.uniform(MIN_KI, MAX_KI),
            "KD": 0, # this will remain 0 for all simulation long
            "integral_error": 0,
            "pv_arr": [], 
            "control_action_arr": [],
            "setpoint": random.uniform(MIN_SETPOINT, MAX_SETPOINT),
            "noise": 0,
            "iterations_counter": 0,
            "max_iterations": SIMULATION_MAX_ITERATIONS,
            "x1": 0,
            "x1_ponto": 0,
            "KP_arr": [-1],
            "KI_arr": [0],
            "KD_arr": [0],
        }

        self.__set_initial_control_action()
        self.__set_initial_setpoint()
        self.__set_initial_pv()

        return self.__serialize_state(), {}

    def __get_reward(self):
        error = self.internal_state["setpoint"] - self.internal_state["pv_arr"][-1]
        last_control_action = self.internal_state['control_action_arr'][-1]*RADS_TO_VOLTS

        error_penalty = -error**2
        # tries to keep control action away from limits
        control_action_penalty = -(3/25_00)*(last_control_action-5)**2
        noise_forgiviness = NOISE_GAMMA*self.internal_state['noise']**2

        return error_penalty + control_action_penalty + noise_forgiviness
    
    def __set_initial_setpoint(self):
        a, b = 0.5, 0.5  # Esses valores dão mais peso às extremidades

        # Gerar valores da distribuição beta
        beta_values = np.random.beta(a, b)
        mapped_value = np.interp(beta_values, (0, 1), (MIN_SETPOINT, MAX_SETPOINT))
        self.internal_state["setpoint"] = mapped_value

    def __set_initial_pv(self):
        # define lower and upper bounds based on configured percentage 
        delta_setpoint_scale = MAX_SETPOINT - MIN_SETPOINT
        max_delta_step = delta_setpoint_scale * MAX_STEP_SIZE_PERCENTAGE

        # random initial reading value (respecting max step size and max read values)
        initial_pv = random.uniform(self.internal_state['setpoint'] - max_delta_step, self.internal_state['setpoint'] + max_delta_step)
        initial_pv = np.clip(initial_pv, MIN_SETPOINT, MAX_SETPOINT)

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
        ran_out_of_time = self.internal_state["iterations_counter"] >= self.internal_state["max_iterations"]

        return ran_out_of_time 

    def __take_action(self, action):
        action = np.clip(action, [MIN_KP, MIN_KI], [MAX_KP, MAX_KI])
        self.internal_state["KP"] = action[0]
        self.internal_state["KI"] = action[1]

        # if(self.plot_results and len(self.internal_state['KP_arr'])>0):
        if(False):
            smooth = 0.4
            self.internal_state['KP'] = smooth*action[0] + (1-smooth) *self.internal_state["KP_arr"][-1]
            self.internal_state['KI'] = smooth*action[1] + (1-smooth) *self.internal_state["KI_arr"][-1]
        
        # These internal states make simulation slower and are only used for plots
        if(self.plot_results):
        # if(True):
            self.internal_state["KP_arr"].append(self.internal_state["KP"])
            self.internal_state["KI_arr"].append(self.internal_state["KI"])
            self.internal_state["KD_arr"].append(self.internal_state["KD"])

    def __calculate_control_action(self):
        error = self.internal_state["setpoint"] - self.internal_state["pv_arr"][-1]
        P = self.internal_state["KP"] * error
        
        self.internal_state["integral_error"] += error * SIMULATION_STEP_PERIOD_SEC
        I = self.internal_state["integral_error"] * self.internal_state["KI"] 

        control_action = np.clip(P+I, MIN_CONTROL_ACTION, MAX_CONTROL_ACTION)

        return control_action
    
    def __simulate_plant(self):
        control_action = self.__calculate_control_action()

        y0 = [self.internal_state["x1"], self.internal_state["x1_ponto"]]

        simulation_noise = self.simulator.generate_noise()
        simulation_result = self.simulator.simulate(y0, control_action, SIMULATION_STEP_PERIOD_SEC, simulation_noise)
        
        
        pv = simulation_result[-1,0]

        self.internal_state["pv_arr"].append(pv)
        self.internal_state["noise"] = simulation_noise
        self.internal_state["x1"] = pv
        self.internal_state["x1_ponto"] = simulation_result[-1,1]
        self.internal_state["control_action_arr"].append(control_action)

        return pv
    
    def __serialize_state(self):
        # [pv, mv, error, error_integral, error_derivative, kp, ki]
        curr_error = self.internal_state["setpoint"] - self.internal_state["pv_arr"][-1]
        last_error = self.internal_state["setpoint"] - self.internal_state["pv_arr"][-2]

        pv = self.internal_state["pv_arr"][-1]
        mv = self.internal_state["control_action_arr"][-1]
        error = curr_error
        error_integral = self.internal_state['integral_error']
        error_derivative = (curr_error - last_error)/SIMULATION_STEP_PERIOD_SEC
        kp = self.internal_state["KP"]
        ki = self.internal_state["KI"]

        output = np.array([pv, mv, error, error_integral, error_derivative, kp, ki]).astype(np.float16)

        return output
    
    def __plot(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        pv_list = self.internal_state['pv_arr']
        control_action_list = self.internal_state['control_action_arr']
        time_list = np.linspace(0, len(pv_list) * SIMULATION_STEP_PERIOD_SEC, len(pv_list))

        # Plot PV x Setpoint in time
        ax1.plot(time_list, pv_list, label='Tensão de Saída (PV)', color='b')
        ax1.set_ylabel('Tensão de Saída [V]', color='b')
        ax1.legend(loc='upper right')

        # Add setpoint line
        setpoint_list = [self.internal_state['setpoint']] * len(time_list)
        ax1.plot(time_list, setpoint_list, label='Setpoint', color='g', linestyle='--')

        # Plot control action
        ax2.plot(time_list, control_action_list, label='Ação de Controle (U)', color='r')
        ax2.set_ylabel('Ação de Controle [V]', color='r')
        ax2.legend(loc='upper right')

        # Plot gains in time
        kp_arr = self.internal_state['KP_arr']
        ki_arr = self.internal_state['KI_arr']
        kd_arr = self.internal_state['KD_arr']
        time_list = np.linspace(0, len(kp_arr) * SIMULATION_STEP_PERIOD_SEC, len(kp_arr))
        ax3.plot(time_list, kp_arr, label='Kp', color='c')
        ax3.plot(time_list, ki_arr, label='Ki', color='r')
        ax3.plot(time_list, kd_arr, label='Kd', color='m')
        ax3.set_xlabel('Tempo [s]')
        ax3.set_ylabel('Ganhos')
        ax3.legend(loc='upper right')

        plt.suptitle('Resultado de simulação em ambiente de aprendizado.')
        plt.tight_layout()
        plt.show()