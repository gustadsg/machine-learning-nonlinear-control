from TapSimulator import TapSimulator
from PIDController import PIDController

import numpy as np
from matplotlib import pyplot as plt

MIN_SETPOINT = 2.353 # min tensao sensor
MAX_SETPOINT = 3.023 # max tensao lido
AVG_SETPOINT = (MAX_SETPOINT+MIN_SETPOINT)/2
AVG_SETPOINT = 2.5

class ClosedLoopSimulator:
    def __init__(self, kp, ki, kd, setpoint, TS):
        self.tap_simulator = TapSimulator()
        self.pid_controller = PIDController(kp, ki, kd, setpoint, TS, min_max=[0,10])
        self.TS = TS
        self.setpoint = setpoint

    def run_simulation(self, simulation_time_sec):
        x0 = 2.5+.134
        dxdt = 0
        y0 = np.array([x0, dxdt])
        num_steps = int(simulation_time_sec / self.TS)
        
        pv_list = []
        control_action_list = []
        time_list = []

        for step in range(num_steps):
            noise = self.tap_simulator.generate_noise()
            pv = y0[0]
            control_action = self.pid_controller.calculate_control_action(pv)
            result = self.tap_simulator.simulate(y0, control_action, self.TS, noise)
            y0 = result[-1]
            t = np.linspace(step * self.TS, (step + 1) * self.TS, len(result))
            pv_list.extend(result[:, 0])
            control_action_list.extend([control_action] * len(result))
            time_list.extend(t)

        self.plot_results(time_list, pv_list, control_action_list)
        self.print_summary(pv_list)

    def plot_results(self, time_list, pv_list, control_action_list):
        fig, ax1 = plt.subplots()

        ax1.plot(time_list, pv_list, label='Plant Output (PV)', color='b')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Output Voltage [V]', color='b')

        # Add setpoint line
        setpoint_list = [self.setpoint] * len(time_list)
        ax1.plot(time_list, setpoint_list, label='Setpoint', color='g', linestyle='--')

        ax2 = ax1.twinx()
        ax2.plot(time_list, control_action_list, label='Control Action (U)', color='r')
        ax2.set_ylabel('Control Action [V]', color='r')

        plt.title('Closed Loop Simulation')
        fig.legend(loc='upper right')
        plt.grid(True)
        plt.savefig("closed_loop.png")

    def print_summary(self,pv_arr):
        print(f"setpoint: {AVG_SETPOINT}")
        print(f"last PV: {pv_arr[-10:]}")

if __name__ == "__main__":
    #this controller has good results
    closed_loop_simulator = ClosedLoopSimulator(kp=-20, ki=-5, kd=0, setpoint=AVG_SETPOINT, TS=0.25)
    closed_loop_simulator.run_simulation(simulation_time_sec=200)  # Simulate for 6 minutes
