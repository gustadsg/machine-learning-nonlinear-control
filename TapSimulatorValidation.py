import math
import random
from scipy import integrate
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import csv

class TapSimulator:
    def __init__(self):
        self.vin = 10
        self.R = 0.1
        self.tau1 = 0.8902
        self.tau2 = 26.8097
        self.r = math.pi / 10
        self.k = -0.1093
        self.Ts = 0.2

        self.noiseVariance = 0.004

    def __F(self, theta: float) -> float:
        return ((self.vin * self.vin) / self.R) * (0.5 - theta / (2 * math.pi) + math.sin(2 * theta) / (4 * math.pi))

    def __dFdt(self, theta: float) -> float:
        return ((self.vin * self.vin) / (self.R * 2 * math.pi)) * (math.cos(2 * theta) - 1) + 0.00001

    def generate_noise(self, number_of_points: int):
        noise_vec = np.random.normal(0, 0, number_of_points)
        return noise_vec

    def __plant(self, y: List[float], t, u: float, u0: float, noise_in: float):
        x1, x2 = y

        u_noisy = u + noise_in[-1]

        return [
            x2, # x1dot
            ((self.k * (self.__F(self.r * u_noisy) - self.__F(self.r * u0))) / (self.r * self.__dFdt(self.r * u0)) - x1 - x2 * (self.tau1 + self.tau2)) / (self.tau1 * self.tau2) # x2dot
        ]

    def simulate(self, y0: List[float], u0: float, u: float, simulationPeriodSec: float):
        num_of_points = math.floor((simulationPeriodSec * 1000) / 5)

        t = np.linspace(0, simulationPeriodSec, num_of_points)
        noise_input = self.generate_noise(len(t))
        noise_output = self.generate_noise(len(t))

        result = integrate.odeint(self.__plant, y0, t, args=(u, u0, noise_input))

        result[:, 0] += noise_output

        return t, result

    def validate(self, data: pd.DataFrame):
        data.columns = data.columns.str.strip()  # Remove leading/trailing spaces in column names
        
        if 'CicloInicio' not in data.columns or 'CicloFim' not in data.columns:
            raise KeyError("As colunas 'CicloInicio' e 'CicloFim' devem estar presentes no arquivo CSV.")

        time_ns = data['CicloInicio'].to_numpy()
        time_sec = (time_ns - time_ns[0])*5 / 1e6  # Convert nanoseconds to seconds and start at 0
        leitura_filtrada = data['Leitura_filtrada'].to_numpy()
        escrita = data['Escrita'].to_numpy()
        media_leitura = np.mean(leitura_filtrada)

        y0 = [leitura_filtrada[0], 0]  # Assume initial derivative is zero
        u0 = escrita[0]

        fig, ax1 = plt.subplots()
        simulated_output = []
        time_vector = []

        for i in range(1, len(time_sec)):
            u = escrita[i]
            period = time_sec[i] - time_sec[i - 1]
            
            # Ensure period is positive
            if period <= 0:
                print(f"Invalid period between time steps {i-1} and {i}: {period}. Skipping this interval.")
                continue
            
            t, result = self.simulate(y0, u0, u, period)
            
            if result.size == 0:
                print(f"No points generated for interval between time steps {i-1} and {i}.")
                continue
            
            y0 = result[-1]
            
            time_vector.extend(t + time_sec[i-1])
            simulated_output.extend(result[:, 0]+2.5725)
            u0 = u

        ax1.plot(time_vector, simulated_output, label='Simulated Data', color='blue')
        ax1.plot(time_sec, leitura_filtrada, 'r-', label='Real Data')

        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Output Voltage [V]')
        ax1.legend()
        ax1.set_ylim([media_leitura-0.2, media_leitura+0.2])
        plt.title('Model Validation')
        plt.grid(True)
        plt.savefig("validation_result.png")


if __name__ == "__main__":
    # Carregar dados do CSV
    data = pd.read_csv('ale_ts500_tb1500_3a7.csv', sep=";")
    #data = pd.read_csv('degraus.csv', sep=";")

    TapSimulator = TapSimulator()

    TapSimulator.validate(data)
