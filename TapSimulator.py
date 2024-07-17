import math
import random
from scipy import integrate
import numpy as np
from typing import List
import matplotlib.pyplot as plt

class TapSimulator:
    def __init__(self):
        self.vin = 10
        self.R = 0.1
        self.tau1 = 0.8902
        self.tau2 = 26.8097
        self.r = math.pi / 10
        self.k = -0.1093
        self.ybar = (3.023 + 2.353) / 2
        self.noiseVariance = 0.2
        self.standartVariation = math.sqrt(self.noiseVariance)
        self.noiseGain = self.k

    def __F(self, theta: float) -> float:
        return (-(math.pi/2 - theta/2 + math.sin(2*theta)/4))

    def generate_noise(self):
        return np.random.normal(0, abs(self.noiseGain))

    def __plant(self, t, y, u, noise):
        x1, x2 = y
        uclipped = np.clip(u, 0, 10)
        F_r_uclipped = self.__F(self.r * uclipped)

        x1dot = x2
        x2dot = (self.k * (F_r_uclipped + math.pi / 4) / self.r + noise * self.noiseVariance - (x1 - self.ybar) - x2 * (self.tau1 + self.tau2)) / (self.tau1 * self.tau2)
        
        return [x1dot, x2dot]

    def simulate(self, y0, u, simulationPeriodSec, noise):
        t = [0, simulationPeriodSec]
        sol = integrate.solve_ivp(self.__plant, t, y0, args=(u, noise), method='RK45', t_eval=np.linspace(0, simulationPeriodSec, 3))
        return sol.y.T
    

if __name__ == "__main__":
    x0 = 0
    dxdt = 0
    y0 = np.array([x0, dxdt])

    TapSimulator = TapSimulator()
    simulationPeriod = 1

    # Lista de valores de u para cada degrau
    u_values = np.concatenate((np.full(180, 5), np.full(180, 3)))
    u0 = 5

    u_vector = []

    fig, ax1 = plt.subplots()

    # Simulação para cada degrau
    for i in range(len(u_values)):
        u = u_values[i]
        num_of_points =  math.floor((simulationPeriod*1000)/5)
        noise = TapSimulator.generate_noise()
        result = TapSimulator.simulate(y0, u, simulationPeriod, noise)  # Comece cada degrau a partir do anterior
        t = np.linspace(i * simulationPeriod, (i + 1) * simulationPeriod, len(result))  # Intervalo de tempo para cada degrau
        y0 = result[-1]

        ax1.plot(t, result[:, 0])
        
        u_vector = np.append(u_vector, np.full(len(result), u))


    t_vector = np.linspace(0, len(u_values), len(u_values)*len(result))

    ax1.set_xlabel('Tempo [s]')
    ax1.set_ylabel('Tensão Saída [V]')

    ax2 = ax1.twinx()
    ax2.plot(t_vector, u_vector)
    ax2.set_ylabel('Tensão Entrada [V]')

    plt.title('Simulação torneira')
    plt.grid(True)

    plt.savefig("result.png")
    plt.show()
    

