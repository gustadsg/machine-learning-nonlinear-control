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
        self.r = math.pi/10
        self.k = -0.1093
        self.ybar=(3.023+2.353)/2

        self.noiseVariance = 2.287e-4

        "Assumindo thetabar=pi/2, F'(thetabar)=-1"
    def __F(self, theta: float) -> float:
        return (-(math.pi/2 - theta/2 + math.sin(2*theta)/4))

    def generate_noise(self, number_of_points: int):
        noise_vec = list()
        noise_rate = 0.01
        for i in range(number_of_points):
            sample_has_noise = random.uniform(0,1) < noise_rate
            sample_noise_value = random.gauss(0, self.noiseVariance) if sample_has_noise else 0
            # noise_vec.append(sample_noise_value)
            noise_vec.append(0)

        return noise_vec

    """
    x1 = output (V)
    x2 = x1dot = output derivative (V/s)
    x2dot = output second derivative (v/s^2)
    """
    def __plant(self, y: List[float], t, u: float, noise: float):
        x1, x2 = y
        noise_variance = 2.287e-4
        uclipped = np.clip(u,0,10)

        return [
            x2, # x1dot
            (self.k*(self.__F(self.r*uclipped)+math.pi/4)/self.r + noise_variance*noise - (x1-self.ybar) - x2*(self.tau1+self.tau2))/(self.tau1*self.tau2) #x2dot
        ]

    def simulate(self, y0: List[int], u: float, simulationPeriodSec: float, noise: float):
        """
        Simulates one plant iteration given a control input and a initial state

        Args:
            y0: the initial state of the plant. Passed in the format [x, dxdt], where x is the process variable.
            u: the current control signal to be applied to the plant
            simulationPeriodSec: total period of simulation in seconds

        Returns:
            The list of points collected during the simulation
        """

        num_of_points =  5

        t = np.linspace(0, simulationPeriodSec, num_of_points)
        return integrate.odeint(self.__plant, y0, t, args=(u, noise))
    

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
        noise = TapSimulator.generate_noise(num_of_points)
        result = TapSimulator.simulate(y0, u, simulationPeriod, noise[-1])  # Comece cada degrau a partir do anterior
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
    

