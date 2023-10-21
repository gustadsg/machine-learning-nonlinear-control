import math
from scipy import integrate
import numpy as np
from typing import List

class TapSimulator:
    def __init__(self):
        self.vin = 10
        self.R = 0.1
        self.tau1 = 5
        self.tau2 = 30
        self.r = math.pi/10
        self.k = 0.111
        self.Ts = 0.2

    def __F(self, theta: float) -> float:
        return ((self.vin*self.vin)/self.R) * (0.5 - theta/(2*math.pi) + math.sin(2*theta)/(4*math.pi))

    def __dFdt(self, theta: float) -> float:
        return ((self.vin*self.vin)/(self.R*2*math.pi)) * (math.cos(2*theta)-1)
    
    def __get_noise(self):
        return 0

    """
    x1 = output (V)
    x2 = x1dot = output derivative (V/s)
    x2dot = output second derivative (v/s^2)
    """
    def __plant(self, y, t, u, u0):
        x1, x2 = y

        # inject noise into simulated output
        x1 += self.__get_noise()

        return [
            x2, # x1dot
            ((self.k*(self.__F(self.r*u)-self.__F(self.r*u0)))/(self.r*self.__dFdt(self.r*u0)) - x1 - x2*(self.tau1+self.tau2))/(self.tau1*self.tau2) #x2dot
        ]

    def simulate(self, y0: List[int], u0: float, u: float):
        """
        Simulates one plant iteration given a control input and a initial state

        Args:
            y0: the initial state of the plant. Passed in the format [x, dxdt], where x is the process variable.
            u0: the previous control signal applied to the plant
            u: the current control signal to be applied to the plant

        Returns:
            The list of points collected during the simulation
        """

        period = 1
        num_of_points = 5

        t = np.linspace(0, period, num_of_points)
        return integrate.odeint(self.__plant, y0, t, args=(u, u0))