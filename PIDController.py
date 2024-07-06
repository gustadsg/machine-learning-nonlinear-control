import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, TS, min_max):
        self.KP = kp
        self.KI = ki
        self.KD = kd
        self.setpoint = setpoint
        self.integral_error = 0
        self.previous_error = 0
        self.TS = TS
        self.min_max = min_max

    def calculate_control_action(self, pv):
        error = self.setpoint - pv
        P = self.KP * error
        self.integral_error += error * self.TS
        I = self.integral_error * self.KI
        D = (error - self.previous_error) * self.KD / self.TS
        self.previous_error = error
        return np.clip(P + I + D, self.min_max[0], self.min_max[1])