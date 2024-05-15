
import numpy as np


class PID():
    def __init__(self, KP, KI, KD, saturation_min, saturation_max):
        self.kp = KP
        self.ki = KI
        self.kd = KD
        self.prev_error = 0
        self.integral_error = 0
        self.saturation_max = saturation_max
        self.saturation_min = saturation_min
        self.prev_out = None

    def tune(self, dKP, dKI, dKD, dSAT):
        self.kp += dKP
        self.ki += dKI
        self.kd += dKD

        # Makes range larger or smaller
        self.saturation_max += dSAT
        self.saturation_min -= dSAT

    def compute(self, curr_error, dt):
        derivative_error = (curr_error - self.prev_error) / dt # Rate of Change of Error

        self.integral_error += curr_error * dt # Sum of error

        # Prevents wind up (if saturated, integral error is not added)
        if self.prev_out != None and not ((curr_error>0 and self.prev_out == self.saturation_max) or (curr_error < 0 and self.prev_out == self.saturation_min)):
            self.integral_error += curr_error * dt # Sum of error


        output = (self.kp * curr_error) + (self.ki * self.integral_error) + (self.kd * derivative_error)

        self.prev_error = curr_error

        output = np.clip(output, self.saturation_min, self.saturation_max)

        self.prev_out = output

        return output
