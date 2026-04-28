import numpy as np

class Vehicle:
    def __init__(self, x, y, theta=0.0, v=0.0, L=2.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.L = L  # wheelbase

    def step(self, action, dt=0.1):
        a, delta = action

        # limits
        delta = np.clip(delta, -0.5, 0.5)
        a = np.clip(a, -3.0, 3.0)

        # update velocity
        self.v += a * dt
        self.v = np.clip(self.v, 0.0, 10.0)

        # update heading FIRST (so steering affects this position update)
        self.theta += (self.v / self.L) * np.tan(delta) * dt

        # then update position using the updated heading
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt

        return np.array([self.x, self.y, self.theta, self.v])