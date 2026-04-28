import numpy as np
import matplotlib.pyplot as plt
from vehicle import Vehicle

def run_test(acceleration, steering, steps=100):
    car = Vehicle(0, 0, theta=0.0, v=0.0)

    xs, ys = [], []

    for _ in range(steps):
        state = car.step([acceleration, steering])
        xs.append(state[0])
        ys.append(state[1])

    return xs, ys


# -------- TEST 1: Straight line --------
xs1, ys1 = run_test(acceleration=1.0, steering=0.0)

# -------- TEST 2: Turning --------
xs2, ys2 = run_test(acceleration=0.5, steering=0.3)

print("Final state (straight):", xs1[-1], ys1[-1])
print("Final state (turning):", xs2[-1], ys2[-1])
# Plot
plt.figure(figsize=(6, 6))
plt.plot(xs1, ys1, label="Straight")
plt.plot(xs2, ys2, label="Turning")
plt.legend()
plt.axis("equal")
plt.title("Vehicle Model Test")
plt.show()