#!/usr/bin/env python3
"""
Debug script - trace what the controller is doing
"""

import numpy as np
from vehicle import Vehicle
from controller import PathFollower

# Create a simple path
path = [
    (-20, -5), (-15, -5), (-10, -5), (-5, -5), (0, -5), (5, -5), (10, -5), (15, -5),
    (15, 0), (15, 5), (15, 10), (15, 15),
]

print("PATH ANALYSIS:")
print(f"Path has {len(path)} waypoints")
print(f"Goal (last waypoint): {path[-1]}")
print()

vehicle = Vehicle(x=-20, y=-5, theta=0, v=0)
controller = PathFollower()
dt = 0.05

print("STEP-BY-STEP CONTROL DEBUG:")
print(f"{'Step':<5} {'X':<8} {'Y':<8} {'V':<6} | {'Lookahead':<12} {'Lookahead Pt':<18} {'Delta':<8} {'Idx':<3}")
print("-" * 95)

idx = 0
for step in range(30):  # Just 30 steps to debug
    # Get lookahead distance
    lookahead_dist = controller.get_lookahead_distance(vehicle.v)
    
    # Find lookahead point (replicate controller logic)
    lookahead_point = None
    for i in range(idx, len(path)):
        dx = path[i][0] - vehicle.x
        dy = path[i][1] - vehicle.y
        dist = np.hypot(dx, dy)
        if dist > lookahead_dist:
            lookahead_point = path[i]
            break
    
    if lookahead_point is None:
        lookahead_point = path[-1]
    
    a, delta, idx = controller.control(vehicle, path, idx, dt=dt)
    vehicle.step([a, delta], dt)
    
    print(f"{step:<5} {vehicle.x:<8.2f} {vehicle.y:<8.2f} {vehicle.v:<6.2f} | "
          f"{lookahead_dist:<12.2f} {str(lookahead_point):<18} {np.degrees(delta):<8.1f} {idx:<3}")

print()
print("ISSUE: Vehicle keeps accelerating and goes past the goal")
print("REASON: The lookahead point always exists (path is continuous)")
print("FIX NEEDED: When close to goal, vehicle should slow down MORE")
