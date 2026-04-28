#!/usr/bin/env python3
"""
Single vehicle path following test - verify Pure Pursuit on curved path
"""

import numpy as np
from vehicle import Vehicle
from controller import PathFollower

print("=" * 60)
print("PATH FOLLOWING TEST (CURVED PATH)")
print("=" * 60)

# Create an L-shaped path (right, then up)
path = [
    (-20, -5), (-15, -5), (-10, -5), (-5, -5), (0, -5), (5, -5), (10, -5), (15, -5),  # Go right
    (15, 0), (15, 5), (15, 10), (15, 15),  # Go up
]

print(f"\n1. SETUP:")
print(f"   Path: L-shaped (right, then up)")
print(f"   Total waypoints: {len(path)}")

# Create vehicle
vehicle = Vehicle(x=-20, y=-5, theta=0, v=0)
print(f"   Vehicle initial state: x={vehicle.x}, y={vehicle.y}, θ={np.degrees(vehicle.theta):.1f}°")

# Create controller
controller = PathFollower()
dt = 0.05

print(f"\n2. SIMULATING (500 steps):")

idx = 0
max_idx = 0
goal_reached = False

for step in range(500):
    a, delta, idx = controller.control(vehicle, path, idx, dt=dt)
    vehicle.step([a, delta], dt)
    
    max_idx = max(max_idx, idx)
    
    # Check if goal reached
    goal = path[-1]
    dist_to_goal = np.hypot(vehicle.x - goal[0], vehicle.y - goal[1])
    if dist_to_goal < 2.0:
        goal_reached = True
    
    if step % 100 == 0:
        print(f"   Step {step:3d}: pos=({vehicle.x:7.2f}, {vehicle.y:7.2f}) | "
              f"θ={np.degrees(vehicle.theta):7.1f}° | v={vehicle.v:5.2f} | "
              f"δ={np.degrees(delta):6.1f}° | idx={idx}")

print(f"\n3. RESULTS:")
print(f"   Final position: ({vehicle.x:.2f}, {vehicle.y:.2f})")
print(f"   Final heading: {np.degrees(vehicle.theta):.1f}°")
print(f"   Final velocity: {vehicle.v:.2f} m/s")
print(f"   Waypoint index reached: {max_idx}/{len(path)-1}")

goal = path[-1]
dist_to_goal = np.hypot(vehicle.x - goal[0], vehicle.y - goal[1])
print(f"   Distance to goal: {dist_to_goal:.2f}")

if dist_to_goal < 3.0:
    print(f"   ✓ Vehicle reached goal!")
else:
    print(f"   ✗ Vehicle did not reach goal")

if max_idx > len(path) // 2:
    print(f"   ✓ Vehicle progressed through path")
else:
    print(f"   ✗ Vehicle did not progress through path")

print("\n" + "=" * 60)
print("PATH FOLLOWING TEST COMPLETE")
print("=" * 60)
