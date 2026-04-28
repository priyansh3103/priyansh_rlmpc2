#!/usr/bin/env python3
"""
Single vehicle control test - verify Pure Pursuit + PID works correctly
"""

import numpy as np
from vehicle import Vehicle
from controller import PathFollower

print("=" * 60)
print("SINGLE VEHICLE CONTROL TEST")
print("=" * 60)

# Create a simple straight path (left to right)
path = [
    (-20, 0),
    (-10, 0),
    (0, 0),
    (10, 0),
    (20, 0),
]

print(f"\n1. SETUP:")
print(f"   Path: {path}")

# Create vehicle
vehicle = Vehicle(x=-20, y=0, theta=0, v=0)
print(f"   Vehicle initial state: x={vehicle.x}, y={vehicle.y}, θ={np.degrees(vehicle.theta):.1f}°, v={vehicle.v}")

# Create controller
controller = PathFollower()
dt = 0.05  # 50ms timestep

print(f"\n2. SIMULATING VEHICLE MOTION (100 steps):")
print(f"   dt = {dt}s")

step_data = []

for step in range(100):
    idx = 0  # Always use first waypoint for this test
    a, delta, idx = controller.control(vehicle, path, idx, dt=dt)
    
    # Apply action
    vehicle.step([a, delta], dt)
    
    step_data.append({
        'step': step,
        'x': vehicle.x,
        'y': vehicle.y,
        'theta': np.degrees(vehicle.theta),
        'v': vehicle.v,
        'a': a,
        'delta': np.degrees(delta),
    })
    
    if step % 20 == 0:
        print(f"   Step {step:3d}: pos=({vehicle.x:6.2f}, {vehicle.y:6.2f}) | "
              f"θ={np.degrees(vehicle.theta):6.1f}° | v={vehicle.v:5.2f} m/s | "
              f"a={a:5.2f} | δ={np.degrees(delta):6.1f}°")

print(f"\n3. RESULTS:")
print(f"   Final position: x={vehicle.x:.2f}, y={vehicle.y:.2f}")
print(f"   Final heading: θ={np.degrees(vehicle.theta):.1f}°")
print(f"   Final velocity: v={vehicle.v:.2f} m/s")

# Check if vehicle is moving
if vehicle.v > 0.1:
    print(f"   ✓ Vehicle accelerating")
else:
    print(f"   ✗ Vehicle NOT accelerating")

# Check if x increased (vehicle moved forward)
if vehicle.x > -20:
    print(f"   ✓ Vehicle moved forward ({vehicle.x - (-20):.2f} units)")
else:
    print(f"   ✗ Vehicle did NOT move forward")

# Check heading is close to 0 (pointing right)
if abs(np.degrees(vehicle.theta)) < 10:
    print(f"   ✓ Vehicle heading correct (should be ~0°)")
else:
    print(f"   ✗ Vehicle heading wrong (should be ~0°, got {np.degrees(vehicle.theta):.1f}°)")

print("\n" + "=" * 60)
print("CONTROL TEST COMPLETE")
print("=" * 60)
