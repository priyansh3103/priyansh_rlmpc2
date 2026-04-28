#!/usr/bin/env python3
"""
Diagnostic script to verify map and graph generation
"""

import numpy as np
from roads import ROADS
from graph import RoadGraph

print("=" * 60)
print("MAP & GRAPH DIAGNOSTIC")
print("=" * 60)

# 1. Check roads
print(f"\n1. ROADS:")
print(f"   Total road segments: {len(ROADS)}")
for i, road in enumerate(ROADS):
    if road[0] == "line":
        start, end = road[1], road[2]
        length = np.linalg.norm(np.array(end) - np.array(start))
        print(f"   Road {i}: {start} → {end} (length: {length:.2f})")

# 2. Build graph
print(f"\n2. GRAPH GENERATION:")
graph = RoadGraph(spacing=2.0)
print(f"   Spacing: {graph.spacing}")
graph.build_graph()

print(f"   Total nodes: {len(graph.nodes)}")
print(f"   Total edges: {sum(len(neighbors) for neighbors in graph.edges.values())}")

# Print min/max coordinates
if graph.nodes:
    xs = [n[0] for n in graph.nodes]
    ys = [n[1] for n in graph.nodes]
    print(f"   X range: [{min(xs):.2f}, {max(xs):.2f}]")
    print(f"   Y range: [{min(ys):.2f}, {max(ys):.2f}]")

# 3. Test connectivity
print(f"\n3. CONNECTIVITY CHECK:")

# Find a start node (leftmost)
if graph.nodes:
    start = min(graph.nodes, key=lambda n: n[0])
    # Find an end node (rightmost)
    end = max(graph.nodes, key=lambda n: n[0])
    
    print(f"   Start node (leftmost): {start}")
    print(f"   End node (rightmost): {end}")
    
    # Test A* pathfinding
    path = graph.astar(start, end)
    print(f"   A* path length: {len(path)} waypoints")
    
    if len(path) > 0:
        print(f"   Path starts: {path[0]}")
        print(f"   Path ends: {path[-1]}")
        print(f"   ✓ Path planning works!")
    else:
        print(f"   ✗ ERROR: No path found!")

# 4. Test waypoint interpolation
print(f"\n4. WAYPOINT INTERPOLATION:")
if len(path) > 0:
    smooth_path = graph.interpolate_waypoints(path, num_points=10)
    print(f"   Discrete waypoints (A*): {len(path)}")
    print(f"   Smooth waypoints (interpolated): {len(smooth_path)}")
    print(f"   ✓ Interpolation works!")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
