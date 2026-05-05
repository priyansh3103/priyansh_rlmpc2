#!/usr/bin/env python3
"""Debug: Analyze which lanes paths use."""

from graph import RoadGraph

g = RoadGraph(spacing=1.5)
g.build_graph()

print("="*60)
print("LANE USAGE ANALYSIS")
print("="*60)

# Route 1: Upper-left to lower-left (vertical crossing on left side)
print("\n1. Upper-left zone (y=12) to Lower-left zone (y=-12):")
start1 = g.get_closest_node((-12, 12))
goal1 = g.get_closest_node((-12, -12))
path1 = g.astar(start1, goal1)

if path1:
    y_coords = [p[1] for p in path1]
    central_y = [y for y in y_coords if abs(y) <= 1]
    print(f"   Path length: {len(path1)} waypoints")
    print(f"   Y-coords in path: min={min(y_coords):.2f}, max={max(y_coords):.2f}")
    if central_y:
        unique_central_y = set([round(y, 1) for y in central_y])
        print(f"   Y-values IN CENTRAL: {sorted(unique_central_y)}")
        print(f"   Uses BOTH lanes (y=±0.5)? {len(unique_central_y) > 1}")
    else:
        print(f"   Does NOT pass through central crossing")

# Route 2: Upper-left to upper-right (horizontal crossing on upper side)
print("\n2. Upper-left zone (x=-12) to Upper-right zone (x=12):")
start2 = g.get_closest_node((-12, 20))
goal2 = g.get_closest_node((12, 20))
path2 = g.astar(start2, goal2)

if path2:
    x_coords = [p[0] for p in path2]
    central_x = [x for x in x_coords if abs(x) <= 1]
    print(f"   Path length: {len(path2)} waypoints")
    print(f"   X-coords in path: min={min(x_coords):.2f}, max={max(x_coords):.2f}")
    if central_x:
        unique_central_x = set([round(x, 1) for x in central_x])
        print(f"   X-values IN CENTRAL: {sorted(unique_central_x)}")
        print(f"   Uses BOTH lanes (x=±0.5)? {len(unique_central_x) > 1}")
    else:
        print(f"   Does NOT pass through central crossing")

# Route 3: Lower-left to upper-right (diagonal crossing)
print("\n3. Lower-left zone to Upper-right zone:")
start3 = g.get_closest_node((-12, -12))
goal3 = g.get_closest_node((12, 12))
path3 = g.astar(start3, goal3)

if path3:
    x_coords = [p[0] for p in path3]
    y_coords = [p[1] for p in path3]
    central_nodes = [(p[0], p[1]) for p in path3 if abs(p[0]) <= 1 or abs(p[1]) <= 1]
    print(f"   Path length: {len(path3)} waypoints")
    print(f"   Central region nodes: {len(central_nodes)}")
    if central_nodes:
        print(f"   Sample central nodes: {central_nodes[:5]}")

print("\n" + "="*60)
print("CRITICAL FINDING:")
print("="*60)
print("Are the two horizontal lanes independent?")
h_plus = [n for n in g.nodes if -22 <= n[0] <= 22 and 0.4 < n[1] < 0.6]
h_minus = [n for n in g.nodes if -22 <= n[0] <= 22 and -0.6 < n[1] < -0.4]
print(f"  y=+0.5 lane: {len(h_plus)} nodes")
print(f"  y=-0.5 lane: {len(h_minus)} nodes")

h_plus_neighbors = set()
for n in h_plus:
    h_plus_neighbors.update(g.edges.get(n, []))

lateral_connections = len([x for x in h_minus if x in h_plus_neighbors])
print(f"  Cross-lane connections: {lateral_connections}")
print(f"  ⟹ Lanes are {'CONNECTED' if lateral_connections > 0 else 'SEPARATE'}")
