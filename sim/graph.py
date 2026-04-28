import numpy as np
import heapq
from roads import ROADS


class RoadGraph:
    def __init__(self, spacing=1.5):
        self.spacing = spacing
        self.nodes = []
        self.edges = {}

    def add_path(self, points):
        prev = None
        for p in points:
            node = (round(p[0], 2), round(p[1], 2))

            if node not in self.edges:
                self.nodes.append(node)
                self.edges[node] = []

            if prev is not None:
                if node not in self.edges[prev]:
                    self.edges[prev].append(node)
                if prev not in self.edges[node]:
                    self.edges[node].append(prev)

            prev = node

    def sample_line(self, start, end):
        start = np.array(start)
        end = np.array(end)

        direction = end - start
        length = np.linalg.norm(direction)

        direction = direction / length
        num_points = int(length / self.spacing)

        return [
            start + direction * i * self.spacing
            for i in range(num_points + 1)
        ]

    def build_graph(self):
        for road in ROADS:
            if road[0] == "line":
                _, start, end = road
                pts = self.sample_line(start, end)
                self.add_path(pts)

    def heuristic(self, n1, n2):
        return np.linalg.norm(np.array(n1) - np.array(n2))

    def get_closest_node(self, point):
        px, py = point
        return min(self.nodes, key=lambda n: (n[0]-px)**2 + (n[1]-py)**2)

    def astar(self, start, goal):
        # Validate inputs
        if start not in self.nodes:
            print(f"Warning: start node {start} not in graph. Finding closest...")
            start = self.get_closest_node(start)
        
        if goal not in self.nodes:
            print(f"Warning: goal node {goal} not in graph. Finding closest...")
            goal = self.get_closest_node(goal)

        if start == goal:
            return [start]  # Already at goal

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {node: float('inf') for node in self.nodes}
        g_score[start] = 0

        f_score = {node: float('inf') for node in self.nodes}
        f_score[start] = self.heuristic(start, goal)

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.edges[current]:
                tentative_g = g_score[current] + self.heuristic(current, neighbor)

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        print(f"No path found from {start} to {goal}")
        return []

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def interpolate_waypoints(self, path, num_points=10):
        """
        Convert discrete node path to continuous waypoint trajectory.
        
        Input: path = [node1, node2, node3, ...] (discrete nodes from A*)
        Output: smooth_path = [waypoint1, waypoint2, ...] (dense waypoints)
        
        Between each pair of nodes, we linearly interpolate with 'num_points' samples.
        """
        if not path:
            return []
        
        if len(path) == 1:
            return path  # Already at goal
        
        smooth_path = []
        
        # Interpolate between consecutive nodes
        for i in range(len(path) - 1):
            start_node = np.array(path[i])
            end_node = np.array(path[i + 1])
            
            # Generate intermediate waypoints
            for t in np.linspace(0, 1, num_points + 1):
                waypoint = start_node + t * (end_node - start_node)
                waypoint = tuple(np.round(waypoint, 2))
                
                # Avoid duplicates
                if not smooth_path or waypoint != smooth_path[-1]:
                    smooth_path.append(waypoint)
        
        return smooth_path