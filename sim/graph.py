import numpy as np
import heapq
from roads import ROADS


class RoadGraph:
    def __init__(self, spacing=1.5):
        self.spacing = spacing
        self.nodes = []
        self.edges = {}
        self.edge_cost = {}
        self.edge_kind = {}
        self.kind_cost = {
            "lane": 1.0,
            "outer_loop": 1.0,
            "lane_change": 2.2,
        }

    def _node_key(self, p):
        # Use stable rounding so floating-point intersections collapse to shared nodes.
        return (round(float(p[0]), 3), round(float(p[1]), 3))

    def _ensure_node(self, p):
        node = self._node_key(p)
        if node not in self.edges:
            self.nodes.append(node)
            self.edges[node] = []
            self.edge_cost[node] = {}
            self.edge_kind[node] = {}
        return node

    def _add_edge(self, a, b, cost, kind):
        if a == b:
            return
        if b not in self.edges[a]:
            self.edges[a].append(b)
        if a not in self.edges[b]:
            self.edges[b].append(a)
        # Keep the cheapest edge if multiple segments connect same node pair.
        current_ab = self.edge_cost[a].get(b, float("inf"))
        current_ba = self.edge_cost[b].get(a, float("inf"))
        if cost < current_ab:
            self.edge_cost[a][b] = cost
            self.edge_kind[a][b] = kind
        if cost < current_ba:
            self.edge_cost[b][a] = cost
            self.edge_kind[b][a] = kind

    def add_path(self, points, weight=1.0, kind="lane"):
        prev = None
        for p in points:
            node = self._ensure_node(p)
            if prev is not None:
                seg_len = self.heuristic(prev, node)
                self._add_edge(prev, node, seg_len * weight, kind)
            prev = node

    def sample_line(self, start, end):
        start = np.array(start)
        end = np.array(end)

        direction = end - start
        length = np.linalg.norm(direction)

        direction = direction / length
        if length < 1e-9:
            return [start]

        num_points = max(1, int(length / self.spacing))

        points = [
            start + direction * i * self.spacing
            for i in range(num_points + 1)
        ]

        # Always include exact segment endpoint.
        if np.linalg.norm(points[-1] - end) > 1e-6:
            points.append(end)

        return points

    def _extract_segments(self):
        segments = []
        for road in ROADS:
            if not road or road[0] != "line":
                continue
            if len(road) == 4:
                _, start, end, kind = road
            else:
                _, start, end = road
                kind = "lane"
            p1 = np.array(start, dtype=float)
            p2 = np.array(end, dtype=float)
            if np.linalg.norm(p2 - p1) > 1e-9:
                segments.append((p1, p2, kind))
        return segments

    def _cross_2d(self, a, b):
        return a[0] * b[1] - a[1] * b[0]

    def _line_intersection(self, p, p2, q, q2, eps=1e-9):
        """Return intersection point if two finite segments intersect."""
        r = p2 - p
        s = q2 - q
        rxs = self._cross_2d(r, s)
        q_p = q - p

        if abs(rxs) < eps:
            # Parallel or collinear; we skip collinear overlap handling for this map.
            return None

        t = self._cross_2d(q_p, s) / rxs
        u = self._cross_2d(q_p, r) / rxs

        if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
            return p + t * r
        return None

    def _project_t(self, start, end, point):
        seg = end - start
        denom = np.dot(seg, seg)
        if denom < 1e-12:
            return 0.0
        return float(np.dot(point - start, seg) / denom)

    def _build_intersection_aware_graph(self):
        segments = self._extract_segments()
        if not segments:
            return

        split_points = {}

        # Seed each segment with its endpoints.
        for idx, (s, e, _) in enumerate(segments):
            split_points[idx] = [s, e]

        # Add pairwise intersections as split points on both segments.
        for i in range(len(segments)):
            p1, p2, _ = segments[i]
            for j in range(i + 1, len(segments)):
                q1, q2, _ = segments[j]
                inter = self._line_intersection(p1, p2, q1, q2)
                if inter is not None:
                    split_points[i].append(inter)
                    split_points[j].append(inter)

        # Split each segment and sample each sub-segment.
        for idx, (start, end, kind) in enumerate(segments):
            pts = split_points[idx]
            weight = self.kind_cost.get(kind, 1.0)

            # Deduplicate split points along this segment.
            unique = {}
            for p in pts:
                key = self._node_key(p)
                if key not in unique:
                    unique[key] = np.array(p, dtype=float)

            ordered = sorted(
                unique.values(),
                key=lambda p: self._project_t(start, end, p),
            )

            for a, b in zip(ordered[:-1], ordered[1:]):
                if np.linalg.norm(b - a) < 1e-9:
                    continue
                sampled = self.sample_line(a, b)
                self.add_path(sampled, weight=weight, kind=kind)

    def _is_driving_edge(self, kind):
        return kind in ("lane", "outer_loop")

    def _lane_change_region_allowed(self, midpoint):
        """Allow lane changes only in intended interaction regions.

        1) Center interaction corridors (zone <-> central intersection)
        2) Perimeter interaction corridors (zone <-> outer loop)
        """
        x, y = abs(midpoint[0]), abs(midpoint[1])

        near_center_horizontal = y <= 5.0 and x <= 18.0
        near_center_vertical = x <= 5.0 and y <= 18.0

        near_outer_vertical = x >= 18.0 and y <= 22.0
        near_outer_horizontal = y >= 18.0 and x <= 22.0

        return (
            near_center_horizontal
            or near_center_vertical
            or near_outer_vertical
            or near_outer_horizontal
        )

    def _node_heading(self, node):
        """Estimate dominant local lane heading from driving edges."""
        vecs = []
        for nb in self.edges.get(node, []):
            kind = self.edge_kind.get(node, {}).get(nb, "lane")
            if not self._is_driving_edge(kind):
                continue
            v = np.array([nb[0] - node[0], nb[1] - node[1]], dtype=float)
            n = np.linalg.norm(v)
            if n > 1e-9:
                vecs.append(v / n)

        if not vecs:
            return None

        # Build an orientation estimate that ignores sign (lane directionality).
        M = np.zeros((2, 2), dtype=float)
        for v in vecs:
            M += np.outer(v, v)
        vals, vecs_eig = np.linalg.eigh(M)
        heading = vecs_eig[:, np.argmax(vals)]
        hn = np.linalg.norm(heading)
        return heading / hn if hn > 1e-9 else None

    def _add_lane_change_edges(self):
        """Add synthetic lane-change links between nearby parallel lanes.

        This allows overtaking/lane switching anywhere along adjacent lanes,
        instead of forcing transitions only at hand-authored connector segments.
        """
        lane_nodes = []
        for node in self.nodes:
            has_driving_neighbor = False
            for nb in self.edges.get(node, []):
                kind = self.edge_kind.get(node, {}).get(nb, "lane")
                if self._is_driving_edge(kind):
                    has_driving_neighbor = True
                    break
            if has_driving_neighbor:
                lane_nodes.append(node)

        min_dist = 0.8
        max_dist = 3.25
        parallel_threshold = 0.85
        lateral_threshold = 0.40
        lane_change_weight = self.kind_cost["lane_change"]

        headings = {n: self._node_heading(n) for n in lane_nodes}

        for i in range(len(lane_nodes)):
            a = lane_nodes[i]
            ha = headings.get(a)
            if ha is None:
                continue

            pa = np.array(a, dtype=float)
            for j in range(i + 1, len(lane_nodes)):
                b = lane_nodes[j]
                if b in self.edge_cost[a]:
                    continue

                pb = np.array(b, dtype=float)
                dvec = pb - pa
                dist = float(np.linalg.norm(dvec))
                if dist < min_dist or dist > max_dist:
                    continue

                hb = headings.get(b)
                if hb is None:
                    continue

                # Lanes should be approximately parallel.
                if abs(float(np.dot(ha, hb))) < parallel_threshold:
                    continue

                # Lane-change should be mostly lateral to each lane heading.
                ddir = dvec / dist
                if abs(float(np.dot(ddir, ha))) > lateral_threshold:
                    continue
                if abs(float(np.dot(ddir, hb))) > lateral_threshold:
                    continue

                midpoint = 0.5 * (pa + pb)
                if not self._lane_change_region_allowed(midpoint):
                    continue

                self._add_edge(a, b, dist * lane_change_weight, "lane_change")

    def build_graph(self):
        self.nodes = []
        self.edges = {}
        self.edge_cost = {}
        self.edge_kind = {}
        self._build_intersection_aware_graph()
        self._add_lane_change_edges()

    def heuristic(self, n1, n2):
        return np.linalg.norm(np.array(n1) - np.array(n2))

    def get_closest_node(self, point):
        if not self.nodes:
            raise ValueError("Graph is empty. Call build_graph() first.")
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
                edge_c = self.edge_cost[current].get(neighbor, self.heuristic(current, neighbor))
                tentative_g = g_score[current] + edge_c

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