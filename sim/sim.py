import pygame
import numpy as np

from vehicle import Vehicle
from world import World
from graph import RoadGraph
from controller import PathFollower
from roads import ROADS

# ---- CONFIG ----
WIDTH, HEIGHT = 800, 800
SCALE = 15
DT = 0.05


def world_to_screen(x, y):
    return WIDTH // 2 + int(x * SCALE), HEIGHT // 2 - int(y * SCALE)


def draw_arrow(screen, start, vec, color):
    end = (start[0] + int(vec[0]), start[1] + int(vec[1]))
    pygame.draw.line(screen, color, start, end, 3)


# ---------- COLOR SYSTEM ----------
def get_color(i):
    COLORS = [
        (0, 200, 255),   # cyan
        (255, 100, 100), # red-ish
        (100, 255, 100), # green
        (255, 255, 100), # yellow
        (200, 100, 255), # purple
    ]
    return COLORS[i % len(COLORS)]


# ---------- VEHICLE DRAW ----------
def draw_vehicle(screen, vehicle, color, vehicle_id=0):
    """Enhanced vehicle visualization with state indicators"""
    car_length = 2.6
    car_width = 1.1

    corners = np.array([
        [0, -car_width/2],
        [0, car_width/2],
        [car_length, car_width/2],
        [car_length, -car_width/2]
    ])

    R = np.array([
        [np.cos(vehicle.theta), -np.sin(vehicle.theta)],
        [np.sin(vehicle.theta),  np.cos(vehicle.theta)]
    ])

    pts = []
    for c in corners:
        p = np.array([vehicle.x, vehicle.y]) + R @ c
        pts.append(world_to_screen(p[0], p[1]))

    # Main body with gradient effect
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, (255, 255, 255), pts, 3)  # white outline

    # Rear axle (reference frame)
    rear = np.array([vehicle.x, vehicle.y])
    rear_screen = world_to_screen(rear[0], rear[1])
    pygame.draw.circle(screen, (100, 100, 100), rear_screen, 4)
    pygame.draw.circle(screen, (200, 200, 200), rear_screen, 2)

    # Front point (wheel reference)
    front = rear + vehicle.L * np.array([np.cos(vehicle.theta), np.sin(vehicle.theta)])
    front_screen = world_to_screen(front[0], front[1])
    pygame.draw.circle(screen, (255, 200, 0), front_screen, 3)

    # Velocity vector (green)
    vel_magnitude = vehicle.v * 20  # scale for visibility
    vel_vec = np.array([np.cos(vehicle.theta), np.sin(vehicle.theta)]) * vel_magnitude
    vel_end_screen = (int(rear_screen[0] + vel_vec[0]), int(rear_screen[1] - vel_vec[1]))
    pygame.draw.line(screen, (0, 255, 100), rear_screen, vel_end_screen, 3)
    pygame.draw.circle(screen, (0, 255, 100), vel_end_screen, 2)

    # Heading vector (blue - shows orientation)
    heading_scale = 25
    heading_vec = np.array([np.cos(vehicle.theta), -np.sin(vehicle.theta)]) * heading_scale
    heading_end = (int(rear_screen[0] + heading_vec[0]), int(rear_screen[1] + heading_vec[1]))
    pygame.draw.line(screen, (0, 150, 255), rear_screen, heading_end, 2)

    # Steering angle indicator (red arc)
    # Shows if vehicle is turning left/right
    steering_indicator_scale = 15
    steering_direction = np.array([np.cos(vehicle.theta + 0.3), -np.sin(vehicle.theta + 0.3)]) * steering_indicator_scale
    steering_end = (int(rear_screen[0] + steering_direction[0]), int(rear_screen[1] + steering_direction[1]))
    pygame.draw.line(screen, (255, 100, 0), rear_screen, steering_end, 2)

    # Vehicle ID label
    font = pygame.font.Font(None, 24)
    id_text = font.render(f"V{vehicle_id}", True, color)
    screen.blit(id_text, (rear_screen[0] + 10, rear_screen[1] - 15))




# ---------- MAP ----------
def draw_roads(screen):
    styles = {
        "lane": {"color": (92, 92, 92), "width": max(2, int(1.6 * SCALE))},
        "outer_loop": {"color": (96, 96, 96), "width": max(2, int(1.7 * SCALE))},
    }

    draw_order = ["lane", "outer_loop"]

    for kind in draw_order:
        style = styles[kind]
        for road in ROADS:
            if not road or road[0] != "line":
                continue

            if len(road) == 4:
                _, start, end, rkind = road
            else:
                _, start, end = road
                rkind = "lane"

            if rkind != kind:
                continue

            x1, y1 = world_to_screen(start[0], start[1])
            x2, y2 = world_to_screen(end[0], end[1])
            pygame.draw.line(screen, style["color"], (x1, y1), (x2, y2), style["width"])

def draw_lane_markings(screen):
    """Draw dashed separators between the two central lanes."""
    marking_color = (115, 115, 115)
    dash_length = 6
    gap_length = 6
    line_width = 1

    # Dashed separator between horizontal central lanes (y=0).
    x_left = world_to_screen(-20, 0)[0]
    x_right = world_to_screen(20, 0)[0]
    y_mid = world_to_screen(0, 0)[1]

    x = x_left
    while x < x_right:
        pygame.draw.line(
            screen,
            marking_color,
            (int(x), y_mid),
            (int(min(x + dash_length, x_right)), y_mid),
            line_width,
        )
        x += dash_length + gap_length

    # Dashed separator between vertical central lanes (x=0).
    y_top = world_to_screen(0, 20)[1]
    y_bottom = world_to_screen(0, -20)[1]
    x_mid = world_to_screen(0, 0)[0]

    y = y_bottom
    while y > y_top:
        pygame.draw.line(
            screen,
            marking_color,
            (x_mid, int(y)),
            (x_mid, int(max(y - dash_length, y_top))),
            line_width,
        )
        y -= dash_length + gap_length


def draw_outer_loop(screen):
    pygame.draw.rect(screen, (80, 80, 80), (100, 100, WIDTH - 200, HEIGHT - 200), 40)


def draw_hud(screen, vehicles):
    """Draw heads-up display with vehicle information"""
    font_small = pygame.font.Font(None, 18)
    y_offset = 10
    x_offset = 10
    
    # Title
    font_large = pygame.font.Font(None, 24)
    title = font_large.render("VEHICLE STATE", True, (255, 255, 255))
    screen.blit(title, (x_offset, y_offset))
    y_offset += 30
    
    for i, v in enumerate(vehicles):
        # Vehicle ID and position
        info_text = f"V{i}: pos=({v.x:.1f},{v.y:.1f}) θ={np.degrees(v.theta):.1f}° v={v.v:.2f}m/s"
        text_surface = font_small.render(info_text, True, get_color(i))
        screen.blit(text_surface, (x_offset, y_offset))
        y_offset += 20


def draw_patrol_status(screen, patrol_state):
    """Display deterministic patrol progress."""
    if not patrol_state:
        return

    font_small = pygame.font.Font(None, 20)
    txt = (
        f"PATROL: {patrol_state['name']} | "
        f"progress_idx={patrol_state['path_index']} | "
        f"loops={patrol_state['loops_completed']}"
    )
    surf = font_small.render(txt, True, (220, 220, 220))
    screen.blit(surf, (10, 80))



# ---------- PATH ----------
def draw_path(screen, path, color):
    for p in path:
        x, y = world_to_screen(p[0], p[1])
        pygame.draw.circle(screen, color, (x, y), 3)


# ---------- COLLISION DEBUG ----------
def draw_collision_debug(screen, world):
    for i, v in enumerate(world.vehicles):
        color = get_color(i)

        rear = np.array([v.x, v.y])
        front = rear + v.L * np.array([np.cos(v.theta), np.sin(v.theta)])

        for p in [rear, front]:
            cx, cy = world_to_screen(p[0], p[1])
            pygame.draw.circle(screen, color, (cx, cy), int(1.5 * SCALE), 1)


# ---------- ROAD CHECK ----------
def is_on_road(x, y):
    road_half_width = 1.8
    p = np.array([x, y], dtype=float)

    for road in ROADS:
        if road[0] != "line":
            continue

        if len(road) >= 4:
            _, start, end, _ = road
        else:
            _, start, end = road
        a = np.array(start, dtype=float)
        b = np.array(end, dtype=float)
        ab = b - a
        ab_norm_sq = float(np.dot(ab, ab))

        if ab_norm_sq < 1e-12:
            continue

        t = float(np.dot(p - a, ab) / ab_norm_sq)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab

        if np.linalg.norm(p - proj) <= road_half_width:
            return True

    return False


def plan_path(graph, start_xy, goal_xy, interp=18):
    start = graph.get_closest_node(start_xy)
    goal = graph.get_closest_node(goal_xy)
    discrete = graph.astar(start, goal)
    return graph.interpolate_waypoints(discrete, num_points=interp)


def create_patrol_mission(name="outer_loop"):
    """Deterministic looping missions using fixed checkpoints."""
    if name == "upper_zone":
        checkpoints = [(-17.0, 19.0), (-7.0, 19.0), (-5.0, 12.0), (-7.0, 5.0), (-17.0, 5.0), (-19.0, 12.0)]
    elif name == "center_cross":
        checkpoints = [(-12.0, 1.5), (0.0, 1.5), (12.0, 1.5), (12.0, -1.5), (0.0, -1.5), (-12.0, -1.5)]
    elif name == "zone_outer_mix":
        # Alternates between zone lanes and nearby outer-loop lanes.
        checkpoints = [
            (-17.0, 19.0), (-18.5, 23.5),
            (17.0, 19.0), (23.5, 18.5),
            (17.0, -19.0), (18.5, -23.5),
            (-17.0, -19.0), (-23.5, -18.5),
        ]
    else:
        # Default: perimeter patrol on outer loop.
        checkpoints = [
            (-18.5, 23.5), (18.5, 23.5), (23.5, 18.5), (23.5, -18.5),
            (18.5, -23.5), (-18.5, -23.5), (-23.5, -18.5), (-23.5, 18.5),
        ]

    return {
        "name": name,
        "checkpoints": checkpoints,
        "path_index": 0,
        "loops_completed": 0,
    }


def build_patrol_loop_path(graph, checkpoints):
    """Build one closed-loop path by concatenating A* segments."""
    if len(checkpoints) < 2:
        return []

    full = []
    n = len(checkpoints)
    for i in range(n):
        s = checkpoints[i]
        e = checkpoints[(i + 1) % n]
        seg = plan_path(graph, s, e, interp=10)
        if not seg:
            continue
        if full and seg[0] == full[-1]:
            full.extend(seg[1:])
        else:
            full.extend(seg)
    return full



# ---------- MAIN ----------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    world = World()
    graph = RoadGraph(spacing=2)
    graph.build_graph()

    controller = PathFollower()

    vehicles = [
        Vehicle(-23.5, 0.0, np.pi / 2.0, 0.0),
    ]

    for v in vehicles:
        world.add_vehicle(v)

    patrol_state = create_patrol_mission(name="zone_outer_mix")
    loop_path = build_patrol_loop_path(graph, patrol_state["checkpoints"])
    paths = [loop_path]
    indices = [0]

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        actions = []

        for i, v in enumerate(world.vehicles):
            path = paths[i]
            idx = indices[i]

            # goal check
            if path:
                goal = path[-1]
                # Wrap loop deterministically when nearing end of closed path.
                patrol_state["path_index"] = idx
                if idx >= len(path) - 15 and np.hypot(v.x - goal[0], v.y - goal[1]) < 3.0:
                    patrol_state["loops_completed"] += 1
                    controller.reset_pid()
                    idx = 0

            a, d, idx = controller.control(v, path, idx, dt=DT)
            indices[i] = idx

            actions.append([a, d])

        world.step(actions, DT)

        collisions = world.check_collision()

        # ---------- DRAW ----------
        screen.fill((20, 20, 20))

        draw_roads(screen)
        draw_lane_markings(screen)
        #draw_outer_loop(screen)

        for i, v in enumerate(world.vehicles):
            color = get_color(i)

            draw_vehicle(screen, v, color, vehicle_id=i)
            draw_path(screen, paths[i], color)

            # off-road debug
            if not is_on_road(v.x, v.y):
                cx, cy = world_to_screen(v.x, v.y)
                pygame.draw.circle(screen, (255, 0, 255), (cx, cy), 5)

        draw_collision_debug(screen, world)

        if collisions:
            pygame.draw.circle(screen, (255, 0, 0), (50, 50), 10)

        # Draw HUD
        draw_hud(screen, world.vehicles)
        draw_patrol_status(screen, patrol_state)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()