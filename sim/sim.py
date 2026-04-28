import pygame
import numpy as np
import random

from vehicle import Vehicle
from world import World
from graph import RoadGraph
from controller import PathFollower
from roads import ROADS

# ---- CONFIG ----
WIDTH, HEIGHT = 800, 800
SCALE = 10
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
    car_length = 5  # Increased from 4
    car_width = 2.5  # Increased from 2

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
    road_width = 6
    half = int((road_width / 2) * SCALE)
    color = (80, 80, 80)

    for road in ROADS:
        if road[0] == "line":
            _, start, end = road

            x1, y1 = world_to_screen(start[0], start[1])
            x2, y2 = world_to_screen(end[0], end[1])

            if x1 == x2:
                # vertical
                pygame.draw.rect(
                    screen,
                    color,
                    (x1 - half, min(y1, y2), 2 * half, abs(y2 - y1))
                )
            else:
                # horizontal
                pygame.draw.rect(
                    screen,
                    color,
                    (min(x1, x2), y1 - half, abs(x2 - x1), 2 * half)
                )

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
    road_width = 3

    if abs(y - 0) < road_width or abs(y - 10) < road_width or abs(y + 10) < road_width:
        return True

    if abs(x - 0) < road_width or abs(x - 10) < road_width or abs(x + 10) < road_width:
        return True

    if abs(abs(x) - 20) < road_width or abs(abs(y) - 20) < road_width:
        return True

    return False


# ---------- GOAL ----------
def assign_new_goal(vehicle, graph):
    """
    Assign next goal with improved logic to avoid sharp U-turns.
    """
    start = graph.get_closest_node((vehicle.x, vehicle.y))

    forward = np.array([np.cos(vehicle.theta), np.sin(vehicle.theta)])

    valid_goals = []
    for g in graph.nodes:
        direction = np.array([g[0] - vehicle.x, g[1] - vehicle.y])
        distance = np.linalg.norm(direction)

        # Filters
        if distance < 5:
            continue  # too close
        
        if distance > 50:
            continue  # too far

        direction_normalized = direction / distance

        # Forward bias: prefer goals more than 60° forward
        # (increased from just any positive dot product)
        forward_bias = np.dot(forward, direction_normalized)
        
        if forward_bias > 0.5:  # ~60° forward
            valid_goals.append(g)

    # If no strictly forward goals, relax to 0° (any direction ahead)
    if not valid_goals:
        for g in graph.nodes:
            direction = np.array([g[0] - vehicle.x, g[1] - vehicle.y])
            distance = np.linalg.norm(direction)

            if distance < 5 or distance > 50:
                continue

            direction_normalized = direction / distance
            forward_bias = np.dot(forward, direction_normalized)
            
            if forward_bias > 0.0:
                valid_goals.append(g)

    # If still nothing, pick any node
    if not valid_goals:
        valid_goals = graph.nodes

    goal = random.choice(valid_goals)

    # Plan path using A*
    discrete_path = graph.astar(start, goal)
    
    # Convert to smooth waypoint trajectory
    path = graph.interpolate_waypoints(discrete_path, num_points=15)

    # Enforce minimum path length (now measured in waypoints)
    if len(path) < 10:
        return assign_new_goal(vehicle, graph)

    return path



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
        Vehicle(-15, 0, 0, 0),  # Single vehicle starting at left-center
    ]

    for v in vehicles:
        world.add_vehicle(v)

    paths = [assign_new_goal(v, graph) for v in vehicles]
    indices = [0 for _ in vehicles]

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
                if np.hypot(v.x - goal[0], v.y - goal[1]) < 2.0:
                    path = assign_new_goal(v, graph)
                    controller.reset_pid()  # Reset PID for new path
                    paths[i] = path
                    idx = 0

            a, d, idx = controller.control(v, path, idx, dt=DT)
            indices[i] = idx

            actions.append([a, d])

        world.step(actions, DT)

        collisions = world.check_collision()

        # ---------- DRAW ----------
        screen.fill((20, 20, 20))

        draw_roads(screen)
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

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()