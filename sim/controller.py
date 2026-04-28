import numpy as np


class PIDController:
    """Simple PID controller for throttle"""
    def __init__(self, kp=1.0, ki=0.1, kd=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        """
        error = target_speed - current_speed
        dt = time step
        """
        self.integral += error * dt
        self.integral = np.clip(self.integral, -2.0, 2.0)  # anti-windup

        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -3.0, 3.0)  # respect vehicle limits

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class PathFollower:
    def __init__(self):
        # Pure Pursuit parameters
        self.base_lookahead = 2.5  # base lookahead distance
        self.max_steer = 0.5  # max steering angle
        self.max_steer_rate = 0.3  # max steering rate (rad/s)
        self.prev_delta = 0.0
        
        # Speed control
        self.target_speed = 4.0
        self.max_speed = 8.0
        self.pid = PIDController(kp=1.5, ki=0.15, kd=0.5)

    def get_lookahead_distance(self, vehicle_speed):
        """Adaptive lookahead based on speed (Stanley method)"""
        return self.base_lookahead + 0.2 * vehicle_speed

    def control(self, vehicle, path, index, dt=0.05):
        """
        Returns: (acceleration, steering_angle, updated_index)
        """
        if not path:
            self.pid.reset()
            return 0.0, 0.0, index

        # ---- FIND LOOKAHEAD POINT ----
        lookahead_dist = self.get_lookahead_distance(vehicle.v)
        lookahead_point = None

        # Start search from current index
        for i in range(index, len(path)):
            dx = path[i][0] - vehicle.x
            dy = path[i][1] - vehicle.y
            dist = np.hypot(dx, dy)

            if dist > lookahead_dist:
                lookahead_point = path[i]
                index = i
                break

        # If no lookahead point found, use end of path
        if lookahead_point is None:
            lookahead_point = path[-1]
            index = len(path) - 1

        # ---- TRANSFORM TO VEHICLE FRAME ----
        dx = lookahead_point[0] - vehicle.x
        dy = lookahead_point[1] - vehicle.y

        # Rotate into vehicle frame
        cos_theta = np.cos(vehicle.theta)
        sin_theta = np.sin(vehicle.theta)
        
        local_x = cos_theta * dx + sin_theta * dy
        local_y = -sin_theta * dx + cos_theta * dy

        # ---- PURE PURSUIT STEERING ----
        if local_x <= 0.1:
            # Lookahead point is behind vehicle (or very close)
            delta = 0.0
        else:
            # Standard Pure Pursuit formula
            curvature = 2.0 * local_y / (local_x**2 + local_y**2)
            delta = np.arctan(curvature * vehicle.L)
            delta = np.clip(delta, -self.max_steer, self.max_steer)

        # ---- STEERING RATE LIMITING ----
        # Prevent unrealistic steering changes
        max_delta_change = self.max_steer_rate * dt
        delta = np.clip(delta, self.prev_delta - max_delta_change, 
                       self.prev_delta + max_delta_change)
        self.prev_delta = delta

        # ---- SPEED CONTROL (PID-based) ----
        goal = path[-1]
        dist_to_goal = np.hypot(vehicle.x - goal[0], vehicle.y - goal[1])

        # Aggressive speed reduction as approaching goal
        if dist_to_goal < 1.0:
            target_speed = 0.0  # STOP when very close
        elif dist_to_goal < 2.5:
            target_speed = 0.5  # Very slow
        elif dist_to_goal < 5.0:
            target_speed = 1.0  # Slow down
        elif dist_to_goal < 8.0:
            target_speed = 2.0  # Moderate
        else:
            target_speed = self.target_speed  # Full speed

        # PID controller
        speed_error = target_speed - vehicle.v
        a = self.pid.update(speed_error, dt)

        return a, delta, index

    def reset_pid(self):
        """Call this when assigning a new path"""
        self.pid.reset()
        self.prev_delta = 0.0
