import numpy as np
from scipy.optimize import minimize


class PathTrajectory:
    """Arc-length parameterized trajectory for smooth path tracking.
    
    Provides methods to query reference state and vehicle frenet coordinates
    relative to the continuous path.
    """
    
    def __init__(self, path):
        self.path = np.array(path, dtype=float)
        if len(self.path) < 2:
            raise ValueError("Path must have at least 2 points")
        
        # Compute arc lengths
        diffs = np.diff(self.path, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self.arc_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self.total_length = self.arc_lengths[-1]
    
    def _interpolate_at_s(self, s):
        """Interpolate position and heading at arc-length s.
        
        Returns: (position, heading) or (None, None) if s is out of bounds.
        """
        s = np.clip(float(s), 0.0, self.total_length)
        
        # Find the segment containing this arc length
        idx = np.searchsorted(self.arc_lengths, s, side='right') - 1
        idx = np.clip(idx, 0, len(self.path) - 2)
        
        s0 = self.arc_lengths[idx]
        s1 = self.arc_lengths[idx + 1]
        p0 = self.path[idx]
        p1 = self.path[idx + 1]
        
        if abs(s1 - s0) < 1e-9:
            return p0, 0.0
        
        t = (s - s0) / (s1 - s0)
        pos = p0 + t * (p1 - p0)
        heading = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        
        return pos, heading
    
    def get_reference(self, s):
        """Get (position, heading) at arc-length s."""
        return self._interpolate_at_s(s)
    
    def project_vehicle(self, x, y, theta):
        """Project vehicle position onto path and return frenet coordinates.
        
        Returns: (s_proj, lateral_error, heading_error)
        where s_proj is the arc-length of the closest point on the path,
        lateral_error is signed perpendicular distance,
        and heading_error is signed angle difference.
        """
        point = np.array([x, y], dtype=float)
        
        # Find closest point on path
        best_s = 0.0
        best_dist = float('inf')
        best_lateral = 0.0
        
        for i in range(len(self.path) - 1):
            p0 = self.path[i]
            p1 = self.path[i + 1]
            seg = p1 - p0
            seg_len_sq = np.dot(seg, seg)
            
            if seg_len_sq < 1e-9:
                dist = np.linalg.norm(point - p0)
                if dist < best_dist:
                    best_dist = dist
                    best_s = self.arc_lengths[i]
                    best_lateral = dist
                continue
            
            t = np.dot(point - p0, seg) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj = p0 + t * seg
            dist = np.linalg.norm(point - proj)
            
            if dist < best_dist:
                best_dist = dist
                best_s = self.arc_lengths[i] + t * (self.arc_lengths[i + 1] - self.arc_lengths[i])
                
                # Signed lateral error using cross product
                to_point = point - p0
                cross = seg[0] * to_point[1] - seg[1] * to_point[0]
                sign = np.sign(cross) if abs(cross) > 1e-9 else 1.0
                best_lateral = sign * dist
        
        _, path_heading = self._interpolate_at_s(best_s)
        heading_err = self._wrap_angle(theta - path_heading)
        
        return best_s, best_lateral, heading_err
    
    @staticmethod
    def _wrap_angle(ang):
        return (ang + np.pi) % (2 * np.pi) - np.pi


class PathFollower:
    """Path-parameterized MPC for robust trajectory tracking.
    
    Tracks a closed patrol path using arc-length parameterization (Frenet frame).
    Explicitly rewards forward progress to prevent parking on segments.
    """
    
    def __init__(self):
        # Vehicle limits
        self.max_steer = 0.5
        self.max_steer_rate = 0.35
        self.max_accel = 1.8
        
        # Horizon and optimization
        self.horizon_steps = 6
        self.control_dt = 0.05
        self.max_iter = 18
        
        # Frenet-frame cost weights
        self.w_lateral = 4.0
        self.w_heading = 2.2
        self.w_speed = 0.9
        self.w_steer = 0.10
        self.w_input_rate = 0.9
        self.w_progress = 2.5
        
        # Target speed
        self.target_speed = 2.1
        self.turn_speed = 1.5
        
        self.prev_delta = 0.0
        self.prev_a = 0.0
        self._warm_start = None
        self._trajectory = None
        self._s_vehicle = 0.0
    
    def _wrap_angle(self, ang):
        return (ang + np.pi) % (2 * np.pi) - np.pi
    
    def _predict_step(self, state, a, delta, L, dt):
        """Kinematic bicycle step."""
        x, y, theta, v = state
        a = float(np.clip(a, -self.max_accel, self.max_accel))
        delta = float(np.clip(delta, -self.max_steer, self.max_steer))
        
        v = float(np.clip(v + a * dt, 0.0, 10.0))
        theta = float(theta + (v / L) * np.tan(delta) * dt)
        x = float(x + v * np.cos(theta) * dt)
        y = float(y + v * np.sin(theta) * dt)
        
        return np.array([x, y, theta, v], dtype=float)
    
    def _target_speed_from_curvature(self, s):
        """Reduce speed on high-curvature segments."""
        if self._trajectory is None or s < 0 or s > self._trajectory.total_length - 2:
            return self.target_speed
        
        pos_now, h_now = self._trajectory.get_reference(s)
        pos_next, h_next = self._trajectory.get_reference(s + 2.0)
        
        if pos_now is None or pos_next is None:
            return self.target_speed
        
        turn = abs(self._wrap_angle(h_next - h_now))
        return self.turn_speed if turn > 0.28 else self.target_speed
    
    def _rollout_cost(self, u_flat, state0, s_init, dt, prev_a, prev_delta, wheelbase):
        """Compute MPC cost over the horizon in Frenet frame."""
        horizon = self.horizon_steps
        a_seq = np.clip(u_flat[:horizon], -self.max_accel, self.max_accel)
        d_seq = np.clip(u_flat[horizon:], -self.max_steer, self.max_steer)
        
        state = state0.copy()
        s = s_init
        cost = 0.0
        last_a = prev_a
        last_d = prev_delta
        
        for k in range(horizon):
            a = float(a_seq[k])
            delta = float(d_seq[k])
            
            # Predict next state
            state = self._predict_step(state, a, delta, wheelbase, dt)
            
            # Project onto path to get frenet coordinates
            s_proj, lat_err, head_err = self._trajectory.project_vehicle(
                state[0], state[1], state[2]
            )
            
            # Update estimated arc-length (smooth progress)
            ds = s_proj - s
            if abs(ds) > self._trajectory.total_length / 2:
                ds = np.sign(ds) * (self._trajectory.total_length - abs(ds))
            s = s_proj
            
            # Get target speed and heading at this arc-length
            ref_speed = self._target_speed_from_curvature(s)
            speed_err = state[3] - ref_speed
            
            # Frenet-frame cost: penalize lateral error, heading error, and reward progress
            cost += (
                self.w_lateral * (lat_err ** 2)
                + self.w_heading * (head_err ** 2)
                + self.w_speed * (speed_err ** 2)
                + self.w_steer * (delta ** 2)
                + 0.08 * (a ** 2)
                + self.w_input_rate * (abs(a - last_a) + abs(delta - last_d))
                - self.w_progress * max(0.0, state[3])
            )
            
            last_a = a
            last_d = delta
        
        return float(cost)
    
    def control(self, vehicle, path, index, dt=0.05):
        """Compute MPC control for path following.
        
        Args:
            vehicle: Vehicle object with x, y, theta, v, L.
            path: List of (x, y) waypoints (used to initialize trajectory if needed).
            index: Unused (kept for interface compatibility).
            dt: Control timestep.
        
        Returns:
            (acceleration, steering_angle, updated_index)
        """
        if not path:
            return 0.0, 0.0, index
        
        # Initialize trajectory on first call
        if self._trajectory is None:
            self._trajectory = PathTrajectory(path)
            self._s_vehicle = 0.0
        
        state0 = np.array([vehicle.x, vehicle.y, vehicle.theta, vehicle.v], dtype=float)
        
        # Project vehicle onto path to update arc-length position
        s_proj, _, _ = self._trajectory.project_vehicle(vehicle.x, vehicle.y, vehicle.theta)
        
        # Smooth arc-length update (handle loop wrapping)
        ds = s_proj - self._s_vehicle
        if abs(ds) > self._trajectory.total_length / 2:
            ds = np.sign(ds) * (self._trajectory.total_length - abs(ds))
        self._s_vehicle = s_proj
        
        # Warm-start from previous solution
        horizon = self.horizon_steps
        if self._warm_start is None or len(self._warm_start) != 2 * horizon:
            a_init = np.full(horizon, self.prev_a, dtype=float)
            d_init = np.full(horizon, self.prev_delta, dtype=float)
            u0 = np.concatenate([a_init, d_init])
        else:
            u0 = self._warm_start.copy()
        
        # Heuristic speed guess if starting from rest
        if np.allclose(u0, 0.0):
            ref_speed = self._target_speed_from_curvature(self._s_vehicle)
            u0[:horizon] = np.clip(ref_speed - vehicle.v, -1.0, 1.0)
        
        # Bounds on controls
        bounds = [(-self.max_accel, self.max_accel)] * horizon + [
            (-self.max_steer, self.max_steer)
        ] * horizon
        
        # Steering rate constraint on the first step
        max_delta_change = self.max_steer_rate * dt
        bounds[horizon] = (
            max(self.prev_delta - max_delta_change, -self.max_steer),
            min(self.prev_delta + max_delta_change, self.max_steer),
        )
        
        # Solve the MPC problem
        try:
            result = minimize(
                lambda u: self._rollout_cost(
                    u, state0, self._s_vehicle, dt, self.prev_a, self.prev_delta, vehicle.L
                ),
                u0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iter, "ftol": 1e-3},
            )
            u_opt = result.x if result.success and result.x is not None else u0
        except Exception:
            u_opt = u0
        
        # Extract first control
        u_opt = np.asarray(u_opt, dtype=float)
        a_seq = np.clip(u_opt[:horizon], -self.max_accel, self.max_accel)
        d_seq = np.clip(u_opt[horizon:], -self.max_steer, self.max_steer)
        
        best_a = float(a_seq[0])
        best_delta = float(d_seq[0])
        
        # Warm-start for next step
        self._warm_start = np.concatenate([
            np.r_[a_seq[1:], a_seq[-1]],
            np.r_[d_seq[1:], d_seq[-1]],
        ])
        self.prev_a = best_a
        self.prev_delta = best_delta
        
        # Return index for compatibility (not used in arc-length tracking)
        return best_a, best_delta, 0
    
    def reset_pid(self):
        """Reset controller state for loop restart."""
        self.prev_delta = 0.0
        self.prev_a = 0.0
        self._warm_start = None
        self._s_vehicle = 0.0


class PIDController:
    """Simple PID controller for throttle (kept for compatibility)."""
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
    
    def update(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -2.0, 2.0)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -3.0, 3.0)
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
