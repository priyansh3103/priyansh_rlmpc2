import numpy as np
from scipy.optimize import minimize


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


class ArcLengthPath:
    """Arc-length parameterized representation of a closed patrol path."""
    
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints, dtype=float)
        self.n = len(self.waypoints)
        
        # Compute cumulative arc length
        self.arc_lengths = np.zeros(self.n)
        for i in range(1, self.n):
            dx = self.waypoints[i, 0] - self.waypoints[i-1, 0]
            dy = self.waypoints[i, 1] - self.waypoints[i-1, 1]
            self.arc_lengths[i] = self.arc_lengths[i-1] + np.sqrt(dx*dx + dy*dy)
        
        self.total_length = self.arc_lengths[-1]
        if self.total_length < 1e-6:
            self.total_length = 1.0
        
        # Precompute segment midpoints for faster lookup
        self.segment_midpoints = (self.arc_lengths[:-1] + self.arc_lengths[1:]) / 2.0
    
    def interpolate(self, s):
        """Return (x, y, heading) at arc-length s (wraps around on closed loop)."""
        s = s % self.total_length
        
        # Find the two waypoints bracketing this arc length
        idx = np.searchsorted(self.arc_lengths, s, side='right') - 1
        idx = np.clip(idx, 0, self.n - 2)
        
        s_a = self.arc_lengths[idx]
        s_b = self.arc_lengths[idx + 1]
        p_a = self.waypoints[idx]
        p_b = self.waypoints[idx + 1]
        
        # Linear interpolation
        if abs(s_b - s_a) < 1e-9:
            return p_a[0], p_a[1], np.arctan2(p_b[1] - p_a[1], p_b[0] - p_a[0])
        
        t = (s - s_a) / (s_b - s_a)
        x = p_a[0] + t * (p_b[0] - p_a[0])
        y = p_a[1] + t * (p_b[1] - p_a[1])
        heading = np.arctan2(p_b[1] - p_a[1], p_b[0] - p_a[0])
        return x, y, heading
    
    def find_closest_arc_length_local(self, x, y, s_hint):
        """Fast local search with limited window."""
        search_window = max(8.0, self.total_length * 0.12)
        best_s = s_hint
        best_d = float('inf')
        
        # Search only nearby segments
        for i in range(self.n - 1):
            s_mid = self.segment_midpoints[i]
            dist_to_hint = abs(s_mid - s_hint)
            wrapped_dist = min(dist_to_hint, self.total_length - dist_to_hint)
            if wrapped_dist > search_window:
                continue
            
            p_a = self.waypoints[i]
            p_b = self.waypoints[i+1]
            ab = p_b - p_a
            ap = np.array([x, y], dtype=float) - p_a
            denom = np.dot(ab, ab)
            if denom < 1e-12:
                t = 0.0
            else:
                t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
            closest = p_a + t * ab
            d = np.linalg.norm(np.array([x, y], dtype=float) - closest)
            
            if d < best_d:
                best_d = d
                best_s = self.arc_lengths[i] + t * (self.arc_lengths[i+1] - self.arc_lengths[i])
        
        return best_s
    
    def find_closest_arc_length(self, x, y, s_hint=None):
        """Fast arc-length search with local-first strategy."""
        if s_hint is not None and abs(self.waypoints[0][0] - x) < 20 and abs(self.waypoints[0][1] - y) < 20:
            # Try local search first (fast path)
            return self.find_closest_arc_length_local(x, y, s_hint)
        
        # Fallback: grid-based global search with early exit
        best_s = 0.0
        best_d = float('inf')
        step = max(1, self.n // 50)  # Sample every Nth point initially
        
        for i in range(0, self.n - 1, step):
            p_a = self.waypoints[i]
            p_b = self.waypoints[i+1]
            ab = p_b - p_a
            ap = np.array([x, y], dtype=float) - p_a
            denom = np.dot(ab, ab)
            if denom < 1e-12:
                t = 0.0
            else:
                t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
            closest = p_a + t * ab
            d = np.linalg.norm(np.array([x, y], dtype=float) - closest)
            
            if d < best_d:
                best_d = d
                best_s = self.arc_lengths[i] + t * (self.arc_lengths[i+1] - self.arc_lengths[i])
        
        # Refine in window around best
        window_rad = max(2.0, self.total_length * 0.05)
        for i in range(self.n - 1):
            s_mid = self.segment_midpoints[i]
            dist_to_best = abs(s_mid - best_s)
            wrapped_dist = min(dist_to_best, self.total_length - dist_to_best)
            if wrapped_dist > window_rad:
                continue
            
            p_a = self.waypoints[i]
            p_b = self.waypoints[i+1]
            ab = p_b - p_a
            ap = np.array([x, y], dtype=float) - p_a
            denom = np.dot(ab, ab)
            if denom < 1e-12:
                t = 0.0
            else:
                t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
            closest = p_a + t * ab
            d = np.linalg.norm(np.array([x, y], dtype=float) - closest)
            
            if d < best_d:
                best_d = d
                best_s = self.arc_lengths[i] + t * (self.arc_lengths[i+1] - self.arc_lengths[i])
        
        return best_s


class PathFollower:
    """Fast arc-length MPC with reduced horizon and optimization steps."""
    
    def __init__(self):
        self.max_steer = 0.5
        self.max_steer_rate = 0.35
        self.max_accel = 1.8
        
        # Aggressive reduction for speed
        self.horizon_steps = 4
        self.max_iter = 10
        self.control_dt = 0.05
        
        self.w_lateral = 1.6
        self.w_heading = 1.4
        self.w_speed = 0.6
        self.w_steer = 0.10
        self.w_input_rate = 0.7
        self.w_forward = 1.8
        
        self.target_speed = 2.2
        self.turn_speed = 1.6
        
        self.path = None
        self.prev_delta = 0.0
        self.prev_a = 0.0
        self._warm_start = None
        self.last_s = 0.0
    
    def set_path(self, waypoints):
        """Initialize arc-length path."""
        self.path = ArcLengthPath(waypoints)
    
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
    
    def _frenet_error(self, x, y, theta, s):
        """Lateral and heading error."""
        ref_x, ref_y, ref_heading = self.path.interpolate(s)
        dx = x - ref_x
        dy = y - ref_y
        lateral = -dx * np.sin(ref_heading) + dy * np.cos(ref_heading)
        heading_err = self._wrap_angle(theta - ref_heading)
        return lateral, heading_err
    
    def _target_speed(self, s):
        """Adaptive target speed."""
        delta_s = 0.7
        _, _, h1 = self.path.interpolate(s - delta_s)
        _, _, h2 = self.path.interpolate(s + delta_s)
        heading_change = abs(self._wrap_angle(h2 - h1))
        if heading_change > 0.22:
            return self.turn_speed
        return self.target_speed
    
    def _rollout_cost(self, u_flat, state0, s0, dt, prev_a, prev_delta, wheelbase):
        """Fast horizon rollout cost."""
        horizon = self.horizon_steps
        a_seq = np.clip(u_flat[:horizon], -self.max_accel, self.max_accel)
        d_seq = np.clip(u_flat[horizon:], -self.max_steer, self.max_steer)
        
        state = state0.copy()
        s = s0
        cost = 0.0
        last_a = prev_a
        last_d = prev_delta
        
        for k in range(horizon):
            a = float(a_seq[k])
            delta = float(d_seq[k])
            state = self._predict_step(state, a, delta, wheelbase, dt)
            
            # Fast arc-length search with hint
            s_new = self.path.find_closest_arc_length(state[0], state[1], s_hint=s)
            
            # Forward progress (simplified)
            forward_progress = s_new - s
            if abs(forward_progress) > self.path.total_length * 0.4:
                forward_progress = np.sign(forward_progress) * (self.path.total_length - abs(forward_progress))
            
            # Frenet errors
            lateral, heading_err = self._frenet_error(state[0], state[1], state[2], s_new)
            ref_speed = self._target_speed(s_new)
            speed_err = 0.5 * (state[3] - ref_speed) ** 2
            
            cost += (
                self.w_lateral * (lateral ** 2)
                + self.w_heading * (heading_err ** 2)
                + self.w_speed * speed_err
                + self.w_steer * abs(delta)
                + 0.06 * abs(a)
                + self.w_input_rate * (abs(a - last_a) + abs(delta - last_d))
                - self.w_forward * forward_progress
            )
            
            s = s_new
            last_a = a
            last_d = delta
        
        return float(cost)
    
    def control(self, vehicle, path, index, dt=0.05):
        """One control step."""
        if self.path is None:
            self.set_path(path)
        
        state0 = np.array([vehicle.x, vehicle.y, vehicle.theta, vehicle.v], dtype=float)
        
        # Find arc-length with hint from last iteration
        s0 = self.path.find_closest_arc_length(vehicle.x, vehicle.y, s_hint=self.last_s)
        
        horizon = self.horizon_steps
        if self._warm_start is None or len(self._warm_start) != 2 * horizon:
            u0 = np.concatenate([
                np.full(horizon, self.prev_a, dtype=float),
                np.full(horizon, self.prev_delta, dtype=float),
            ])
        else:
            u0 = self._warm_start.copy()
        
        bounds = [(-self.max_accel, self.max_accel)] * horizon + [(-self.max_steer, self.max_steer)] * horizon
        
        max_delta_change = self.max_steer_rate * dt
        bounds[horizon] = (
            max(self.prev_delta - max_delta_change, -self.max_steer),
            min(self.prev_delta + max_delta_change, self.max_steer),
        )
        
        try:
            result = minimize(
                lambda u: self._rollout_cost(u, state0, s0, dt, self.prev_a, self.prev_delta, vehicle.L),
                u0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iter, "ftol": 5e-3},
            )
            u_opt = result.x if result.success and result.x is not None else u0
        except Exception:
            u_opt = u0
        
        u_opt = np.asarray(u_opt, dtype=float)
        a_seq = np.clip(u_opt[:horizon], -self.max_accel, self.max_accel)
        d_seq = np.clip(u_opt[horizon:], -self.max_steer, self.max_steer)
        
        best_a = float(a_seq[0])
        best_delta = float(d_seq[0])
        
        self._warm_start = np.concatenate([
            np.r_[a_seq[1:], a_seq[-1]],
            np.r_[d_seq[1:], d_seq[-1]],
        ])
        
        self.prev_a = best_a
        self.prev_delta = best_delta
        self.last_s = s0
        
        return best_a, best_delta, 0
    
    def reset_pid(self):
        """Reset for loop restart."""
        self.prev_delta = 0.0
        self.prev_a = 0.0
        self._warm_start = None
        self.last_s = 0.0
    
    def _target_speed_from_curvature(self, waypoints, segment_index):
        """Compute target speed based on path curvature (for evaluator compatibility)."""
        if segment_index < 0 or segment_index >= len(waypoints) - 1:
            return self.target_speed
        
        p0 = np.array(waypoints[segment_index], dtype=float)
        p1 = np.array(waypoints[segment_index + 1], dtype=float)
        p2 = np.array(waypoints[min(segment_index + 2, len(waypoints) - 1)], dtype=float)
        
        h1 = np.arctan2((p1 - p0)[1], (p1 - p0)[0])
        h2 = np.arctan2((p2 - p1)[1], (p2 - p1)[0])
        
        def wrap_angle(ang):
            return (ang + np.pi) % (2 * np.pi) - np.pi
        
        heading_change = abs(wrap_angle(h2 - h1))
        if heading_change > 0.25:
            return self.turn_speed
        return self.target_speed
