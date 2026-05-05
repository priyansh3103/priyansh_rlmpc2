#!/usr/bin/env python3
"""Headless tracking evaluation and simple tuner for PathFollower.

Usage:
  python sim/evaluate_tracking.py --mission zone_outer_mix --steps 2000
  python sim/evaluate_tracking.py --tune --grid-pos 1.0,2.0,3.0 --grid-rate 0.2,0.6,1.0

The script runs a closed-loop patrol mission without rendering and computes
metrics: RMS lateral error, max lateral error, mean heading error (deg),
control effort, off-road fraction. Optionally performs a small grid search
over controller weights and reports the best setting.
"""
import argparse
import math
import numpy as np
from vehicle import Vehicle
from graph import RoadGraph
from controller import PathFollower
from sim import create_patrol_mission, build_patrol_loop_path, is_on_road


def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _closest_seg_projection(p, path):
    # Returns (seg_idx, proj_point, signed_cross_track)
    best_d = float("inf")
    best_proj = None
    best_idx = 0
    px, py = p
    for i in range(len(path) - 1):
        a = np.array(path[i], dtype=float)
        b = np.array(path[i + 1], dtype=float)
        ab = b - a
        ab2 = ab.dot(ab)
        if ab2 < 1e-12:
            continue
        t = np.dot(np.array([px, py]) - a, ab) / ab2
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        d2 = np.sum((proj - np.array([px, py])) ** 2)
        if d2 < best_d:
            best_d = d2
            best_proj = proj
            best_idx = i

    # signed cross-track: sign from cross(ab, vehicle - a)
    a = np.array(path[best_idx], dtype=float)
    b = np.array(path[best_idx + 1], dtype=float)
    ab = b - a
    to_p = np.array([px, py]) - a
    cross = ab[0] * to_p[1] - ab[1] * to_p[0]
    sign = np.sign(cross) if abs(cross) > 1e-9 else 1.0
    lateral = sign * math.sqrt(best_d)
    return best_idx, best_proj, lateral


def evaluate(controller, v, path, steps=1500, dt=0.05, verbose=False):
    idx = 0
    lateral_errs = []
    heading_errs = []
    speed_errs = []
    acc_cmds = []
    steer_cmds = []
    offroad_count = 0
    indices = []

    for step in range(steps):
        a, d, idx = controller.control(v, path, idx, dt=dt)
        acc_cmds.append(a)
        steer_cmds.append(d)

        v.step([a, d], dt)

        # nearest segment projection for lateral & heading
        si, proj, lat = _closest_seg_projection((v.x, v.y), path)
        lateral_errs.append(lat)

        # path heading at segment
        p0 = np.array(path[si], dtype=float)
        p1 = np.array(path[si + 1], dtype=float)
        path_h = math.atan2((p1 - p0)[1], (p1 - p0)[0])
        head_err = _wrap_angle(v.theta - path_h)
        heading_errs.append(head_err)

        # speed error wrt controller target
        ref_speed = controller._target_speed_from_curvature(path, si)
        speed_errs.append(v.v - ref_speed)

        if not is_on_road(v.x, v.y):
            offroad_count += 1

        indices.append(idx)

        # loop reset to emulate closed patrol
        goal = path[-1]
        if idx >= len(path) - 15 and np.hypot(v.x - goal[0], v.y - goal[1]) < 3.0:
            idx = 0
            controller.reset_pid()

    lateral_arr = np.array(lateral_errs)
    heading_arr = np.array(heading_errs)
    acc_arr = np.array(acc_cmds)
    steer_arr = np.array(steer_cmds)

    metrics = {
        'rms_lateral': float(np.sqrt(np.mean(lateral_arr ** 2))),
        'max_lateral': float(np.max(np.abs(lateral_arr))),
        'mean_abs_heading_deg': float(np.degrees(np.mean(np.abs(heading_arr)))),
        'rms_heading_deg': float(np.degrees(np.sqrt(np.mean(heading_arr ** 2)))),
        'mean_abs_accel': float(np.mean(np.abs(acc_arr))),
        'mean_abs_steer': float(np.mean(np.abs(steer_arr))),
        'steer_slew': float(np.mean(np.abs(np.diff(steer_arr)))) if len(steer_arr) > 1 else 0.0,
        'offroad_frac': float(offroad_count / steps),
        'final_index': int(indices[-1]) if indices else 0,
    }

    if verbose:
        print('Metrics:')
        for k, v in metrics.items():
            print(f'  {k}: {v}')

    return metrics


def baseline_run(mission='zone_outer_mix', steps=1500, dt=0.05):
    g = RoadGraph(spacing=2)
    g.build_graph()

    controller = PathFollower()
    v = Vehicle(-23.5, 0.0, np.pi / 2, 0.0)

    patrol = create_patrol_mission(mission)
    path = build_patrol_loop_path(g, patrol['checkpoints'])

    metrics = evaluate(controller, v, path, steps=steps, dt=dt, verbose=True)
    return metrics


def grid_tune(mission='zone_outer_mix', steps=1200, dt=0.05, grid_pos=(1.0, 2.0, 3.0), grid_rate=(0.2, 0.8)):
    g = RoadGraph(spacing=2)
    g.build_graph()

    patrol = create_patrol_mission(mission)
    path = build_patrol_loop_path(g, patrol['checkpoints'])

    best = None
    results = []
    for wpos in grid_pos:
        for wrate in grid_rate:
            controller = PathFollower()
            controller.w_pos = float(wpos)
            controller.w_input_rate = float(wrate)

            v = Vehicle(-23.5, 0.0, np.pi / 2, 0.0)
            metrics = evaluate(controller, v, path, steps=steps, dt=dt, verbose=False)

            # objective: prefer low RMS lateral, penalize offroad and steering slew
            obj = metrics['rms_lateral'] + 1.5 * metrics['offroad_frac'] + 0.4 * metrics['steer_slew']
            results.append({'w_pos': wpos, 'w_input_rate': wrate, 'metrics': metrics, 'obj': obj})

            if best is None or obj < best['obj']:
                best = results[-1]

    # print summary
    print('Grid search results (top 5):')
    results_sorted = sorted(results, key=lambda r: r['obj'])
    for r in results_sorted[:5]:
        print(f"w_pos={r['w_pos']} w_input_rate={r['w_input_rate']} obj={r['obj']:.4f} rms_lat={r['metrics']['rms_lateral']:.3f} offroad={r['metrics']['offroad_frac']:.3f} steer_slew={r['metrics']['steer_slew']:.3f}")

    print('\nBest:')
    print(f"w_pos={best['w_pos']} w_input_rate={best['w_input_rate']} obj={best['obj']:.4f}")
    print('Best metrics:')
    for k, v in best['metrics'].items():
        print(f'  {k}: {v}')

    return best, results


def _parse_floats(s):
    return tuple(float(x) for x in s.split(','))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mission', default='zone_outer_mix')
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--tune', action='store_true')
    p.add_argument('--grid-pos', default='1.0,2.0,3.0')
    p.add_argument('--grid-rate', default='0.2,0.6,1.0')
    args = p.parse_args()

    if args.tune:
        grid_pos = _parse_floats(args.grid_pos)
        grid_rate = _parse_floats(args.grid_rate)
        grid_tune(mission=args.mission, steps=args.steps, dt=args.dt, grid_pos=grid_pos, grid_rate=grid_rate)
    else:
        baseline_run(mission=args.mission, steps=args.steps, dt=args.dt)


if __name__ == '__main__':
    main()
