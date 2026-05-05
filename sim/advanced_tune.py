#!/usr/bin/env python3
"""Advanced tuner: sweep target_speed, max_steer_rate, and steering candidate sets.

Produces a ranked table and prints the best parameter set.
"""
import itertools
import numpy as np
from vehicle import Vehicle
from graph import RoadGraph
from controller import PathFollower
from sim import create_patrol_mission, build_patrol_loop_path, is_on_road


def evaluate_once(controller, v, path, steps=1200, dt=0.05):
    # use evaluate implementation from evaluate_tracking by reimplementing minimal loop
    idx = 0
    lateral_errs = []
    steer_cmds = []
    offroad = 0

    for _ in range(steps):
        a, d, idx = controller.control(v, path, idx, dt=dt)
        steer_cmds.append(d)
        v.step([a, d], dt)

        # lateral via simple closest segment
        # reuse same projection logic but approximate with linear search
        best_d = float('inf')
        best_lat = 0.0
        px, py = v.x, v.y
        for i in range(len(path)-1):
            a_pt = np.array(path[i], dtype=float)
            b_pt = np.array(path[i+1], dtype=float)
            ab = b_pt - a_pt
            ab2 = ab.dot(ab)
            if ab2 < 1e-12:
                continue
            t = np.dot(np.array([px, py]) - a_pt, ab) / ab2
            t = max(0.0, min(1.0, t))
            proj = a_pt + t*ab
            d2 = np.sum((proj - np.array([px, py]))**2)
            if d2 < best_d:
                best_d = d2
                # signed
                to_p = np.array([px, py]) - a_pt
                cross = ab[0]*to_p[1] - ab[1]*to_p[0]
                sign = np.sign(cross) if abs(cross) > 1e-9 else 1.0
                best_lat = sign * float(np.sqrt(d2))

        lateral_errs.append(best_lat)
        if not is_on_road(v.x, v.y):
            offroad += 1

    lateral = np.array(lateral_errs)
    steer = np.array(steer_cmds)
    metrics = {
        'rms_lateral': float(np.sqrt(np.mean(lateral**2))),
        'max_lateral': float(np.max(np.abs(lateral))),
        'mean_abs_steer': float(np.mean(np.abs(steer))),
        'steer_slew': float(np.mean(np.abs(np.diff(steer)))) if len(steer)>1 else 0.0,
        'offroad_frac': float(offroad/steps),
    }
    return metrics


def run_advanced_tune(mission='zone_outer_mix', steps=1200):
    g = RoadGraph(spacing=2)
    g.build_graph()
    patrol = create_patrol_mission(mission)
    path = build_patrol_loop_path(g, patrol['checkpoints'])

    # baseline controller to extract defaults
    base = PathFollower()

    # grids around baseline
    target_speeds = [max(0.8, base.target_speed - 0.8), base.target_speed, base.target_speed + 0.6]
    steer_rates = [max(0.05, base.max_steer_rate * 0.5), base.max_steer_rate, min(1.2, base.max_steer_rate * 1.8)]

    # steering candidate sets: baseline, narrow, coarse
    baseline_candidates = base.steer_candidates
    narrow = baseline_candidates[(np.abs(baseline_candidates) <= 0.22)]
    if len(narrow) < 3:
        narrow = np.linspace(-0.22, 0.22, 7)
    coarse = np.array([baseline_candidates[0], baseline_candidates[len(baseline_candidates)//2], 0.0, baseline_candidates[-1]])

    candidate_sets = [baseline_candidates, narrow, coarse]

    # include baseline tuple explicitly
    combos = []
    for ts, sr, cand in itertools.product(target_speeds, steer_rates, candidate_sets):
        combos.append({'target_speed': float(ts), 'max_steer_rate': float(sr), 'steer_set': np.array(cand)})

    results = []
    for c in combos:
        ctrl = PathFollower()
        ctrl.target_speed = c['target_speed']
        ctrl.max_steer_rate = c['max_steer_rate']
        ctrl.steer_candidates = c['steer_set']

        v = Vehicle(-23.5, 0.0, np.pi/2, 0.0)
        m = evaluate_once(ctrl, v, path, steps=steps)

        # objective: prioritize off-road then lateral
        obj = 10.0 * m['offroad_frac'] + m['rms_lateral'] + 0.5 * m['steer_slew']
        results.append({'params': c, 'metrics': m, 'obj': obj})

    results_sorted = sorted(results, key=lambda r: r['obj'])

    print('Top 8 results:')
    for r in results_sorted[:8]:
        p = r['params']
        m = r['metrics']
        print(f"ts={p['target_speed']:.2f} sr={p['max_steer_rate']:.3f} cand_len={len(p['steer_set'])} obj={r['obj']:.3f} rms_lat={m['rms_lateral']:.3f} off={m['offroad_frac']:.3f} slew={m['steer_slew']:.4f}")

    best = results_sorted[0]
    print('\nBest param set:')
    p = best['params']
    print(f"target_speed={p['target_speed']} max_steer_rate={p['max_steer_rate']} steer_candidates={p['steer_set']}")
    print('Best metrics:')
    for k, v in best['metrics'].items():
        print(f'  {k}: {v}')

    return best, results_sorted


if __name__ == '__main__':
    run_advanced_tune()
