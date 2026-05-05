#!/usr/bin/env python3
import numpy as np
from vehicle import Vehicle
from graph import RoadGraph
from controller import PathFollower
from sim import create_patrol_mission, build_patrol_loop_path


def run_validation(steps=1200, mission_name='outer_loop'):
    g = RoadGraph(spacing=2)
    g.build_graph()

    controller = PathFollower()
    v = Vehicle(-23.5, 0.0, np.pi / 2, 0.0)

    patrol = create_patrol_mission(mission_name)
    path = build_patrol_loop_path(g, patrol['checkpoints'])
    idx = 0

    for _ in range(steps):
        a, d, idx = controller.control(v, path, idx, dt=0.05)
        v.step([a, d], 0.05)

        goal = path[-1]
        if idx >= len(path) - 15 and np.hypot(v.x - goal[0], v.y - goal[1]) < 3.0:
            patrol['loops_completed'] += 1
            idx = 0
            controller.reset_pid()

    print('final_pos', round(v.x, 2), round(v.y, 2), 'v', round(v.v, 2))
    print('mission', patrol['name'])
    print('loops_completed', patrol['loops_completed'])
    print('path_len', len(path))


if __name__ == '__main__':
    run_validation()
