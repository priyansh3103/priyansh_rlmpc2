import numpy as np

class World:
    def __init__(self):
        self.vehicles = []

    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)

    def step(self, actions, dt):
        for v, a in zip(self.vehicles, actions):
            v.step(a, dt)

    def get_collision_points(self, vehicle):
        # rear axle
        rear = np.array([vehicle.x, vehicle.y])

        # front point (along heading)
        front = rear + vehicle.L * np.array([
            np.cos(vehicle.theta),
            np.sin(vehicle.theta)
        ])

        return [rear, front]

    def check_collision(self):
        collisions = []

        RADIUS = 1.5  # tuned for 2-circle model

        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                v1 = self.vehicles[i]
                v2 = self.vehicles[j]

                pts1 = self.get_collision_points(v1)
                pts2 = self.get_collision_points(v2)

                collided = False

                for p1 in pts1:
                    for p2 in pts2:
                        dist = np.linalg.norm(p1 - p2)
                        if dist < 2 * RADIUS:
                            collisions.append((i, j))
                            collided = True
                            break
                    if collided:
                        break

        return collisions