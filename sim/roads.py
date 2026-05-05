"""Connector-free lane map.

Road tuple formats:
- ("line", start, end) -> defaults to kind "lane"
- ("line", start, end, kind)

Kinds used in this file:
- "lane": drivable regular lanes
- "outer_loop": single-lane perimeter loop
"""


def _add_polyline(roads, points, kind="lane"):
    for i in range(len(points) - 1):
        roads.append(("line", points[i], points[i + 1], kind))


def _add_rounded_rect_loop(roads, cx, cy, hx, hy, chamfer=3.0, kind="lane"):
    c = chamfer
    pts = [
        (cx - hx + c, cy + hy),
        (cx + hx - c, cy + hy),
        (cx + hx, cy + hy - c),
        (cx + hx, cy - hy + c),
        (cx + hx - c, cy - hy),
        (cx - hx + c, cy - hy),
        (cx - hx, cy - hy + c),
        (cx - hx, cy + hy - c),
        (cx - hx + c, cy + hy),
    ]
    _add_polyline(roads, pts, kind)


ROADS = []

zone_centers = [(-12.0, 12.0), (12.0, 12.0), (-12.0, -12.0), (12.0, -12.0)]

# Two parallel hugging lanes per zone (no explicit connector segments).
for cx, cy in zone_centers:
    _add_rounded_rect_loop(ROADS, cx, cy, hx=8.5, hy=8.5, chamfer=2.5, kind="lane")
    _add_rounded_rect_loop(ROADS, cx, cy, hx=6.5, hy=6.5, chamfer=2.0, kind="lane")

# Shared central intersection with two lanes per direction.
_add_polyline(ROADS, [(-20.0, 1.5), (20.0, 1.5)], kind="lane")
_add_polyline(ROADS, [(-20.0, -1.5), (20.0, -1.5)], kind="lane")
_add_polyline(ROADS, [(1.5, -20.0), (1.5, 20.0)], kind="lane")
_add_polyline(ROADS, [(-1.5, -20.0), (-1.5, 20.0)], kind="lane")

# Single-lane outer ring that hugs zone outer lanes (3 units offset).
_add_rounded_rect_loop(ROADS, 0.0, 0.0, hx=23.5, hy=23.5, chamfer=5.0, kind="outer_loop")
