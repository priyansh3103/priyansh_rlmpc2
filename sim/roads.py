# roads.py
# SIMPLE 4-ZONE PARKING LOT (matches reference image)
# Layout:
#   ┌─────────────────┬─────────────────┐
#   │                 │                 │
#   │     ZONE 1      │     ZONE 2      │  (y = 0 to 25)
#   │                 │                 │
#   ├─────────────────┼─────────────────┤
#   │                 │                 │
#   │     ZONE 3      │     ZONE 4      │  (y = -25 to 0)
#   │                 │                 │
#   └─────────────────┴─────────────────┘
#   x = -25 to 0     x = 0 to 25

ROADS = [
    # ============ OUTER PERIMETER ============
    # Forms a rectangle boundary
    ("line", (-25, -25), (25, -25)),   # Bottom edge
    ("line", (25, -25), (25, 25)),     # Right edge
    ("line", (25, 25), (-25, 25)),     # Top edge
    ("line", (-25, 25), (-25, -25)),   # Left edge

    # ============ MAIN DIVIDING ROADS ============
    # Horizontal divider (separates top and bottom zones)
    ("line", (-25, 0), (25, 0)),

    # Vertical divider (separates left and right zones)
    ("line", (0, -25), (0, 25)),

    # ============ INNER ZONE ROADS (optional for more paths) ============
    # Horizontal inner roads
    ("line", (-25, 12.5), (25, 12.5)),   # Upper inner horizontal
    ("line", (-25, -12.5), (25, -12.5)), # Lower inner horizontal

    # Vertical inner roads
    ("line", (-12.5, -25), (-12.5, 25)), # Left inner vertical
    ("line", (12.5, -25), (12.5, 25)),   # Right inner vertical
]
