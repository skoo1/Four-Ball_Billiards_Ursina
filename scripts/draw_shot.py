"""끌어치기 (Draw shot) — 백스핀으로 수구가 충돌 후 뒤로 복귀"""

SCRIPT = {
    "setup": {
        "white":  ( 0.0, -0.4),
        "yellow": ( 0.3,  0.2),
        "red1":   ( 0.0,  0.1),
        "red2":   (-0.2,  0.5),
    },
    "shot": {
        "ball":      "white",
        "azimuth":   0.0,
        "elevation": 0.0,
        "tip_x":     0.0,
        "tip_y":    -0.5,    # 아래 당점 → 백스핀 (끌기)
        "force":     1.0,
        "stroke_profile": {
            "aim_delay":      0.7,
            "backswing_dist": 0.10,
            "backswing_time": 0.35,
            "strike_time":    0.15,
        },
    },
}
