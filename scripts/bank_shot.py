"""빈쿠션 (Bank shot) — 오른쪽 쿠션 경유 후 적구 공략"""

SCRIPT = {
    "setup": {
        "white":  ( 0.0, -0.5),
        "yellow": ( 0.3,  0.0),
        "red1":   (-0.3,  0.6),
        "red2":   ( 0.1,  0.9),
    },
    "shot": {
        "ball":      "white",
        "azimuth":   70.0,   # 오른쪽으로 향해 오른 쿠션에 맞힘
        "elevation":  0.0,
        "tip_x":      0.0,
        "tip_y":      0.0,
        "force":      1.0,
        "stroke_profile": {
            "aim_delay":      0.6,
            "backswing_dist": 0.10,
            "backswing_time": 0.35,
            "strike_time":    0.15,
        },
    },
}
