"""마세 (Masse shot) — 큐를 세워서 수구에 wy 회전 부여, 경로 커브"""

SCRIPT = {
    "setup": {
        "white":  ( 0.0, -0.3),
        "yellow": ( 0.3,  0.1),
        "red1":   (-0.1,  0.6),
        "red2":   ( 0.2,  0.7),
    },
    "shot": {
        "ball":      "white",
        "azimuth":   10.0,
        "elevation": 30.0,   # 큐를 30도 세움 → wy 스핀 발생 → 커브
        "tip_x":     0.3,    # 오른쪽 사이드 English → 반시계 방향 커브
        "tip_y":     0.0,
        "force":     0.9,
        "stroke_profile": {
            "aim_delay":      0.8,
            "backswing_dist": 0.08,
            "backswing_time": 0.35,
            "strike_time":    0.15,
        },
    },
}
