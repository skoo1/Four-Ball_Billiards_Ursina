"""
Shot Preset System — Phase 3
5개 시나리오(밀어치기, 끌어치기, 죽여치기, 빈쿠션, 3쿠션 대회전)를
자동 세팅 + 시뮬레이션하는 프리셋 함수 모듈.
"""

import numpy as np
from physics import Ball, PhysicsEngine, BALL_RADIUS

# Simulation timestep (smaller for accuracy)
_DT = 0.0005


def _pick_corners(exclude_positions):
    """시나리오에 관여하지 않는 나머지 공을 테이블 구석에 배치."""
    corners = [
        [0.5, 0.0, 1.0],
        [-0.5, 0.0, 1.0],
        [0.5, 0.0, -1.0],
        [-0.5, 0.0, -1.0],
    ]
    available = []
    for c in corners:
        too_close = False
        for ep in exclude_positions:
            if np.linalg.norm(np.array(c) - np.array(ep)) < 0.15:
                too_close = True
                break
        if not too_close:
            available.append(c)
    return available


class ShotPreset:
    """각 프리셋은 balls 배치 → 큐 타격 → simulate → 결과 dict 반환."""

    @staticmethod
    def scenario_1_follow(run=True) -> dict:
        """밀어치기 (Follow shot): 탑스핀으로 수구가 충돌점 너머 전진."""
        engine = PhysicsEngine()

        white = Ball("white", position=[0.0, 0.0, -0.8])
        red1  = Ball("red1",  position=[0.0, 0.0, -0.3])

        extras   = _pick_corners([[0.0, 0.0, -0.8], [0.0, 0.0, -0.3]])
        yellow   = Ball("yellow", position=extras[0])
        red2     = Ball("red2",   position=extras[1])

        engine.apply_cue(
            white,
            force=0.5,
            direction=[0.0, 0.0, 1.0],
            offset=[0.0, 0.02],
        )

        balls = [white, red1, yellow, red2]
        elapsed = 0.0
        peak_z_white = float(white.position[2])
        if run:
            t = 0.0
            while t < 10.0:
                engine.update([white, red1], _DT)
                t += _DT
                elapsed = t
                if white.position[2] > peak_z_white:
                    peak_z_white = float(white.position[2])
                if all(not b.is_moving() for b in [white, red1]):
                    break
        return {"white": white, "red1": red1, "balls": balls, "engine": engine,
                "elapsed": elapsed, "peak_z_white": peak_z_white}

    @staticmethod
    def scenario_2_draw(run=True) -> dict:
        """끌어치기 (Draw shot): 백스핀으로 수구가 출발점 뒤로 복귀."""
        engine = PhysicsEngine()

        white = Ball("white", position=[0.0, 0.0, -0.5])
        red1  = Ball("red1",  position=[0.0, 0.0,  0.0])

        extras = _pick_corners([[0.0, 0.0, -0.5], [0.0, 0.0, 0.0]])
        yellow = Ball("yellow", position=extras[0])
        red2   = Ball("red2",   position=extras[1])

        engine.apply_cue(
            white,
            force=0.8,
            direction=[0.0, 0.0, 1.0],
            offset=[0.0, -0.028],
        )

        balls = [white, red1, yellow, red2]
        elapsed = 0.0
        if run:
            elapsed = engine.simulate([white, red1], dt=_DT)
        return {"white": white, "red1": red1, "balls": balls, "engine": engine, "elapsed": elapsed}

    @staticmethod
    def scenario_3_stop(run=True) -> dict:
        """죽여치기 (Stop shot): 수구가 충돌점 근처에서 정지."""
        engine = PhysicsEngine()

        white = Ball("white", position=[0.0, 0.0, -0.075])
        red1  = Ball("red1",  position=[0.0, 0.0,  0.075])

        extras = _pick_corners([[0.0, 0.0, -0.075], [0.0, 0.0, 0.075]])
        yellow = Ball("yellow", position=extras[0])
        red2   = Ball("red2",   position=extras[1])

        engine.apply_cue(
            white,
            force=0.2,
            direction=[0.0, 0.0, 1.0],
            offset=None,
        )

        balls = [white, red1, yellow, red2]
        elapsed = 0.0
        if run:
            elapsed = engine.simulate([white, red1], dt=_DT)
        return {"white": white, "red1": red1, "balls": balls, "engine": engine, "elapsed": elapsed}

    @staticmethod
    def scenario_4_bank(english: float = 0.0, run=True) -> dict:
        """빈쿠션 (Bank shot): 수구를 쿠션에 먼저 맞힌 후 적구 방향으로.

        Args:
            english: 좌우 회전 오프셋. 0.0=무회전, 양수=우회전, 음수=좌회전.
        """
        engine = PhysicsEngine()

        white = Ball("white", position=[0.0, 0.0, -0.5])
        red1  = Ball("red1",  position=[0.3, 0.0,  0.5])

        extras = _pick_corners([[0.0, 0.0, -0.5], [0.3, 0.0, 0.5]])
        yellow = Ball("yellow", position=extras[0])
        red2   = Ball("red2",   position=extras[1])

        direction = np.array([1.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)

        offset = [english, 0.0] if english != 0.0 else None

        engine.apply_cue(
            white,
            force=0.5,
            direction=direction,
            offset=offset,
        )

        balls = [white, red1, yellow, red2]
        elapsed = 0.0
        if run:
            elapsed = engine.simulate([white, red1], dt=_DT)
        return {"white": white, "red1": red1, "balls": balls, "engine": engine, "elapsed": elapsed}

    @staticmethod
    def scenario_5_nejire(run=True) -> dict:
        """3쿠션 대회전 (Nejire): 강타 + 회전으로 3쿠션 이상 경유."""
        engine = PhysicsEngine()

        white = Ball("white", position=[ 0.0, 0.0, -0.8])
        red1  = Ball("red1",  position=[-0.3, 0.0,  0.8])

        extras = _pick_corners([[0.0, 0.0, -0.8], [-0.3, 0.0, 0.8]])
        yellow = Ball("yellow", position=extras[0])
        red2   = Ball("red2",   position=extras[1])

        direction = np.array([0.6, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)

        engine.apply_cue(
            white,
            force=0.5,
            direction=direction,
            offset=[-0.005, 0.0],
        )

        balls = [white, red1, yellow, red2]
        elapsed = 0.0
        if run:
            elapsed = engine.simulate([white, red1], dt=_DT)
        return {"white": white, "red1": red1, "balls": balls, "engine": engine, "elapsed": elapsed}
