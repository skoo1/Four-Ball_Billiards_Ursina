"""
Physics Engine Tests — Verification of Demo Scenarios A, B, C.

Scenario A: Follow Shot (밀어치기) — cue ball continues forward after collision.
Scenario B: Draw Shot (끌어치기) — cue ball returns backward after collision.
Scenario C: Stop Shot (죽여치기) — cue ball stops at collision point.

All scenario tests use a large virtual table to avoid cushion interference.
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from physics import (
    Ball, BallState, PhysicsEngine, BALL_RADIUS, BALL_MASS,
    INERTIA, VELOCITY_THRESHOLD,
    RAIL_H_OFFSET, CUSHION_RESTITUTION, BALL_RESTITUTION,
    SQUIRT_FACTOR, MU_BALL,
)


# ── Helpers ──────────────────────────────────────────────

def make_straight_pair(separation: float = 0.5):
    """Create cue ball and object ball on the z-axis, separated by `separation` m."""
    cue = Ball(name="cue", position=[0.0, 0.0, 0.0])
    obj = Ball(name="object", position=[0.0, 0.0, separation])
    return cue, obj


def simulate_shot(cue: Ball, obj: Ball, table_size: float = 20.0) -> float:
    """Simulate until all balls stop. Uses a large table to avoid cushion interference."""
    engine = PhysicsEngine(table_width=table_size, table_length=table_size)
    balls = [cue, obj]
    elapsed = engine.simulate(balls, dt=0.0005, max_time=15.0)
    return elapsed


# ── Scenario A: Follow Shot (밀어치기) ───────────────────

class TestFollowShot:
    """
    Cue ball is struck on its upper half (topspin).
    After hitting the object ball, the cue ball should CONTINUE FORWARD
    past the collision point.
    """

    def test_cue_ball_continues_forward(self):
        cue, obj = make_straight_pair(separation=0.5)
        collision_z = obj.position[2]

        # Strike top of ball: offset y > 0 → topspin
        # Moderate force so the ball has time to develop rolling + topspin
        PhysicsEngine.apply_cue(
            cue,
            force=0.8,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.02]),  # top half
        )

        simulate_shot(cue, obj)

        # Cue ball should end up BEYOND the collision point
        assert cue.position[2] > collision_z, (
            f"Follow shot failed: cue z={cue.position[2]:.4f} "
            f"should be > collision z={collision_z:.4f}"
        )

    def test_cue_further_than_no_spin(self):
        """Follow shot cue ball should end further forward than a center-hit cue ball."""
        # Topspin shot
        cue_top, obj_top = make_straight_pair(separation=0.5)
        PhysicsEngine.apply_cue(
            cue_top, force=0.8,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.02]),
        )
        simulate_shot(cue_top, obj_top)

        # Center (no-spin) shot
        cue_center, obj_center = make_straight_pair(separation=0.5)
        PhysicsEngine.apply_cue(
            cue_center, force=0.8,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.0]),
        )
        simulate_shot(cue_center, obj_center)

        assert cue_top.position[2] > cue_center.position[2], (
            f"Topspin cue z={cue_top.position[2]:.4f} should be > "
            f"center-hit cue z={cue_center.position[2]:.4f}"
        )

    def test_cue_has_topspin_before_collision(self):
        """Verify that the cue strike generates topspin.
        For +z motion, topspin means wx > 0 (excess forward rotation).
        """
        cue = Ball(name="cue", position=[0.0, 0.0, 0.0])
        PhysicsEngine.apply_cue(
            cue,
            force=0.8,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.02]),
        )
        assert cue.angular_velocity[0] > 0, (
            f"Expected topspin (wx > 0), got wx={cue.angular_velocity[0]:.4f}"
        )


# ── Scenario B: Draw Shot (끌어치기) ─────────────────────

class TestDrawShot:
    """
    Cue ball is struck on its lower half (backspin).
    After hitting the object ball, the cue ball should RETURN BACKWARD.
    """

    def test_cue_ball_returns_backward(self):
        cue, obj = make_straight_pair(separation=0.5)
        initial_cue_z = cue.position[2]

        # Strike bottom of ball: offset y < 0 → backspin, strong force
        PhysicsEngine.apply_cue(
            cue,
            force=1.5,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, -0.025]),
        )

        simulate_shot(cue, obj)

        # Cue ball should end up BEHIND its starting position
        assert cue.position[2] < initial_cue_z, (
            f"Draw shot failed: cue z={cue.position[2]:.4f} "
            f"should be < start z={initial_cue_z:.4f}"
        )

    def test_cue_has_backspin_before_collision(self):
        """Verify that the low strike generates backspin.
        For +z motion, backspin means wx < 0.
        """
        cue = Ball(name="cue", position=[0.0, 0.0, 0.0])
        PhysicsEngine.apply_cue(
            cue,
            force=1.5,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, -0.025]),
        )
        assert cue.angular_velocity[0] < 0, (
            f"Expected backspin (wx < 0), got wx={cue.angular_velocity[0]:.4f}"
        )

    def test_object_ball_moves_forward_on_draw(self):
        """Object ball should still go forward even in a draw shot.

        With high force the object ball may reach the far cushion and bounce back,
        so we track the peak z position rather than the final resting position.
        """
        cue, obj = make_straight_pair(separation=0.5)
        initial_obj_z = obj.position[2]

        PhysicsEngine.apply_cue(
            cue,
            force=1.5,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, -0.025]),
        )

        # Track peak forward position — ball may bounce off far cushion and return
        engine = PhysicsEngine(table_width=20.0, table_length=20.0)
        balls = [cue, obj]
        peak_obj_z = initial_obj_z
        t = 0.0
        while t < 15.0:
            engine.update(balls, dt=0.0005)
            t += 0.0005
            peak_obj_z = max(peak_obj_z, float(obj.position[2]))
            if all(not b.is_moving() for b in balls):
                break

        assert peak_obj_z > initial_obj_z + 0.05, (
            f"Object ball should move forward: peak z={peak_obj_z:.4f}"
        )


# ── Scenario C: Stop Shot (죽여치기) ─────────────────────

class TestStopShot:
    """
    Cue ball is struck at the center (or very slightly below) with moderate force.
    After hitting the object ball head-on, the cue ball should STOP near the
    collision point because all linear momentum is transferred and there's
    minimal residual spin.
    """

    def test_cue_ball_stops_near_collision(self):
        """
        For a stop shot, the ideal is hitting dead center so the ball transitions
        to rolling just before collision, then all velocity transfers to the object ball.
        We use center hit (no offset) which naturally reaches rolling before collision.
        """
        cue, obj = make_straight_pair(separation=0.5)
        collision_z = obj.position[2]

        # Center hit: no spin offset. Ball will transition to rolling,
        # then upon collision transfer nearly all velocity.
        PhysicsEngine.apply_cue(
            cue,
            force=0.5,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.0]),
        )

        simulate_shot(cue, obj)

        # Cue ball should stop reasonably near the collision point
        # Allow generous tolerance since some residual motion is expected
        distance_from_collision = abs(cue.position[2] - collision_z)
        assert distance_from_collision < 0.25, (
            f"Stop shot failed: cue z={cue.position[2]:.4f}, "
            f"collision z={collision_z:.4f}, "
            f"distance={distance_from_collision:.4f} (should be < 0.25)"
        )

    def test_cue_ball_is_stationary(self):
        """Cue ball should be stationary at end of simulation."""
        cue, obj = make_straight_pair(separation=0.5)

        PhysicsEngine.apply_cue(
            cue,
            force=0.5,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.0]),
        )

        simulate_shot(cue, obj)

        assert cue.state == BallState.STATIONARY, (
            f"Cue ball should be stationary, got {cue.state}"
        )

    def test_stop_shot_less_movement_than_follow(self):
        """Stop shot cue ball should move less than a follow (topspin) shot."""
        # Stop shot (center hit)
        cue_stop, obj_stop = make_straight_pair(separation=0.5)
        PhysicsEngine.apply_cue(
            cue_stop, force=0.8,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.0]),
        )
        simulate_shot(cue_stop, obj_stop)

        # Follow shot (topspin)
        cue_follow, obj_follow = make_straight_pair(separation=0.5)
        PhysicsEngine.apply_cue(
            cue_follow, force=0.8,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.02]),
        )
        simulate_shot(cue_follow, obj_follow)

        assert cue_stop.position[2] < cue_follow.position[2], (
            f"Stop cue z={cue_stop.position[2]:.4f} should be < "
            f"follow cue z={cue_follow.position[2]:.4f}"
        )

    def test_object_ball_takes_momentum(self):
        """Object ball should carry away most of the momentum."""
        cue, obj = make_straight_pair(separation=0.5)
        initial_obj_z = obj.position[2]

        PhysicsEngine.apply_cue(
            cue,
            force=0.5,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.0]),
        )

        simulate_shot(cue, obj)

        assert obj.position[2] > initial_obj_z + 0.05, (
            f"Object ball should have moved forward: z={obj.position[2]:.4f}"
        )


# ── Additional unit tests for physics primitives ─────────

class TestCueImpact:
    """Basic cue impact tests."""

    def test_center_hit_no_spin(self):
        """Hitting dead center should produce minimal angular velocity."""
        ball = Ball(name="test")
        PhysicsEngine.apply_cue(
            ball,
            force=2.0,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.0]),
        )
        assert ball.speed > 0
        assert np.linalg.norm(ball.angular_velocity) < 1.0

    def test_velocity_direction(self):
        """Ball should move in the cue direction."""
        ball = Ball(name="test")
        PhysicsEngine.apply_cue(
            ball,
            force=2.0,
            direction=np.array([1.0, 0.0, 0.0]),
        )
        assert ball.velocity[0] > 0
        assert abs(ball.velocity[2]) < 1e-6


class TestFloorFriction:
    """Test sliding-to-rolling transition."""

    def test_sliding_to_rolling_transition(self):
        """A sliding ball should eventually transition to rolling."""
        engine = PhysicsEngine()
        ball = Ball(name="test", velocity=[0.0, 0.0, 2.0])
        ball.state = BallState.SLIDING

        states_seen = set()
        for _ in range(5000):
            engine.update([ball], dt=0.001)
            states_seen.add(ball.state)
            if ball.state == BallState.ROLLING:
                break

        assert BallState.ROLLING in states_seen or BallState.STATIONARY in states_seen

    def test_rolling_ball_decelerates(self):
        """A rolling ball should gradually slow down."""
        engine = PhysicsEngine()
        ball = Ball(name="test", velocity=[0.0, 0.0, 1.0])
        ball.state = BallState.ROLLING
        R = ball.radius
        ball.angular_velocity = np.array([-ball.velocity[2] / R, 0.0, ball.velocity[0] / R])

        initial_speed = ball.speed
        for _ in range(1000):
            engine.update([ball], dt=0.001)

        assert ball.speed < initial_speed

    def test_ball_eventually_stops(self):
        """A ball should eventually come to rest."""
        engine = PhysicsEngine()
        ball = Ball(name="test", velocity=[0.0, 0.0, 1.0])
        ball.state = BallState.SLIDING
        engine.simulate([ball], dt=0.001, max_time=10.0)
        assert ball.state == BallState.STATIONARY


class TestCushionCollision:
    """Test cushion bouncing."""

    def test_ball_bounces_off_cushion(self):
        """Ball heading toward cushion should bounce back."""
        engine = PhysicsEngine()
        ball = Ball(name="test",
                    position=[engine.x_max - 0.01, 0.0, 0.0],
                    velocity=[2.0, 0.0, 0.0])
        ball.state = BallState.SLIDING

        for _ in range(100):
            engine.update([ball], dt=0.001)

        assert ball.velocity[0] < 0 or ball.position[0] < engine.x_max - BALL_RADIUS


class TestBallCollision:
    """Test ball-ball collision."""

    def test_head_on_collision_transfers_velocity(self):
        """Head-on collision should transfer velocity to stationary ball."""
        b1 = Ball(name="cue", position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 2.0])
        b1.state = BallState.ROLLING
        b2 = Ball(name="obj", position=[0.0, 0.0, 2 * BALL_RADIUS + 0.0001])

        PhysicsEngine()._resolve_ball_collision(b1, b2)

        assert b2.velocity[2] > 0
        assert b1.velocity[2] < 1.0


# ══════════════════════════════════════════════════════════
# 강화 테스트: 엣지 케이스 & 물리적 정합성
# ══════════════════════════════════════════════════════════


# ── 강화 1, 8: Curve Force Edge Cases ──────────────────

class TestCurveForce:
    """커브(맛세이) 힘의 상태별 작동 및 안전성 검증."""

    def test_no_curve_when_rolling(self):
        """ROLLING 상태에서 사이드 스핀(wy)이 있어도 횡방향 속도 불변."""
        R = BALL_RADIUS
        ball = Ball("test", velocity=[0.0, 0.0, 1.0])
        ball.state = BallState.ROLLING
        ball.angular_velocity = np.array([1.0 / R, 50.0, 0.0])  # rolling + strong wy
        vx_before = ball.velocity[0]

        PhysicsEngine._apply_curve_force(ball, 0.01)

        assert ball.velocity[0] == vx_before, (
            f"ROLLING에서 커브 힘이 작용하면 안 됨: vx={ball.velocity[0]}"
        )

    def test_curve_active_when_sliding(self):
        """SLIDING 상태에서 사이드 스핀(wy)이 있으면 횡방향 속도 발생."""
        ball = Ball("test", velocity=[0.0, 0.0, 1.0])
        ball.state = BallState.SLIDING
        ball.angular_velocity = np.array([0.0, 50.0, 0.0])

        PhysicsEngine._apply_curve_force(ball, 0.01)

        assert ball.velocity[0] != 0.0, (
            f"SLIDING에서 wy≠0이면 커브 발생해야 함: vx={ball.velocity[0]}"
        )

    def test_curve_direction_follows_spin_sign(self):
        """wy 부호에 따라 커브 방향이 반대여야 함."""
        ball_pos = Ball("p", velocity=[0.0, 0.0, 1.0])
        ball_pos.state = BallState.SLIDING
        ball_pos.angular_velocity = np.array([0.0, 50.0, 0.0])
        PhysicsEngine._apply_curve_force(ball_pos, 0.01)

        ball_neg = Ball("n", velocity=[0.0, 0.0, 1.0])
        ball_neg.state = BallState.SLIDING
        ball_neg.angular_velocity = np.array([0.0, -50.0, 0.0])
        PhysicsEngine._apply_curve_force(ball_neg, 0.01)

        # Opposite wy → opposite lateral deflection
        assert ball_pos.velocity[0] * ball_neg.velocity[0] < 0, (
            f"wy 부호 반전 시 커브 방향도 반전: "
            f"vx(+wy)={ball_pos.velocity[0]:.6f}, vx(-wy)={ball_neg.velocity[0]:.6f}"
        )

    def test_no_blowup_at_low_speed(self):
        """속도가 매우 작을 때 커브 힘이 발산하지 않아야 함 (분모 +0.10 감쇠)."""
        ball = Ball("test", velocity=[0.0, 0.0, 0.0001])
        ball.state = BallState.SLIDING
        ball.angular_velocity = np.array([0.0, 100.0, 0.0])

        PhysicsEngine._apply_curve_force(ball, 0.01)

        assert np.all(np.isfinite(ball.velocity)), "속도가 유한해야 함"
        assert np.linalg.norm(ball.velocity) < 1.0, (
            f"속도 폭발: |v|={np.linalg.norm(ball.velocity):.4f}"
        )


# ── 강화 2, 5, 7: Cushion Physics ──────────────────────

class TestCushionPhysics:
    """쿠션 충돌의 물리적 정합성 (토크 수치, 스핀 누적, 좌우 대칭) 검증."""

    def test_normal_torque_magnitude(self):
        """쿠션 법선 임펄스 → ωz 변화량이 수식 예측값과 일치.

        우측 쿠션, 순수 수직 입사(vz=0) → 마찰 없음 → 순수 법선 토크만 작용.
        Δωz = h × (1+e) × |vx| × m / I
        """
        engine = PhysicsEngine()
        pos_x = engine.x_max - BALL_RADIUS + 0.001
        ball = Ball("test", position=[pos_x, 0.0, 0.0])
        ball.velocity = np.array([1.0, 0.0, 0.0])   # pure perpendicular → no friction
        ball.state = BallState.SLIDING
        wz_before = ball.angular_velocity[2]

        engine._check_cushion_collisions(ball)

        expected_dw = (
            RAIL_H_OFFSET * (1 + CUSHION_RESTITUTION)
            * 1.0 * BALL_MASS / INERTIA
        )
        actual_dw = ball.angular_velocity[2] - wz_before
        assert math.isclose(actual_dw, expected_dw, rel_tol=0.01), (
            f"Normal torque Δωz: actual={actual_dw:.4f}, expected={expected_dw:.4f}"
        )

    def test_multi_cushion_peak_spin_bounded(self):
        """다중 쿠션 충돌 시 각속도가 비현실적으로 누적되면 안 됨."""
        engine = PhysicsEngine()
        ball = Ball("test", position=[0.0, 0.0, 0.0])
        ball.velocity = np.array([2.5, 0.0, 2.0])
        ball.state = BallState.SLIDING

        peak_w = 0.0
        for _ in range(40000):
            engine.update([ball], dt=0.0005)
            w_mag = float(np.linalg.norm(ball.angular_velocity))
            peak_w = max(peak_w, w_mag)
            if ball.state == BallState.STATIONARY:
                break

        # 초기 속도 ~3.2 m/s → 순수 롤링 ω ≈ v/R ≈ 98 rad/s
        # 쿠션 토크 포함해도 200 rad/s 초과는 비현실적
        assert peak_w < 200.0, (
            f"Peak angular velocity {peak_w:.1f} rad/s exceeds realistic bound (200)"
        )
        assert ball.cushion_hits >= 2, (
            f"Ball should hit ≥2 cushions: got {ball.cushion_hits}"
        )

    def test_left_right_symmetry(self):
        """좌측/우측 쿠션 충돌이 거울 대칭 결과를 내야 함."""
        engine = PhysicsEngine()

        # Right cushion
        b_r = Ball("r", position=[engine.x_max - BALL_RADIUS + 0.001, 0.0, 0.0])
        b_r.velocity = np.array([1.0, 0.0, 1.0])
        b_r.state = BallState.SLIDING
        engine._check_cushion_collisions(b_r)

        # Left cushion (mirror)
        b_l = Ball("l", position=[engine.x_min + BALL_RADIUS - 0.001, 0.0, 0.0])
        b_l.velocity = np.array([-1.0, 0.0, 1.0])
        b_l.state = BallState.SLIDING
        engine._check_cushion_collisions(b_l)

        # vx: opposite signs
        assert math.isclose(b_r.velocity[0], -b_l.velocity[0], abs_tol=1e-9), (
            f"vx symmetry: right={b_r.velocity[0]:.6f}, left={b_l.velocity[0]:.6f}"
        )
        # vz: same
        assert math.isclose(b_r.velocity[2], b_l.velocity[2], abs_tol=1e-9), (
            f"vz symmetry: right={b_r.velocity[2]:.6f}, left={b_l.velocity[2]:.6f}"
        )
        # ωz: opposite signs (normal torque)
        assert math.isclose(
            b_r.angular_velocity[2], -b_l.angular_velocity[2], abs_tol=1e-9
        ), (
            f"ωz symmetry: right={b_r.angular_velocity[2]:.6f}, "
            f"left={b_l.angular_velocity[2]:.6f}"
        )
        # ωx: same (friction torque symmetry)
        assert math.isclose(
            b_r.angular_velocity[0], b_l.angular_velocity[0], abs_tol=1e-9
        ), (
            f"ωx symmetry: right={b_r.angular_velocity[0]:.6f}, "
            f"left={b_l.angular_velocity[0]:.6f}"
        )
        # ωy: opposite signs
        assert math.isclose(
            b_r.angular_velocity[1], -b_l.angular_velocity[1], abs_tol=1e-9
        ), (
            f"ωy symmetry: right={b_r.angular_velocity[1]:.6f}, "
            f"left={b_l.angular_velocity[1]:.6f}"
        )

    def test_far_near_symmetry(self):
        """전방/후방 쿠션 충돌이 거울 대칭 결과를 내야 함."""
        engine = PhysicsEngine()

        # Far cushion
        b_f = Ball("f", position=[0.0, 0.0, engine.z_max - BALL_RADIUS + 0.001])
        b_f.velocity = np.array([1.0, 0.0, 1.0])
        b_f.state = BallState.SLIDING
        engine._check_cushion_collisions(b_f)

        # Near cushion (mirror)
        b_n = Ball("n", position=[0.0, 0.0, engine.z_min + BALL_RADIUS - 0.001])
        b_n.velocity = np.array([1.0, 0.0, -1.0])
        b_n.state = BallState.SLIDING
        engine._check_cushion_collisions(b_n)

        # vz: opposite signs
        assert math.isclose(b_f.velocity[2], -b_n.velocity[2], abs_tol=1e-9), (
            f"vz symmetry: far={b_f.velocity[2]:.6f}, near={b_n.velocity[2]:.6f}"
        )
        # vx: same
        assert math.isclose(b_f.velocity[0], b_n.velocity[0], abs_tol=1e-9), (
            f"vx symmetry: far={b_f.velocity[0]:.6f}, near={b_n.velocity[0]:.6f}"
        )
        # ωx: opposite signs (normal torque)
        assert math.isclose(
            b_f.angular_velocity[0], -b_n.angular_velocity[0], abs_tol=1e-9
        ), (
            f"ωx symmetry: far={b_f.angular_velocity[0]:.6f}, "
            f"near={b_n.angular_velocity[0]:.6f}"
        )


# ── 강화 4: Spin Preservation ──────────────────────────

class TestSpinPreservation:
    """SLIDING→ROLLING 전환 시 사이드 스핀(wy) 보존 검증."""

    def test_side_spin_preserved_through_transition(self):
        """SLIDING→ROLLING 전환 시 wy가 _snap_to_rolling에 의해 제거되면 안 됨."""
        engine = PhysicsEngine(table_width=20.0, table_length=20.0)
        ball = Ball("test", velocity=[0.0, 0.0, 1.0])
        ball.state = BallState.SLIDING
        ball.angular_velocity = np.array([0.0, 30.0, 0.0])

        for _ in range(5000):
            engine.update([ball], dt=0.001)
            if ball.state == BallState.ROLLING:
                break

        assert ball.state == BallState.ROLLING, "ROLLING 전환이 발생해야 함"
        # wy는 spin decel(~30 rad/s²)에 의해 감소하지만 ~0.15초 전환 시점에는 아직 충분히 큼
        assert ball.angular_velocity[1] > 5.0, (
            f"Side spin 보존 실패: wy={ball.angular_velocity[1]:.4f}"
        )

    def test_rolling_constraint_correct_at_transition(self):
        """전환 직후 wx, wz가 rolling constraint를 정확히 만족해야 함."""
        engine = PhysicsEngine(table_width=20.0, table_length=20.0)
        ball = Ball("test", velocity=[0.3, 0.0, 1.0])
        ball.state = BallState.SLIDING
        ball.angular_velocity = np.array([0.0, 20.0, 0.0])

        for _ in range(5000):
            engine.update([ball], dt=0.001)
            if ball.state == BallState.ROLLING:
                break

        assert ball.state == BallState.ROLLING
        R = BALL_RADIUS
        expected_wx = ball.velocity[2] / R
        expected_wz = -ball.velocity[0] / R
        assert math.isclose(ball.angular_velocity[0], expected_wx, rel_tol=0.01), (
            f"wx={ball.angular_velocity[0]:.4f}, expected={expected_wx:.4f}"
        )
        assert math.isclose(ball.angular_velocity[2], expected_wz, rel_tol=0.01), (
            f"wz={ball.angular_velocity[2]:.4f}, expected={expected_wz:.4f}"
        )


# ── 강화 6: Energy Conservation ────────────────────────

class TestEnergyConservation:
    """볼-볼 충돌의 에너지 보존 검증."""

    def test_head_on_collision_energy(self):
        """Head-on 충돌: 운동 에너지 비율 ≈ (1+e²)/2.

        등질량 정면충돌에서 KE_after/KE_before = (1+e²)/2 ≈ 0.94.
        Throw에 의한 추가 에너지 이동을 고려해 회전 에너지 포함.
        """
        b1 = Ball("cue", position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 2.0])
        b1.state = BallState.SLIDING
        b2 = Ball("obj", position=[0.0, 0.0, 2 * BALL_RADIUS + 0.0001])

        KE_before = 0.5 * BALL_MASS * np.dot(b1.velocity, b1.velocity)

        PhysicsEngine()._resolve_ball_collision(b1, b2)

        # Linear KE
        KE_linear = (
            0.5 * BALL_MASS * np.dot(b1.velocity, b1.velocity)
            + 0.5 * BALL_MASS * np.dot(b2.velocity, b2.velocity)
        )
        # Rotational KE
        KE_rot = (
            0.5 * INERTIA * np.dot(b1.angular_velocity, b1.angular_velocity)
            + 0.5 * INERTIA * np.dot(b2.angular_velocity, b2.angular_velocity)
        )
        KE_total = KE_linear + KE_rot
        ratio = KE_total / KE_before
        expected = (1 + BALL_RESTITUTION ** 2) / 2

        # Total energy (linear+rotational) should not exceed initial
        assert ratio <= 1.01, f"Energy increased: ratio={ratio:.4f}"
        # Should be close to theoretical value (with some tolerance for throw)
        assert ratio > 0.85, f"Energy ratio {ratio:.4f} too low"

    def test_oblique_collision_energy_nonincreasing(self):
        """빗맞음 충돌에서도 총 에너지가 증가하면 안 됨."""
        R = BALL_RADIUS
        b1 = Ball("cue", position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 2.0])
        b1.state = BallState.SLIDING
        b2 = Ball("obj", position=[R, 0.0, R * np.sqrt(3) - 0.0001])

        KE_before = 0.5 * BALL_MASS * np.dot(b1.velocity, b1.velocity)

        PhysicsEngine()._resolve_ball_collision(b1, b2)

        KE_linear = (
            0.5 * BALL_MASS * np.dot(b1.velocity, b1.velocity)
            + 0.5 * BALL_MASS * np.dot(b2.velocity, b2.velocity)
        )
        KE_rot = (
            0.5 * INERTIA * np.dot(b1.angular_velocity, b1.angular_velocity)
            + 0.5 * INERTIA * np.dot(b2.angular_velocity, b2.angular_velocity)
        )
        KE_total = KE_linear + KE_rot

        assert KE_total <= KE_before * 1.01, (
            f"Energy increased in oblique collision: "
            f"before={KE_before:.6f}, after={KE_total:.6f}"
        )


# ── 강화 9: Squirt ─────────────────────────────────────

class TestSquirt:
    """Cue squirt (스쿼트) 편향 방향 및 크기 검증."""

    def test_right_english_deflects_left(self):
        """우측 당점(offset_x > 0) → vx < 0 (좌측 편향)."""
        ball = Ball("test")
        PhysicsEngine.apply_cue(
            ball,
            force=3.0,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.02, 0.0]),
        )
        assert ball.velocity[2] > 0, "주 진행 방향(+z) 유지"
        assert ball.velocity[0] < 0, (
            f"Right english → left deflection: vx={ball.velocity[0]:.6f}"
        )

    def test_left_english_deflects_right(self):
        """좌측 당점(offset_x < 0) → vx > 0 (우측 편향)."""
        ball = Ball("test")
        PhysicsEngine.apply_cue(
            ball,
            force=3.0,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([-0.02, 0.0]),
        )
        assert ball.velocity[2] > 0
        assert ball.velocity[0] > 0, (
            f"Left english → right deflection: vx={ball.velocity[0]:.6f}"
        )

    def test_larger_offset_more_squirt(self):
        """더 큰 당점 오프셋 → 더 큰 스쿼트 편향량."""
        ball_small = Ball("s")
        PhysicsEngine.apply_cue(
            ball_small, force=3.0,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.01, 0.0]),
        )
        ball_large = Ball("l")
        PhysicsEngine.apply_cue(
            ball_large, force=3.0,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.025, 0.0]),
        )
        assert abs(ball_large.velocity[0]) > abs(ball_small.velocity[0]), (
            f"More offset → more squirt: "
            f"|vx_large|={abs(ball_large.velocity[0]):.6f}, "
            f"|vx_small|={abs(ball_small.velocity[0]):.6f}"
        )

    def test_center_hit_no_squirt(self):
        """중앙 당점 → 횡방향 편향 없음."""
        ball = Ball("test")
        PhysicsEngine.apply_cue(
            ball,
            force=3.0,
            direction=np.array([0.0, 0.0, 1.0]),
            offset=np.array([0.0, 0.0]),
        )
        assert abs(ball.velocity[0]) < 1e-9, (
            f"Center hit should have no x-deflection: vx={ball.velocity[0]:.9f}"
        )


# ── 강화 10: Ball-Ball Throw ──────────────────────────

class TestBallBallThrow:
    """볼-볼 충돌 시 Coulomb 마찰(throw)에 의한 스핀 전달 및 편향 검증."""

    @staticmethod
    def _setup_half_ball_hit():
        """Half-ball (30°) 충돌 세팅."""
        R = BALL_RADIUS
        b1 = Ball("cue", position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 2.0])
        b1.state = BallState.SLIDING
        b2 = Ball("obj", position=[R, 0.0, R * np.sqrt(3) - 0.0001])
        return b1, b2

    def test_throw_transfers_spin_to_object_ball(self):
        """비스듬한 충돌 시 MU_BALL 마찰로 타구에 각속도 전달."""
        b1, b2 = self._setup_half_ball_hit()

        PhysicsEngine()._resolve_ball_collision(b1, b2)

        w_obj = float(np.linalg.norm(b2.angular_velocity))
        assert w_obj > 0.1, (
            f"Throw로 타구에 스핀 전달되어야 함: |ω|={w_obj:.4f}"
        )

    def test_throw_changes_departure_angle(self):
        """Throw에 의해 타구 분리각이 μ=0일 때와 달라야 함."""
        import physics as phys

        # With throw (MU_BALL default)
        b1_t, b2_t = self._setup_half_ball_hit()
        PhysicsEngine()._resolve_ball_collision(b1_t, b2_t)

        # Without throw (MU_BALL = 0)
        saved_mu = phys.MU_BALL
        phys.MU_BALL = 0.0
        try:
            b1_n, b2_n = self._setup_half_ball_hit()
            PhysicsEngine()._resolve_ball_collision(b1_n, b2_n)
        finally:
            phys.MU_BALL = saved_mu

        angle_throw = float(np.arctan2(b2_t.velocity[0], b2_t.velocity[2]))
        angle_no = float(np.arctan2(b2_n.velocity[0], b2_n.velocity[2]))
        diff_deg = abs(np.degrees(angle_throw - angle_no))

        assert diff_deg > 0.1, (
            f"Throw가 분리각 변경해야 함: "
            f"with={np.degrees(angle_throw):.2f}°, "
            f"without={np.degrees(angle_no):.2f}°, diff={diff_deg:.2f}°"
        )

    def test_no_throw_without_friction(self):
        """MU_BALL=0이면 타구에 스핀 전달 없음 (순수 탄성)."""
        import physics as phys

        saved_mu = phys.MU_BALL
        phys.MU_BALL = 0.0
        try:
            b1, b2 = self._setup_half_ball_hit()
            PhysicsEngine()._resolve_ball_collision(b1, b2)
        finally:
            phys.MU_BALL = saved_mu

        w_obj = float(np.linalg.norm(b2.angular_velocity))
        assert w_obj < 1e-6, (
            f"MU_BALL=0에서 타구 스핀 없어야 함: |ω|={w_obj:.6f}"
        )
