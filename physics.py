"""
3D 4-Ball Billiards Custom Physics Engine
Phase 1: Angular Velocity, Friction Model, Collision
"""

import enum
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# ──────────────────────────────────────────────
# Constants (SI units)
# ──────────────────────────────────────────────
BALL_RADIUS: float = 0.03275  # m  (65.5mm regulation ball)
BALL_MASS: float = 0.21  # kg
MU_SLIDE: float = 0.2  # sliding friction coefficient
MU_ROLL: float = 0.015  # rolling friction coefficient
MU_SPIN: float = 0.04  # pivot friction coefficient (wy deceleration)
GRAVITY: float = 9.8  # m/s^2
INERTIA: float = (2 / 5) * BALL_MASS * BALL_RADIUS ** 2  # moment of inertia

# Table dimensions (inner playing surface, m)
TABLE_WIDTH: float = 1.27  # ~50 inches
TABLE_LENGTH: float = 2.54  # ~100 inches

# Cushion restitution
CUSHION_RESTITUTION: float = 0.75
# Ball-ball restitution
BALL_RESTITUTION: float = 0.94

# Numerical thresholds
VELOCITY_THRESHOLD: float = 1e-5
ANGULAR_VELOCITY_THRESHOLD: float = 1e-4
CONTACT_VELOCITY_THRESHOLD: float = 1e-4

# ── Runtime-editable behavior constants ───────────────────────────────────────
# These are read by name every call, so main.py can mutate them live via:
#   import physics as _phys;  _phys.MU_CUSHION = 0.18
MU_CUSHION: float = 0.18            # ball-cushion Coulomb friction coefficient
MU_BALL: float = 0.08               # ball-ball throw Coulomb friction coefficient
SQUIRT_FACTOR: float = -0.02        # cue squirt: velocity deflection per arctan(ox/R)
RAIL_H_OFFSET: float = 0.003        # cushion contact height above ball center (m)
TABLE_BOUNCE_REST: float = 0.35     # table surface restitution for jump landing


class BallState(enum.Enum):
    STATIONARY = 0
    SLIDING = 1
    ROLLING = 2


@dataclass
class Ball:
    """Billiard ball with linear and angular velocity."""
    name: str
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    state: BallState = BallState.STATIONARY
    radius: float = BALL_RADIUS
    mass: float = BALL_MASS
    cushion_hits: int = 0

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)
        self.angular_velocity = np.array(self.angular_velocity, dtype=float)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def is_moving(self) -> bool:
        return (self.speed > VELOCITY_THRESHOLD or
                np.linalg.norm(self.angular_velocity) > ANGULAR_VELOCITY_THRESHOLD)


class PhysicsEngine:
    """Custom billiards physics engine using numpy."""

    def __init__(self, table_width: float = TABLE_WIDTH, table_length: float = TABLE_LENGTH):
        self.table_width = table_width
        self.table_length = table_length
        # Table bounds: x in [-w/2, w/2], z in [-l/2, l/2], y = 0 (table surface)
        self.x_min = -table_width / 2
        self.x_max = table_width / 2
        self.z_min = -table_length / 2
        self.z_max = table_length / 2
        self.events: list = []

    # ──────────────────────────────────────────
    # 3.1 Cue Impact
    # ──────────────────────────────────────────
    @staticmethod
    def apply_cue(ball: Ball, force: float, direction: np.ndarray,
                  offset: np.ndarray = None) -> None:
        """
        Apply cue strike to a ball.

        Args:
            ball: The ball being struck.
            force: Scalar force magnitude.
            direction: Unit vector of cue direction (in xz-plane typically).
            offset: Strike point offset from center [x, y] on the ball face.
                    x > 0 = right english, y > 0 = top (follow), y < 0 = bottom (draw).
        """
        direction = np.array(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)

        # Squirt effect: side-offset deflects velocity direction opposite to English.
        # theta_sq = SQUIRT_FACTOR * arctan(ox / R); rotate direction around Y axis.
        # Torque calculation below still uses the original 'direction' (cue axis).
        vel_direction = direction.copy()
        if offset is not None:
            ox_sq = float(offset[0]) if len(offset) >= 1 else 0.0
            if abs(ox_sq) > 1e-9:
                theta_sq = SQUIRT_FACTOR * np.arctan2(ox_sq, ball.radius)
                cs, sn = np.cos(theta_sq), np.sin(theta_sq)
                vd = direction.copy()
                vel_direction = np.array([vd[0] * cs + vd[2] * sn,
                                          vd[1],
                                          -vd[0] * sn + vd[2] * cs])
                n = np.linalg.norm(vel_direction)
                if n > 1e-9:
                    vel_direction /= n

        # Linear velocity change: delta_v = F / m * vel_direction (squirt-deflected)
        delta_v = (force / ball.mass) * vel_direction
        ball.velocity = ball.velocity + delta_v

        # Angular velocity from offset (uses original cue direction, not squirt-deflected)
        if offset is not None:
            offset = np.array(offset, dtype=float)
            # Convert 2D offset [x, y] to 3D contact point on the ball surface.
            # offset.x = horizontal (left-right), offset.y = vertical (up-down)
            # The contact point relative to the ball center, in the frame where
            # the cue approaches along the direction vector.
            # For a cue pointing along +z direction:
            #   contact_offset_3d = [offset_x, offset_y, -sqrt(R^2 - ox^2 - oy^2)]
            # We apply torque = r x F / I
            ox, oy = offset[0], offset[1]
            r_sq = ball.radius ** 2
            oz_sq = r_sq - ox ** 2 - oy ** 2
            if oz_sq < 0:
                oz_sq = 0.0
            oz = -np.sqrt(oz_sq)

            # Contact point in ball-local coordinates
            # We need to rotate this into world coordinates based on cue direction.
            # Build a 3D orthonormal frame from the actual cue direction vector.
            fwd = direction.copy()  # use true 3D direction (supports Masse/Jump)

            global_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(global_up, fwd)
            right_len = np.linalg.norm(right)
            if right_len > 1e-9:
                right = right / right_len
            else:
                right = np.array([1.0, 0.0, 0.0])  # fallback for near-vertical cue

            local_up = np.cross(fwd, right)  # true local-up perpendicular to cue axis

            # Contact point in world coordinates
            contact_point = right * ox + local_up * oy + fwd * oz

            # Force vector (applied at contact point)
            force_vec = force * direction

            # Torque = r x F
            torque = np.cross(contact_point, force_vec)

            # Angular velocity change = torque / I
            delta_w = torque / INERTIA
            ball.angular_velocity = ball.angular_velocity + delta_w

        ball.state = BallState.SLIDING

    # ──────────────────────────────────────────
    # 3.2 Floor Interaction (Friction)
    # ──────────────────────────────────────────
    @staticmethod
    def _compute_contact_velocity(ball: Ball) -> np.ndarray:
        """Compute horizontal contact velocity at the ball's floor contact point."""
        R = ball.radius
        r_contact = np.array([0.0, -R, 0.0])
        vc = ball.velocity + np.cross(ball.angular_velocity, r_contact)
        vc[1] = 0.0  # only horizontal components
        return vc

    @staticmethod
    def _snap_to_rolling(ball: Ball) -> None:
        """Enforce pure rolling constraint: set angular velocity consistent with velocity.

        Derivation (r_contact = [0, -R, 0]):
          vc = v + (w × r) = 0  →  [vx + wz·R,  0,  vz - wx·R] = 0
          ∴  wx = +vz/R,  wz = -vx/R          (right-hand rule, Y-up, Z-forward)
        Previous sign was inverted → SLIDING→ROLLING transition appeared reversed.
        """
        R = ball.radius
        ball.angular_velocity[0] = ball.velocity[2] / R    # wx = +vz/R
        ball.angular_velocity[2] = -ball.velocity[0] / R   # wz = -vx/R

    def _apply_floor_friction(self, ball: Ball, dt: float) -> None:
        """Apply floor friction forces based on sliding/rolling state."""
        if ball.state == BallState.STATIONARY:
            return

        R = ball.radius
        r_contact = np.array([0.0, -R, 0.0])

        vc = self._compute_contact_velocity(ball)
        vc_mag = np.linalg.norm(vc)

        transitioned_this_frame = False

        if ball.state == BallState.SLIDING or (ball.state != BallState.ROLLING and vc_mag > CONTACT_VELOCITY_THRESHOLD):
            ball.state = BallState.SLIDING

            if vc_mag <= CONTACT_VELOCITY_THRESHOLD:
                # Already at rolling condition — snap
                self._snap_to_rolling(ball)
                ball.state = BallState.ROLLING
                # transitioned_this_frame stays False → full dt rolling friction applied below
            else:
                vc_dir = vc / vc_mag

                # Friction force: opposes contact velocity
                friction_force = -MU_SLIDE * ball.mass * GRAVITY * vc_dir

                # Compute how vc changes this step to detect zero-crossing
                # dv/dt = F/m, dw/dt = (r × F) / I
                accel = friction_force / ball.mass
                torque = np.cross(r_contact, friction_force)
                alpha = torque / INERTIA

                # Predicted new contact velocity
                new_v = ball.velocity + accel * dt
                new_w = ball.angular_velocity + alpha * dt
                new_vc = new_v + np.cross(new_w, r_contact)
                new_vc[1] = 0.0

                # Check if contact velocity changed direction (zero-crossing)
                if np.dot(vc, new_vc) < 0 or np.linalg.norm(new_vc) < CONTACT_VELOCITY_THRESHOLD:
                    # Would overshoot — snap to rolling condition
                    # Solve for t* when vc = 0, then apply rolling for remainder
                    # Approximate: find fraction of dt to reach vc = 0
                    dvc = new_vc - vc
                    dvc_mag = np.linalg.norm(dvc)
                    if dvc_mag > 1e-12:
                        # fraction where |vc + frac * dvc| is minimized ≈ -dot(vc, dvc)/|dvc|^2
                        frac = max(0.0, min(1.0, -np.dot(vc, dvc) / (dvc_mag ** 2)))
                    else:
                        frac = 1.0
                    partial_dt = frac * dt

                    # Apply sliding friction for partial_dt
                    ball.velocity = ball.velocity + accel * partial_dt
                    ball.angular_velocity = ball.angular_velocity + alpha * partial_dt

                    # Snap to rolling
                    self._snap_to_rolling(ball)
                    ball.state = BallState.ROLLING

                    # Apply rolling friction for remaining dt
                    remaining_dt = dt - partial_dt
                    if remaining_dt > 1e-9:
                        self._apply_rolling_friction(ball, remaining_dt)
                    transitioned_this_frame = True  # rolling friction already accounted for
                else:
                    # Normal sliding update
                    ball.velocity = ball.velocity + accel * dt
                    ball.angular_velocity = ball.angular_velocity + alpha * dt

        if ball.state == BallState.ROLLING and not transitioned_this_frame:
            self._apply_rolling_friction(ball, dt)

        # Spin deceleration (y-axis spin / english)
        # Factor 2.5 = 5/2: from angular momentum analysis of pivot friction
        # (equivalent to I = 2/5·mR² with full Coulomb model at contact circle)
        wy = ball.angular_velocity[1]
        if abs(wy) > ANGULAR_VELOCITY_THRESHOLD:
            spin_decel = 2.5 * MU_SPIN * GRAVITY / R
            if wy > 0:
                ball.angular_velocity[1] = max(0.0, wy - spin_decel * dt)
            else:
                ball.angular_velocity[1] = min(0.0, wy + spin_decel * dt)

    def _apply_rolling_friction(self, ball: Ball, dt: float) -> None:
        """Apply rolling friction deceleration."""
        R = ball.radius
        v_horizontal = np.array([ball.velocity[0], 0.0, ball.velocity[2]])
        v_speed = np.linalg.norm(v_horizontal)

        if v_speed > VELOCITY_THRESHOLD:
            v_dir = v_horizontal / v_speed
            friction_decel = MU_ROLL * GRAVITY
            new_speed = max(0.0, v_speed - friction_decel * dt)
            ball.velocity[0] = v_dir[0] * new_speed
            ball.velocity[2] = v_dir[2] * new_speed

            # Maintain rolling constraint
            self._snap_to_rolling(ball)
        else:
            ball.velocity[:] = 0.0
            ball.angular_velocity[0] = 0.0  # wx (rolling constraint)
            ball.angular_velocity[2] = 0.0  # wz (rolling constraint)
            if abs(ball.angular_velocity[1]) <= ANGULAR_VELOCITY_THRESHOLD:
                ball.state = BallState.STATIONARY
                ball.angular_velocity[1] = 0.0
            # else: keep ROLLING so spin decel in _apply_floor_friction continues

    # ──────────────────────────────────────────
    # 3.3 Masse (Curving Force)
    # ──────────────────────────────────────────
    @staticmethod
    def _apply_curve_force(ball: Ball, dt: float) -> None:
        """Apply curving lateral force from vertical spin (masse effect).

        Physical model: wy·R creates a relative sliding speed at the floor contact.
        The Coulomb friction force is perpendicular to the ball velocity (lateral),
        causing the path to curve. Lateral direction = v_hat × y_hat.

        a_masse = (wy · R / (v + 0.10)) · MU_SLIDE · g · lateral

        SLIDING 상태일 때만 적용: ROLLING 상태에서는 접촉점 미끄럼이 없으므로
        wy로 인한 횡방향 마찰력이 발생하지 않음. ROLLING에서 적용 시 v→0 구간에서
        J자형 궤적 꺾임 버그 발생.
        """
        if ball.state != BallState.SLIDING:
            return

        wy = ball.angular_velocity[1]
        if abs(wy) < ANGULAR_VELOCITY_THRESHOLD:
            return

        v_horizontal = np.array([ball.velocity[0], 0.0, ball.velocity[2]])
        v_speed = np.linalg.norm(v_horizontal)
        if v_speed < VELOCITY_THRESHOLD:
            return

        # Lateral direction: v_hat × y_hat = (-vz/v, 0, vx/v)
        v_hat = v_horizontal / v_speed
        lateral = np.array([-v_hat[2], 0.0, v_hat[0]])

        a_mag = wy * BALL_RADIUS / (v_speed + 0.10) * MU_SLIDE * GRAVITY
        ball.velocity[0] += a_mag * lateral[0] * dt
        ball.velocity[2] += a_mag * lateral[2] * dt

    # ──────────────────────────────────────────
    # Ball-Ball Collision
    # ──────────────────────────────────────────
    @staticmethod
    def _check_ball_collision(b1: Ball, b2: Ball) -> bool:
        """Check if two balls are overlapping."""
        diff = b1.position - b2.position
        dist = np.linalg.norm(diff)
        return dist <= (b1.radius + b2.radius)

    def _resolve_ball_collision(self, b1: Ball, b2: Ball) -> None:
        """
        Resolve elastic collision between two balls.
        Transfers velocity along the line of centers.
        """
        diff = b1.position - b2.position
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return

        normal = diff / dist

        # Separate overlapping balls
        overlap = (b1.radius + b2.radius) - dist
        if overlap > 0:
            b1.position = b1.position + normal * (overlap / 2)
            b2.position = b2.position - normal * (overlap / 2)

        # Relative velocity along normal
        rel_vel = b1.velocity - b2.velocity
        vel_along_normal = np.dot(rel_vel, normal)

        # Only resolve if balls are approaching
        if vel_along_normal > 0:
            return

        rel_speed = float(np.linalg.norm(b1.velocity - b2.velocity))
        self.events.append({
            "type": "ball_ball", "ball1": b1.name, "ball2": b2.name,
            "speed": rel_speed,
        })

        # Impulse for equal-mass elastic collision with restitution
        e = BALL_RESTITUTION
        j = -(1 + e) * vel_along_normal / (1 / b1.mass + 1 / b2.mass)

        impulse = j * normal
        b1.velocity = b1.velocity + impulse / b1.mass
        b2.velocity = b2.velocity - impulse / b2.mass

        # Ball-ball throw: Coulomb friction at contact (v2 model).
        # Contact vectors from each ball center to contact point:
        #   r_c1 = -R·n,  r_c2 = +R·n  (n = normal from b2→b1)
        # Slip velocity (tangential surface velocity difference at contact):
        #   v_slip = (v1 + ω1 × r_c1) - (v2 + ω2 × r_c2), tangential component
        # Effective tangential mass (equal solid spheres, I = 2mR²/5):
        #   1/m_eff = 1/m + R²/I + 1/m + R²/I  = 7/m  →  m_eff = m/7
        # Sticking impulse: J_stop = |v_slip_tang| · m_eff
        # Coulomb limit:    J_max  = MU_BALL · j
        # Angular impulse identical for both balls: R·Jt·cross(normal, tang_dir)/I
        r_c1 = -b1.radius * normal
        r_c2 =  b2.radius * normal
        v_surf1 = b1.velocity + np.cross(b1.angular_velocity, r_c1)
        v_surf2 = b2.velocity + np.cross(b2.angular_velocity, r_c2)
        v_slip = v_surf1 - v_surf2
        v_slip_tang = v_slip - np.dot(v_slip, normal) * normal
        tang_mag = float(np.linalg.norm(v_slip_tang))
        if tang_mag > 1e-6:
            tang_dir = v_slip_tang / tang_mag
            m_eff_tang = b1.mass / 7.0
            J_stop = tang_mag * m_eff_tang
            J_max  = MU_BALL * j
            Jt = min(J_stop, J_max)
            tang_impulse = -tang_dir * Jt
            b1.velocity += tang_impulse / b1.mass
            b2.velocity -= tang_impulse / b2.mass
            # Angular impulse = r_c × ΔJ / I; same vector for both balls
            ang_impulse = np.cross(r_c1, tang_impulse)
            b1.angular_velocity += ang_impulse / INERTIA
            b2.angular_velocity += ang_impulse / INERTIA

        # Mark both as sliding after collision
        if b1.speed > VELOCITY_THRESHOLD:
            b1.state = BallState.SLIDING
        if b2.speed > VELOCITY_THRESHOLD:
            b2.state = BallState.SLIDING

    # ──────────────────────────────────────────
    # 3.4 Cushion Collision (with Spin)
    # ──────────────────────────────────────────
    def _check_cushion_collisions(self, ball: Ball) -> None:
        """Check and resolve cushion collisions with Coulomb impulse + 3D contact.

        The cushion contacts the ball at height h = RAIL_H_OFFSET above ball center,
        and R_xy = sqrt(R²-h²) to the side. Contact vectors from ball center:
          Right (x_max): r_c = (+R_xy, +h,    0)
          Left  (x_min): r_c = (-R_xy, +h,    0)
          Far   (z_max): r_c = (  0,   +h, +R_xy)
          Near  (z_min): r_c = (  0,   +h, -R_xy)

        Normal impulse creates extra torque (Δω_z for x-cushions, Δω_x for z-cushions).
        Effective tangential mass m_eff = m/3.5 (unchanged: R_xy²+h²=R²).

        Slip velocities (v_contact tangential, derived from ω × r_c):
          Right: slip_z = vz + ωx·h  − ωy·R_xy
          Left:  slip_z = vz + ωx·h  + ωy·R_xy   ← both terms positive
          Far:   slip_x = vx + ωy·R_xy − ωz·h
          Near:  slip_x = vx − ωy·R_xy − ωz·h

        Friction torques per cushion (Δω = r_c × Jf_vec / I):
          Right: Δωx += h·Jf/I,   Δωy += −R_xy·Jf/I,  Δωz += +h·Jn/I  (normal)
          Left:  Δωx += h·Jf/I,   Δωy += +R_xy·Jf/I,  Δωz += −h·Jn/I  (normal)
          Far:   Δωy += R_xy·Jf/I, Δωz += −h·Jf/I,    Δωx += −h·Jn/I  (normal)
          Near:  Δωy += −R_xy·Jf/I, Δωz += −h·Jf/I,   Δωx += +h·Jn/I  (normal)
        """
        R = ball.radius
        h = RAIL_H_OFFSET
        R_xy = float(np.sqrt(max(0.0, R * R - h * h)))
        m = ball.mass
        e = CUSHION_RESTITUTION
        eff_mass = m / 3.5   # = 1 / (1/m + R²/I) for solid sphere; R_xy²+h²=R² keeps it

        def _fric(J_n, v_slip):
            """Coulomb friction impulse (signed, in tangential direction)."""
            if abs(v_slip) < 1e-9:
                return 0.0
            J_stop = -v_slip * eff_mass
            J_max  = MU_CUSHION * J_n
            return J_stop if abs(J_stop) <= J_max else -float(np.sign(v_slip)) * J_max

        # Right cushion (x_max), r_c = (+R_xy, +h, 0)
        if ball.position[0] + R >= self.x_max and ball.velocity[0] > 0:
            impact_speed = abs(ball.velocity[0])
            ball.position[0] = self.x_max - R
            J_n = (1.0 + e) * impact_speed * m
            ball.velocity[0] = -ball.velocity[0] * e
            ball.state = BallState.SLIDING  # cushion breaks rolling constraint
            ball.angular_velocity[2] +=  h * J_n / INERTIA        # normal torque → Δωz
            v_slip = ball.velocity[2] + ball.angular_velocity[0] * h - ball.angular_velocity[1] * R_xy
            Jf = _fric(J_n, v_slip)
            ball.velocity[2]          += Jf / m
            ball.angular_velocity[0]  +=  h * Jf / INERTIA
            ball.angular_velocity[1]  += -R_xy * Jf / INERTIA
            ball.cushion_hits += 1
            self.events.append({"type": "cushion", "ball": ball.name, "speed": float(impact_speed)})

        # Left cushion (x_min), r_c = (-R_xy, +h, 0)
        if ball.position[0] - R <= self.x_min and ball.velocity[0] < 0:
            impact_speed = abs(ball.velocity[0])
            ball.position[0] = self.x_min + R
            J_n = (1.0 + e) * impact_speed * m
            ball.velocity[0] = -ball.velocity[0] * e
            ball.state = BallState.SLIDING  # cushion breaks rolling constraint
            ball.angular_velocity[2] += -h * J_n / INERTIA        # normal torque → Δωz (opposite)
            v_slip = ball.velocity[2] + ball.angular_velocity[0] * h + ball.angular_velocity[1] * R_xy
            Jf = _fric(J_n, v_slip)
            ball.velocity[2]          += Jf / m
            ball.angular_velocity[0]  +=  h * Jf / INERTIA
            ball.angular_velocity[1]  +=  R_xy * Jf / INERTIA
            ball.cushion_hits += 1
            self.events.append({"type": "cushion", "ball": ball.name, "speed": float(impact_speed)})

        # Far cushion (z_max), r_c = (0, +h, +R_xy)
        if ball.position[2] + R >= self.z_max and ball.velocity[2] > 0:
            impact_speed = abs(ball.velocity[2])
            ball.position[2] = self.z_max - R
            J_n = (1.0 + e) * impact_speed * m
            ball.velocity[2] = -ball.velocity[2] * e
            ball.state = BallState.SLIDING  # cushion breaks rolling constraint
            ball.angular_velocity[0] += -h * J_n / INERTIA        # normal torque → Δωx
            v_slip = ball.velocity[0] + ball.angular_velocity[1] * R_xy - ball.angular_velocity[2] * h
            Jf = _fric(J_n, v_slip)
            ball.velocity[0]          += Jf / m
            ball.angular_velocity[1]  +=  R_xy * Jf / INERTIA
            ball.angular_velocity[2]  += -h * Jf / INERTIA
            ball.cushion_hits += 1
            self.events.append({"type": "cushion", "ball": ball.name, "speed": float(impact_speed)})

        # Near cushion (z_min), r_c = (0, +h, -R_xy)
        if ball.position[2] - R <= self.z_min and ball.velocity[2] < 0:
            impact_speed = abs(ball.velocity[2])
            ball.position[2] = self.z_min + R
            J_n = (1.0 + e) * impact_speed * m
            ball.velocity[2] = -ball.velocity[2] * e
            ball.state = BallState.SLIDING  # cushion breaks rolling constraint
            ball.angular_velocity[0] +=  h * J_n / INERTIA        # normal torque → Δωx (opposite)
            v_slip = ball.velocity[0] - ball.angular_velocity[1] * R_xy - ball.angular_velocity[2] * h
            Jf = _fric(J_n, v_slip)
            ball.velocity[0]          += Jf / m
            ball.angular_velocity[1]  += -R_xy * Jf / INERTIA
            ball.angular_velocity[2]  += -h * Jf / INERTIA
            ball.cushion_hits += 1
            self.events.append({"type": "cushion", "ball": ball.name, "speed": float(impact_speed)})

    # ──────────────────────────────────────────
    # Main Update Loop
    # ──────────────────────────────────────────
    # ──────────────────────────────────────────
    # 3.5 Vertical (jump/bounce) helpers
    # ──────────────────────────────────────────
    @staticmethod
    def _is_airborne(ball: Ball) -> bool:
        """True when ball is above the table or has upward velocity."""
        return ball.position[1] > 1e-5 or ball.velocity[1] > 1e-5

    def _apply_vertical(self, ball: Ball, dt: float) -> None:
        """Apply gravity and table-surface bounce for airborne or down-hit balls."""
        has_vertical = abs(ball.velocity[1]) > 1e-6 or ball.position[1] > 1e-6
        if not has_vertical:
            ball.position[1] = 0.0
            return
        # Gravity
        ball.velocity[1] -= GRAVITY * dt
        if ball.position[1] < 0.0:
            ball.position[1] = 0.0
            if ball.velocity[1] < -0.05:
                ball.velocity[1] = -ball.velocity[1] * TABLE_BOUNCE_REST
            else:
                ball.velocity[1] = 0.0

    def update(self, balls: List[Ball], dt: float) -> None:
        """Advance physics simulation by dt seconds."""
        self.events.clear()
        # Apply floor friction and curve force (skip when airborne)
        for ball in balls:
            if ball.state != BallState.STATIONARY:
                if not self._is_airborne(ball):
                    self._apply_floor_friction(ball, dt)
                    self._apply_curve_force(ball, dt)

        # Move balls
        for ball in balls:
            if ball.state != BallState.STATIONARY:
                ball.position = ball.position + ball.velocity * dt
                self._apply_vertical(ball, dt)

        # Check ball-ball collisions
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                if self._check_ball_collision(balls[i], balls[j]):
                    self._resolve_ball_collision(balls[i], balls[j])

        # Check cushion collisions
        for ball in balls:
            self._check_cushion_collisions(ball)

        # Final state check
        for ball in balls:
            if not ball.is_moving():
                ball.state = BallState.STATIONARY
                ball.velocity[:] = 0.0
                ball.angular_velocity[:] = 0.0

    def simulate(self, balls: List[Ball], dt: float = 0.001,
                 max_time: float = 10.0) -> float:
        """
        Run simulation until all balls stop or max_time is reached.

        Returns:
            Elapsed time in seconds.
        """
        t = 0.0
        while t < max_time:
            self.update(balls, dt)
            t += dt
            if all(not b.is_moving() for b in balls):
                break
        return t
