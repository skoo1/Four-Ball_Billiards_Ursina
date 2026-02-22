"""
BilliardsController — Layer 2 (Game Logic)

Owns all game state, physics state, and business logic.
Communicates with Layer 3 (main.py / Ursina renderer) via two queues:
  - pending_events  : rendering commands (spawn_ball, clear_balls, show_result, …)
  - physics_events  : collision events for sound playback

Layer 3 calls:
  ctrl.step(dt)               — advance physics + state machine each frame
  ctrl.pending_events         — list of dicts to consume and act on
  ctrl.physics_events         — list of collision dicts for sounds
  ctrl.<state properties>     — read-only references to aim_angle, mode, etc.
"""

import math
import json
import csv
import os
import random
from pathlib import Path
import numpy as np

from physics import PhysicsEngine, Ball, BallState, BALL_RADIUS
import physics as _phys


# ── Fast ball copy (used by simulate_shot — 3× faster than deepcopy) ─────────
def _copy_ball(b: "Ball") -> "Ball":
    nb = Ball(b.name)
    nb.position         = b.position.copy()
    nb.velocity         = b.velocity.copy()
    nb.angular_velocity = b.angular_velocity.copy()
    nb.state            = b.state
    nb.radius           = b.radius
    nb.mass             = b.mass
    nb.cushion_hits     = b.cushion_hits
    return nb


# ── Default info-bar message ───────────────────────────────────────────────────
DEFAULT_INFO_MSG = (
    "[1-5] Scenario  [G] Game  [T] Practice  [R] Reset  "
    "[P] Params([/])  [Shift+S] Script  [Shift+A] Adv"
)


class BilliardsController:
    """Layer 2: game-state machine + physics orchestration."""

    # ── Class-level constants ─────────────────────────────────────────────────
    SIM_DT            = 0.001
    SIM_SUBSTEPS      = 4
    WIN_SCORE         = 10
    AI_THINK_DELAY    = 1.5
    AIM_ROTATE_SPEED  = 60.0
    AIM_ELEVATION_SPEED = 45.0
    MAX_ELEVATION     = 45.0
    POWER_CHARGE_RATE = 0.7
    MAX_POWER         = 1.5
    TRAIL_MAX_POINTS  = 200

    GAME_POSITIONS = {
        "white":  np.array([-0.25, 0.0,  0.0]),
        "yellow": np.array([ 0.25, 0.0,  0.0]),
        "red1":   np.array([ 0.0,  0.0,  0.5]),
        "red2":   np.array([ 0.0,  0.0, -0.5]),
    }

    _STROKE_DEFAULTS = {
        "aim_delay":      0.6,
        "backswing_dist": 0.10,
        "backswing_time": 0.35,
        "strike_time":    0.15,
    }

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(self):
        # Physics
        self.physics_balls: list[Ball] = []
        self.engine = PhysicsEngine()
        self._sim_engine = PhysicsEngine()   # reused for headless simulate_shot()

        # Game state
        self.mode = "idle"          # "idle"|"aiming"|"running"|"ai_calculating"
        self.game_mode    = False
        self.practice_mode = False
        self.player_turn  = True
        self.player_score = 0
        self.ai_score     = 0

        # Aiming
        self.aim_angle      = 0.0
        self.aim_elevation  = 0.0
        self.aim_power      = 0.0
        self.power_charging = False
        self.hit_offset     = [0.0, 0.0]
        self.shot_touched: set = set()

        # AI
        self.ai_think_timer = 0.0
        self.ai_shot_data: dict = {}

        # Script state
        self._script_state      = "idle"
        self._script_timer      = 0.0
        self._script_params: dict = {}
        self._last_script_path  = ""
        self._last_script: dict = {}
        self._script_picker_active = False
        self._script_picker_files: list = []

        # Advanced panel state
        self._adv_active         = False
        self._adv_activate_field = False
        self._adv_wrapping       = False

        # Session recording
        self._session_recording = False
        self._session_rows: list = []
        self._session_header: list = []
        self._session_file  = ""
        self._session_t     = 0.0

        # Trail data (positions only; entities stay in L3)
        self.trail_positions: dict[str, list] = {}

        # Status / info messages (L3 reads these to update text entities)
        self.status_msg = ""
        self.info_msg   = DEFAULT_INFO_MSG

        # Event queues
        self.pending_events: list[dict] = []   # L3 rendering commands
        self.physics_events: list[dict] = []   # collision sounds

    # ──────────────────────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, dt_frame: float) -> None:
        """Advance physics + state machine. Called every frame by L3."""
        if self.mode != "running":
            return

        self.physics_events.clear()

        for _ in range(self.SIM_SUBSTEPS):
            self.engine.update(self.physics_balls, self.SIM_DT)
            for ev in self.engine.events:
                self.physics_events.append(ev)
                # Track ball-ball touches for game scoring
                if ev["type"] == "ball_ball" and self.game_mode:
                    active = "white" if self.player_turn else "yellow"
                    if ev["ball1"] == active:
                        self.shot_touched.add(ev["ball2"])
                    elif ev["ball2"] == active:
                        self.shot_touched.add(ev["ball1"])

        if self._session_recording:
            self._session_record_frame()

        all_stopped = all(not b.is_moving() for b in self.physics_balls)
        if all_stopped:
            self.mode = "idle"
            self.ai_think_timer = 0.0
            self._on_shot_finished()

    def _on_shot_finished(self) -> None:
        if self._session_recording:
            saved = self._session_file
            self._session_write_csv()
            self.pending_events.append({"type": "session_saved", "file": saved})

        if self.game_mode:
            self._check_shot_result()
            self.pending_events.append({"type": "update_game_ui"})
            if self.game_mode:    # may have ended in _check_shot_result
                if self.player_turn:
                    self.status_msg = "Your turn! Aim with arrows, hold Space."
                else:
                    self.status_msg = "AI thinking..."
        elif self.practice_mode:
            self.status_msg = "Practice: aim with arrows, hold Space to fire."
        else:
            self.status_msg = "Stopped. Left-drag cue ball to shoot."

    # ──────────────────────────────────────────────────────────────────────────
    # Ball management
    # ──────────────────────────────────────────────────────────────────────────

    def clear_balls(self) -> None:
        """Clear physics state and emit clear_balls + clear_trails events."""
        self.physics_balls.clear()
        self.trail_positions.clear()
        self.mode = "idle"
        self.pending_events.append({"type": "clear_balls"})

    def load_scenario(self, scenario_fn, label: str) -> None:
        """Load a shot-preset scenario (keys 1-5)."""
        if self.mode == "running":
            return
        self.clear_balls()

        result = scenario_fn(run=False)
        self.physics_balls = result["balls"]
        self.engine        = result["engine"]

        for pb in self.physics_balls:
            self.trail_positions[pb.name] = []
            self.pending_events.append({"type": "spawn_ball", "ball": pb})

        self.mode = "running"
        self.info_msg   = f"Scenario {label}"
        self.status_msg = "Running..."

    # ──────────────────────────────────────────────────────────────────────────
    # Game / practice mode start
    # ──────────────────────────────────────────────────────────────────────────

    def start_game(self) -> None:
        if self.mode == "running":
            return
        self.clear_balls()
        self.pending_events.append({"type": "clear_game_visuals"})

        self.game_mode     = True
        self.practice_mode = False
        self.player_turn   = True
        self.player_score  = 0
        self.ai_score      = 0
        self.aim_angle = self.aim_power = self.aim_elevation = 0.0
        self.power_charging = False
        self.shot_touched   = set()
        self.engine = PhysicsEngine()

        for name, pos in self.GAME_POSITIONS.items():
            b = Ball(name, position=pos.copy())
            self.physics_balls.append(b)
            self.trail_positions[b.name] = []
            self.pending_events.append({"type": "spawn_ball", "ball": b})

        self.mode = "idle"
        self.pending_events.append({"type": "init_game_ui"})
        self.pending_events.append({"type": "update_game_ui"})
        self.info_msg   = (
            "[G] New game  [R] Quit  LR=aim  UD=elev  WASD=hit  Shift=fine  Space=fire"
        )
        self.status_msg = "Your turn! LR arrows=aim, UD=elevation, hold Space=charge."

    def start_practice(self) -> None:
        """Start practice mode: all four balls, no scoring."""
        if self.mode == "running":
            return
        self.clear_balls()
        self.pending_events.append({"type": "clear_game_visuals"})
        self.pending_events.append({"type": "destroy_game_ui"})

        self.practice_mode  = True
        self.game_mode      = False
        self.aim_angle = self.aim_power = self.aim_elevation = 0.0
        self.power_charging = False
        self.engine = PhysicsEngine()

        for name, pos in self.GAME_POSITIONS.items():
            b = Ball(name, position=pos.copy())
            self.physics_balls.append(b)
            self.trail_positions[b.name] = []
            self.pending_events.append({"type": "spawn_ball", "ball": b})

        self.mode = "idle"
        self.pending_events.append({"type": "init_game_ui"})
        self.pending_events.append({"type": "hide_game_score_ui"})
        self.info_msg   = (
            "[T] Exit practice  [R] Reset  LR=aim  UD=elev  WASD=hit  Shift=fine  Space=fire"
        )
        self.status_msg = "Practice: aim with arrows, hold Space to charge, release to fire."

    def toggle_practice(self) -> None:
        if self.game_mode or self.mode == "running":
            return
        if self.practice_mode:
            self.clear_balls()
            self.practice_mode = False
            self.pending_events.append({"type": "clear_game_visuals"})
            self.pending_events.append({"type": "destroy_game_ui"})
            self.info_msg   = DEFAULT_INFO_MSG
            self.status_msg = ""
        else:
            self.start_practice()

    def reset_game(self) -> None:
        """Handle R key: quit game mode, reset practice, or clear balls."""
        if self.mode == "running":
            return
        if self.practice_mode:
            # Reset → restart practice from scratch
            self.clear_balls()
            self.pending_events.append({"type": "clear_game_visuals"})
            self.start_practice()
        elif self.game_mode:
            self.clear_balls()
            self.game_mode = False
            self.pending_events.append({"type": "clear_game_visuals"})
            self.pending_events.append({"type": "destroy_game_ui"})
            self.info_msg   = DEFAULT_INFO_MSG
            self.status_msg = ""
        else:
            self.clear_balls()
            self.info_msg   = DEFAULT_INFO_MSG
            self.status_msg = ""

    # ──────────────────────────────────────────────────────────────────────────
    # Shooting
    # ──────────────────────────────────────────────────────────────────────────

    def fire_shot(self, direction, force: float, offset=None) -> None:
        """Fire cue ball (mouse-drag manual shot in non-game mode)."""
        ball = next((b for b in self.physics_balls if b.name == "white"), None)
        if ball is None:
            return
        ball.velocity[:] = 0.0
        ball.angular_velocity[:] = 0.0
        ball.state = BallState.STATIONARY
        self.engine.apply_cue(ball, force=force, direction=np.asarray(direction, dtype=float),
                              offset=offset)
        for n in self.trail_positions:
            self.trail_positions[n] = []
        self.mode = "running"
        self.pending_events.append({"type": "clear_aim"})
        self.status_msg = "Running..."

    def fire_game_shot(self) -> None:
        """Fire white ball in game/practice mode (keyboard aiming)."""
        ball = next((b for b in self.physics_balls if b.name == "white"), None)
        if not ball:
            return
        angle_rad = math.radians(self.aim_angle)
        elev_rad  = math.radians(self.aim_elevation)
        cos_e = math.cos(elev_rad)
        direction = np.array([
            math.sin(angle_rad) * cos_e,
            math.sin(elev_rad),
            math.cos(angle_rad) * cos_e,
        ])
        force = max(self.aim_power * self.MAX_POWER, 0.05)
        offset = None
        if abs(self.hit_offset[0]) > 0.001 or abs(self.hit_offset[1]) > 0.001:
            offset = [self.hit_offset[0], self.hit_offset[1]]

        ball.velocity[:] = 0
        ball.angular_velocity[:] = 0
        ball.state = BallState.STATIONARY
        self.engine.apply_cue(ball, force=force, direction=direction, offset=offset)

        self.shot_touched   = set()
        self.aim_power      = 0.0
        self.aim_elevation  = 0.0
        self.power_charging = False
        for n in self.trail_positions:
            self.trail_positions[n] = []

        self.mode = "running"
        self.pending_events.append({"type": "clear_game_visuals"})
        self.pending_events.append({"type": "update_power_bar"})
        self.status_msg = "Running..."

    # ──────────────────────────────────────────────────────────────────────────
    # Per-frame aiming + AI updates (called by L3 update)
    # ──────────────────────────────────────────────────────────────────────────

    def update_aim(self, dt: float,
                   rotate_left: bool, rotate_right: bool,
                   elev_up: bool, elev_down: bool,
                   fine: bool = False) -> None:
        """Update aim state from held keys. L3 passes key states."""
        speed_mul = 0.25 if fine else 1.0
        rot_spd = self.AIM_ROTATE_SPEED  * speed_mul * dt
        elv_spd = self.AIM_ELEVATION_SPEED * speed_mul * dt

        if rotate_left:
            self.aim_angle = (self.aim_angle - rot_spd) % 360
        if rotate_right:
            self.aim_angle = (self.aim_angle + rot_spd) % 360
        if elev_up:
            self.aim_elevation = min(self.MAX_ELEVATION, self.aim_elevation + elv_spd)
        if elev_down:
            self.aim_elevation = max(-self.MAX_ELEVATION, self.aim_elevation - elv_spd)

        if self.power_charging:
            self.aim_power = min(1.0, self.aim_power + self.POWER_CHARGE_RATE * dt)

    def update_ai_turn(self, dt: float) -> None:
        """Tick AI decision making. Called when game_mode and not player_turn."""
        if self.mode == "idle" and self.ai_think_timer == 0.0:
            self.status_msg = "AI thinking..."
            self.ai_shot_data = self._ai_calculate_shot()
            self.ai_think_timer = self.AI_THINK_DELAY
            self.mode = "ai_calculating"

        if self.mode == "ai_calculating":
            self.ai_think_timer -= dt
            if self.ai_think_timer <= 0.0:
                self.ai_think_timer = 0.0
                self._run_ai_shot()

    def tick_script(self, dt: float) -> dict | None:
        """Advance script cue animation. Returns render state dict or None."""
        if self._script_state == "idle":
            return None

        params   = self._script_params
        profile  = params.get("_profile", self._STROKE_DEFAULTS.copy())
        ball_name = params.get("ball", "white")
        ball     = next((b for b in self.physics_balls if b.name == ball_name), None)
        az_deg   = float(params.get("azimuth",   0.0))
        el_deg   = float(params.get("elevation", 0.0))

        if self._script_state == "aiming":
            self._script_timer -= dt
            if self._script_timer <= 0:
                self._script_timer = profile["backswing_time"]
                self._script_state = "backswing"
                self.status_msg = "Script: backswing..."
            return {
                "ball": ball, "az_deg": az_deg, "el_deg": el_deg,
                "dist_back": profile["backswing_dist"] * 0.5, "show_aim": True,
            }

        if self._script_state == "backswing":
            bt   = profile["backswing_time"]
            frac = max(0.0, min(1.0, 1.0 - self._script_timer / bt)) if bt > 0 else 1.0
            self._script_timer -= dt
            if self._script_timer <= 0:
                self._script_timer = profile["strike_time"]
                self._script_state = "strike"
                self.status_msg = "Script: striking..."
            return {
                "ball": ball, "az_deg": az_deg, "el_deg": el_deg,
                "dist_back": profile["backswing_dist"] * frac, "show_aim": True,
            }

        if self._script_state == "strike":
            st   = profile["strike_time"]
            frac = max(0.0, min(1.0, self._script_timer / st)) if st > 0 else 0.0
            self._script_timer -= dt
            if self._script_timer <= 0:
                self.fire_script_impulse()
                return None   # impulse fired; _script_state already "idle"
            return {
                "ball": ball, "az_deg": az_deg, "el_deg": el_deg,
                "dist_back": profile["backswing_dist"] * frac, "show_aim": False,
            }

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # AI
    # ──────────────────────────────────────────────────────────────────────────

    def _ai_simulate_shot(self, positions_snap: dict, angle_deg: float,
                          power: float) -> float:
        """Simulate one candidate AI angle. Returns score 1.0/0.5/0.0/-1.0."""
        sim_balls = [Ball(n, position=p.copy()) for n, p in positions_snap.items()]
        opp = next(b for b in sim_balls if b.name == "yellow")
        angle_rad = math.radians(angle_deg)
        direction = np.array([math.sin(angle_rad), 0.0, math.cos(angle_rad)])
        sim_engine = PhysicsEngine()
        sim_engine.apply_cue(opp, force=power * self.MAX_POWER, direction=direction)

        touched: set = set()
        foul = False
        t = 0.0
        while t < 3.0 and not all(not b.is_moving() for b in sim_balls):
            sim_engine.update(sim_balls, 0.01)
            t += 0.01
            for evt in sim_engine.events:
                if evt["type"] == "ball_ball":
                    if evt["ball1"] == "yellow":
                        other = evt["ball2"]
                    elif evt["ball2"] == "yellow":
                        other = evt["ball1"]
                    else:
                        other = None
                    if other:
                        if other == "white":
                            foul = True
                            break
                        else:
                            touched.add(other)
            if foul:
                break

        if foul:
            return -1.0
        hits = len({"red1", "red2"} & touched)
        return 1.0 if hits == 2 else (0.5 if hits == 1 else 0.0)

    def _ai_calculate_shot(self) -> dict:
        """Two-phase angle scan. Returns {"angle": float, "power": float}."""
        snap  = {b.name: b.position.copy() for b in self.physics_balls}
        power = 0.6
        best_score, best_angle = -999.0, 0.0

        for i in range(36):
            a = i * 10.0
            s = self._ai_simulate_shot(snap, a, power)
            if s > best_score:
                best_score, best_angle = s, a

        for i in range(21):
            a = best_angle - 10 + i
            s = self._ai_simulate_shot(snap, a, power)
            if s > best_score:
                best_score, best_angle = s, a

        rng  = random.Random()
        final_angle = best_angle + rng.gauss(0, 6.0)
        final_power = max(0.3, min(0.9, power + rng.gauss(0, 0.05)))
        return {"angle": final_angle, "power": final_power}

    def _run_ai_shot(self) -> None:
        ball = next((b for b in self.physics_balls if b.name == "yellow"), None)
        if not ball:
            return
        angle_rad = math.radians(self.ai_shot_data["angle"])
        direction = np.array([math.sin(angle_rad), 0.0, math.cos(angle_rad)])
        ball.velocity[:] = 0
        ball.angular_velocity[:] = 0
        ball.state = BallState.STATIONARY
        self.engine.apply_cue(
            ball, force=self.ai_shot_data["power"] * self.MAX_POWER, direction=direction
        )
        self.shot_touched = set()
        for n in self.trail_positions:
            self.trail_positions[n] = []
        self.mode = "running"
        self.pending_events.append({"type": "clear_game_visuals"})
        self.status_msg = "AI shooting..."

    # ──────────────────────────────────────────────────────────────────────────
    # Game rules
    # ──────────────────────────────────────────────────────────────────────────

    def _check_shot_result(self) -> None:
        t = self.shot_touched
        if self.player_turn:
            if "yellow" in t:
                self.pending_events.append({"type": "show_result", "msg": "Foul!", "color_name": "red"})
                self.player_turn = False
            elif "red1" in t and "red2" in t:
                self.player_score += 1
                self.pending_events.append({
                    "type": "show_result",
                    "msg":  f"Score! Player: {self.player_score}",
                    "color_name": "green",
                })
                if self.player_score >= self.WIN_SCORE:
                    self._end_game("Player Wins!")
                    return
            else:
                self.pending_events.append({"type": "show_result", "msg": "Miss", "color_name": "orange"})
                self.player_turn = False
        else:
            if "white" in t:
                self.pending_events.append({"type": "show_result", "msg": "AI Foul!", "color_name": "cyan"})
                self.player_turn = True
            elif "red1" in t and "red2" in t:
                self.ai_score += 1
                self.pending_events.append({
                    "type": "show_result",
                    "msg":  f"AI Score! AI: {self.ai_score}",
                    "color_name": "red",
                })
                if self.ai_score >= self.WIN_SCORE:
                    self._end_game("AI Wins!")
                    return
            else:
                self.pending_events.append({"type": "show_result", "msg": "AI Miss", "color_name": "gray"})
                self.player_turn = True
        self.shot_touched = set()

    def _end_game(self, msg: str) -> None:
        self.game_mode = False
        self.pending_events.append({"type": "show_result", "msg": msg, "color_name": "yellow"})
        self.info_msg = f"{msg}  [G] Play again  [1-5] Scenarios  [T] Practice"

    # ──────────────────────────────────────────────────────────────────────────
    # Advanced Command Panel
    # ──────────────────────────────────────────────────────────────────────────

    def get_state_json(self) -> str:
        """Return current ball state as compact single-line set-command JSON."""
        balls = {}
        for b in self.physics_balls:
            balls[b.name] = {
                "pos":  [round(float(b.position[0]), 4), round(float(b.position[2]), 4)],
                "spin": [round(float(b.angular_velocity[0]), 3),
                         round(float(b.angular_velocity[1]), 3),
                         round(float(b.angular_velocity[2]), 3)],
            }
        return json.dumps({"cmd": "set", "balls": balls}, separators=(',', ':'))

    def execute_command(self, text: str) -> None:
        """Parse a JSON command string and dispatch to handlers."""
        if not text:
            print("[ADV] execute_command: empty text")
            return
        text = text.replace('\r', '')
        print(f"[ADV] execute_command: parsing JSON len={len(text)}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            print(f"[ADV] JSON parse error: {exc}")
            self.status_msg = f"JSON error: {exc}"
            return
        cmd = str(data.get("cmd", "")).lower().strip()
        print(f"[ADV] cmd={cmd}")
        if cmd == "set":
            self._adv_cmd_set(data)
        elif cmd == "shot":
            self._adv_cmd_shot(data)
        elif cmd == "save":
            self._adv_cmd_save(data)
        elif cmd == "load":
            self._adv_cmd_load(data)
        else:
            self.status_msg = f"Unknown cmd '{cmd}'. Use set/shot/save/load."

    def _adv_cmd_set(self, data: dict) -> None:
        """set: update ball positions/spins OR physics params."""
        params_data = data.get("params")
        if params_data is not None:
            self._adv_cmd_params(params_data)
            return

        balls_data = data.get("balls", {})
        if not balls_data:
            self.status_msg = "set: 'balls' or 'params' field required."
            return

        existing = {b.name: b for b in self.physics_balls}
        print(f"[ADV] _adv_cmd_set: existing={list(existing.keys())}  keys={list(balls_data.keys())}")
        updated = []

        for name, bd in balls_data.items():
            pos = bd.get("pos")
            if pos is None:
                continue
            if len(pos) == 2:
                x, y, z = float(pos[0]), 0.0, float(pos[1])
            else:
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

            vel    = bd.get("vel", bd.get("velocity", [0.0, 0.0, 0.0]))
            angvel = bd.get("angvel", bd.get("spin", [0.0, 0.0, 0.0]))
            vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
            wx, wy, wz = float(angvel[0]), float(angvel[1]), float(angvel[2])
            try:
                ball_state = BallState[bd.get("state", "STATIONARY")]
            except KeyError:
                ball_state = BallState.STATIONARY

            if name in existing:
                b = existing[name]
                b.position        = np.array([x, y, z])
                b.velocity        = np.array([vx, vy, vz])
                b.angular_velocity = np.array([wx, wy, wz])
                b.state           = ball_state
                self.trail_positions[name] = []
                print(f"[ADV]   update {name} -> ({x:.3f},{z:.3f})")
            else:
                b = Ball(name, position=np.array([x, y, z]))
                b.velocity        = np.array([vx, vy, vz])
                b.angular_velocity = np.array([wx, wy, wz])
                b.state           = ball_state
                self.physics_balls.append(b)
                self.trail_positions[name] = []
                self.pending_events.append({"type": "spawn_ball", "ball": b})
                print(f"[ADV]   spawn NEW {name} -> ({x:.3f},{z:.3f})")
            updated.append(name)

        print(f"[ADV] done. updated={updated}  mode={self.mode}")
        self.status_msg = f"set: {updated} updated."
        self._adv_active = False
        self._adv_activate_field = False
        self.pending_events.append({"type": "adv_close"})

    def _adv_cmd_shot(self, data: dict) -> None:
        """shot: fire a ball with azimuth/elevation/tip/power parameters."""
        ball_name = str(data.get("ball", "white"))
        ball = next((b for b in self.physics_balls if b.name == ball_name), None)
        if ball is None:
            self.status_msg = f"shot: ball '{ball_name}' not found."
            return

        az = math.radians(float(data.get("azimuth",   0.0)))
        el = math.radians(float(data.get("elevation", 0.0)))
        direction = np.array([
            math.sin(az) * math.cos(el),
            -math.sin(el),
            math.cos(az) * math.cos(el),
        ])
        n = float(np.linalg.norm(direction))
        if n > 1e-9:
            direction /= n

        tip = data.get("tip", [0.0, 0.0])
        R   = ball.radius
        tx, ty = float(tip[0]) * R, float(tip[1]) * R
        offset = [tx, ty] if (abs(tx) > 1e-9 or abs(ty) > 1e-9) else None

        power_pct = float(data.get("power", 100.0))
        force     = (power_pct / 100.0) * self.MAX_POWER

        ball.velocity[:] = 0.0
        ball.angular_velocity[:] = 0.0
        ball.state = BallState.STATIONARY
        self.engine.apply_cue(ball, force=force, direction=direction, offset=offset)

        for nm in self.trail_positions:
            self.trail_positions[nm] = []

        # Session recording
        record_opt = data.get("record", False)
        if record_opt:
            from datetime import datetime
            self._session_recording = True
            self._session_rows      = []
            self._session_t         = 0.0
            self._session_header    = self._session_make_header(self.physics_balls)
            if isinstance(record_opt, str):
                fname = record_opt if record_opt.endswith(".csv") else record_opt + ".csv"
            else:
                fname = datetime.now().strftime("%H%M%S") + ".csv"
            self._session_file = fname
            print(f"[REC] Recording started → {fname}  balls={[b.name for b in self.physics_balls]}")
        else:
            self._session_recording = False

        self.mode = "running"
        self.status_msg = (
            f"shot: {ball_name}  az={data.get('azimuth', 0):.1f}°  "
            f"el={data.get('elevation', 0):.1f}°  "
            f"tip=[{float(tip[0]):.2f},{float(tip[1]):.2f}]  "
            f"power={power_pct:.0f}%"
        )
        self._adv_active = False
        self._adv_activate_field = False
        self.pending_events.append({"type": "adv_close"})

    def _adv_cmd_save(self, data: dict) -> None:
        """save: write all ball physical states to a JSON file."""
        file_opt = data.get("file", "")
        if not file_opt:
            from datetime import datetime
            fname = datetime.now().strftime("%H%M%S") + "_state.json"
        else:
            fname = file_opt if file_opt.endswith(".json") else file_opt + ".json"

        balls_data = {}
        for b in self.physics_balls:
            balls_data[b.name] = {
                "pos":    b.position.tolist(),
                "vel":    b.velocity.tolist(),
                "angvel": b.angular_velocity.tolist(),
                "state":  b.state.name,
            }
        payload = {"cmd": "set", "balls": balls_data}
        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[ADV] save → {fname}")
            self.status_msg = f"Saved → {fname}"
        except Exception as e:
            self.status_msg = f"Save error: {e}"

        self._adv_active = False
        self._adv_activate_field = False
        self.pending_events.append({"type": "adv_close"})

    def _adv_cmd_load(self, data: dict) -> None:
        """load: restore ball states from a JSON file saved by 'save' command."""
        file_opt = data.get("file", "")
        if not file_opt:
            self.status_msg = "load: 'file' field required."
            return
        fname = file_opt if file_opt.endswith(".json") else file_opt + ".json"
        try:
            with open(fname, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            print(f"[ADV] load ← {fname}")
            self._adv_cmd_set(loaded)
        except FileNotFoundError:
            self.status_msg = f"load: not found: {fname}"
        except Exception as e:
            self.status_msg = f"Load error: {e}"

    def _adv_cmd_params(self, params: dict) -> None:
        """set params: update physics module constants by name."""
        # Allowed params (mirrors PhysicsParamsEditor.PARAMS in L3)
        allowed = {
            "MU_SLIDE", "MU_ROLL", "MU_SPIN", "CUSHION_RESTITUTION",
            "BALL_RESTITUTION", "GRAVITY", "MU_CUSHION", "MU_BALL",
            "SQUIRT_FACTOR", "RAIL_H_OFFSET", "TABLE_BOUNCE_REST",
        }
        updated, skipped = [], []
        for k, v in params.items():
            if k in allowed and hasattr(_phys, k):
                try:
                    setattr(_phys, k, float(v))
                    updated.append(f"{k}={float(v):.4g}")
                except Exception as e:
                    print(f"[ADV] setattr {k} failed: {e}")
                    skipped.append(k)
            else:
                skipped.append(k)

        # Notify L3 to refresh PhysicsParamsEditor display
        self.pending_events.append({
            "type": "refresh_params_editor",
            "params": list(params.keys()),
        })
        msg = f"params: set {updated}"
        if skipped:
            msg += f"  (unknown: {skipped})"
        print(f"[ADV] {msg}")
        self.status_msg = msg

        self._adv_active = False
        self._adv_activate_field = False
        self.pending_events.append({"type": "adv_close"})

    # ──────────────────────────────────────────────────────────────────────────
    # Script system
    # ──────────────────────────────────────────────────────────────────────────

    def collect_script_files(self) -> list:
        """Return sorted list of .py files from scripts/ dir + shot_script.py."""
        files = []
        scripts_dir = Path("scripts")
        if scripts_dir.is_dir():
            files.extend(sorted(scripts_dir.glob("*.py")))
        local = Path("shot_script.py")
        if local.exists():
            files.insert(0, local)
        return files

    def execute_script(self, script: dict) -> None:
        """Execute a billiard script dict (setup + optional shot)."""
        if self.mode == "running":
            return
        self._last_script = script

        # Exit any active mode cleanly
        if self.game_mode or self.practice_mode:
            self.pending_events.append({"type": "clear_game_visuals"})
            self.pending_events.append({"type": "destroy_game_ui"})
            self.game_mode     = False
            self.practice_mode = False

        self.clear_balls()
        self.pending_events.append({"type": "clear_script_cue"})
        self.engine = PhysicsEngine()

        setup = script.get("setup", {})
        for name, pos in setup.items():
            x, z = float(pos[0]), float(pos[1])
            b = Ball(name, position=np.array([x, 0.0, z]))
            self.physics_balls.append(b)
            self.trail_positions[b.name] = []
            self.pending_events.append({"type": "spawn_ball", "ball": b})

        self.mode = "idle"
        self.info_msg   = DEFAULT_INFO_MSG
        self.status_msg = f"Script: {len(setup)} ball(s) placed."

        shot = script.get("shot")
        if shot is None:
            return

        raw     = shot.get("stroke_profile", {})
        profile = self._STROKE_DEFAULTS.copy()
        if isinstance(raw, dict):
            profile.update(raw)

        self._script_params          = shot.copy()
        self._script_params["_profile"] = profile
        self._script_timer           = profile["aim_delay"]
        self._script_state           = "aiming"
        self.status_msg              = "Script: aiming..."

    def load_script_file(self, path: str) -> None:
        """Load and execute a billiard script from a .py file."""
        import importlib.util
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            self.status_msg = f"Script not found: {abs_path}"
            return
        spec = importlib.util.spec_from_file_location("_user_billiard_script", abs_path)
        mod  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            self.status_msg = f"Script error: {exc}"
            return
        script = getattr(mod, "SCRIPT", None)
        if script is None:
            self.status_msg = f"No SCRIPT variable in {os.path.basename(abs_path)}"
            return
        self._last_script_path = abs_path
        self.execute_script(script)

    def reload_script(self) -> None:
        """Re-execute the last loaded script."""
        if self._last_script_path:
            self.load_script_file(self._last_script_path)
        elif self._last_script:
            self.execute_script(self._last_script)
        else:
            self.status_msg = (
                "No script loaded yet.  "
                "Create shot_script.py or call load_script_file(path)."
            )

    def fire_script_impulse(self) -> None:
        """Apply cue impulse from _script_params and transition to running."""
        params    = self._script_params
        ball_name = params.get("ball", "white")
        ball      = next((b for b in self.physics_balls if b.name == ball_name), None)
        if ball is None:
            self._script_state = "idle"
            return

        az = math.radians(float(params.get("azimuth",   0.0)))
        el = math.radians(float(params.get("elevation", 0.0)))
        direction = np.array([
            math.sin(az) * math.cos(el),
            -math.sin(el),
            math.cos(az) * math.cos(el),
        ])
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            direction /= norm

        R  = ball.radius
        tx = float(params.get("tip_x", 0.0)) * R
        ty = float(params.get("tip_y", 0.0)) * R
        offset = [tx, ty] if (abs(tx) > 1e-9 or abs(ty) > 1e-9) else None
        force  = float(params.get("force", 1.0))

        ball.velocity[:] = 0
        ball.angular_velocity[:] = 0
        ball.state = BallState.STATIONARY
        self.engine.apply_cue(ball, force=force, direction=direction, offset=offset)

        for n in self.trail_positions:
            self.trail_positions[n] = []

        self._script_state = "idle"
        self.pending_events.append({"type": "clear_script_cue"})
        self.mode = "running"
        self.status_msg = "Script shot running..."

    # ──────────────────────────────────────────────────────────────────────────
    # Session recording
    # ──────────────────────────────────────────────────────────────────────────

    def _session_make_header(self, balls) -> list:
        cols = ["t"]
        for b in balls:
            n = b.name
            cols += [
                f"{n}_px", f"{n}_py", f"{n}_pz",
                f"{n}_vx", f"{n}_vy", f"{n}_vz",
                f"{n}_wx", f"{n}_wy", f"{n}_wz",
                f"{n}_vc_x", f"{n}_vc_z",
                f"{n}_speed", f"{n}_state",
            ]
        return cols

    def _session_record_frame(self) -> None:
        R   = BALL_RADIUS
        row = [f"{self._session_t:.4f}"]
        for b in self.physics_balls:
            px, py, pz = b.position
            vx, vy, vz = b.velocity
            wx, wy, wz = b.angular_velocity
            vc_x = vx + wz * R
            vc_z = vz - wx * R
            row += [
                f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}",
                f"{vx:.6f}", f"{vy:.6f}", f"{vz:.6f}",
                f"{wx:.6f}", f"{wy:.6f}", f"{wz:.6f}",
                f"{vc_x:.6f}", f"{vc_z:.6f}",
                f"{b.speed:.6f}", b.state.name,
            ]
        self._session_rows.append(row)
        self._session_t += self.SIM_DT * self.SIM_SUBSTEPS

    def _session_write_csv(self) -> None:
        path = self._session_file
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self._session_header)
                writer.writerows(self._session_rows)
            print(f"[REC] Saved {len(self._session_rows)} frames → {path}")
        except Exception as e:
            print(f"[REC] Write failed: {e}")
        self._session_recording = False
        self._session_rows.clear()
        self._session_header    = []
        self._session_file      = ""
        self._session_t         = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Headless RL / AI API
    # ──────────────────────────────────────────────────────────────────────────

    # Canonical ball order for obs vectors (Gym observation space).
    _BALL_ORDER = ["white", "yellow", "red1", "red2"]

    # Reward constants (tunable without changing code logic)
    REWARD_SCORE      =  1.0   # both target balls touched
    REWARD_ONE_RED    =  0.3   # only one target ball touched
    REWARD_MISS       = -0.1   # no target ball touched
    REWARD_FOUL       = -0.5   # foul ball touched

    def simulate_shot(
        self,
        ball_name: str,
        azimuth: float,
        tip: list | None = None,
        power: float = 50.0,
        *,
        elevation: float = 0.0,
        balls_state: dict | None = None,
        sim_dt: float = 0.005,
        max_t: float = 10.0,
    ) -> dict:
        """Headless shot simulation for RL training.

        Non-destructive: operates on deep copies — does NOT change
        ``self.physics_balls`` or any controller state.

        Args:
            ball_name:   Ball to strike ("white" or "yellow").
            azimuth:     Horizontal cue direction in degrees.
                         0 = toward +Z (far cushion), 90 = toward +X (right).
            tip:         [tip_x, tip_y] as fractions of ball radius (−1 … +1).
                         tip_x: right=+, left=−.  tip_y: top=+, bottom=−.
                         ``None`` → centre-ball hit.
            power:       Shot power as percent of MAX_POWER (0–100). Default 50.
            elevation:   Vertical cue angle in degrees (0 = horizontal).
                         Positive = cue angled downward (masse/jump).
            balls_state: Optional starting-state override before the shot.
                         Format::

                             {
                               "white":  {"pos": [x, z]},
                               "red1":   {"pos": [x, z], "vel": [vx, vz]},
                             }

                         Keys not present fall back to ``self.physics_balls``.
                         Missing balls (not in self.physics_balls and not here)
                         are silently omitted.
            sim_dt:      Physics timestep in seconds (default 0.001).
                         Use 0.005 for ~5× speedup at slight accuracy cost.
            max_t:       Maximum simulation wall-time (seconds). Default 30.

        Returns:
            ``dict`` with keys:

            scored (bool)
                True if the cue ball legally touched all target balls
                (red1 AND red2) without first triggering a foul.
            foul (bool)
                True if the cue ball touched a foul ball at any point.
                (white → foul_ball = yellow; yellow → foul_ball = white)
            touched (list[str])
                All balls the cue ball contacted during the shot.
            cushion_hits (int)
                Total cushion hits by the cue ball.
            sim_time (float)
                Elapsed simulated time in seconds until all balls stopped.
            reward (float)
                Composite scalar reward ready for ``gym.Env.step()``:
                  +1.0  scored (both reds hit)
                  +0.3  one red hit
                  −0.1  miss
                  −0.5  foul
            balls (dict)
                Final state of every ball::

                    {
                      "white":  {"pos": [x, z], "vel": [vx, vz],
                                 "speed": float, "state": "STATIONARY"},
                      ...
                    }

            obs (np.ndarray, float32, shape=(16,))
                Flat observation vector in canonical ball order
                (white, yellow, red1, red2), each as [x, z, vx, vz].
                Missing balls are represented as zeros.
                Suitable for ``gym.spaces.Box`` with::

                    low  = np.full(16, -2.0, dtype=np.float32)
                    high = np.full(16,  2.0, dtype=np.float32)

        Example — RL training loop::

            ctrl = BilliardsController()
            obs  = ctrl.reset()
            while True:
                action = agent.predict(obs)
                result = ctrl.simulate_shot(
                    "white",
                    azimuth=action[0] * 360,
                    tip=[action[1], action[2]],
                    power=action[3] * 100,
                )
                next_obs = ctrl.get_obs()   # positions are unchanged (headless)
                # --- or, to advance to the returned final state: ---
                ctrl.set_balls(result["balls"])
                next_obs = result["obs"]
                reward   = result["reward"]
                done     = result["scored"]
        """
        # ── 1. Build simulation ball list (manual copy — 3× faster than deepcopy)
        balls: list[Ball] = [_copy_ball(b) for b in self.physics_balls]

        # Apply optional positional override
        if balls_state is not None:
            existing = {b.name: b for b in balls}
            for name, bd in balls_state.items():
                pos = bd.get("pos")
                if pos is None:
                    continue
                if name in existing:
                    b = existing[name]
                else:
                    # Create a new ball from scratch with default physics params
                    b = Ball(name)
                    balls.append(b)
                    existing[name] = b

                if len(pos) == 2:
                    b.position = np.array([float(pos[0]), 0.0, float(pos[1])])
                else:
                    b.position = np.array([float(pos[0]), float(pos[1]), float(pos[2])])

                vel = bd.get("vel", [0.0, 0.0])
                if len(vel) == 2:
                    b.velocity = np.array([float(vel[0]), 0.0, float(vel[1])])
                else:
                    b.velocity = np.array([float(v) for v in vel])

                angvel = bd.get("angvel", [0.0, 0.0, 0.0])
                b.angular_velocity = np.array([float(v) for v in angvel])
                try:
                    b.state = BallState[bd.get("state", "STATIONARY")]
                except KeyError:
                    b.state = BallState.STATIONARY

        # ── 2. Identify cue ball and rule sets ────────────────────────────
        cue = next((b for b in balls if b.name == ball_name), None)
        if cue is None:
            raise ValueError(f"simulate_shot: ball '{ball_name}' not found in current state")

        # 4-ball billiards rules: opposite cue ball = foul, both reds = score
        foul_ball    = "yellow" if ball_name == "white" else "white"
        target_balls = {"red1", "red2"}

        # ── 3. Apply cue impulse ───────────────────────────────────────────
        az  = math.radians(azimuth)
        el  = math.radians(elevation)
        cos_e = math.cos(el)
        direction = np.array([
            math.sin(az) * cos_e,
            -math.sin(el),         # negative → cue tilted downward into ball
            math.cos(az) * cos_e,
        ])
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            direction /= norm

        R      = cue.radius
        tip    = tip or [0.0, 0.0]
        tx, ty = float(tip[0]) * R, float(tip[1]) * R
        offset = [tx, ty] if (abs(tx) > 1e-9 or abs(ty) > 1e-9) else None
        force  = (power / 100.0) * self.MAX_POWER

        cue.velocity[:]         = 0.0
        cue.angular_velocity[:] = 0.0
        cue.state               = BallState.STATIONARY
        eng = self._sim_engine   # reuse pre-allocated engine (events cleared in update)
        eng.apply_cue(cue, force=force, direction=direction, offset=offset)

        # ── 4. Run simulation, collect events ─────────────────────────────
        touched:      set = set()   # other balls the cue ball hit
        foul          = False
        cushion_hits  = 0           # cue ball cushion hits
        sim_t         = 0.0

        while sim_t < max_t:
            eng.update(balls, sim_dt)
            sim_t += sim_dt

            for ev in eng.events:
                if ev["type"] == "ball_ball":
                    # Track contacts involving the cue ball
                    if ev["ball1"] == ball_name:
                        touched.add(ev["ball2"])
                        if ev["ball2"] == foul_ball:
                            foul = True
                    elif ev["ball2"] == ball_name:
                        touched.add(ev["ball1"])
                        if ev["ball1"] == foul_ball:
                            foul = True
                elif ev["type"] == "cushion" and ev["ball"] == ball_name:
                    cushion_hits += 1

            if all(not b.is_moving() for b in balls):
                break

        # ── 5. Evaluate outcome ───────────────────────────────────────────
        reds_hit  = target_balls & touched          # {"red1"} or {"red1","red2"}
        scored    = (not foul) and (reds_hit == target_balls)

        if foul:
            reward = self.REWARD_FOUL
        elif scored:
            reward = self.REWARD_SCORE
        elif len(reds_hit) == 1:
            reward = self.REWARD_ONE_RED
        else:
            reward = self.REWARD_MISS

        # ── 6. Build result dict ──────────────────────────────────────────
        ball_map  = {b.name: b for b in balls}
        balls_out = {}
        for b in balls:
            balls_out[b.name] = {
                "pos":   [round(float(b.position[0]), 6),
                          round(float(b.position[2]), 6)],
                "vel":   [round(float(b.velocity[0]),  6),
                          round(float(b.velocity[2]),  6)],
                "speed": round(float(b.speed), 6),
                "state": b.state.name,
            }

        obs = self._make_obs(ball_map)

        return {
            "scored":       scored,
            "foul":         foul,
            "touched":      sorted(touched),
            "cushion_hits": cushion_hits,
            "sim_time":     round(sim_t, 4),
            "reward":       round(reward, 4),
            "balls":        balls_out,
            "obs":          obs,
        }

    # ── RL helper: flat observation vector ───────────────────────────────────

    def get_obs(self, ball_order: list | None = None) -> np.ndarray:
        """Return current physics state as a flat float32 observation vector.

        Each ball contributes 4 values: [x, z, vx, vz].
        Balls missing from ``self.physics_balls`` contribute four zeros.

        Args:
            ball_order: List of ball names defining order.
                        Default: ``["white", "yellow", "red1", "red2"]``.

        Returns:
            ``np.ndarray`` of shape ``(len(ball_order) * 4,)``, dtype ``float32``.
            Suitable for ``gym.spaces.Box(low=-2, high=2, shape=(16,))``.
        """
        ball_map = {b.name: b for b in self.physics_balls}
        return self._make_obs(ball_map, ball_order)

    def _make_obs(self, ball_map: dict, ball_order: list | None = None) -> np.ndarray:
        """Internal: build obs vector from a name→Ball dict."""
        order  = ball_order or self._BALL_ORDER
        values = []
        for name in order:
            b = ball_map.get(name)
            if b is not None:
                values.extend([
                    float(b.position[0]),  # x
                    float(b.position[2]),  # z
                    float(b.velocity[0]),  # vx
                    float(b.velocity[2]),  # vz
                ])
            else:
                values.extend([0.0, 0.0, 0.0, 0.0])
        return np.array(values, dtype=np.float32)

    # ── RL helper: reset to standard game positions ───────────────────────────

    def reset(self, balls_state: dict | None = None) -> np.ndarray:
        """Reset physics balls to standard positions and return obs vector.

        Equivalent to Gym ``env.reset()``.  Does NOT touch Ursina entities
        (call ``ctrl.start_game()`` from L3 to also reset the renderer).

        Args:
            balls_state: Optional override.  Same format as ``simulate_shot``'s
                         ``balls_state`` parameter.  Keys not present fall back
                         to ``GAME_POSITIONS``.

        Returns:
            Flat float32 obs vector (shape ``(16,)``).
        """
        self.physics_balls.clear()
        self.trail_positions.clear()
        self.engine = PhysicsEngine()

        for name, pos in self.GAME_POSITIONS.items():
            b = Ball(name, position=pos.copy())
            if balls_state and name in balls_state:
                bd  = balls_state[name]
                raw = bd.get("pos")
                if raw is not None:
                    if len(raw) == 2:
                        b.position = np.array([float(raw[0]), 0.0, float(raw[1])])
                    else:
                        b.position = np.array([float(v) for v in raw])
            self.physics_balls.append(b)
            self.trail_positions[b.name] = []

        self.mode       = "idle"
        self.shot_touched = set()
        return self.get_obs()

    # ── RL helper: set arbitrary ball positions ───────────────────────────────

    def set_balls(self, balls_info: dict) -> "BilliardsController":
        """Update physics ball positions/velocities without firing a shot.

        Accepts the same format as ``simulate_shot``'s ``balls_state`` **or**
        the ``balls`` sub-dict from a ``simulate_shot`` result::

            # From simulate_shot result:
            result = ctrl.simulate_shot("white", 45.0, power=60)
            ctrl.set_balls(result["balls"])

            # Manual placement:
            ctrl.set_balls({
                "white":  {"pos": [-0.2, -0.3]},
                "red1":   {"pos": [ 0.0,  0.5]},
            })

        Balls listed here but absent from ``self.physics_balls`` are created.
        Balls absent from this dict are left unchanged.

        Returns:
            ``self`` for chaining.
        """
        existing = {b.name: b for b in self.physics_balls}
        for name, bd in balls_info.items():
            # Accept both simulate_shot result format AND balls_state format
            pos = bd.get("pos")
            if pos is None:
                continue

            if name in existing:
                b = existing[name]
            else:
                b = Ball(name)
                self.physics_balls.append(b)
                self.trail_positions[b.name] = []
                existing[name] = b

            if len(pos) == 2:
                b.position = np.array([float(pos[0]), 0.0, float(pos[1])])
            else:
                b.position = np.array([float(pos[0]), float(pos[1]), float(pos[2])])

            vel = bd.get("vel", [0.0, 0.0])
            if len(vel) == 2:
                b.velocity = np.array([float(vel[0]), 0.0, float(vel[1])])
            else:
                b.velocity = np.array([float(v) for v in vel])

            b.angular_velocity[:] = 0.0
            b.state               = BallState.STATIONARY

        return self
