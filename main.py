"""
3D 4-Ball Billiards Visualizer -- Phase 2 Enhanced (3-Tier Architecture)
Layer 3: Ursina rendering / input handling.
Layer 2: controller.py (BilliardsController)
Layer 1: physics.py (PhysicsEngine)

Press 1-5 for shot presets, R to reset.
After balls stop, left-drag cue ball to aim and shoot manually.
"""

import math
import random
import tempfile
import wave
import os
from pathlib import Path
import numpy as np
from ursina import (
    Ursina, Entity, Text, camera, color, window,
    held_keys, application, Vec3, EditorCamera,
    mouse, Button, Audio, Mesh, destroy,
    time as ursina_time,
)
from PIL import Image, ImageDraw
from panda3d.core import Quat, LVector3f

from physics import PhysicsEngine, TABLE_WIDTH, TABLE_LENGTH, BALL_RADIUS, BallState, Ball
import physics as _phys          # used by PhysicsParamsEditor for live attribute mutation
from shot_presets import ShotPreset
from controller import BilliardsController, DEFAULT_INFO_MSG

# ── Layer 2: controller instance ──────────────────────────────────────────────
ctrl = BilliardsController()

# ──────────────────────────────────────────
# Dot texture generation (PIL)
# ──────────────────────────────────────────

def _make_dot_texture(base_rgb, dot_rgb=(30, 30, 30), size=256, num_dots=30, seed=0):
    """Scattered dots on UV map -> visible rotation on sphere."""
    img = Image.new("RGB", (size, size), base_rgb)
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)
    dot_r = size // 28
    for _ in range(num_dots):
        cx = rng.randint(dot_r, size - dot_r - 1)
        cy = rng.randint(dot_r, size - dot_r - 1)
        draw.ellipse(
            [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
            fill=dot_rgb,
        )
    return img


_tex_cache = {}

# Seed map for consistent textures per ball name
_BALL_SEEDS = {"white": 42, "yellow": 99, "red1": 7, "red2": 23}

# Ball base colors (for dot-texture generation)
_BALL_COLORS = {
    "white":  ((240, 240, 240), (160, 160, 200)),
    "yellow": ((240, 220, 80), (140, 120, 20)),
    "red1":   ((220, 40, 40), (80, 10, 10)),
    "red2":   ((200, 50, 50), (70, 15, 15)),
}

# Solid colors used for entity rendering (PIL→RGBA alpha=0 bug workaround)
_BALL_DISPLAY_COLORS = {
    "white":  color.white,
    "yellow": color.yellow,
    "red1":   color.red,
    "red2":   color.hsv(0, 0.8, 0.7),
}


def _get_texture(name):
    """Return or create a dot texture for the given ball name."""
    if name in _tex_cache:
        return _tex_cache[name]

    from ursina import Texture

    base, dot = _BALL_COLORS.get(name, ((200, 200, 200), (100, 100, 100)))
    seed = _BALL_SEEDS.get(name, 0)
    img = _make_dot_texture(base, dot, seed=seed)
    tex_path = os.path.join(_sound_dir, f"tex_{name}.png")
    img.save(tex_path)
    tex = Texture(tex_path)
    _tex_cache[name] = tex
    return tex


# ──────────────────────────────────────────
# Synthesized Sound Effects (numpy + wave)
# ──────────────────────────────────────────

_sound_dir = tempfile.mkdtemp(prefix="kbilliards_snd_")


def _synth_wav(filename, samples):
    """Write mono 16-bit 44100Hz WAV and return Path object."""
    path = os.path.join(_sound_dir, filename)
    data = np.clip(samples, -1.0, 1.0)
    data_int = (data * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data_int.tobytes())
    return Path(path)


def _synth_click():
    sr = 44100; dur = 0.08
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    env = np.exp(-t * 60)
    sig = env * np.sin(2 * np.pi * 800 * t + 5 * np.sin(2 * np.pi * 200 * t))
    return _synth_wav("click.wav", sig * 0.8)


def _synth_cushion():
    sr = 44100; dur = 0.1
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    env = np.exp(-t * 40)
    sig = env * np.sin(2 * np.pi * 400 * t)
    return _synth_wav("cushion.wav", sig * 0.6)


def _synth_roll():
    sr = 44100; dur = 0.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    rng = np.random.RandomState(123)
    noise = rng.randn(len(t))
    kernel = np.ones(200) / 200
    noise = np.convolve(noise, kernel, mode="same")
    noise = noise / (np.max(np.abs(noise)) + 1e-9)
    return _synth_wav("roll.wav", noise * 0.3)


# ──────────────────────────────────────────
# Scenario descriptions
# ──────────────────────────────────────────

SCENARIO_NAMES = {
    "1": "1: Follow Shot",
    "2": "2: Draw Shot",
    "3": "3: Stop Shot",
    "4": "4: Bank Shot",
    "5": "5: 3-Cushion Nejire",
}

# ──────────────────────────────────────────
# Ursina App
# ──────────────────────────────────────────

app = Ursina(borderless=False, title="4-Ball Billiards Visualizer", size=(1280, 800))

# ── Table ─────────────────────────────────
HW = TABLE_WIDTH / 2   # 0.635
HL = TABLE_LENGTH / 2  # 1.27

table_surface = Entity(
    model="plane",
    color=color.hsv(120, 0.7, 0.45),
    scale=(TABLE_WIDTH, 1, TABLE_LENGTH),
    position=(0, -0.001, 0),
)

CUSHION_THICK = 0.04
CUSHION_HEIGHT = 0.03

cushion_data = [
    ((0, CUSHION_HEIGHT / 2, HL + CUSHION_THICK / 2),
     (TABLE_WIDTH + CUSHION_THICK * 2, CUSHION_HEIGHT, CUSHION_THICK)),
    ((0, CUSHION_HEIGHT / 2, -HL - CUSHION_THICK / 2),
     (TABLE_WIDTH + CUSHION_THICK * 2, CUSHION_HEIGHT, CUSHION_THICK)),
    ((HW + CUSHION_THICK / 2, CUSHION_HEIGHT / 2, 0),
     (CUSHION_THICK, CUSHION_HEIGHT, TABLE_LENGTH)),
    ((-HW - CUSHION_THICK / 2, CUSHION_HEIGHT / 2, 0),
     (CUSHION_THICK, CUSHION_HEIGHT, TABLE_LENGTH)),
]

for pos, scl in cushion_data:
    Entity(model="cube", color=color.hsv(25, 0.6, 0.35), position=pos, scale=scl)

# ── Ball entities (L3 owns these) ─────────────────────────────────────────────
ball_entities: dict[str, Entity] = {}

# ── Trail system ─────────────────────────────────────────────────────────────
TRAIL_COLORS = {
    "white":  color.white,
    "yellow": color.yellow,
    "red1":   color.red,
    "red2":   color.hsv(0, 0.8, 0.7),
}
trail_entities: list[Entity] = []

# ── Angular velocity arrow system ─────────────────────────────────────────────
SPIN_ARROW_SCALE = 0.003
SPIN_ARROW_MIN   = 2.0
spin_arrow_entities: dict[str, list] = {}

# ── Cue stick aiming (mouse drag) ─────────────────────────────────────────────
aim_start = None
cue_stick_entity = None
aim_line_entity = None
power_text = None

# ── Sound effects ─────────────────────────────────────────────────────────────
click_path   = _synth_click()
cushion_path = _synth_cushion()
roll_path    = _synth_roll()
snd_click   = None
snd_cushion = None
snd_roll    = None
roll_playing = False

# ── Game UI entities (L3 owns) ────────────────────────────────────────────────
score_text    = None
turn_text     = None
power_bar_bg  = None
power_bar_fill = None
result_text   = None
result_text_timer: float = 0.0
RESULT_DISPLAY_TIME: float = 2.0

# ── Game cue visuals (persistent per-frame update) ────────────────────────────
game_cue_stick = None
game_aim_line  = None

# ── Script cue entities ────────────────────────────────────────────────────────
_script_cue_ent  = None
_script_aim_ent  = None

# ── Script picker UI ──────────────────────────────────────────────────────────
_script_picker_ent = None

# ── Advanced panel entities ───────────────────────────────────────────────────
_adv_entities: list = []
_adv_input = None

# ── UI ────────────────────────────────────────────────────────────────────────
info_text = Text(
    text=DEFAULT_INFO_MSG,
    position=(-0.7, 0.47),
    scale=1.2,
    color=color.white,
)

status_text = Text(
    text="",
    position=(-0.7, 0.42),
    scale=1.0,
    color=color.light_gray,
)

# ── Camera ────────────────────────────────────────────────────────────────────
camera.position = (0, 0, -3)
ec = EditorCamera(enabled=True)
ec.position = (0, 0, 0)
ec.rotation = (90, 0, 0)
camera.fov = 80

# ── ADV wrap columns ──────────────────────────────────────────────────────────
_ADV_WRAP_COLS = 85


# ──────────────────────────────────────────
# Hit-Point (Dangjeom) UI Panel
# ──────────────────────────────────────────

class HitPointUI:
    """Small circle in corner of screen for selecting cue strike offset."""

    def __init__(self):
        self.panel_size = 0.16
        self.panel_pos  = (0.62, -0.36)
        self.max_offset = 0.025

        self.bg = Entity(
            parent=camera.ui, model="circle",
            color=color.hsv(0, 0, 0.9),
            scale=self.panel_size,
            position=(self.panel_pos[0], self.panel_pos[1]), z=-0.1,
        )
        self.border = Entity(
            parent=camera.ui, model="circle",
            color=color.hsv(0, 0, 0.3),
            scale=self.panel_size * 1.05,
            position=(self.panel_pos[0], self.panel_pos[1]), z=-0.05,
        )
        self.dot = Entity(
            parent=camera.ui, model="circle",
            color=color.red, scale=0.015,
            position=(self.panel_pos[0], self.panel_pos[1]), z=-0.2,
        )
        label_scale  = 0.7
        label_offset = self.panel_size * 0.65
        self.labels  = []
        for (dx, dy), txt in [
            ((0, label_offset), "Top"), ((0, -label_offset), "Bot"),
            ((-label_offset, 0), "L"), ((label_offset, 0), "R"),
        ]:
            t = Text(text=txt, parent=camera.ui, scale=label_scale,
                     position=(self.panel_pos[0] + dx - 0.01, self.panel_pos[1] + dy),
                     color=color.hsv(0, 0, 0.4))
            self.labels.append(t)
        self.title = Text(text="Hit Point", parent=camera.ui, scale=0.8,
                          position=(self.panel_pos[0] - 0.03,
                                    self.panel_pos[1] + self.panel_size * 0.85),
                          color=color.white)
        self.dragging = False

    def handle_click(self, mx, my):
        cx, cy = self.bg.x, self.bg.y
        dx = mx - cx; dy = my - cy
        radius = self.panel_size / 2
        if dx * dx + dy * dy <= radius * radius:
            self.dragging = True
            self._update_dot(mx, my)
            return True
        return False

    def handle_drag(self, mx, my):
        if self.dragging:
            self._update_dot(mx, my)

    def handle_release(self):
        self.dragging = False

    def _update_dot(self, mx, my):
        cx, cy = self.bg.x, self.bg.y
        dx = mx - cx; dy = my - cy
        radius = self.panel_size / 2
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > radius:
            dx = dx / dist * radius; dy = dy / dist * radius
        self.dot.position = (cx + dx, cy + dy)
        ctrl.hit_offset[0] = (dx / radius) * self.max_offset
        ctrl.hit_offset[1] = (dy / radius) * self.max_offset

    def reset(self):
        ctrl.hit_offset[0] = 0.0
        ctrl.hit_offset[1] = 0.0
        self.dot.position = (self.bg.x, self.bg.y)


hitpoint_ui = None


def _ensure_hitpoint_ui():
    global hitpoint_ui
    if hitpoint_ui is None:
        hitpoint_ui = HitPointUI()
    return hitpoint_ui


# ──────────────────────────────────────────
# Physics Params Editor
# ──────────────────────────────────────────

class PhysicsParamsEditor:
    """Right-side panel listing all physics constants for live editing."""

    PARAMS = [
        ("MU_SLIDE",           "Slide Frict.",    0.01,   1.0,   0.01 ),
        ("MU_ROLL",            "Roll Frict.",     0.001,  0.2,   0.002),
        ("MU_SPIN",            "Spin Frict.",     0.001,  0.5,   0.005),
        ("CUSHION_RESTITUTION","Cushion Rest.",   0.10,   1.0,   0.01 ),
        ("BALL_RESTITUTION",   "Ball Rest.",      0.10,   1.0,   0.01 ),
        ("GRAVITY",            "Gravity",         1.0,   20.0,   0.2  ),
        ("MU_CUSHION",         "Cushion Fric.",   0.0,    0.5,   0.005),
        ("MU_BALL",            "Ball Throw",      0.0,    0.3,   0.005),
        ("SQUIRT_FACTOR",      "Squirt",         -0.1,    0.0,   0.005),
        ("RAIL_H_OFFSET",      "Rail Height",     0.0,    0.015, 0.001),
        ("TABLE_BOUNCE_REST",  "Floor Bounce",    0.0,    1.0,   0.01 ),
    ]
    DEFAULTS = {p[0]: getattr(_phys, p[0]) for p in PARAMS}

    PANEL_X = 0.63
    PANEL_W = 0.27
    ROW_H   = 0.054
    ROW_Y0  = 0.37

    def __init__(self):
        self.selected = 0
        self.visible  = True
        self._ents     = []
        self._val_txts = []
        self._row_bgs  = []
        self._build()

    def _build(self):
        n = len(self.PARAMS)
        panel_h  = n * self.ROW_H + 0.10
        panel_cy = self.ROW_Y0 - (n - 1) * self.ROW_H / 2 + 0.03
        self._add(Entity(parent=camera.ui, model="quad",
                         color=color.rgba(0, 0, 0, 0.65),
                         scale=(self.PANEL_W, panel_h),
                         position=(self.PANEL_X, panel_cy), z=0.05))
        self._add(Text(text="--- Physics Params ---", parent=camera.ui, scale=0.72,
                       position=(self.PANEL_X - 0.11, self.ROW_Y0 + 0.058),
                       color=color.cyan))
        self._add(Text(text="Click row  ]=up  [=down  Shift=fine  (Wheel=camera zoom)",
                       parent=camera.ui, scale=0.56,
                       position=(self.PANEL_X - 0.11, self.ROW_Y0 + 0.030),
                       color=color.gray))
        for i, (attr, label, mn, mx, step) in enumerate(self.PARAMS):
            y = self.ROW_Y0 - i * self.ROW_H
            row_bg = Entity(parent=camera.ui, model="quad",
                            color=color.rgba(1, 1, 0, 0.22) if i == 0 else color.rgba(1, 1, 1, 0.04),
                            scale=(self.PANEL_W - 0.01, self.ROW_H - 0.007),
                            position=(self.PANEL_X, y), z=0.04)
            self._add(row_bg)
            self._row_bgs.append(row_bg)
            self._add(Text(text=label, parent=camera.ui, scale=0.70,
                           position=(self.PANEL_X - 0.11, y - 0.008),
                           color=color.white))
            val = getattr(_phys, attr)
            vtxt = Text(text=self._fmt(val), parent=camera.ui, scale=0.70,
                        position=(self.PANEL_X + 0.04, y - 0.008),
                        color=color.yellow)
            self._add(vtxt)
            self._val_txts.append(vtxt)
        bot_y = self.ROW_Y0 - n * self.ROW_H - 0.005
        self._add(Text(text="[Shift+P] Reset all to defaults", parent=camera.ui, scale=0.56,
                       position=(self.PANEL_X - 0.11, bot_y), color=color.gray))

    def _add(self, entity):
        self._ents.append(entity)
        return entity

    @staticmethod
    def _fmt(val):
        if val == 0.0:
            return "0"
        return f"{val:.4g}"

    def _refresh_row(self, i):
        attr = self.PARAMS[i][0]
        val  = getattr(_phys, attr)
        self._val_txts[i].text  = self._fmt(val)
        dflt = self.DEFAULTS[attr]
        self._val_txts[i].color = color.yellow if abs(val - dflt) < 1e-9 else color.orange

    def _update_selection(self):
        for i, bg in enumerate(self._row_bgs):
            bg.color = (color.rgba(1, 1, 0, 0.22) if i == self.selected
                        else color.rgba(1, 1, 1, 0.04))

    def try_click(self, mx, my):
        if not self.visible:
            return False
        for i in range(len(self.PARAMS)):
            row_y = self.ROW_Y0 - i * self.ROW_H
            if (abs(mx - self.PANEL_X) <= self.PANEL_W / 2 and
                    abs(my - row_y) <= self.ROW_H / 2):
                self.selected = i
                self._update_selection()
                return True
        return False

    def adjust(self, direction, fine=False):
        i = self.selected
        attr, _, mn, mx, step = self.PARAMS[i]
        s   = step / 10.0 if fine else step
        cur = getattr(_phys, attr)
        setattr(_phys, attr, max(mn, min(mx, cur + direction * s)))
        self._refresh_row(i)

    def reset_all(self):
        for attr, dflt in self.DEFAULTS.items():
            setattr(_phys, attr, dflt)
        for i in range(len(self.PARAMS)):
            self._refresh_row(i)

    def toggle_visible(self):
        self.visible = not self.visible
        for e in self._ents:
            e.enabled = self.visible

    def destroy_all(self):
        for e in self._ents:
            destroy(e)
        self._ents.clear()
        self._val_txts.clear()
        self._row_bgs.clear()


physics_params_editor = None


def _ensure_params_editor():
    global physics_params_editor
    if physics_params_editor is None:
        physics_params_editor = PhysicsParamsEditor()
    return physics_params_editor


# ──────────────────────────────────────────
# Sound loading (deferred until app is running)
# ──────────────────────────────────────────

_sounds_loaded = False


def _load_sounds():
    global snd_click, snd_cushion, snd_roll, _sounds_loaded
    if _sounds_loaded:
        return
    try:
        snd_click   = Audio(click_path,   autoplay=False)
        snd_cushion = Audio(cushion_path, autoplay=False)
        snd_roll    = Audio(roll_path, loop=True, autoplay=False)
        _sounds_loaded = True
    except Exception:
        pass


# ──────────────────────────────────────────
# Helper functions (L3 only)
# ──────────────────────────────────────────

def _clear_spin_arrows():
    global spin_arrow_entities
    for entities in spin_arrow_entities.values():
        for ent in entities:
            try:
                destroy(ent)
            except Exception:
                pass
    spin_arrow_entities.clear()


def _update_spin_arrows():
    global spin_arrow_entities
    _clear_spin_arrows()
    for pb in ctrl.physics_balls:
        ent = ball_entities.get(pb.name)
        if ent is None:
            continue
        w     = pb.angular_velocity
        w_mag = float(np.linalg.norm(w))
        if w_mag < SPIN_ARROW_MIN:
            continue
        cx = float(pb.position[0])
        cy = BALL_RADIUS
        cz = float(pb.position[2])
        w_hat     = w / w_mag
        arrow_len = w_mag * SPIN_ARROW_SCALE
        tip = Vec3(cx + w_hat[0] * arrow_len,
                   cy + w_hat[1] * arrow_len,
                   cz + w_hat[2] * arrow_len)
        try:
            shaft_mesh = Mesh(vertices=[Vec3(cx, cy, cz), tip], mode='line', thickness=3)
            shaft    = Entity(model=shaft_mesh, color=color.black)
            tip_ent  = Entity(model='cube', color=color.black,
                              scale=BALL_RADIUS * 0.4, position=tip)
            spin_arrow_entities[pb.name] = [shaft, tip_ent]
        except Exception:
            pass


def _do_clear_balls():
    """Destroy all ball entities (L3 side). Called via pending_event 'clear_balls'."""
    global ball_entities
    _clear_spin_arrows()
    for ent in ball_entities.values():
        for child in list(ent.children):
            destroy(child)
        ent.disable()
        destroy(ent)
    ball_entities.clear()
    _clear_trails()
    _clear_aim()


# ── Axis constants ────────────────────────────────────────────────────────────
_AX_L = 1.6
_AX_T = 0.14
_AX_H = 0.32


def _add_axes(parent_ent):
    L, T, H = _AX_L, _AX_T, _AX_H
    for shaft_pos, shaft_scale, tip_pos, clr in [
        (Vec3(L/2,   0,   0), Vec3(L, T, T), Vec3(L,   0,   0), color.red),
        (Vec3(  0, L/2,   0), Vec3(T, L, T), Vec3(0,   L,   0), color.green),
        (Vec3(  0,   0, L/2), Vec3(T, T, L), Vec3(0,   0,   L), color.azure),
    ]:
        Entity(parent=parent_ent, model='cube', color=clr,
               scale=shaft_scale, position=shaft_pos)
        Entity(parent=parent_ent, model='cube', color=clr,
               scale=Vec3(H, H, H), position=tip_pos)


def _spawn_ball(phys_ball):
    """Create an Ursina entity for a physics ball."""
    ent = Entity(
        model="sphere",
        texture=_get_texture(phys_ball.name),
        scale=BALL_RADIUS * 2,
        position=Vec3(phys_ball.position[0], BALL_RADIUS, phys_ball.position[2]),
    )
    ball_entities[phys_ball.name] = ent
    _add_axes(ent)
    return ent


# ── Trail management ──────────────────────────────────────────────────────────

def _clear_trails():
    global trail_entities
    for ent in trail_entities:
        if ent.enabled:
            destroy(ent)
    trail_entities.clear()


def _update_trails():
    global trail_entities
    for pb in ctrl.physics_balls:
        if pb.is_moving():
            positions = ctrl.trail_positions.get(pb.name)
            if positions is not None:
                x, z = float(pb.position[0]), float(pb.position[2])
                if not positions or (abs(positions[-1][0] - x) > 0.001
                                     or abs(positions[-1][1] - z) > 0.001):
                    positions.append((x, z))
                    if len(positions) > ctrl.TRAIL_MAX_POINTS:
                        positions.pop(0)

    for ent in trail_entities:
        if ent.enabled:
            destroy(ent)
    trail_entities.clear()

    for name, positions in ctrl.trail_positions.items():
        if len(positions) < 2:
            continue
        verts = [Vec3(x, 0.001, z) for x, z in positions]
        trail_color = TRAIL_COLORS.get(name, color.white)
        try:
            mesh = Mesh(vertices=verts, mode="line", thickness=2)
            ent  = Entity(model=mesh, color=trail_color)
            trail_entities.append(ent)
        except Exception:
            pass


# ── Aim / Cue Stick (mouse drag, non-game mode) ───────────────────────────────

def _clear_aim():
    global cue_stick_entity, aim_line_entity, power_text
    if cue_stick_entity:
        destroy(cue_stick_entity)
        cue_stick_entity = None
    if aim_line_entity:
        destroy(aim_line_entity)
        aim_line_entity = None
    if power_text:
        destroy(power_text)
        power_text = None


def _get_mouse_table_pos():
    if mouse.world_point is not None:
        return mouse.world_point
    return None


def _start_aim(world_pos):
    global aim_start
    if ctrl.mode != "idle" or not ctrl.physics_balls:
        return False
    cue_ball = next((b for b in ctrl.physics_balls if b.name == "white"), None)
    if cue_ball is None:
        return False
    dx = world_pos.x - cue_ball.position[0]
    dz = world_pos.z - cue_ball.position[2]
    if math.sqrt(dx * dx + dz * dz) > BALL_RADIUS * 15:
        return False
    aim_start   = Vec3(cue_ball.position[0], 0, cue_ball.position[2])
    ctrl.mode   = "aiming"
    ec.enabled  = False
    return True


def _update_aim(world_pos):
    global cue_stick_entity, aim_line_entity, power_text
    if aim_start is None or world_pos is None:
        return
    dx   = world_pos.x - aim_start.x
    dz   = world_pos.z - aim_start.z
    dist = math.sqrt(dx * dx + dz * dz)
    if dist < 0.005:
        return
    dir_x = -dx / dist
    dir_z = -dz / dist
    power = min(dist * 2.0, 1.5)
    _clear_aim()
    angle = math.degrees(math.atan2(dir_x, dir_z))
    stick_length = 0.4
    stick_cx = aim_start.x - dir_x * (stick_length / 2 + BALL_RADIUS + dist * 0.3)
    stick_cz = aim_start.z - dir_z * (stick_length / 2 + BALL_RADIUS + dist * 0.3)
    cue_stick_entity = Entity(
        model="cube", color=color.hsv(30, 0.5, 0.6),
        scale=(0.006, 0.006, stick_length),
        position=(stick_cx, BALL_RADIUS, stick_cz),
        rotation_y=angle,
    )
    line_len    = 0.5
    line_end_x  = aim_start.x + dir_x * line_len
    line_end_z  = aim_start.z + dir_z * line_len
    try:
        mesh = Mesh(
            vertices=[Vec3(aim_start.x, 0.002, aim_start.z),
                      Vec3(line_end_x,  0.002, line_end_z)],
            mode="line", thickness=1,
        )
        aim_line_entity = Entity(model=mesh, color=color.hsv(0, 0, 1.0))
    except Exception:
        pass
    power_pct = int(power / 1.5 * 100)
    power_text = Text(text=f"Power: {power_pct}%",
                      position=(-0.7, 0.37), scale=1.0, color=color.orange)


def _fire_shot(world_pos):
    global aim_start
    if aim_start is None or world_pos is None:
        ctrl.mode = "idle"
        ec.enabled = True
        _clear_aim()
        return
    dx   = world_pos.x - aim_start.x
    dz   = world_pos.z - aim_start.z
    dist = math.sqrt(dx * dx + dz * dz)
    if dist < 0.005:
        ctrl.mode  = "idle"
        ec.enabled = True
        _clear_aim()
        return
    dir_x     = -dx / dist
    dir_z     = -dz / dist
    direction = np.array([dir_x, 0.0, dir_z])
    power     = min(dist * 2.0, 1.5)
    offset    = None
    if abs(ctrl.hit_offset[0]) > 0.001 or abs(ctrl.hit_offset[1]) > 0.001:
        offset = [ctrl.hit_offset[0], ctrl.hit_offset[1]]

    aim_start  = None
    ec.enabled = True
    _clear_aim()
    ctrl.fire_shot(direction, power, offset)


# ── Sound playback ────────────────────────────────────────────────────────────

def _play_collision_sounds(events):
    global roll_playing
    for evt in events:
        vol = min(1.0, evt["speed"] * 1.5)
        if vol < 0.05:
            continue
        if evt["type"] == "ball_ball" and snd_click:
            snd_click.volume = vol
            snd_click.pitch  = 0.8 + evt["speed"] * 0.5
            snd_click.play()
        elif evt["type"] == "cushion" and snd_cushion:
            snd_cushion.volume = vol * 0.7
            snd_cushion.pitch  = 0.7 + evt["speed"] * 0.3
            snd_cushion.play()


def _update_roll_sound():
    global roll_playing
    if snd_roll is None:
        return
    any_rolling = False
    max_speed   = 0.0
    for b in ctrl.physics_balls:
        if b.state == BallState.ROLLING and b.speed > 0.01:
            any_rolling = True
            max_speed   = max(max_speed, b.speed)
    if any_rolling and not roll_playing:
        snd_roll.volume = min(0.3, max_speed * 0.5)
        snd_roll.play()
        roll_playing = True
    elif any_rolling and roll_playing:
        snd_roll.volume = min(0.3, max_speed * 0.5)
    elif not any_rolling and roll_playing:
        snd_roll.stop()
        roll_playing = False


# ── Game Mode UI ──────────────────────────────────────────────────────────────

def _destroy_game_ui():
    global score_text, turn_text, power_bar_bg, power_bar_fill, result_text
    for e in [score_text, turn_text, power_bar_bg, power_bar_fill, result_text]:
        if e:
            try:
                destroy(e)
            except Exception:
                pass
    score_text = turn_text = power_bar_bg = power_bar_fill = result_text = None


def _clear_game_visuals():
    global game_cue_stick, game_aim_line
    if game_cue_stick:
        try:
            destroy(game_cue_stick)
        except Exception:
            pass
        game_cue_stick = None
    if game_aim_line:
        try:
            destroy(game_aim_line)
        except Exception:
            pass
        game_aim_line = None


def _init_game_ui():
    global score_text, turn_text, power_bar_bg, power_bar_fill, result_text
    _destroy_game_ui()
    score_text   = Text(text="Player: 0  |  AI: 0",
                        position=(0, 0.47), origin=(0, 0), scale=1.4, color=color.white)
    turn_text    = Text(text=">> Player's Turn",
                        position=(0, 0.42), origin=(0, 0), scale=1.1, color=color.yellow)
    power_bar_bg = Entity(parent=camera.ui, model="cube",
                          color=color.hsv(0, 0, 0.25), scale=(0.4, 0.02, 1),
                          position=(0, -0.45), z=-0.1)
    power_bar_fill = Entity(parent=camera.ui, model="cube",
                            color=color.orange, scale=(0.001, 0.018, 1),
                            position=(-0.2, -0.45), z=-0.2)
    result_text  = Text(text="", position=(0, 0.1), origin=(0, 0),
                        scale=2.5, color=color.white)


def _update_game_ui():
    if score_text:
        score_text.text = f"Player: {ctrl.player_score}  |  AI: {ctrl.ai_score}"
    if turn_text:
        if ctrl.player_turn:
            turn_text.text  = ">> Player's Turn"
            turn_text.color = color.yellow
        else:
            turn_text.text  = ">> AI's Turn"
            turn_text.color = color.cyan


def _update_power_bar():
    if not power_bar_fill:
        return
    w = ctrl.aim_power * 0.4
    power_bar_fill.scale_x = max(w, 0.001)
    power_bar_fill.x       = -0.2 + w / 2
    power_bar_fill.color   = color.hsv(30 - ctrl.aim_power * 30, 1.0, 1.0)


_COLOR_MAP = {
    "red":    color.red,
    "green":  color.green,
    "orange": color.orange,
    "cyan":   color.cyan,
    "gray":   color.gray,
    "yellow": color.yellow,
    "white":  color.white,
}


def _show_result(msg, clr=None):
    global result_text_timer
    if result_text:
        result_text.text  = msg
        result_text.color = clr or color.white
        result_text_timer = RESULT_DISPLAY_TIME


# ── Game Mode Cue Visual ──────────────────────────────────────────────────────

def _update_game_cue_visual(active_name="white", angle=None, show_aim=True):
    global game_cue_stick, game_aim_line
    _clear_game_visuals()
    if angle is None:
        angle = ctrl.aim_angle

    ball = next((b for b in ctrl.physics_balls if b.name == active_name), None)
    if ball is None:
        return

    cx = float(ball.position[0])
    cz = float(ball.position[2])
    cy = BALL_RADIUS + float(ball.position[1])

    elev     = ctrl.aim_elevation if active_name == "white" else 0.0
    elev_rad = math.radians(elev)
    angle_rad = math.radians(angle)
    cos_e = math.cos(elev_rad)
    sin_e = math.sin(elev_rad)
    dx = math.sin(angle_rad) * cos_e
    dy = sin_e
    dz = math.cos(angle_rad) * cos_e

    stick_dist = 0.25 + ctrl.aim_power * 0.10
    stick_len  = 0.4
    back       = stick_dist + stick_len / 2
    sx = cx - dx * back; sy = cy - dy * back; sz = cz - dz * back

    game_cue_stick = Entity(model="cube", color=color.hsv(30, 0.5, 0.7),
                            scale=(0.008, 0.008, stick_len),
                            position=Vec3(sx, sy, sz),
                            rotation_x=-elev, rotation_y=angle)
    if show_aim:
        try:
            mesh = Mesh(
                vertices=[Vec3(cx, 0.003, cz),
                          Vec3(cx + math.sin(angle_rad) * 0.5,
                               0.003,
                               cz + math.cos(angle_rad) * 0.5)],
                mode="line", thickness=2,
            )
            game_aim_line = Entity(model=mesh, color=color.rgba(1, 1, 1, 0.7))
        except Exception:
            pass


# ── Script cue visuals ────────────────────────────────────────────────────────

def _clear_script_cue():
    global _script_cue_ent, _script_aim_ent
    if _script_cue_ent:
        destroy(_script_cue_ent); _script_cue_ent = None
    if _script_aim_ent:
        destroy(_script_aim_ent); _script_aim_ent = None


def _build_script_cue(ball, az_deg: float, el_deg: float,
                      dist_back: float, show_aim: bool = True) -> None:
    global _script_cue_ent, _script_aim_ent
    _clear_script_cue()
    if ball is None:
        return
    az = math.radians(az_deg); el = math.radians(el_deg)
    dx = math.sin(az) * math.cos(el)
    dy = -math.sin(el)
    dz = math.cos(az) * math.cos(el)
    bx = float(ball.position[0])
    by = BALL_RADIUS + float(ball.position[1])
    bz = float(ball.position[2])
    stick_len = 0.45
    total = BALL_RADIUS + 0.01 + dist_back + stick_len * 0.5
    cx, cy, cz = bx - dx * total, by - dy * total, bz - dz * total
    _script_cue_ent = Entity(model="cube", color=color.hsv(25, 0.55, 0.85),
                             scale=(0.009, 0.009, stick_len),
                             position=Vec3(cx, cy, cz),
                             rotation_y=az_deg, rotation_x=-el_deg)
    if show_aim:
        try:
            mesh = Mesh(vertices=[Vec3(bx, by, bz),
                                  Vec3(bx + dx * 0.5, by + dy * 0.5, bz + dz * 0.5)],
                        mode="line", thickness=2)
            _script_aim_ent = Entity(model=mesh, color=color.rgba(1, 1, 0, 0.65))
        except Exception:
            pass


# ── Script picker UI ──────────────────────────────────────────────────────────

def _open_script_picker():
    global _script_picker_ent
    files = ctrl.collect_script_files()
    if not files:
        status_text.text = ("No scripts found.  "
                            "Create shot_script.py or add .py files to scripts/")
        return
    if len(files) == 1:
        ctrl.load_script_file(str(files[0]))
        return

    ctrl._script_picker_files  = files
    ctrl._script_picker_active = True

    lines = ["--- Select script (1-9, Esc=cancel) ---"]
    for i, p in enumerate(files[:9], 1):
        lines.append(f"  {i}. {p.name}")
    _script_picker_ent = Text(text="\n".join(lines),
                              position=(-0.1, 0.15), origin=(0, 0),
                              scale=1.3, color=color.yellow, background=True)


def _close_script_picker():
    global _script_picker_ent
    ctrl._script_picker_active = False
    if _script_picker_ent:
        destroy(_script_picker_ent)
        _script_picker_ent = None


# ── Advanced Command Panel ────────────────────────────────────────────────────

def _adv_open():
    global _adv_entities, _adv_input
    if ctrl._adv_active or ctrl.mode == "running":
        return
    ctrl._adv_active   = True
    ctrl._adv_wrapping = False

    state_json = ctrl.get_state_json()

    bg = Entity(parent=camera.ui, model="quad",
                color=color.rgba(0, 0, 0, 0.90),
                scale=(1.62, 0.44), position=(0, -0.200), z=0.10)
    _adv_entities.append(bg)

    _adv_entities.append(Text(
        parent=camera.ui,
        text="[ Advanced Command Panel ]   Esc=close   Enter=execute",
        position=(-0.78, 0.010), scale=0.88, color=color.cyan, z=0.09,
    ))

    _hint_lines = [
        'set balls:  {"cmd":"set","balls":{"white":{"pos":[x,z]},"red1":{"pos":[x,z]}}}',
        'set params: {"cmd":"set","params":{"MU_ROLL":0.015,"MU_CUSHION":0.18}}',
        'shot:       {"cmd":"shot","ball":"white","azimuth":0.0,"tip":[0.0,0.5],"power":35}',
        '  tip:[right/left, top/bottom]  topspin=+y  backspin=-y  az: 0=fwd 90=right',
    ]
    for _i, _line in enumerate(_hint_lines):
        _adv_entities.append(Text(parent=camera.ui, text=_line,
                                  position=(-0.78, -0.278 - _i * 0.040),
                                  scale=0.74, color=color.gray, z=0.09))

    try:
        from ursina.prefabs.input_field import InputField as _IF
        inp = _IF(parent=camera.ui, scale=(1.58, 0.25),
                  position=(0, -0.135), z=0.08,
                  max_lines=6, character_limit=10000)

        ctrl._adv_activate_field = True
        _adv_input = inp

        def _do_wrap():
            if ctrl._adv_wrapping:
                return
            raw = inp.text_field.text.replace('\n', '')
            if len(raw) <= _ADV_WRAP_COLS:
                return
            ctrl._adv_wrapping = True
            try:
                chunks = []
                while len(raw) > _ADV_WRAP_COLS:
                    bp = _ADV_WRAP_COLS
                    for ch in (',', ':', ' '):
                        p = raw.rfind(ch, bp - 15, bp)
                        if p > 0:
                            bp = p + 1; break
                    chunks.append(raw[:bp])
                    raw = raw[bp:]
                chunks.append(raw)
                wrapped = '\n'.join(chunks)
                if inp.text_field.text != wrapped:
                    inp.text = wrapped
            finally:
                ctrl._adv_wrapping = False

        inp.on_value_changed = _do_wrap
        inp.text = state_json
        _adv_entities.append(inp)
    except Exception as exc:
        _adv_entities.append(Text(parent=camera.ui,
                                  text=f"InputField error: {exc}",
                                  position=(-0.78, -0.135), scale=0.80,
                                  color=color.red, z=0.09))


def _adv_close():
    global _adv_entities, _adv_input
    ctrl._adv_active         = False
    ctrl._adv_activate_field = False
    ctrl._adv_wrapping       = False
    _adv_input = None
    for e in _adv_entities:
        if e:
            try:
                destroy(e)
            except Exception:
                pass
    _adv_entities.clear()


# ──────────────────────────────────────────
# Controller event dispatcher (L2 → L3)
# ──────────────────────────────────────────

def _handle_controller_event(ev: dict):
    t = ev["type"]
    if t == "spawn_ball":
        _spawn_ball(ev["ball"])
    elif t == "clear_balls":
        _do_clear_balls()
    elif t == "clear_aim":
        _clear_aim()
    elif t == "clear_game_visuals":
        _clear_game_visuals()
    elif t == "init_game_ui":
        _init_game_ui()
    elif t == "destroy_game_ui":
        _destroy_game_ui()
    elif t == "hide_game_score_ui":
        if score_text:
            score_text.enabled = False
        if turn_text:
            turn_text.enabled = False
    elif t == "show_result":
        clr = _COLOR_MAP.get(ev.get("color_name", "white"), color.white)
        _show_result(ev["msg"], clr)
    elif t == "update_game_ui":
        _update_game_ui()
    elif t == "update_power_bar":
        _update_power_bar()
    elif t == "adv_close":
        _adv_close()
    elif t == "clear_script_cue":
        _clear_script_cue()
    elif t == "refresh_params_editor":
        ppe = _ensure_params_editor()
        for k in ev.get("params", []):
            for i, row in enumerate(ppe.PARAMS):
                if row[0] == k:
                    ppe._refresh_row(i)
    elif t == "session_saved":
        status_text.text = f"Recorded → {ev['file']}"


# ──────────────────────────────────────────
# Input handler
# ──────────────────────────────────────────

def input(key):
    # ── Advanced panel: captures ALL keys ─────────────────────────────────────
    if ctrl._adv_active:
        if key == "escape":
            _adv_close()
            status_text.text = "Advanced panel closed."
        elif key == "enter" and _adv_input is not None:
            _txt = _adv_input.text.replace('\n', '').strip()
            print(f"[ADV] Enter pressed. text={repr(_txt[:80])}")
            ctrl.execute_command(_txt)
        return

    # ── Shift+A: open advanced command panel ──────────────────────────────────
    if key == "a" and held_keys["shift"] and ctrl.mode != "running":
        _adv_open()
        return

    # ── Physics params editor ─────────────────────────────────────────────────
    ppe = _ensure_params_editor()
    if key == "p":
        if held_keys["shift"]:
            ppe.reset_all()
        else:
            ppe.toggle_visible()
        return
    if ppe.visible:
        if key == "]":
            ppe.adjust(+1, fine=bool(held_keys["shift"]))
            return
        if key == "[":
            ppe.adjust(-1, fine=bool(held_keys["shift"]))
            return

    # ── Script picker ─────────────────────────────────────────────────────────
    if ctrl._script_picker_active:
        if key == "escape":
            _close_script_picker()
            status_text.text = "Script picker cancelled."
        elif key.isdigit() and key != "0":
            idx = int(key) - 1
            if idx < len(ctrl._script_picker_files):
                path = str(ctrl._script_picker_files[idx])
                _close_script_picker()
                ctrl.load_script_file(path)
            else:
                status_text.text = f"No script at slot {key}."
        return

    # ── Shift+S: open script picker ───────────────────────────────────────────
    if key == "s" and held_keys["shift"] and ctrl.mode != "running":
        _open_script_picker()
        return

    # ── Game start ────────────────────────────────────────────────────────────
    if key == "g" and not ctrl.practice_mode and ctrl.mode != "running":
        ctrl.start_game()
        return

    # ── Practice start/exit ───────────────────────────────────────────────────
    if key == "t" and not ctrl.game_mode and ctrl.mode != "running":
        ctrl.toggle_practice()
        return

    # ── Game mode: handle r/space/mouse ───────────────────────────────────────
    if ctrl.game_mode:
        if key == "r" and ctrl.mode != "running":
            ctrl.reset_game()
            return
        if key == "space" and ctrl.player_turn and ctrl.mode == "idle":
            ctrl.power_charging = True
            ctrl.aim_power = 0.0
            return
        if key == "space up" and ctrl.power_charging:
            ctrl.power_charging = False
            if ctrl.player_turn and ctrl.mode == "idle":
                ctrl.fire_game_shot()
            return
        if key == "left mouse down":
            if mouse.position is not None:
                mx, my = mouse.position[0], mouse.position[1]
                if ppe.visible and ppe.try_click(mx, my):
                    return
            hp = _ensure_hitpoint_ui()
            if mouse.position is not None:
                hp.handle_click(mouse.position[0], mouse.position[1])
            return
        if key == "left mouse up":
            _ensure_hitpoint_ui().handle_release()
            return
        return

    # ── Practice mode: r/space ────────────────────────────────────────────────
    if ctrl.practice_mode:
        if key == "r" and ctrl.mode != "running":
            ctrl.reset_game()
            return
        if key == "space" and ctrl.mode == "idle":
            ctrl.power_charging = True
            ctrl.aim_power = 0.0
            return
        if key == "space up" and ctrl.power_charging:
            ctrl.power_charging = False
            if ctrl.mode == "idle":
                ctrl.fire_game_shot()
            return
        if key == "left mouse down":
            if mouse.position is not None:
                mx, my = mouse.position[0], mouse.position[1]
                if ppe.visible and ppe.try_click(mx, my):
                    return
            hp = _ensure_hitpoint_ui()
            if mouse.position is not None:
                hp.handle_click(mouse.position[0], mouse.position[1])
            return
        if key == "left mouse up":
            _ensure_hitpoint_ui().handle_release()
            return
        return

    # ── Non-game-mode input ───────────────────────────────────────────────────
    if ctrl.mode != "aiming":
        if key == "1":
            ctrl.load_scenario(ShotPreset.scenario_1_follow, SCENARIO_NAMES["1"]); return
        elif key == "2":
            ctrl.load_scenario(ShotPreset.scenario_2_draw,   SCENARIO_NAMES["2"]); return
        elif key == "3":
            ctrl.load_scenario(ShotPreset.scenario_3_stop,   SCENARIO_NAMES["3"]); return
        elif key == "4":
            ctrl.load_scenario(
                lambda run: ShotPreset.scenario_4_bank(english=0.0, run=run),
                SCENARIO_NAMES["4"]); return
        elif key == "5":
            ctrl.load_scenario(ShotPreset.scenario_5_nejire, SCENARIO_NAMES["5"]); return
        elif key == "r":
            ctrl.reset_game(); return

    # ── Mouse: cue aim (non-game mode) ────────────────────────────────────────
    hp = _ensure_hitpoint_ui()
    if key == "left mouse down":
        if mouse.position is not None:
            mx, my = mouse.position[0], mouse.position[1]
            if ppe.visible and ppe.try_click(mx, my):
                return
            if hp.handle_click(mx, my):
                return
        world_pos = _get_mouse_table_pos()
        if world_pos is not None and ctrl.mode == "idle":
            _start_aim(world_pos)

    elif key == "left mouse up":
        if hp.dragging:
            hp.handle_release()
            return
        if ctrl.mode == "aiming":
            world_pos = _get_mouse_table_pos()
            _fire_shot(world_pos)


# ──────────────────────────────────────────
# Update loop
# ──────────────────────────────────────────

_frame_count = 0


def update():
    global _frame_count, result_text_timer, aim_start

    _frame_count += 1
    _load_sounds()

    # ── Deferred InputField activation (prevents Shift+A typing 'A') ──────────
    if ctrl._adv_activate_field and _adv_input is not None:
        ctrl._adv_activate_field = False
        _adv_input.text_field.active = True

    # ── Sync status/info text from controller ─────────────────────────────────
    if ctrl.info_msg and info_text.text != ctrl.info_msg:
        info_text.text = ctrl.info_msg

    # ── Result text fade ──────────────────────────────────────────────────────
    if result_text_timer > 0:
        result_text_timer -= ursina_time.dt
        if result_text_timer <= 0 and result_text:
            result_text.text = ""

    # ── Game / Practice: player arrow-key aiming + power charging ─────────────
    if (ctrl.game_mode and ctrl.player_turn and ctrl.mode == "idle") or \
       (ctrl.practice_mode and ctrl.mode == "idle"):
        fine = bool(held_keys["shift"])
        ctrl.update_aim(
            ursina_time.dt,
            rotate_left  = bool(held_keys["left arrow"]),
            rotate_right = bool(held_keys["right arrow"]),
            elev_up      = bool(held_keys["up arrow"]),
            elev_down    = bool(held_keys["down arrow"]),
            fine         = fine,
        )
        _update_game_cue_visual("white")
        _update_power_bar()
        elev_sign = "+" if ctrl.aim_elevation >= 0 else ""
        status_text.text = (
            f"Aim:{ctrl.aim_angle:.0f}deg  Elev:{elev_sign}{ctrl.aim_elevation:.0f}deg"
            f"  Hit:({ctrl.hit_offset[0]*1000:.0f},{ctrl.hit_offset[1]*1000:.0f})mm"
            f"  WASD=hit point  Shift=fine"
        )

    # ── Game: AI turn ─────────────────────────────────────────────────────────
    if ctrl.game_mode and not ctrl.player_turn and ctrl.mode in ("idle", "ai_calculating"):
        ctrl.update_ai_turn(ursina_time.dt)
        if ctrl.mode == "ai_calculating" and ctrl.ai_shot_data:
            _update_game_cue_visual("yellow", ctrl.ai_shot_data.get("angle", 0))

    # ── Script shot animation ─────────────────────────────────────────────────
    if ctrl._script_state != "idle" and ctrl.mode != "running":
        render_state = ctrl.tick_script(ursina_time.dt)
        if render_state:
            _build_script_cue(
                render_state["ball"],
                render_state["az_deg"],
                render_state["el_deg"],
                render_state["dist_back"],
                render_state["show_aim"],
            )

    # ── HitPointUI drag ───────────────────────────────────────────────────────
    hp = _ensure_hitpoint_ui()
    _ensure_params_editor()
    if hp.dragging and mouse.position:
        hp.handle_drag(mouse.position[0], mouse.position[1])

    if not ctrl.game_mode and not ctrl.practice_mode:
        if ctrl.mode == "aiming" and not hp.dragging:
            world_pos = _get_mouse_table_pos()
            if world_pos is not None:
                _update_aim(world_pos)

    # ── WASD: hit-point keyboard control ──────────────────────────────────────
    if ctrl.mode != "running":
        max_off = hp.max_offset
        spd     = max_off * 2.0 * ursina_time.dt
        changed = False
        if held_keys["w"]:
            ctrl.hit_offset[1] = min(max_off, ctrl.hit_offset[1] + spd);  changed = True
        if held_keys["s"] and not ctrl.game_mode and not ctrl.practice_mode:
            ctrl.hit_offset[1] = max(-max_off, ctrl.hit_offset[1] - spd); changed = True
        if held_keys["a"] and not ctrl._adv_active:
            ctrl.hit_offset[0] = max(-max_off, ctrl.hit_offset[0] - spd); changed = True
        if held_keys["d"]:
            ctrl.hit_offset[0] = min(max_off, ctrl.hit_offset[0] + spd);  changed = True
        if changed:
            radius = hp.panel_size / 2
            nx = hp.bg.x + (ctrl.hit_offset[0] / max_off) * radius
            ny = hp.bg.y + (ctrl.hit_offset[1] / max_off) * radius
            hp.dot.position = (nx, ny)

    # ── Physics step → controller ─────────────────────────────────────────────
    ctrl.step(ursina_time.dt)

    # ── Process pending events (L2 → L3 rendering commands) ──────────────────
    for ev in ctrl.pending_events:
        _handle_controller_event(ev)
    ctrl.pending_events.clear()

    # ── Sounds ────────────────────────────────────────────────────────────────
    _play_collision_sounds(ctrl.physics_events)
    _update_roll_sound()

    # ── Sync controller status_msg → status_text ─────────────────────────────
    # (only when not in active aiming mode, which writes status_text directly)
    if ctrl.mode != "idle" or not (
        (ctrl.game_mode and ctrl.player_turn) or ctrl.practice_mode
    ):
        if ctrl.status_msg:
            status_text.text = ctrl.status_msg

    # ── Ball entity position/rotation sync ────────────────────────────────────
    if ctrl.mode == "running" and ctrl.physics_balls:
        dt_total = ctrl.SIM_DT * ctrl.SIM_SUBSTEPS
        for pb in ctrl.physics_balls:
            ent = ball_entities.get(pb.name)
            if ent is None:
                continue
            ent.position = Vec3(pb.position[0], BALL_RADIUS + pb.position[1], pb.position[2])
            w     = pb.angular_velocity
            w_mag = float(np.linalg.norm(w))
            if w_mag > 1e-6:
                angle_deg = math.degrees(w_mag * dt_total)
                axis = w / w_mag
                dq = Quat()
                dq.setFromAxisAngle(angle_deg, LVector3f(axis[0], axis[1], axis[2]))
                ent.setQuat(ent.getQuat() * dq)

        if _frame_count % 3 == 0:
            _update_trails()
            _update_spin_arrows()

    elif ctrl.mode == "idle" and ctrl.physics_balls:
        for pb in ctrl.physics_balls:
            ent = ball_entities.get(pb.name)
            if ent is not None:
                ent.position = Vec3(pb.position[0], BALL_RADIUS + pb.position[1], pb.position[2])


# ──────────────────────────────────────────
# Public scripting API (backward-compat shims)
# ──────────────────────────────────────────

def execute_script(script: dict) -> None:
    """Backward-compat: delegate to ctrl.execute_script()."""
    ctrl.execute_script(script)


def load_script_file(path: str) -> None:
    """Backward-compat: delegate to ctrl.load_script_file()."""
    ctrl.load_script_file(path)


def reload_script() -> None:
    """Backward-compat: delegate to ctrl.reload_script()."""
    ctrl.reload_script()


# ──────────────────────────────────────────
# Run
# ──────────────────────────────────────────

if __name__ == "__main__":
    app.run()
