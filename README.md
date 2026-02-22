# Four-ball Billiards (Ursina)

3D four-ball carom billiards simulator with real-time physics, AI opponent, and a headless RL API.

Built with **Ursina Engine** (Panda3D) — fully custom physics, no external physics library.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Custom physics engine** — sliding/rolling friction, spin (English), cushion rebound, ball-ball throw, masse/jump shots, curve force
- **3D desktop rendering** — Ursina engine with orbit camera, trail visualization, angular velocity arrows
- **Game mode** — Player vs AI, first to 10 points
- **Practice mode** — free shot practice with full parameter control
- **5 shot presets** — Follow, Draw, Stop, Bank, Nejire (3-cushion)
- **Scripting system** — Python-based shot scripts with stroke animation
- **Advanced command panel** — JSON-based ball placement, precision shots
- **Physics parameter editor** — tune friction, restitution, and more in real time
- **Hit-point selector** — visual tip-offset indicator for precise English control
- **Synthesized audio** — click, cushion, and rolling sounds generated via NumPy (no audio files)
- **Headless RL API** — rendering-free simulation for reinforcement learning, OpenAI Gym compatible

---

## Architecture

```
┌──────────────────┐   events / state   ┌──────────────┐     ┌──────────────┐
│  Ursina Renderer │ ◄════════════════► │  Controller  │ ──► │  Physics     │
│  main.py         │   pending_events   │  controller  │     │  Engine      │
│  (Layer 3)       │   physics_events   │  .py (L2)    │     │  physics.py  │
└──────────────────┘                    └──────────────┘     │  (L1)        │
                                                              └──────────────┘
```

| Layer | File | Role |
|-------|------|------|
| **Layer 1** | `physics.py` | Pure physics engine (friction, collision, cushion, spin) |
| **Layer 2** | `controller.py` | Game logic, state machine, AI, RL API |
| **Layer 3** | `main.py` | Ursina rendering, input handling, UI panels |

Layers 1 & 2 have no rendering dependency, enabling headless RL training.

---

## Quick Start

```bash
# Clone
git clone https://github.com/skoo1/KBilliards_Ursina.git
cd KBilliards_Ursina

# Virtual environment (recommended)
python -m venv venv
source venv/Scripts/activate    # Windows
# source venv/bin/activate      # macOS / Linux

# Install dependencies
pip install ursina numpy pillow

# Run
python main.py
```

---

## Controls

### Mouse (Free Play)

| Action | Input |
|--------|-------|
| Aim & shoot | Left-click drag on cue ball |
| Orbit camera | Left-click drag (off ball) |
| Pan camera | Right-click / wheel-click drag |
| Zoom | Scroll wheel |
| Tip offset | Drag the circle indicator (bottom-right) |

### Keyboard

| Key | Action |
|-----|--------|
| `1`–`5` | Load shot preset (Follow / Draw / Stop / Bank / Nejire) |
| `G` | Start game mode (Player vs AI, first to 10) |
| `T` | Toggle practice mode |
| `R` | Reset |
| `P` | Open/close physics parameter editor |
| `[` / `]` | Adjust selected parameter (-/+) |
| `Shift+P` | Reset all parameters to defaults |
| `Shift+S` | Open script picker |
| `Shift+A` | Open advanced command panel |

### Aiming & Shooting (Game / Practice)

| Key | Action |
|-----|--------|
| `←` / `→` | Aim left / right |
| `↑` / `↓` | Cue elevation (masse / jump) |
| `W` / `S` | Tip offset up / down (topspin / backspin) |
| `A` / `D` | Tip offset left / right (English) |
| `Shift` | Fine adjustment (1/4 speed) |
| `Space` | Hold to charge power, release to shoot |

---

## Headless RL API

The controller provides a rendering-free API for reinforcement learning.

```python
from controller import BilliardsController

ctrl = BilliardsController()
obs = ctrl.reset()  # shape (16,): [x, z, vx, vz] x 4 balls

result = ctrl.simulate_shot(
    ball_name="white",
    azimuth=45.0,       # degrees, 0 = forward (+Z)
    tip=[0.2, 0.5],     # [side, vertical], -1 to +1
    power=60.0,          # 0-100 (% of max)
)

print(result["scored"], result["reward"])
ctrl.set_balls(result["balls"])  # apply result
```

**Reward structure:** +1.0 scored, +0.3 one red hit, -0.1 miss, -0.5 foul.

**Performance:** ~1,000 shots/sec (`sim_dt=0.005`).

---

## Scripting

Place Python scripts in the `scripts/` directory and load them with `Shift+S`.

```python
# scripts/my_shot.py
SCRIPT = {
    "setup": {
        "white":  ( 0.0, -0.5),
        "yellow": ( 0.3,  0.3),
        "red1":   (-0.2,  0.6),
        "red2":   ( 0.2,  0.8),
    },
    "shot": {
        "ball":      "white",
        "azimuth":   5.0,       # 0=+Z, 90=+X, clockwise
        "elevation": 0.0,       # 0=horizontal, >0=masse
        "tip_x":     0.0,       # -1(left) to +1(right)
        "tip_y":     0.15,      # -1(draw) to +1(follow)
        "force":     1.1,       # impulse (N*s), typical 0.3-1.5
    },
}
```

---

## Project Structure

```
KBilliards_Ursina/
├── main.py               # Layer 3 — Ursina renderer, input, UI
├── controller.py         # Layer 2 — game logic, AI, RL API
├── physics.py            # Layer 1 — physics engine
├── shot_presets.py       # 5 built-in shot scenarios
├── scripts/
│   ├── example_shot.py   # Documented script template
│   ├── draw_shot.py      # Draw shot example
│   ├── bank_shot.py      # Bank shot example
│   └── masse_shot.py     # Masse shot example
├── tests/
│   ├── test_physics.py       # 22 edge-case physics tests
│   ├── test_controller.py    # Controller state machine tests
│   ├── test_shot_presets.py  # Preset scenario tests
│   └── test_visual.py        # Visual system tests
├── docs/
│   └── physics_mechanics.md  # Physics derivations & documentation
└── billiards_spec.md         # Project specification (Korean)
```

---

## Dependencies

- Python 3.10+
- [Ursina](https://www.ursinaengine.org/) (3D engine, wraps Panda3D)
- NumPy (physics computations)
- Pillow (procedural ball textures)

---

## Documentation

- [docs/physics_mechanics.md](docs/physics_mechanics.md) — Physics derivations and formulas
- [billiards_spec.md](billiards_spec.md) — Project specification (Korean)
- [scripts/example_shot.py](scripts/example_shot.py) — Annotated script template

---

## License

MIT
