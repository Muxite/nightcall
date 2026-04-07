# Nightcall — Training & Model Guide

## Overview

Nightcall is a deterministic fixed-point RTS simulation with a PPO-trained AI.
The full pipeline goes: **C++ simulation engine → Python bindings → gymnasium env → PPO training → ONNX export → in-game inference**.

```
engine/          C++20 sim kernel (collision, combat, heal, win)
python/
  training/
    env.py       gymnasium wrapper around nightcall_sim
    model.py     actor-critic network (MLP backbone)
    train.py     standalone PPO training loop
    export.py    export checkpoint → ONNX
  sweep.py       train + render comparison GIFs across configs
```

---

## Simulation Rules

| Rule | Value |
|------|-------|
| Minimum centre-to-centre separation | 1 tile (`COLLISION_RADIUS` = 256 fixed units = 1 tile) |
| Geometric body radius (each unit as a disk) | 0.5 tiles (two disks touch at separation 1 tile) |
| Attack range | 1.8 tiles (= 1.8 × separation constant; combat starts just before contact) |
| Unit speed | 0.25 tiles/tick |
| Unit HP | 100 |
| Damage vs single enemy | 5 HP/tick |
| Damage vs N enemies | floor(5/N) each (min 1) |
| Heal rate | 1 HP/tick, starts 20 ticks after last hit |
| Tick rate (logical) | 10 Hz |

**Collision:** when two alive units have centre distance **below** 1 tile, they are pushed apart. In the sweep GIF renderer, the semi-transparent ring and scatter markers are both scaled to a **0.5-tile** radius in **map coordinates** so they match this model (fixed point size alone would not match data units on large maps).

**Spawns:** `GameState.make_default(..., spawn_seed=0)` uses the legacy fixed columns. A **non-zero** `spawn_seed` applies a small per-unit tile jitter (used by `NightcallEnv` on each `reset()` for varied training episodes). Pass `spawn_seed` explicitly if you need a deterministic layout from Python.

**Win condition:**
- One side loses all units → the other side wins
- Both sides lose all units on the same tick → **draw** (`PlayerId::None`)
- Episode reaches `MAX_TICKS` (1 000) without a winner → **draw** (no terminal bonus for either side)

---

## Quick Start

### 1. Build the C++ bindings

```bash
cmake --preset default  # configures with vcpkg + MinGW
cmake --build build  # builds nightcall_engine + nightcall_sim.pyd
```

The `.pyd` lands in `build/python/` and is auto-found by the Python code.

### 2. Install Python dependencies

```bash
pip install -r python/requirements.txt
# If torch is not available for your Python version, use Python 3.10–3.12:
# pip install torch numpy gymnasium onnx pybind11
```

### 3. Run the sweep (train + encode MP4)

Requires **ffmpeg** on your PATH for H.264 output. Training uses **CUDA** when available (`--cpu` to force the CPU).

```bash
# Full sweep — 5 minutes budget per config, 4 parallel envs
python python/sweep.py --seconds 300 --envs 4

# Specific configs only
python python/sweep.py --seconds 300 --envs 4 --configs 8v8 16v16 64v64

# Quick smoke-test (10 s budget per config, 1 env)
python python/sweep.py --quick --configs 8v8

# Force CPU even if a GPU is present
python python/sweep.py --seconds 60 --cpu
```

The sweep measures **wall-clock time** with `time.perf_counter`, compares it to the requested **budget**, and writes both to `metrics.txt`. It stops before starting a new rollout (or mid-rollout / mid–PPO epoch) when the budget is exhausted.

Output layout:
```
sweep_results/
  8v8/
    baseline.mp4          random-policy replay (video)
    trained.mp4           trained-policy replay
    replay_baseline.jsonl frame data (one JSON object per tick)
    replay_trained.jsonl
    model.pt              PyTorch checkpoint
    metrics.txt           budget vs wall time, throughput, win-rate, …
  16v16/
    ...
```

---

## Sweep Configurations

| Name  | Map    | Units  | Notes |
|-------|--------|--------|-------|
| 8v8   | 40×40  | 8v8    | Baseline — fast to train |
| 16v16 | 60×60  | 16v16  | Medium scale |
| 32v32 | 80×80  | 32v32  | Large scale |
| 64v64 | 100×100 | 64v64 | Heavy — long runs; sweep video replays cap at 500 ticks for speed |

The policy has **`MAX_P1_UNITS` discrete slots** (64 in `python/training/env.py`). Each tick it issues one action per slot; slots map to the first N alive P1 units in engine order. P2 uses a chase heuristic every 25 ticks (`P2_RETARGET`).

---

## Standalone Training (`train.py`)

```bash
# Default: 1 M steps, 4 envs, 4v4 on 30×30
python python/training/train.py

# Larger run
python python/training/train.py --total-steps 5_000_000 --envs 8

# All hyperparameters
python python/training/train.py \
    --total-steps 2_000_000 \
    --num-envs 4 \
    --rollout-steps 128 \
    --update-epochs 8 \
    --lr-start 3e-4 \
    --lr-end 5e-5 \
    --ent-coef-start 0.05 \
    --ent-coef-end 0.005 \
    --clip-coef 0.2 \
    --checkpoint-dir checkpoints/my_run
```

Checkpoints are written to `checkpoints/ckpt_<step>.pt` every 100 updates
and a final `ckpt_final.pt` at the end.

### Key hyperparameters

| Flag | Default | Effect |
|------|---------|--------|
| `--total-steps` | 1 000 000 | Total environment steps |
| `--num-envs` | 4 | Parallel environments (more = faster wall-clock) |
| `--rollout-steps` | 128 | Steps collected per env before each PPO update |
| `--update-epochs` | 8 | PPO gradient passes per rollout |
| `--lr-start` / `--lr-end` | 3e-4 / 5e-5 | Linear LR decay over training |
| `--ent-coef-start` / `--ent-coef-end` | 0.05 / 0.005 | Entropy coef decay (explore → exploit) |
| `--opponent-update-interval` | 30 | Updates between P2 snapshot refreshes |

---

## Loading a Trained Model in Python

```python
import torch
import sys
sys.path.insert(0, 'build/python')
sys.path.insert(0, 'python/training')

from model import NightcallActorCritic
from env   import NightcallEnv

# Load checkpoint
ckpt  = torch.load('sweep_results/8v8/model.pt', map_location='cpu')
model = NightcallActorCritic()
model.load_state_dict(ckpt['model'])
model.eval()

# Run one episode greedily
env = NightcallEnv(map_size=40, p1_units=8, p2_units=8)
obs, _ = env.reset()

for _ in range(1000):
    with torch.no_grad():
        obs_t  = torch.from_numpy(obs).unsqueeze(0)
        logits, _ = model(obs_t)           # (1, MAX_P1_UNITS, N_ACTIONS)
        action = logits.argmax(dim=-1).squeeze(0).numpy()   # (MAX_P1_UNITS,)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print('Winner:', info['winner'])
        break
```

### Greedy vs stochastic inference

```python
from torch.distributions import Categorical
from env import MAX_P1_UNITS

# Stochastic (samples from learned distribution — better for exploration):
logits, _ = model(obs_t)                    # (1, MAX_P1_UNITS, N_ACTIONS)
action = torch.stack([
    Categorical(logits=logits[0, i]).sample()
    for i in range(MAX_P1_UNITS)
]).numpy()

# Greedy (argmax — best for evaluation):
action = logits.argmax(dim=-1).squeeze(0).numpy()
```

---

## Exporting to ONNX (for in-game C++ inference)

```bash
python python/training/export.py sweep_results/8v8/model.pt \
    --out assets/models/rts_policy.onnx \
    --opset 17
```

The exported model (see `FEATURE_DIM`, `MAX_P1_UNITS`, `N_UNIT_ACTIONS` in `env.py`):
- **Input:** `observation` — `float32 [batch, FEATURE_DIM]` (1026 = 2 + 128×8)
- **Output:** `action_logits` — `float32 [batch, MAX_P1_UNITS * N_UNIT_ACTIONS]` (e.g. 64×50 = 3200)

In-game usage:
```cpp
// logits shape: [1, MAX_P1_UNITS * N_ACTIONS] → reshape to [MAX_P1_UNITS, N_ACTIONS]
// action[i] = argmax over row i
auto logits = session.Run({"observation"}, obs)[0];
auto mat    = logits.GetTensorData<float>();
const int A = 50;  // N_UNIT_ACTIONS
for (int i = 0; i < MAX_P1_UNITS; ++i) {
    int best = std::max_element(mat + i*A, mat + (i+1)*A) - (mat + i*A);
    actions[i] = best;   // 0=NoOp, 1..49=move to coarse tile
}
```

---

## Rendering a Replay from JSONL

The sweep saves `.jsonl` replay files. You can re-render them with `tools/visualize.py`:

```bash
python tools/visualize.py sweep_results/8v8/replay_trained.jsonl \
    --out my_replay.gif \
    --fps 20
```

Or render interactively (without `--out`) to open a live matplotlib window.

---

## Action Space

Each tick, the policy outputs one action per unit slot (`MAX_P1_UNITS` slots, e.g. 64):

| Value | Meaning |
|-------|---------|
| 0 | NoOp — hold current position |
| 1–49 | Move to coarse tile (7×7 grid) |

The 7×7 grid divides the map into equal cells; action `k` (1-indexed) maps to cell
`(col = (k-1) % 7, row = (k-1) // 7)` → unit moves to the cell centre.

---

## Observation Space

A `float32` vector of length **FEATURE_DIM** (**1026** = 2 + 128 × 8), produced by `Observation::to_feature_vector()` in `engine/src/observation.cpp`:

| Indices | Content |
|---------|---------|
| 0 | P1 gold / 1000 (normalised) |
| 1 | P1 wood / 500 (normalised) |
| 2 + i×8 + 0 | Unit i: `owner_norm` (+1 = ally, −1 = enemy, 0 = empty) |
| 2 + i×8 + 1 | Unit i: `type` (scaled) |
| 2 + i×8 + 2 | Unit i: `pos_x` (scaled; see encoder) |
| 2 + i×8 + 3 | Unit i: `pos_y` (scaled) |
| 2 + i×8 + 4 | Unit i: HP fraction (current / max) |
| 2 + i×8 + 5 | Unit i: `moving` (1.0 or 0.0) |
| 2 + i×8 + 6 | Unit i: velocity x / `max(unit.max_speed, 1)` in [−1, 1] (direction + speed along x) |
| 2 + i×8 + 7 | Unit i: velocity y / `max(unit.max_speed, 1)` (same for y) |

Allies and enemies use the same encoding, so P1 sees **both** teams’ motion. Up to **128** unit slots (64v64). Unused slots are zero-padded. Unit order follows the observation builder (alive units only).

---

## Reward Signal

Per tick (dense):
```
r = (Δenemy_HP − Δown_HP) / 100
  + 3.0 × enemy_kills_this_tick
  − 1.25 × own_deaths_this_tick
```

Terminal (in addition to dense rewards on the last tick):
```
+100.0 + 15.0 × (surviving_P1 / starting_P1)   P1 wins — large win bonus plus conservation
 0.0                                            draw (time limit or mutual wipe)
−70.0                                           P2 wins
```

Constants live in `python/training/env.py` (`WIN_BONUS`, `CONSERVE_BONUS`, `ENEMY_KILL_BONUS`, …).

---

## Extending the System

### Add a new unit type

1. Add an enum value in [engine/include/engine/types.hpp](engine/include/engine/types.hpp)
2. Append a `UnitDef` entry in [engine/include/engine/unit_def.hpp](engine/include/engine/unit_def.hpp)
3. Rebuild — all systems pick up the new type automatically

### Change map size or unit counts

Pass `map_size`, `p1_units`, `p2_units` to `NightcallEnv`, or add a new entry to `CONFIGS` in [python/sweep.py](python/sweep.py). For scripted layouts from C++, use `GameState.make_default(w, h, p1, p2, spawn_seed=0)` for a fixed spawn or a non-zero seed for jittered spawns.

### Tune game balance

Edit the `k_unit_defs` table in [engine/include/engine/unit_def.hpp](engine/include/engine/unit_def.hpp).
Rebuild the bindings (`cmake --build build --target nightcall_sim`) — no Python changes needed.

### Run TensorBoard during training

```bash
# In a separate terminal:
tensorboard --logdir runs

# Then train with logging (train.py logs to runs/ by default):
python python/training/train.py --total-steps 2_000_000 --num-envs 4
```
