# Nightcall — C++20 RTS Game Architecture Plan

## Context

Building a multiplatform real-time strategy game ("nightcall") from scratch. The game's central design goal is AI that is *strategically intelligent* (trained via reinforcement learning) rather than artificially handicapped. To enable this, the game must be fully deterministic, have a clean command abstraction layer shared by both the human player and the AI agent, and the simulation engine must be runnable headlessly in a Python training loop via C++ bindings.

**Tech stack:** C++20, SDL3, bgfx, ONNX Runtime (in-game inference), PyTorch + pybind11 (training pipeline), vcpkg, CMake.

**Training (implemented):** `python/bindings/sim_bindings.cpp` exposes `SimEngine`, `GameState.make_default` (optional `spawn_seed` for jittered spawns), and observation `to_feature_vector()`. Collision separation is **1 tile** between centres (`engine/src/systems/collision_system.cpp`); sweep videos draw each unit with a **0.5-tile** radius in map space. The flat observation pads **128** units × **8** floats each (position, HP, motion, velocity, …). Policy I/O sizes live in `python/training/env.py`. Details: [TRAINING.md](TRAINING.md).

---

## Directory Structure

```
nightcall/
├── CMakeLists.txt
├── vcpkg.json
├── engine/                    # Core sim — zero platform deps
│   ├── include/engine/
│   │   ├── types.hpp          # UnitId, PlayerId, TileCoord, FixedVec2, enums
│   │   ├── game_state.hpp     # GameState, Unit, Tile, Player, ResourceNode
│   │   ├── command.hpp        # Command variant + all command structs
│   │   ├── command_validator.hpp
│   │   ├── sim_engine.hpp
│   │   └── observation.hpp
│   └── src/
│       ├── sim_engine.cpp
│       ├── command_validator.cpp
│       ├── observation.cpp
│       └── systems/
│           ├── movement_system.cpp
│           ├── combat_system.cpp
│           ├── gathering_system.cpp
│           ├── training_system.cpp
│           └── building_system.cpp
├── ai/                        # In-game inference — depends on engine + ONNX RT
│   ├── include/ai/
│   │   ├── ai_agent.hpp
│   │   └── feature_encoder.hpp
│   └── src/
│       ├── ai_agent.cpp
│       └── feature_encoder.cpp
├── renderer/                  # bgfx rendering — read-only access to GameState
│   ├── include/renderer/
│   │   ├── renderer.hpp
│   │   ├── tile_renderer.hpp
│   │   ├── unit_renderer.hpp
│   │   └── ui_renderer.hpp
│   └── src/...
├── input/                     # SDL3 events → Command API
│   └── src/input_handler.cpp
├── audio/
│   └── src/audio_system.cpp
├── app/
│   └── src/
│       ├── main.cpp
│       ├── game_loop.hpp
│       └── game_loop.cpp
├── python/
│   ├── bindings/
│   │   ├── sim_bindings.cpp   # pybind11: SimEngine, GameState, Command
│   │   └── obs_bindings.cpp   # pybind11: Observation, feature vectors
│   └── training/
│       ├── env.py             # gym.Env wrapper
│       ├── model.py           # PyTorch actor-critic
│       ├── train.py           # PPO self-play loop
│       └── export.py          # torch.onnx.export → rts_policy.onnx
├── assets/
│   ├── shaders/               # bgfx .sc shader sources
│   ├── maps/                  # JSON map definitions
│   └── models/rts_policy.onnx
└── tests/
    └── engine/
        ├── test_sim_determinism.cpp   # golden checksum test — most critical
        ├── test_commands.cpp
        └── test_observation.cpp
```

---

## Key Design Decisions

### 1. Fixed-Point Arithmetic (FixedVec2, int32_t positions in 1/256 tile units)
Guarantees bit-identical results across platforms, compilers, and optimization levels. Required for Python training loop to match in-game behavior exactly.

### 2. C++ Bindings (pybind11) Over Python Reimplementation
Training runs the same C++ `SimEngine` via `nightcall_sim` pybind11 module. No divergence possible. Any gameplay fix in C++ is automatically reflected in training.

### 3. Command API as the Only Mutation Path
Both player and AI submit `CommandBatch` through `CommandValidator` before `SimEngine::tick()`. Replay = record batches per tick. Cheating structurally impossible. Testing trivial.

### 4. Observation API Enforces Fog of War
AI receives `Observation` (filtered by sight range), not raw `GameState`. Trains the policy to handle incomplete information — produces smarter, more robust AI behavior.

### 5. Non-Blocking AI on Dedicated Thread
Game loop runs at 10 sim ticks/sec and never waits on AI inference. `AIAgent` uses condition variable + double-buffer: game loop posts observation and retrieves last computed batch. If inference is slow, the AI repeats its last command. Keeps game responsive at 60Hz.

### 6. Deterministic Systems — Fixed Order Every Tick
`apply_commands → movement → combat → gathering → training → building → win_condition → cleanup`
This ordering is a hard spec. Tests enforce it with golden checksums.

---

## Core Type Sketches

### types.hpp
```cpp
enum class UnitId    : uint32_t { Invalid = 0 };
enum class PlayerId  : uint8_t  { None=0, P1=1, P2=2 };
enum class UnitType  : uint8_t  { Worker, Soldier, Archer, Siege };
enum class TerrainType: uint8_t { Plains, Forest, Water, Mountain };
enum class ResourceType:uint8_t { Gold, Wood };

struct FixedVec2 { int32_t x, y; static constexpr int32_t SCALE = 256; };
struct TileCoord { int16_t x, y; };
```

### command.hpp
```cpp
struct MoveCmd   { PlayerId issuer; UnitId unit; TileCoord dest; };
struct AttackCmd { PlayerId issuer; UnitId attacker; UnitId target; };
struct GatherCmd { PlayerId issuer; UnitId worker; TileCoord resource_coord; };
struct TrainCmd  { PlayerId issuer; StructId barracks; UnitType unit; };
struct BuildCmd  { PlayerId issuer; UnitId worker; TileCoord dest; StructureType type; };
struct NoOpCmd   { PlayerId issuer; };
using Command = std::variant<MoveCmd,AttackCmd,GatherCmd,TrainCmd,BuildCmd,NoOpCmd>;
struct CommandBatch { std::vector<Command> commands; };
```

### sim_engine.hpp
```cpp
class SimEngine {
public:
    void tick(GameState& state, CommandBatch const& batch);
    // For training: run full game, return result
    struct RolloutResult { GameState final_state; uint32_t ticks; };
    RolloutResult rollout(GameState initial,
        std::function<CommandBatch(GameState const&, PlayerId)> p1_policy,
        std::function<CommandBatch(GameState const&, PlayerId)> p2_policy,
        uint32_t max_ticks);
};
```

---

## Data Flow

```
SDL Events → InputHandler → CommandBatch[P1] ─┐
                                               ├→ CommandValidator → SimEngine::tick(GameState)
AIAgent::poll_commands() → CommandBatch[P2] ──┘         │
        ↑                                        GameState (tick N+1)
        └── AIAgent::push_observation() ←── observe(state, P2)
              (wakes inference thread)

Renderer reads GameState const* each frame → bgfx draw calls → swapchain
```

---

## Build System

### vcpkg.json dependencies
- `sdl3` ≥3.2.0
- `bgfx` (custom overlay port — not in main registry; use bgfx.cmake FetchContent)
- `onnxruntime` ≥1.20.0
- `pybind11` ≥2.13.0
- `nlohmann-json` ≥3.11.0
- `catch2` ≥3.7.0
- `spdlog` ≥1.14.0

### CMake targets
- `nightcall_engine` — static lib, zero platform deps
- `nightcall_ai` — static lib, links engine + onnxruntime
- `nightcall_renderer` — static lib, links engine + bgfx
- `nightcall_input` — static lib, links engine + SDL3
- `nightcall_audio` — static lib, links SDL3
- `nightcall` — executable, links all above
- `nightcall_sim` — pybind11 module, links engine only
- `test_engine` — Catch2 test runner

**bgfx note:** Shaders must be compiled offline with `shaderc` per backend (HLSL/GLSL/Metal/SPIR-V). This is the most operationally complex part of the stack — add a CMake custom command to invoke shaderc as part of the build.

---

## Python Training Pipeline

```python
# env.py — wraps nightcall_sim (pybind11 module)
class NightcallEnv(gym.Env):
    def reset(): state = sim.GameState.load_map(map); return observe(state, P1).to_feature_vector()
    def step(action_index): engine.tick(state, decode(action)); return obs, reward, done, ...

# train.py — PPO self-play
envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
# Standard PPO with GAE; periodically freeze P2 as historical opponent

# export.py
torch.onnx.export(actor_only_model, dummy_obs, "assets/models/rts_policy.onnx",
    input_names=["observation"], output_names=["action_probs"], opset_version=17)
```

**Action space:** Start with flat discrete over a fixed action vocabulary (which unit × action type × target tile from coarse grid). Expand to autoregressive if unit count grows.

---

## Implementation Order

### Phase 1 — Engine (build this first, no platform deps)
1. `engine/include/engine/types.hpp`
2. `engine/include/engine/game_state.hpp` + serialize/deserialize
3. `engine/include/engine/command.hpp`
4. `engine/src/systems/` — all 5 systems
5. `engine/src/command_validator.cpp`
6. `engine/src/sim_engine.cpp`
7. `tests/engine/test_sim_determinism.cpp` ← **most important test**
8. `engine/src/observation.cpp`

### Phase 2 — Python Bindings + Training Smoke Test
9. `python/bindings/sim_bindings.cpp` + `obs_bindings.cpp`
10. `python/training/env.py` + `model.py`
11. Run 1M ticks from Python, assert determinism
12. `python/training/train.py` basic PPO, `export.py` ONNX export

### Phase 3 — In-Game AI
13. `ai/src/feature_encoder.cpp` — must match Python's `to_feature_vector()`
14. `ai/src/ai_agent.cpp` — ONNX RT session + inference thread

### Phase 4 — Platform Layer (headless game loop first)
15. `app/src/main.cpp` — SDL3 init
16. `input/src/input_handler.cpp`
17. `app/src/game_loop.cpp` — wire sim + input + AI, **no renderer yet**

### Phase 5 — Rendering
18. `renderer/src/renderer.cpp` — bgfx init/frame
19. `assets/shaders/` — compile vs_tile/fs_tile with shaderc
20. `renderer/src/tile_renderer.cpp` → `unit_renderer.cpp` → `ui_renderer.cpp`
21. Wire renderer into game loop

### Phase 6 — Polish
22. Audio, map loading, full training run, AI asset packaging

---

## Verification

- **Determinism test:** Run same initial state + same command sequence twice; assert `GameState::serialize()` output is byte-identical both times, and matches a golden checksum. Run on CI for all platforms.
- **Python parity test:** Run 1000 ticks from C++ test, serialize final state. Run same sequence from Python via pybind11, compare checksums.
- **AI integration test:** Load a dummy ONNX model with correct input/output shapes, run one game loop tick, assert no crash and commands are valid.
- **End-to-end:** Play a full game (human vs AI agent loaded from ONNX) and verify winner is determined within reasonable tick count.

---

## Critical Files

| File | Why Critical |
|------|-------------|
| `engine/include/engine/game_state.hpp` | Ground truth data model; everything depends on this being stable |
| `engine/include/engine/command.hpp` | Command API contract; changes ripple everywhere |
| `engine/src/sim_engine.cpp` | Deterministic tick; must match Python training exactly |
| `ai/src/ai_agent.cpp` | ONNX session lifecycle + threading; most concurrency-sensitive |
| `python/bindings/sim_bindings.cpp` | C++↔Python bridge; type mismatches cause silent training bugs |
