# nightcall

Deterministic fixed-point RTS simulation with a **PPO-trained** policy: C++ engine, Python training, ONNX export for in-game inference.

## Docs

| Document | Contents |
|----------|----------|
| [TRAINING.md](TRAINING.md) | Env, observation/action shapes, sweep, PPO, export, rewards |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Engine layout, determinism, planned game stack |

## Quick links

- **Build:** `cmake --preset default` then `cmake --build build` (produces `build/python/nightcall_sim.pyd`).
- **Train / sweep:** `pip install -r python/requirements.txt` and **ffmpeg** on `PATH` for sweep videos; e.g. `python python/sweep.py --quick --configs 8v8` (uses CUDA when available).
- **Tests:** `build/tests/test_engine.exe` (Catch2).

## Repo layout (high level)

```
engine/     simulation (collision, combat, observation, …)
python/     pybind11 module + training (env, model, train, export) + sweep.py
tests/      C++ engine tests
tools/      replay_logger, visualize.py
```
