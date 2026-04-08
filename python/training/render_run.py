"""
render_run.py — after training, produce the same artifacts as sweep (MP4 + JSONL + metrics).

Reads a ``train.py`` checkpoint (``model``, optional ``opponent``, ``p2_policy``).
Scenario defaults match :class:`NightcallEnv` (30×30, 4v4); override with env vars
``RENDER_MAP_SIZE``, ``RENDER_P1_UNITS``, ``RENDER_P2_UNITS``.

CLI:
    python python/training/render_run.py checkpoints/ckpt_final.pt --output-dir results/demo
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
from pathlib import Path

import torch

_PY = Path(__file__).resolve().parent.parent
_REPO = _PY.parent
sys.path.insert(0, str(_PY))
sys.path.insert(0, str(_PY / "training"))
sys.path.insert(0, str(_REPO / "build" / "python"))

import sweep  # noqa: E402
from model import NightcallActorCritic  # noqa: E402
from train import make_selfplay_greedy  # noqa: E402


def post_train_render(
    ckpt_path: Path,
    train_cfg: dict,
    results_root: Path,
) -> None:
    """
    Write ``baseline.mp4``, ``trained.mp4``, replay JSONLs, ``metrics.txt`` under
    ``results_root / <run_name>/`` (same layout as ``sweep.py``).

    :param ckpt_path: Path to ``ckpt_final.pt`` (or any compatible checkpoint).
    :param train_cfg: Training config dict (uses ``p2_policy``).
    :param results_root: Parent directory (created if missing).
    """
    results_root = Path(results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    map_size = int(os.environ.get("RENDER_MAP_SIZE", "30"))
    p1u = int(os.environ.get("RENDER_P1_UNITS", "4"))
    p2u = int(os.environ.get("RENDER_P2_UNITS", "4"))
    fps = int(os.environ.get("RENDER_FPS", "15"))
    name = (os.environ.get("RENDER_RUN_NAME") or "train_default").strip() or "train_default"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    p2_policy = train_cfg.get("p2_policy") or ckpt.get("p2_policy") or "scripted"

    run_cfg = {
        "name": name,
        "map_size": map_size,
        "p1_units": p1u,
        "p2_units": p2u,
        "p2_policy": p2_policy,
    }

    out = results_root / name
    out.mkdir(parents=True, exist_ok=True)

    model = NightcallActorCritic().to(device)
    model.load_state_dict(ckpt["model"])
    opponent = NightcallActorCritic().to(device)
    if p2_policy == "neural":
        if "opponent" in ckpt:
            opponent.load_state_dict(ckpt["opponent"])
        else:
            opponent.load_state_dict(copy.deepcopy(model.state_dict()))
    else:
        opponent.load_state_dict(copy.deepcopy(model.state_dict()))
    opponent.eval()
    model.eval()

    print(f"Rendering sweep-style artifacts -> {out}  (device={device})")

    print("  Generating baseline replay...")
    baseline_frames = sweep.generate_replay(run_cfg, policy=None)
    sweep.save_jsonl(baseline_frames, out / "replay_baseline.jsonl")
    sweep.render_episode_mp4(
        baseline_frames,
        out / "baseline.mp4",
        fps=fps,
        label=f"{name} — random policy",
    )

    print("  Generating trained replay...")
    if p2_policy == "neural":
        pol = make_selfplay_greedy(model, opponent, device)
        trained_frames = sweep.generate_replay(run_cfg, policy=pol)
    else:
        pol = sweep._make_policy(model, device)
        trained_frames = sweep.generate_replay(run_cfg, policy=pol)
    sweep.save_jsonl(trained_frames, out / "replay_trained.jsonl")
    sweep.render_episode_mp4(
        trained_frames,
        out / "trained.mp4",
        fps=fps,
        label=f"{name} — trained",
    )

    print("  Evaluating (20 episodes)...")
    eval_metrics = sweep.evaluate_winrate(
        model, run_cfg, n_episodes=20, device=device, opponent=opponent,
    )

    shutil.copy2(ckpt_path, out / "checkpoint_used.pt")
    metrics_path = out / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint source: {ckpt_path.resolve()}\n")
        f.write(f"Copied to: {out / 'checkpoint_used.pt'}\n")
        f.write(f"Map: {map_size}x{map_size}  Units: {p1u}v{p2u}  P2 policy: {p2_policy}\n\n")
        f.write("--- Evaluation (20 episodes) ---\n")
        f.write(f"Win rate:   {eval_metrics['win_rate']:.0%}\n")
        f.write(f"Loss rate:  {eval_metrics['loss_rate']:.0%}\n")
        f.write(f"Draw rate:  {eval_metrics['draw_rate']:.0%}\n")
        f.write(f"Mean length: {eval_metrics['mean_length']:.0f} ticks\n")

    print(f"  Saved: {out}/  (baseline.mp4, trained.mp4, replays, metrics.txt)")


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render MP4 replays from a train.py checkpoint.")
    p.add_argument("checkpoint", type=Path, help="Path to .pt checkpoint")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Root output directory (default: results/)",
    )
    p.add_argument(
        "--p2-policy",
        choices=("scripted", "neural"),
        default=None,
        help="Override checkpoint p2_policy if needed (default: read from checkpoint)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_cli()
    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    cfg: dict = {}
    if args.p2_policy is not None:
        cfg["p2_policy"] = args.p2_policy
    post_train_render(ckpt, cfg, args.output_dir.resolve())


if __name__ == "__main__":
    main()
