"""
export.py — export a trained nightcall actor to ONNX.

The exported model:
    input  "observation"   float32  [1, FEATURE_DIM]
    output "action_logits" float32  [1, MAX_P1_UNITS * N_UNIT_ACTIONS]

Downstream use (in-game AI, Phase 3):
    logits = session.run(["action_logits"], {"observation": obs})[0]
    logits = logits.reshape(MAX_P1_UNITS, N_UNIT_ACTIONS)
    actions = logits.argmax(axis=-1)   # greedy

Usage:
    python python/training/export.py checkpoints/ckpt_final.pt
    python python/training/export.py checkpoints/ckpt_final.pt --out assets/models/rts_policy.onnx
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import torch

# Force UTF-8 stdout so torch.onnx's emoji-laden progress messages don't crash
# on Windows consoles that use cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from model import NightcallActorCritic, ActorOnly
from env   import FEATURE_DIM, MAX_P1_UNITS, N_UNIT_ACTIONS


def export(checkpoint_path: str, out_path: str, opset: int = 17):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt)   # handle both raw and wrapped dicts

    full_model = NightcallActorCritic()
    full_model.load_state_dict(state_dict)
    full_model.eval()

    actor = ActorOnly(full_model)
    actor.eval()

    dummy_obs = torch.zeros(1, FEATURE_DIM)

    # Verify forward pass shape before exporting
    with torch.no_grad():
        logits = actor(dummy_obs)
    assert logits.shape == (1, MAX_P1_UNITS * N_UNIT_ACTIONS), \
        f"Unexpected output shape: {logits.shape}"

    print(f"Exporting to ONNX (opset {opset}): {out}")
    # Use torch.jit.trace + legacy export path to avoid the dynamo-based exporter
    # which prints Unicode emoji and may fail on newer opset versions.
    torch.onnx.export(
        actor,
        dummy_obs,
        str(out),
        input_names=["observation"],
        output_names=["action_logits"],
        dynamic_axes={
            "observation":   {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
        opset_version=opset,
        dynamo=False,   # use legacy TorchScript-based exporter; stable across versions
    )

    print(f"Exported successfully.")
    print(f"  Input:  observation    float32 [batch, {FEATURE_DIM}]")
    print(f"  Output: action_logits  float32 [batch, {MAX_P1_UNITS * N_UNIT_ACTIONS}]")
    print(f"          → reshape to [{MAX_P1_UNITS}, {N_UNIT_ACTIONS}] per sample")


def _parse():
    p = argparse.ArgumentParser(description="Export nightcall actor to ONNX")
    p.add_argument("checkpoint", help="Path to .pt checkpoint file")
    p.add_argument("--out",   default="assets/models/rts_policy.onnx",
                   help="Output .onnx path (default: assets/models/rts_policy.onnx)")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version (default: 17)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    export(args.checkpoint, args.out, args.opset)
