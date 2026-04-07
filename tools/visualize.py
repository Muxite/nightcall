#!/usr/bin/env python3
"""
visualize.py — animate a nightcall JSONL replay produced by replay_logger.

Usage:
    python tools/visualize.py [replay.jsonl] [--fps N] [--save out.gif]

Requires: matplotlib
    pip install matplotlib
"""

import json
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

OWNER_COLOUR = {
    1: "#4a90d9",   # P1 — blue
    2: "#e05c5c",   # P2 — red
    0: "#aaaaaa",   # None / draw
}
OWNER_LABEL = {1: "P1", 2: "P2", 0: "None"}


def load_frames(path: str) -> list[dict]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def build_animation(frames: list[dict], fps: int) -> animation.FuncAnimation:
    map_w = frames[0]["map_w"]
    map_h = frames[0]["map_h"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, map_w - 0.5)
    ax.set_ylim(-0.5, map_h - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (tiles)")
    ax.set_ylabel("Y (tiles)")
    ax.grid(True, color="#dddddd", linewidth=0.5)

    # Draw grid lines at tile boundaries
    for x in range(map_w + 1):
        ax.axvline(x - 0.5, color="#eeeeee", linewidth=0.3)
    for y in range(map_h + 1):
        ax.axhline(y - 0.5, color="#eeeeee", linewidth=0.3)

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=OWNER_COLOUR[1], label="P1"),
        mpatches.Patch(color=OWNER_COLOUR[2], label="P2"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    title = ax.set_title("")
    scatter_p1 = ax.scatter([], [], s=180, color=OWNER_COLOUR[1], zorder=3,
                             edgecolors="white", linewidths=1.5)
    scatter_p2 = ax.scatter([], [], s=180, color=OWNER_COLOUR[2], zorder=3,
                             edgecolors="white", linewidths=1.5)

    # HP bars: a thin horizontal line below each unit
    hp_lines_p1: list[plt.Line2D] = []
    hp_lines_p2: list[plt.Line2D] = []

    def update(frame_idx: int):
        frame = frames[frame_idx]
        tick  = frame["tick"]
        units = frame["units"]

        pos_p1, pos_p2 = [], []
        hp_p1,  hp_p2  = [], []

        for u in units:
            owner = u["owner"]
            x, y  = u["x"], u["y"]
            hp_frac = u["hp"] / u["max_hp"] if u["max_hp"] > 0 else 0
            if owner == 1:
                pos_p1.append((x, y))
                hp_p1.append(hp_frac)
            elif owner == 2:
                pos_p2.append((x, y))
                hp_p2.append(hp_frac)

        # Update scatter positions (must always be shape (N, 2))
        scatter_p1.set_offsets(np.array(pos_p1).reshape(-1, 2) if pos_p1 else np.empty((0, 2)))
        scatter_p2.set_offsets(np.array(pos_p2).reshape(-1, 2) if pos_p2 else np.empty((0, 2)))

        # Redraw HP bars (remove old ones, add new ones)
        for ln in hp_lines_p1 + hp_lines_p2:
            ln.remove()
        hp_lines_p1.clear()
        hp_lines_p2.clear()

        bar_h    = 0.18   # height above unit centre
        bar_half = 0.4    # half-width of full bar

        for (x, y), frac in zip(pos_p1, hp_p1):
            full, = ax.plot([x - bar_half, x + bar_half], [y + bar_h, y + bar_h],
                            color="#cccccc", linewidth=2, zorder=2)
            filled, = ax.plot([x - bar_half, x - bar_half + 2 * bar_half * frac],
                               [y + bar_h, y + bar_h],
                               color="#00cc44", linewidth=2, zorder=2)
            hp_lines_p1.extend([full, filled])

        for (x, y), frac in zip(pos_p2, hp_p2):
            full, = ax.plot([x - bar_half, x + bar_half], [y + bar_h, y + bar_h],
                            color="#cccccc", linewidth=2, zorder=2)
            filled, = ax.plot([x - bar_half, x - bar_half + 2 * bar_half * frac],
                               [y + bar_h, y + bar_h],
                               color="#00cc44", linewidth=2, zorder=2)
            hp_lines_p2.extend([full, filled])

        winner = frame.get("winner")
        if winner is not None:
            label = OWNER_LABEL.get(winner, str(winner))
            title.set_text(f"Tick {tick}  —  Winner: {label}")
        else:
            title.set_text(f"Tick {tick}  —  P1:{len(pos_p1)}u  P2:{len(pos_p2)}u")

        return [scatter_p1, scatter_p2, title] + hp_lines_p1 + hp_lines_p2

    interval_ms = max(16, int(1000 / fps))
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=interval_ms, blit=False, repeat=True,
    )
    return anim


def main():
    parser = argparse.ArgumentParser(description="Animate a nightcall replay.")
    parser.add_argument("replay", nargs="?", default="replay.jsonl",
                        help="Path to the JSONL replay file (default: replay.jsonl)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Playback speed in frames-per-second (default: 30)")
    parser.add_argument("--save", metavar="FILE",
                        help="Save animation to FILE (.gif or .mp4) instead of showing")
    args = parser.parse_args()

    print(f"Loading {args.replay} ...", end=" ", flush=True)
    frames = load_frames(args.replay)
    print(f"{len(frames)} frames, map {frames[0]['map_w']}×{frames[0]['map_h']}")

    anim = build_animation(frames, args.fps)

    if args.save:
        print(f"Saving to {args.save} ...")
        writer = "pillow" if args.save.endswith(".gif") else "ffmpeg"
        anim.save(args.save, writer=writer, fps=args.fps)
        print("Done.")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
