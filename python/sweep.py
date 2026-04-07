#!/usr/bin/env python3
"""
sweep.py — train multiple unit-group configurations and render comparison videos.

For each configuration, the script:
  1. Records a BASELINE replay (random policy) and encodes baseline.mp4
  2. Trains a PPO agent for a wall-clock budget (uses CUDA when available)
  3. Records a TRAINED replay and encodes trained.mp4

Output layout:
  {output_dir}/
    {config_name}/
      baseline.mp4
      trained.mp4
      replay_baseline.jsonl
      replay_trained.jsonl
      metrics.txt        <- timing, win-rate, episode length, reward

Requires ffmpeg on PATH for MP4 encoding.

Usage:
    # Run all configs with default 60 s of training each
    python python/sweep.py

    # Quick demo: 10 s per config
    python python/sweep.py --seconds 10

    # Only specific configs
    python python/sweep.py --configs 8v8 16v16

    # More training time, GPU (default if CUDA available)
    python python/sweep.py --seconds 300 --envs 4

    # Force CPU
    python python/sweep.py --cpu --seconds 60
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")                         # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
_REPO     = _HERE.parent
_BUILD_PY = _REPO / "build" / "python"
sys.path.insert(0, str(_BUILD_PY))
sys.path.insert(0, str(_HERE / "training"))

import nightcall_sim as nc                    # noqa: E402
import torch                                  # noqa: E402
import torch.nn.functional as F               # noqa: E402

from env   import NightcallEnv, MAX_P1_UNITS, N_UNIT_ACTIONS, FEATURE_DIM, MAX_TICKS  # noqa: E402
from model import NightcallActorCritic        # noqa: E402
from train import get_rollout_actions, make_selfplay_greedy  # noqa: E402

# Shorter than MAX_TICKS keeps GIF generation tractable for large battles.
_REPLAY_MAX_TICKS = 500


def _ensure_ffmpeg_on_path() -> None:
    """
    If ``ffmpeg`` is not found, rebuild ``PATH`` from the Machine and User
    ``Path`` values in the registry (same merge as a new logon). IDE shells
    often keep a stale ``PATH`` after editing environment variables until
    the app is restarted.

    :returns: None
    """
    if shutil.which("ffmpeg"):
        return
    if os.name != "nt":
        return
    import winreg

    parts: list[str] = []
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ) as key:
            machine_path, _ = winreg.QueryValueEx(key, "Path")
            parts.extend(p for p in machine_path.split(os.pathsep) if p.strip())
    except OSError:
        pass
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            user_path, _ = winreg.QueryValueEx(key, "Path")
            parts.extend(p for p in user_path.split(os.pathsep) if p.strip())
    except OSError:
        pass
    if not parts:
        return
    seen: set[str] = set()
    merged: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            merged.append(p)
    os.environ["PATH"] = os.pathsep.join(merged)


# ── Preset configurations ─────────────────────────────────────────────────────
# Each entry defines a scenario to train and render.
# "name"      : short slug used as directory name and GIF label
# "map_size"  : width = height (square map)
# "p1_units"  : P1 unit count
# "p2_units"  : P2 unit count

CONFIGS: list[dict] = [
    dict(name="8v8",   map_size=40, p1_units=8,  p2_units=8),
    dict(name="16v16", map_size=60, p1_units=16, p2_units=16),
    dict(name="32v32", map_size=80, p1_units=32, p2_units=32),
    dict(name="64v64", map_size=100, p1_units=64, p2_units=64),
]

# ── Colours (shared with tools/visualize.py) ──────────────────────────────────
OWNER_COLOUR = {1: "#4a90d9", 2: "#e05c5c", 0: "#aaaaaa"}
OWNER_LABEL  = {1: "P1",      2: "P2",      0: "Draw"}

# Engine: overlap when centre distance < 1 tile (COLLISION_RADIUS = SCALE). Two
# equal disks then have radius 0.5 tiles; outline + scatter use this in data coords.
_UNIT_BODY_RADIUS_TILES = 0.5


def _scatter_area_for_radius_tiles(ax, fig, r_tiles: float) -> float:
    """
    Return matplotlib scatter ``s`` (marker area in points^2) so a circle marker
    matches ``r_tiles`` radius in data coordinates (same as ``plt.Circle``).

    :param ax: axes with limits and aspect already set
    :param fig: parent figure
    :param r_tiles: radius in tile units
    :returns: ``s`` suitable for ``ax.scatter(..., s=s)``
    """
    x0, y0 = ax.transData.transform((0.0, 0.0))
    x1, y1 = ax.transData.transform((float(r_tiles), 0.0))
    r_px = float(np.hypot(x1 - x0, y1 - y0))
    r_pt = r_px * (72.0 / float(fig.dpi))
    return float(np.pi * (r_pt ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Replay helpers
# ─────────────────────────────────────────────────────────────────────────────

def _state_to_frame(state: nc.GameState) -> dict:
    """Convert a GameState to the same dict format as replay_logger JSONL."""
    obs = nc.observe(state, nc.PlayerId.P1)    # full visibility; gives moving flag
    winner_val = int(state.winner) if state.winner is not None else None
    return {
        "tick":   state.tick,
        "map_w":  state.map_width,
        "map_h":  state.map_height,
        "winner": winner_val,
        "units": [
            {
                "id":     int(u.id),
                "owner":  int(u.owner),
                "x":      u.pos.x / nc.FixedVec2.SCALE,
                "y":      u.pos.y / nc.FixedVec2.SCALE,
                "hp":     u.hp,
                "max_hp": u.max_hp,
                "moving": u.moving,
            }
            for u in obs.units
        ],
    }


def generate_replay(
    cfg: dict,
    policy,
    max_ticks: int = _REPLAY_MAX_TICKS,
) -> list[dict]:
    """
    Run one episode and return frame dicts. For ``p2_policy=neural``, ``policy``
    must be ``callable(obs_p1, env) -> action`` (e.g. from ``make_selfplay_greedy``).
    """
    pp = cfg.get("p2_policy", "scripted")
    env = NightcallEnv(
        map_size=cfg["map_size"],
        p1_units=cfg["p1_units"],
        p2_units=cfg["p2_units"],
        p2_policy=pp,
    )
    obs, _ = env.reset()
    frames = [_state_to_frame(env._state)]

    for _ in range(max_ticks):
        if policy is None:
            action = env.action_space.sample()
        elif pp == "neural":
            action = policy(obs, env)
        else:
            action = policy(obs)

        obs, _, terminated, truncated, _ = env.step(action)
        frames.append(_state_to_frame(env._state))
        if terminated or truncated:
            break

    return frames


def save_jsonl(frames: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame in frames:
            f.write(json.dumps(frame) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Video rendering (MP4 via ffmpeg; same scene layout as legacy GIF path)
# ─────────────────────────────────────────────────────────────────────────────

def render_episode_mp4(frames: list[dict], out_path: Path, fps: int = 15, label: str = "") -> None:
    """
    Render frame dicts to an H.264 MP4. Requires ffmpeg on PATH.

    :param frames: replay frames from generate_replay / JSONL
    :param out_path: output path (``.mp4`` added if missing)
    :param fps: playback frames per second
    :param label: optional figure title
    :raises RuntimeError: if ffmpeg is not available to Matplotlib
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    map_w = frames[0]["map_w"]
    map_h = frames[0]["map_h"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, map_w - 0.5)
    ax.set_ylim(-0.5, map_h - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (tiles)")
    ax.set_ylabel("Y (tiles)")
    ax.grid(True, color="#dddddd", linewidth=0.4)
    for x in range(map_w + 1):
        ax.axvline(x - 0.5, color="#eeeeee", linewidth=0.25)
    for y in range(map_h + 1):
        ax.axhline(y - 0.5, color="#eeeeee", linewidth=0.25)

    if label:
        fig.suptitle(label, fontsize=9, y=0.98)

    legend_handles = [
        mpatches.Patch(color=OWNER_COLOUR[1], label="P1"),
        mpatches.Patch(color=OWNER_COLOUR[2], label="P2"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    fig.canvas.draw()
    _marker_s = _scatter_area_for_radius_tiles(ax, fig, _UNIT_BODY_RADIUS_TILES)

    title      = ax.set_title("")
    scatter_p1 = ax.scatter([], [], s=_marker_s, color=OWNER_COLOUR[1], zorder=4,
                             edgecolors="white", linewidths=1.0)
    scatter_p2 = ax.scatter([], [], s=_marker_s, color=OWNER_COLOUR[2], zorder=4,
                             edgecolors="white", linewidths=1.0)
    hp_lines:    list = []
    col_circles: list = []

    def update(fi: int):
        frame = frames[fi]
        tick  = frame["tick"]
        pos_p1, pos_p2 = [], []
        hp_p1,  hp_p2  = [], []
        all_units = []
        for u in frame["units"]:
            owner   = u["owner"]
            x, y    = u["x"], u["y"]
            hp_frac = u["hp"] / u["max_hp"] if u["max_hp"] > 0 else 0.0
            if   owner == 1: pos_p1.append((x, y)); hp_p1.append(hp_frac)
            elif owner == 2: pos_p2.append((x, y)); hp_p2.append(hp_frac)
            if owner in (1, 2):
                all_units.append((x, y, owner))

        scatter_p1.set_offsets(np.array(pos_p1).reshape(-1, 2) if pos_p1 else np.empty((0, 2)))
        scatter_p2.set_offsets(np.array(pos_p2).reshape(-1, 2) if pos_p2 else np.empty((0, 2)))

        # Remove previous collision circles and HP bars.
        for artist in col_circles + hp_lines:
            artist.remove()
        col_circles.clear()
        hp_lines.clear()

        for x, y, owner in all_units:
            colour = OWNER_COLOUR[owner]
            circ = plt.Circle(
                (x, y), _UNIT_BODY_RADIUS_TILES,
                color=colour, fill=False,
                linewidth=0.45, alpha=0.4, zorder=2,
            )
            ax.add_patch(circ)
            col_circles.append(circ)

        # HP bars above each unit.
        bar_half = 0.38
        bar_h    = 0.62   # above the collision circle
        for (x, y), frac in [*zip(pos_p1, hp_p1), *zip(pos_p2, hp_p2)]:
            bg,   = ax.plot([x - bar_half, x + bar_half], [y + bar_h]*2,
                            color="#cccccc", lw=2, zorder=3)
            fill, = ax.plot([x - bar_half, x - bar_half + 2*bar_half*frac], [y + bar_h]*2,
                            color="#22cc55", lw=2, zorder=3)
            hp_lines.extend([bg, fill])

        winner = frame.get("winner")
        if winner is not None:
            title.set_text(f"Tick {tick} — Winner: {OWNER_LABEL.get(winner, str(winner))}")
        else:
            title.set_text(f"Tick {tick}   P1:{len(pos_p1)}   P2:{len(pos_p2)}")
        return [scatter_p1, scatter_p2, title] + col_circles + hp_lines

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=max(16, 1000 // fps), blit=False, repeat=True,
    )
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    _ensure_ffmpeg_on_path()
    try:
        from matplotlib.animation import FFMpegWriter

        writer = FFMpegWriter(fps=fps)
        anim.save(str(out_path), writer=writer)
    except Exception as err:
        raise RuntimeError(
            "MP4 export needs ffmpeg on your PATH (install ffmpeg and retry). "
            "See https://ffmpeg.org/download.html"
        ) from err
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Training  (inline — avoids subprocess overhead)
# ─────────────────────────────────────────────────────────────────────────────

def _make_policy(model: NightcallActorCritic, device: torch.device):
    """Return a callable obs_np -> action_np using the model greedily."""
    model.eval()
    def policy(obs_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t  = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            logits, _ = model(obs_t)                       # (1, MAX_UNITS, N_ACTIONS)
            action = logits.argmax(dim=-1).squeeze(0)      # (MAX_UNITS,)
        return action.cpu().numpy()
    return policy


class _SyncVecEnv:
    """Tiny synchronous vectorised env used only during the sweep's training."""
    def __init__(self, num: int, cfg: dict):
        pp = cfg.get("p2_policy", "scripted")
        self.envs = [
            NightcallEnv(
                map_size=cfg["map_size"],
                p1_units=cfg["p1_units"],
                p2_units=cfg["p2_units"],
                p2_policy=pp,
            )
            for _ in range(num)
        ]
        self.n = num

    def observe_p2_batch(self) -> np.ndarray:
        return np.stack([e.observe_p2() for e in self.envs])

    def reset(self):
        return np.stack([e.reset()[0] for e in self.envs])

    def step(self, acts):
        obs, rew, done = [], [], []
        for i, e in enumerate(self.envs):
            o, r, term, trunc, _ = e.step(acts[i])
            if term or trunc:
                o, _ = e.reset()
            obs.append(o); rew.append(r); done.append(term or trunc)
        return np.stack(obs), np.array(rew, np.float32), np.array(done, bool)


def _compute_gae(rewards, values, next_val, dones, gamma=0.99, lam=0.97):
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    g   = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        nv = next_val if t == T - 1 else values[t + 1]
        nd = dones[t + 1].float() if t < T - 1 else torch.zeros(N, device=rewards.device)
        delta = rewards[t] + gamma * nv * (1 - nd) - values[t]
        g     = delta + gamma * lam * (1 - nd) * g
        adv[t] = g
    return adv, adv + values


def train_timed(
    cfg: dict,
    seconds: float,
    num_envs: int = 4,
    rollout_steps: int = 256,
    update_epochs: int = 6,
    num_minibatches: int = 8,
    lr_start: float = 2.5e-4,
    lr_end: float = 1e-5,
    ent_start: float = 0.06,
    ent_end: float = 0.002,
    clip_coef: float = 0.2,
    vf_coef: float = 0.4,
    device: torch.device = torch.device("cpu"),
) -> tuple[NightcallActorCritic, NightcallActorCritic, dict]:
    """
    Train a PPO agent for at most ``seconds`` wall-clock seconds (``time.perf_counter``).

    Stops before starting a new rollout or mid-rollout when the budget is exhausted;
    does not apply a PPO update on a partial rollout. LR/entropy anneal against the
    requested budget.

    :param cfg: sweep config dict (map_size, p1_units, p2_units, optional p2_policy)
    :param seconds: wall-clock training budget (seconds)
    :param device: torch device (CUDA recommended when available)
    :returns: ``model``, ``opponent`` snapshot, metrics (``opponent`` mirrors P1 when scripted)
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model     = NightcallActorCritic().to(device)
    opponent  = copy.deepcopy(model)
    opponent.eval()
    opt       = torch.optim.Adam(model.parameters(), lr=lr_start, eps=1e-5)
    envs       = _SyncVecEnv(num_envs, cfg)
    obs_np     = envs.reset()
    neural_p2  = cfg.get("p2_policy", "scripted") == "neural"

    T, N = rollout_steps, num_envs
    obs_buf  = torch.zeros(T, N, FEATURE_DIM, device=device)
    act_buf  = torch.zeros(T, N, MAX_P1_UNITS, dtype=torch.long, device=device)
    logp_buf = torch.zeros(T, N, device=device)
    rew_buf  = torch.zeros(T, N, device=device)
    val_buf  = torch.zeros(T, N, device=device)
    done_buf = torch.zeros(T, N, dtype=torch.bool, device=device)

    total_steps = 0
    updates     = 0
    ep_rewards: list[float] = []
    t0          = time.perf_counter()
    deadline    = t0 + float(seconds)

    def over_budget() -> bool:
        return time.perf_counter() >= deadline

    while not over_budget():
        elapsed = time.perf_counter() - t0
        frac    = min(elapsed / max(seconds, 1e-9), 1.0)
        lr      = lr_start + (lr_end - lr_start) * frac
        ent     = ent_start + (ent_end - ent_start) * frac
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.eval()
        obs_t = torch.from_numpy(obs_np).to(device)

        for step in range(T):
            if over_budget():
                if device.type == "cuda":
                    torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                return model, opponent, {
                    "budget_seconds":   float(seconds),
                    "wall_seconds":     wall,
                    "steps":            total_steps,
                    "updates":          updates,
                    "mean_ep_reward":   float(np.mean(ep_rewards)) if ep_rewards else 0.0,
                    "device":           str(device),
                    "stopped_early":    True,
                    "p2_policy":        cfg.get("p2_policy", "scripted"),
                }

            actions, logp, val = get_rollout_actions(
                model, opponent, obs_t, envs, neural_p2
            )
            acts_np = actions.cpu().numpy()
            obs_np, rew_np, done_np = envs.step(acts_np)

            obs_buf[step]  = obs_t
            act_buf[step]  = actions[:, :MAX_P1_UNITS]
            logp_buf[step] = logp
            val_buf[step]  = val.squeeze(-1)
            rew_buf[step]  = torch.from_numpy(rew_np).to(device)
            done_buf[step] = torch.from_numpy(done_np).to(device)
            obs_t = torch.from_numpy(obs_np).to(device)
            total_steps += N

        ep_rewards.extend(rew_buf[done_buf].cpu().tolist())

        with torch.no_grad():
            nv = model.get_value(obs_t).squeeze(-1)
        adv, ret = _compute_gae(rew_buf, val_buf, nv, done_buf)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        model.train()
        flat_obs  = obs_buf.view(T * N, FEATURE_DIM)
        flat_act  = act_buf.view(T * N, MAX_P1_UNITS)
        flat_logp = logp_buf.view(T * N)
        flat_adv  = adv.view(T * N)
        flat_ret  = ret.view(T * N)
        mb = max(1, T * N // num_minibatches)

        stop_opt = False
        for _ in range(update_epochs):
            if over_budget():
                stop_opt = True
                break
            perm = torch.randperm(T * N, device=device)
            for s in range(0, T * N, mb):
                if over_budget():
                    stop_opt = True
                    break
                idx = perm[s:s + mb]
                _, nlp, entropy, nval = model.get_action_and_value(
                    flat_obs[idx], flat_act[idx]
                )
                ratio = torch.exp(nlp - flat_logp[idx])
                a     = flat_adv[idx]
                pg    = torch.max(
                    -a * ratio,
                    -a * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef),
                ).mean()
                vf   = F.mse_loss(nval.squeeze(-1), flat_ret[idx])
                loss = pg + vf_coef * vf - ent * entropy.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
            if stop_opt:
                break

        updates += 1
        if neural_p2 and updates % 30 == 0:
            opponent.load_state_dict(copy.deepcopy(model.state_dict()))

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    metrics = {
        "budget_seconds": float(seconds),
        "wall_seconds":   wall,
        "steps":          total_steps,
        "updates":        updates,
        "mean_ep_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "device":         str(device),
        "p2_policy":      cfg.get("p2_policy", "scripted"),
    }
    return model, opponent, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_winrate(
    model: NightcallActorCritic,
    cfg: dict,
    n_episodes: int = 20,
    device=torch.device("cpu"),
    opponent: NightcallActorCritic | None = None,
) -> dict:
    """
    Run n_episodes with greedy P1 (and greedy P2 from ``opponent`` when neural).

    :param opponent: required when ``cfg['p2_policy'] == 'neural'``
    :returns: win/loss/draw rates and mean episode length
    """
    pp = cfg.get("p2_policy", "scripted")
    if pp == "neural":
        if opponent is None:
            raise ValueError("neural P2 evaluation requires opponent model")
        policy = make_selfplay_greedy(model, opponent, device)
    else:
        policy = _make_policy(model, device)

    wins = 0
    losses = 0
    draws = 0
    lengths = []

    for _ in range(n_episodes):
        env = NightcallEnv(
            map_size=cfg["map_size"],
            p1_units=cfg["p1_units"],
            p2_units=cfg["p2_units"],
            p2_policy=pp,
        )
        obs, _ = env.reset()
        for t in range(MAX_TICKS):
            if pp == "neural":
                step_action = policy(obs, env)
            else:
                step_action = policy(obs)
            obs, _, term, trunc, info = env.step(step_action)
            if term or trunc:
                w = info["winner"]
                if   w == nc.PlayerId.P1:
                    wins += 1
                elif w == nc.PlayerId.P2:
                    losses += 1
                else:
                    draws += 1
                lengths.append(t + 1)
                break
        else:
            draws += 1
            lengths.append(MAX_TICKS)

    return {
        "win_rate":    wins / n_episodes,
        "loss_rate":   losses / n_episodes,
        "draw_rate":   draws / n_episodes,
        "mean_length": float(np.mean(lengths)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    configs:    list[dict],
    seconds:    float,
    num_envs:   int,
    fps:        int,
    output_dir: Path,
    force_cpu:  bool = False,
    p2_policy:  str = "scripted",
) -> None:
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_msg = f"{device}"
    if device.type == "cuda" and torch.cuda.is_available():
        dev_msg = f"{device} ({torch.cuda.get_device_name(0)})"
    print(f"Torch device: {dev_msg}")
    print(f"P2 policy mode: {p2_policy}  (scripted=bot, neural=AI vs AI)")

    summary_rows: list[dict] = []

    for i, cfg in enumerate(configs):
        name = cfg["name"]
        run_cfg = {**cfg, "p2_policy": p2_policy}
        print(f"\n[{i+1}/{len(configs)}] Config: {name}  "
              f"({cfg['p1_units']}v{cfg['p2_units']} on {cfg['map_size']}x{cfg['map_size']})")
        cfg_dir = output_dir / name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        # ── Baseline replay (random policy) ──────────────────────────────────
        print(f"  Generating baseline replay...")
        baseline_frames = generate_replay(run_cfg, policy=None)
        save_jsonl(baseline_frames, cfg_dir / "replay_baseline.jsonl")
        print(f"  Encoding baseline.mp4  ({len(baseline_frames)} frames)...")
        render_episode_mp4(
            baseline_frames,
            cfg_dir / "baseline.mp4",
            fps=fps,
            label=f"{name} — random policy",
        )

        # ── Train ─────────────────────────────────────────────────────────────
        print(f"  Training (budget {seconds:.1f}s wall-clock)...")
        model, opponent, train_metrics = train_timed(
            run_cfg, seconds=seconds, num_envs=num_envs, device=device,
        )
        sps = train_metrics["steps"] / max(train_metrics["wall_seconds"], 1e-9)
        print(f"  Training done: {train_metrics['steps']:,} env-steps in "
              f"{train_metrics['wall_seconds']:.2f}s wall ({sps:.0f} steps/s)  "
              f"{train_metrics['updates']} updates  "
              f"mean_ep_reward={train_metrics['mean_ep_reward']:.3f}")

        # ── Evaluate ──────────────────────────────────────────────────────────
        print(f"  Evaluating (20 episodes)...")
        eval_metrics = evaluate_winrate(
            model, run_cfg, n_episodes=20, device=device, opponent=opponent,
        )
        print(f"  Win-rate: {eval_metrics['win_rate']:.0%}  "
              f"Loss-rate: {eval_metrics['loss_rate']:.0%}  "
              f"Mean length: {eval_metrics['mean_length']:.0f} ticks")

        # ── Trained replay ────────────────────────────────────────────────────
        print(f"  Generating trained replay...")
        if p2_policy == "neural":
            trained_frames = generate_replay(
                run_cfg,
                policy=make_selfplay_greedy(model, opponent, device),
            )
        else:
            trained_frames = generate_replay(
                run_cfg,
                policy=_make_policy(model, device),
            )
        save_jsonl(trained_frames, cfg_dir / "replay_trained.jsonl")
        print(f"  Encoding trained.mp4  ({len(trained_frames)} frames)...")
        render_episode_mp4(
            trained_frames,
            cfg_dir / "trained.mp4",
            fps=fps,
            label=f"{name} — trained ({train_metrics['steps']:,} steps)",
        )

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt_path = cfg_dir / "model.pt"
        ckpt_payload = {"model": model.state_dict(), **train_metrics}
        if p2_policy == "neural":
            ckpt_payload["opponent"] = opponent.state_dict()
        torch.save(ckpt_payload, ckpt_path)

        # ── Write metrics ─────────────────────────────────────────────────────
        metrics = {**train_metrics, **eval_metrics}
        with open(cfg_dir / "metrics.txt", "w") as f:
            f.write(f"Config: {name}\n")
            f.write(f"Map: {cfg['map_size']}x{cfg['map_size']}\n")
            f.write(f"Units: P1={cfg['p1_units']} P2={cfg['p2_units']}\n")
            f.write(f"P2 policy:        {p2_policy}\n")
            f.write(f"\n--- Training ---\n")
            f.write(f"Budget (requested): {train_metrics.get('budget_seconds', seconds):.3f}s\n")
            f.write(f"Wall time (actual): {train_metrics['wall_seconds']:.3f}s\n")
            f.write(f"Device:             {train_metrics.get('device', str(device))}\n")
            f.write(f"Total env-steps:    {train_metrics['steps']:,}\n")
            f.write(f"Throughput:         {train_metrics['steps'] / max(train_metrics['wall_seconds'], 1e-9):.0f} steps/s\n")
            f.write(f"PPO updates:        {train_metrics['updates']}\n")
            f.write(f"Mean ep reward:     {train_metrics['mean_ep_reward']:.4f}\n")
            f.write(f"\n--- Evaluation (20 episodes) ---\n")
            f.write(f"Win rate:         {eval_metrics['win_rate']:.0%}\n")
            f.write(f"Loss rate:        {eval_metrics['loss_rate']:.0%}\n")
            f.write(f"Draw rate:        {eval_metrics['draw_rate']:.0%}\n")
            f.write(f"Mean ep length:   {eval_metrics['mean_length']:.0f} ticks\n")

        summary_rows.append({"name": name, "p2_policy": p2_policy, **cfg, **metrics})
        print(f"  Saved: {cfg_dir}/")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"{'Config':<14} {'Map':>5} {'Units':>7} {'Steps':>9} "
          f"{'WinRate':>8} {'MeanLen':>9} {'AvgRew':>8}")
    print(f"{'-'*72}")
    for r in summary_rows:
        print(f"{r['name']:<14} {r['map_size']:>5} "
              f"{r['p1_units']}v{r['p2_units']:>3}  "
              f"{r['steps']:>9,} "
              f"{r['win_rate']:>8.0%} "
              f"{r['mean_length']:>9.0f} "
              f"{r['mean_ep_reward']:>8.3f}")
    print(f"{'='*72}")
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Train + render MP4 comparisons across unit configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--seconds",    type=float, default=60.0,
                   help="Wall-clock training time per config (default: 60)")
    p.add_argument("--envs",       type=int,   default=4,
                   help="Parallel envs per config during training (default: 4)")
    p.add_argument("--fps",        type=int,   default=15,
                   help="Video playback speed in frames-per-second (default: 15)")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU training even if CUDA is available")
    p.add_argument("--output-dir", type=Path,  default=Path("sweep_results"),
                   help="Root output directory (default: sweep_results/)")
    p.add_argument("--configs",    nargs="*",  default=None,
                   metavar="NAME",
                   help="Subset of config names to run (default: all). "
                        f"Available: {[c['name'] for c in CONFIGS]}")
    p.add_argument("--quick", action="store_true",
                   help="Alias for --seconds 10 --envs 1 (fast smoke test)")
    p.add_argument(
        "--p2-policy",
        type=str,
        default="scripted",
        choices=["scripted", "neural"],
        help="scripted: chase-bot P2; neural: both sides AI (frozen opponent snapshot)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.quick:
        args.seconds = 10.0
        args.envs    = 1

    selected = CONFIGS
    if args.configs:
        name_set  = set(args.configs)
        selected  = [c for c in CONFIGS if c["name"] in name_set]
        missing   = name_set - {c["name"] for c in selected}
        if missing:
            print(f"Warning: unknown config names: {missing}")
            print(f"Available: {[c['name'] for c in CONFIGS]}")
        if not selected:
            sys.exit("No configs to run.")

    print(f"Nightcall sweep: {len(selected)} config(s), "
          f"{args.seconds:.0f}s budget each, "
          f"{args.envs} env(s), {args.fps} fps")

    run_sweep(
        configs=selected,
        seconds=args.seconds,
        num_envs=args.envs,
        fps=args.fps,
        output_dir=args.output_dir,
        force_cpu=args.cpu,
        p2_policy=args.p2_policy,
    )
