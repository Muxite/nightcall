"""
train.py — PPO training for nightcall.

Modes (``--p2-policy``)
-----------------------
**scripted** — P2 uses the built-in chase heuristic (non-learning bot). P1 is
trained with PPO on P1 observations only.

**neural** — P2 actions come from a **frozen** copy of the policy evaluated on
P2-centric observations; the copy refreshes every ``opponent_update_interval``
updates. Same network architecture serves both sides (shared weights for P1
learning; opponent mirror for P2 moves).

Typical invocation:
    python python/training/train.py --p2-policy scripted
    python python/training/train.py --p2-policy neural --steps 2_000_000 --envs 8

Checkpoints go under ``--checkpoint-dir`` (default ``checkpoints``), created if missing;
``ckpt_<step>.pt`` every ``--save-interval`` updates, then ``ckpt_final.pt``.
Use ``--load-checkpoint PATH`` for warm-start, or add ``--resume`` to restore step count
and optimizer when saved. Docker defaults live in ``docker/train.env`` (same keys as env vars).
Logs include remaining env-steps and ETA from measured SPS.
TensorBoard logs are written to runs/ (tensorboard --logdir runs).
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))                          # training/ for env, model
sys.path.insert(0, str(_HERE.parent.parent / "build" / "python"))  # nightcall_sim

from env   import NightcallEnv, MAX_P1_UNITS, N_UNIT_ACTIONS, FEATURE_DIM
from model import NightcallActorCritic


def _ensure_output_dir(path_str: str) -> Path:
    """
    Resolve ``path_str`` against the process working directory if relative,
    create all missing directories, and return an absolute Path.

    :param path_str: Directory where checkpoints and ``ckpt_final.pt`` are written.
    :returns: Absolute, normalized path.
    """
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_path(path_str: str) -> Path:
    """
    Resolve a file or directory path for reading (must exist for files checked by caller).

    :param path_str: Relative (to cwd) or absolute path string.
    :returns: Absolute normalized Path.
    """
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    return p


def _format_duration(seconds: float) -> str:
    """
    Format a duration in seconds for ETA display.

    :param seconds: Non-negative wall time in seconds.
    :returns: Short human-readable string.
    """
    if seconds < 0 or not math.isfinite(seconds):
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    if minutes < 120:
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins:02d}m"


# ── Hyper-parameters ──────────────────────────────────────────────────────────

DEFAULTS = dict(
    p2_policy                = "scripted", # "scripted" = chase bot; "neural" = P2 from frozen opponent net
    total_steps              = 1_000_000,
    num_envs                 = 8,
    rollout_steps            = 256,        # long horizon vs MAX_TICKS for stabler GAE / credit
    num_minibatches          = 8,
    update_epochs            = 6,
    gamma                    = 0.99,
    gae_lambda               = 0.97,
    clip_coef                = 0.2,
    vf_coef                  = 0.4,
    ent_coef_start           = 0.06,
    ent_coef_end             = 0.002,
    max_grad_norm            = 0.5,
    lr_start                 = 2.5e-4,
    lr_end                   = 1e-5,
    opponent_update_interval = 40,
    save_interval            = 100,        # updates between checkpoint saves
    checkpoint_dir           = "checkpoints",
    log_interval             = 10,
)


def _truthy_env(name: str) -> bool:
    """
    Return True if the environment variable is set to a common affirmative value.

    :param name: Variable name.
    :returns: Whether the value looks like true/on/yes/1.
    """
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _maybe_post_train_render(cfg: dict, final_ckpt: Path) -> None:
    """
    If ``RENDER_AFTER_TRAIN`` is set, generate baseline/trained MP4s under ``RESULTS_DIR``.

    :param cfg: Training configuration.
    :param final_ckpt: Path to ``ckpt_final.pt`` just written.
    """
    if not _truthy_env("RENDER_AFTER_TRAIN"):
        return
    rd = (os.environ.get("RESULTS_DIR") or "results").strip() or "results"
    try:
        from render_run import post_train_render

        post_train_render(final_ckpt, cfg, Path(rd))
    except Exception as exc:
        print(f"WARNING: post-training render failed: {exc}")


def _cli_defaults() -> dict:
    """
    Merge :data:`DEFAULTS` with the process environment (e.g. ``docker/train.env``).

    Variables: ``TOTAL_STEPS``, ``NUM_ENVS``, ``P2_POLICY``, ``CHECKPOINT_DIR``.
    Post-render: ``RENDER_AFTER_TRAIN``, ``RESULTS_DIR``, ``RENDER_*`` (see ``render_run.py``).
    CLI flags still override these when passed explicitly.

    :returns: A copy of defaults with env overrides applied.
    """
    d = dict(DEFAULTS)
    ts = os.environ.get("TOTAL_STEPS", "").strip()
    if ts:
        d["total_steps"] = int(ts)
    ne = os.environ.get("NUM_ENVS", "").strip()
    if ne:
        d["num_envs"] = int(ne)
    p2 = os.environ.get("P2_POLICY", "").strip()
    if p2:
        d["p2_policy"] = p2
    cd = os.environ.get("CHECKPOINT_DIR", "").strip()
    if cd:
        d["checkpoint_dir"] = cd
    return d


# ── Vectorised environment (simple synchronous wrapper) ───────────────────────

class VecEnv:
    """Synchronous vector environment — no multiprocessing, safe on Windows."""

    def __init__(self, num_envs: int, **env_kwargs):
        self.envs = [NightcallEnv(**env_kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs

    def observe_p2_batch(self) -> np.ndarray:
        return np.stack([e.observe_p2() for e in self.envs])

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.stack(obs_list)   # (N, FEATURE_DIM)

    def step(self, actions: np.ndarray):
        obs_list, rew_list, term_list, trunc_list = [], [], [], []
        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, _ = env.step(actions[i])
            if term or trunc:
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
        return (
            np.stack(obs_list),
            np.array(rew_list, dtype=np.float32),
            np.array(term_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
        )


# ── GAE advantage estimation ──────────────────────────────────────────────────

def compute_gae(
    rewards:     torch.Tensor,   # (T, N)
    values:      torch.Tensor,   # (T, N)
    next_value:  torch.Tensor,   # (N,)
    dones:       torch.Tensor,   # (T, N) bool
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae   = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val  = next_value
            next_done = torch.zeros(N, device=rewards.device)
        else:
            next_val  = values[t + 1]
            next_done = dones[t + 1].float()

        delta    = rewards[t] + gamma * next_val * (1 - next_done) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def get_rollout_actions(
    model: NightcallActorCritic,
    opponent: NightcallActorCritic,
    obs_p1_t: torch.Tensor,
    envs: VecEnv,
    neural_p2: bool,
):
    """
    Sample P1 actions (and P2 actions if neural_p2) for one vectorised step.

    :param model: learning policy (P1)
    :param opponent: frozen copy used as P2 when neural_p2
    :param obs_p1_t: P1 observations (N, FEATURE_DIM)
    :param envs: vector env (for P2 observations)
    :param neural_p2: if True, concatenate P2 actions from opponent on P2-obs
    :returns: actions tensor (N, MAX_P1_UNITS) or (N, 2*MAX_P1_UNITS), log_prob (N,), value (N,1)
    """
    with torch.no_grad():
        if neural_p2:
            obs_p2_t = torch.from_numpy(envs.observe_p2_batch()).to(obs_p1_t.device)
            a1, logp, _, val = model.get_action_and_value(obs_p1_t)
            a2, _, _, _ = opponent.get_action_and_value(obs_p2_t)
            actions = torch.cat([a1, a2], dim=1)
        else:
            actions, logp, _, val = model.get_action_and_value(obs_p1_t)
    return actions, logp, val


def make_selfplay_greedy(
    model: NightcallActorCritic,
    opponent: NightcallActorCritic,
    device: torch.device,
):
    """
    Greedy joint policy for evaluation: P1 from ``model``, P2 from ``opponent``
    on each side's observation.

    :returns: callable ``(obs_p1_np, env) -> action_np`` for neural envs
    """
    model.eval()
    opponent.eval()

    def policy(obs_p1: np.ndarray, env: NightcallEnv) -> np.ndarray:
        with torch.no_grad():
            o1 = torch.from_numpy(obs_p1).float().unsqueeze(0).to(device)
            o2 = torch.from_numpy(env.observe_p2()).float().unsqueeze(0).to(device)
            logits1, _ = model(o1)
            logits2, _ = opponent(o2)
            a1 = logits1.argmax(dim=-1).squeeze(0).cpu().numpy()
            a2 = logits2.argmax(dim=-1).squeeze(0).cpu().numpy()
        return np.concatenate([a1, a2])

    return policy


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    dev_name = str(device)
    if device.type == "cuda" and torch.cuda.is_available():
        dev_name = f"{device} ({torch.cuda.get_device_name(0)})"

    checkpoint_root = _ensure_output_dir(cfg["checkpoint_dir"])
    steps_per_update = cfg["rollout_steps"] * cfg["num_envs"]

    print(f"Training on {dev_name}  |  {cfg['num_envs']} envs  |  "
          f"target {cfg['total_steps']:,} env-steps")
    print(f"Checkpoints -> {checkpoint_root}")

    # Models
    model    = NightcallActorCritic().to(device)
    opponent = copy.deepcopy(model)
    opponent.eval()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr_start"], eps=1e-5)

    total_steps = 0
    update_num = 0
    load_path = (cfg.get("load_checkpoint") or "").strip()
    if load_path:
        ckpt_path = _resolve_path(load_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_p2 = ckpt.get("p2_policy")
        if ckpt_p2 is not None and ckpt_p2 != cfg["p2_policy"]:
            raise ValueError(
                f"Checkpoint p2_policy={ckpt_p2!r} does not match --p2-policy {cfg['p2_policy']!r}"
            )
        model.load_state_dict(ckpt["model"])
        neural_p2 = cfg.get("p2_policy", "scripted") == "neural"
        if neural_p2:
            if "opponent" in ckpt:
                opponent.load_state_dict(ckpt["opponent"])
            else:
                opponent.load_state_dict(copy.deepcopy(model.state_dict()))
        opponent.eval()
        if cfg.get("resume"):
            total_steps = int(ckpt.get("step", 0))
            update_num = int(ckpt.get("update", 0))
            if "opt" in ckpt:
                optimiser.load_state_dict(ckpt["opt"])
            else:
                print("  (checkpoint has no optimizer state; continuing with fresh optimizer)")
            print(
                f"Resumed from {ckpt_path}  |  step={total_steps:,}  update={update_num}  "
                f"target steps={cfg['total_steps']:,}"
            )
        else:
            print(f"Loaded weights from {ckpt_path} (fresh step count and optimizer)")
    else:
        neural_p2 = cfg.get("p2_policy", "scripted") == "neural"

    total_updates = max(1, cfg["total_steps"] // steps_per_update)
    print(
        f"Schedule: ~{total_updates} PPO updates  |  {steps_per_update:,} env-steps per update  "
        f"(ETA logged from measured throughput)"
    )

    envs = VecEnv(
        cfg["num_envs"],
        p2_policy="neural" if neural_p2 else "scripted",
    )
    obs_np = envs.reset()
    if neural_p2:
        print("P2 policy: neural (frozen opponent snapshot vs P1)")
    else:
        print("P2 policy: scripted chase bot")

    # Storage buffers
    T, N   = cfg["rollout_steps"], cfg["num_envs"]
    obs_buf    = torch.zeros(T, N, FEATURE_DIM, device=device)
    act_buf    = torch.zeros(T, N, MAX_P1_UNITS, dtype=torch.long, device=device)
    logp_buf   = torch.zeros(T, N, device=device)
    rew_buf    = torch.zeros(T, N, device=device)
    val_buf    = torch.zeros(T, N, device=device)
    done_buf   = torch.zeros(T, N, dtype=torch.bool, device=device)

    t_start = time.time()

    while total_steps < cfg["total_steps"]:
        # ── Collect rollout ───────────────────────────────────────────────────
        model.eval()
        obs_t = torch.from_numpy(obs_np).to(device)

        for step in range(T):
            actions, log_prob, value = get_rollout_actions(
                model, opponent, obs_t, envs, neural_p2
            )
            act_np = actions.cpu().numpy()
            obs_np, rew_np, term_np, trunc_np = envs.step(act_np)
            done_np = term_np | trunc_np

            obs_buf[step]  = obs_t
            act_buf[step]  = actions[:, :MAX_P1_UNITS]
            logp_buf[step] = log_prob
            val_buf[step]  = value.squeeze(-1)
            rew_buf[step]  = torch.from_numpy(rew_np).to(device)
            done_buf[step] = torch.from_numpy(done_np).to(device)

            obs_t = torch.from_numpy(obs_np).to(device)
            total_steps += N

        # Bootstrap value for the last step
        with torch.no_grad():
            next_val = model.get_value(obs_t).squeeze(-1)

        advantages, returns = compute_gae(
            rew_buf, val_buf, next_val, done_buf,
            cfg["gamma"], cfg["gae_lambda"]
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Anneal LR and entropy coefficient ────────────────────────────────
        frac = min(update_num / max(total_updates, 1), 1.0)
        lr   = cfg["lr_start"]    + (cfg["lr_end"]      - cfg["lr_start"])    * frac
        ent_coef = cfg["ent_coef_start"] + (cfg["ent_coef_end"] - cfg["ent_coef_start"]) * frac
        for pg in optimiser.param_groups:
            pg["lr"] = lr

        # ── PPO update ────────────────────────────────────────────────────────
        model.train()
        batch_size  = T * N
        minibatch   = batch_size // cfg["num_minibatches"]

        flat_obs  = obs_buf.view(batch_size, FEATURE_DIM)
        flat_act  = act_buf.view(batch_size, MAX_P1_UNITS)
        flat_logp = logp_buf.view(batch_size)
        flat_adv  = advantages.view(batch_size)
        flat_ret  = returns.view(batch_size)

        total_pg_loss = total_vf_loss = total_ent = 0.0
        n_updates = 0

        for _ in range(cfg["update_epochs"]):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch):
                idx = perm[start : start + minibatch]
                _, new_logp, entropy, new_val = model.get_action_and_value(
                    flat_obs[idx], flat_act[idx]
                )
                ratio = torch.exp(new_logp - flat_logp[idx])
                adv   = flat_adv[idx]

                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * ratio.clamp(1 - cfg["clip_coef"],
                                               1 + cfg["clip_coef"])
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss  = F.mse_loss(new_val.squeeze(-1), flat_ret[idx])
                ent_loss = entropy.mean()

                loss = pg_loss + cfg["vf_coef"] * vf_loss - ent_coef * ent_loss

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimiser.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent     += ent_loss.item()
                n_updates     += 1

        update_num += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if update_num % cfg["log_interval"] == 0:
            elapsed = time.time() - t_start
            sps     = total_steps / elapsed if elapsed > 0 else 0.0
            remain  = max(0, cfg["total_steps"] - total_steps)
            eta_s   = remain / sps if sps > 1e-9 else float("nan")
            eta_h   = _format_duration(eta_s) if math.isfinite(eta_s) else "?"
            print(
                f"step={total_steps:>8,}  "
                f"update={update_num:>5}  "
                f"pg={total_pg_loss/n_updates:.4f}  "
                f"vf={total_vf_loss/n_updates:.4f}  "
                f"ent={total_ent/n_updates:.4f}  "
                f"lr={lr:.2e}  ent_c={ent_coef:.4f}  "
                f"sps={sps:.0f}  "
                f"rem={remain:>8,}  eta≈{eta_h}"
            )

        # ── Update opponent snapshot ──────────────────────────────────────────
        if neural_p2 and update_num % cfg["opponent_update_interval"] == 0:
            opponent.load_state_dict(copy.deepcopy(model.state_dict()))
            print(f"  -> P2 opponent snapshot updated at update {update_num}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if update_num % cfg["save_interval"] == 0:
            path = checkpoint_root / f"ckpt_{total_steps}.pt"
            ckpt = {
                "model":      model.state_dict(),
                "opt":        optimiser.state_dict(),
                "step":       total_steps,
                "update":     update_num,
                "p2_policy":  cfg["p2_policy"],
            }
            if neural_p2:
                ckpt["opponent"] = opponent.state_dict()
            torch.save(ckpt, path)
            print(f"  -> saved {path}")

    path = checkpoint_root / "ckpt_final.pt"
    ckpt_final = {
        "model": model.state_dict(),
        "opt": optimiser.state_dict(),
        "step": total_steps,
        "update": update_num,
        "p2_policy": cfg["p2_policy"],
    }
    if neural_p2:
        ckpt_final["opponent"] = opponent.state_dict()
    torch.save(ckpt_final, path)
    elapsed_total = time.time() - t_start
    print(f"Training complete in {_format_duration(elapsed_total)}. Final checkpoint: {path}")
    _maybe_post_train_render(cfg, path)
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> dict:
    base = _cli_defaults()
    p = argparse.ArgumentParser(description="Train nightcall PPO agent")
    for k, v in DEFAULTS.items():
        if k == "p2_policy":
            p.add_argument(
                "--p2-policy",
                type=str,
                default=base["p2_policy"],
                choices=["scripted", "neural"],
                dest="p2_policy",
                help="scripted=chase heuristic P2; neural=P2 from frozen policy on P2-obs",
            )
            continue
        t = type(v)
        if t == bool:
            p.add_argument(f"--{k}", action="store_true", default=base[k])
        else:
            p.add_argument(
                f"--{k.replace('_','-')}",
                type=t,
                default=base[k],
                dest=k,
            )
    p.add_argument(
        "--load-checkpoint",
        type=str,
        default=os.environ.get("LOAD_CHECKPOINT", "").strip(),
        metavar="PATH",
        dest="load_checkpoint",
        help="Load model weights from a .pt file; use --resume to restore step count and optimizer.",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=_truthy_env("RESUME"),
        help="With --load-checkpoint, restore env-step count, update index, and optimizer state if saved.",
    )
    return vars(p.parse_args())


if __name__ == "__main__":
    train(_parse())
