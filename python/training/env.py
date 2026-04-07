"""
NightcallEnv — gymnasium-compatible wrapper around nightcall_sim.

Observation space: Box(FEATURE_DIM,) float32  — feature vector (per-unit: pos, hp, moving, vel)
Action space:      MultiDiscrete([N_UNIT_ACTIONS] * MAX_P1_UNITS)
    N_UNIT_ACTIONS = COARSE_SIZE**2 + 1   (0=NoOp, 1..49 = coarse tile index)
    MAX_P1_UNITS   = 64
    COARSE_SIZE    = 7  →  7×7 = 49 possible destinations per unit slot

One step = one engine tick.  ``p2_policy="scripted"`` (default): P1 commands
MAX_P1_UNITS slots; P2 uses a chase heuristic.  ``p2_policy="neural"``: step
expects a flat vector of length ``2 * MAX_P1_UNITS`` — first half P1, second
half P2 (same coarse move semantics per side).

Reward:
    Dense per tick: (ENEMY_HP_REWARD_COEF×Δenemy_hp − OWN_HP_PENALTY_COEF×Δown_hp) / HP_SCALE
    + ENEMY_KILL_BONUS per enemy kill, − OWN_DEATH_PENALTY per own loss (tick)
    Terminal: +WIN_BONUS + CONSERVE_BONUS×(surviving P1 / starting P1) if P1 wins;
              −LOSS_PENALTY if P2 wins; draws unchanged

Usage:
    import sys
    sys.path.insert(0, 'build/python')   # or add to PYTHONPATH

    from python.training.env import NightcallEnv
    env = NightcallEnv()
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import gymnasium as gym

# Locate the built module: check common output paths in priority order.
_MODULE_SEARCH = [
    Path(__file__).parent.parent.parent / "build" / "python",   # repo root/build/python
    Path(os.environ.get("NIGHTCALL_BUILD_DIR", ".")) / "python",
]
for _p in _MODULE_SEARCH:
    if (_p).exists():
        sys.path.insert(0, str(_p))
        break

import nightcall_sim as nc  # noqa: E402  (must be after path setup)

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_DIM    : int = 1026         # 2 + 128 * 8  (must match observation.cpp)
MAX_P1_UNITS   : int = 64           # unit slots in the action vector
COARSE_SIZE    : int = 7            # coarse grid resolution (COARSE×COARSE)
N_UNIT_ACTIONS : int = COARSE_SIZE * COARSE_SIZE + 1   # 0=NoOp, 1..49=move
MAX_TICKS      : int = 1000         # episode horizon (longer for large maps)
HP_SCALE               : float = 100.0   # unit max_hp — normalises dense HP term
ENEMY_HP_REWARD_COEF   : float = 2.5      # weight damage dealt to enemies vs taken (aggression)
OWN_HP_PENALTY_COEF    : float = 1.0      # weight own HP lost in dense term
WIN_BONUS              : float = 100.0    # terminal reward when P1 wins
LOSS_PENALTY           : float = 50.0     # terminal magnitude when P2 wins (lower vs damage focus)
ENEMY_KILL_BONUS       : float = 14.0     # per enemy unit eliminated in a tick
OWN_DEATH_PENALTY      : float = 1.0      # per own unit lost in a tick
CONSERVE_BONUS         : float = 15.0     # extra on P1 win, scaled by surviving P1 count / p1_units
P2_RETARGET         : int  = 25       # ticks between P2 re-targeting (scripted P2 only)

P2Policy = Literal["scripted", "neural"]

P1 = nc.PlayerId.P1
P2 = nc.PlayerId.P2


class NightcallEnv(gym.Env):
    """
    P1-centric RL interface: observations are always P1's view; reward shapes
    learning for P1. P2 is either a scripted bot or a second policy (neural).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_size:   int = 30,
        p1_units:   int = 4,
        p2_units:   int = 4,
        p2_policy:  P2Policy = "scripted",
    ):
        super().__init__()
        self.map_size = map_size
        self.p1_units = p1_units
        self.p2_units = p2_units
        self.p2_policy: P2Policy = p2_policy
        if p2_policy not in ("scripted", "neural"):
            raise ValueError("p2_policy must be 'scripted' or 'neural'")

        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(FEATURE_DIM,), dtype=np.float32
        )
        n_act = MAX_P1_UNITS if p2_policy == "scripted" else 2 * MAX_P1_UNITS
        self.action_space = gym.spaces.MultiDiscrete(
            [N_UNIT_ACTIONS] * n_act, dtype=np.int64
        )

        self._engine: nc.SimEngine = nc.SimEngine()
        self._state:  Optional[nc.GameState] = None
        self._prev_enemy_hp: int = 0
        self._prev_own_hp:   int = 0
        self._prev_enemy_alive: int = 0
        self._prev_own_alive:   int = 0

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        seed=None,
        options=None,
    ):
        """
        Start a new episode with a randomised spawn layout (reproducible if ``seed`` is set).

        :param seed: optional RNG seed for reproducible layouts and actions
        :param options: unused; reserved for gymnasium compatibility
        :returns: observation vector and an empty info dict
        """
        super().reset(seed=seed)
        spawn_seed = int(self.np_random.integers(1, 2**32, dtype=np.uint32))
        self._state = nc.GameState.make_default(
            self.map_size,
            self.map_size,
            self.p1_units,
            self.p2_units,
            spawn_seed,
        )
        self._prev_enemy_hp, self._prev_own_hp = self._hp_totals()
        self._prev_enemy_alive = len(self._alive_p2())
        self._prev_own_alive   = len(self._alive_p1())
        return self._observe(), {}

    def step(self, action: np.ndarray):
        assert self._state is not None, "call reset() before step()"
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        expect = MAX_P1_UNITS if self.p2_policy == "scripted" else 2 * MAX_P1_UNITS
        if action.shape[0] != expect:
            raise ValueError(
                f"action length {action.shape[0]} != {expect} for p2_policy={self.p2_policy!r}"
            )

        batch = self._build_batch(action)
        self._engine.tick(self._state, batch)

        new_enemy_hp, new_own_hp = self._hp_totals()
        enemy_hp_removed = float(self._prev_enemy_hp - new_enemy_hp)
        own_hp_lost      = float(self._prev_own_hp - new_own_hp)
        reward = (
            enemy_hp_removed * ENEMY_HP_REWARD_COEF - own_hp_lost * OWN_HP_PENALTY_COEF
        ) / HP_SCALE
        self._prev_enemy_hp = new_enemy_hp
        self._prev_own_hp   = new_own_hp

        new_enemy_alive = len(self._alive_p2())
        new_own_alive   = len(self._alive_p1())
        kills  = self._prev_enemy_alive - new_enemy_alive
        deaths = self._prev_own_alive   - new_own_alive
        reward += kills * ENEMY_KILL_BONUS
        reward -= deaths * OWN_DEATH_PENALTY
        self._prev_enemy_alive = new_enemy_alive
        self._prev_own_alive   = new_own_alive

        terminated = self._state.winner is not None
        if terminated:
            if self._state.winner == P1:
                reward += WIN_BONUS
                surv = max(new_own_alive, 0)
                ratio = float(surv) / float(max(self.p1_units, 1))
                reward += CONSERVE_BONUS * ratio
            elif self._state.winner == P2:
                reward -= LOSS_PENALTY

        # Time-limit draw: treat as terminated with no bonus (neutral outcome).
        truncated = False
        if not terminated and self._state.tick >= MAX_TICKS:
            terminated = True   # episode ends; reward stays 0 (draw by time limit)

        info = {
            "tick":   self._state.tick,
            "winner": self._state.winner,
        }
        return self._observe(), reward, terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _observe(self) -> np.ndarray:
        obs = nc.observe(self._state, P1)
        return obs.to_feature_vector().copy()   # numpy array float32

    def observe_p2(self) -> np.ndarray:
        """
        P2-centric feature vector (same layout as P1 obs, roles flipped).

        :returns: float32 vector of length FEATURE_DIM
        """
        assert self._state is not None, "call reset() before observe_p2()"
        obs = nc.observe(self._state, P2)
        return obs.to_feature_vector().copy()

    def _hp_totals(self):
        enemy_hp = sum(u.hp for u in self._state.units if u.alive and u.owner == P2)
        own_hp   = sum(u.hp for u in self._state.units if u.alive and u.owner == P1)
        return enemy_hp, own_hp

    def _alive_p1(self):
        return [u for u in self._state.units if u.alive and u.owner == P1]

    def _alive_p2(self):
        return [u for u in self._state.units if u.alive and u.owner == P2]

    def _coarse_to_dest(self, coarse_idx: int) -> nc.FixedVec2:
        """Convert a coarse tile index (0..COARSE²-1) to a world FixedVec2."""
        cx     = coarse_idx % COARSE_SIZE
        cy     = coarse_idx // COARSE_SIZE
        cell_w = self.map_size // COARSE_SIZE
        cell_h = self.map_size // COARSE_SIZE
        tile_x = cx * cell_w + cell_w // 2
        tile_y = cy * cell_h + cell_h // 2
        return nc.FixedVec2.from_tile(tile_x, tile_y)

    def _push_side_moves(
        self,
        batch: nc.CommandBatch,
        player: nc.PlayerId,
        alive: list,
        slot_actions: np.ndarray,
    ) -> None:
        for slot, unit_action in enumerate(slot_actions):
            if slot >= len(alive):
                break
            if int(unit_action) == 0:
                continue
            dest = self._coarse_to_dest(int(unit_action) - 1)
            batch.push_move(player, alive[slot].id, dest)

    def _build_batch(self, action: np.ndarray) -> nc.CommandBatch:
        batch    = nc.CommandBatch()
        p1_alive = self._alive_p1()
        p2_alive = self._alive_p2()

        p1_slots = action[:MAX_P1_UNITS]
        self._push_side_moves(batch, P1, p1_alive, p1_slots)

        if self.p2_policy == "neural":
            p2_slots = action[MAX_P1_UNITS : MAX_P1_UNITS * 2]
            self._push_side_moves(batch, P2, p2_alive, p2_slots)
        else:
            if self._state.tick % P2_RETARGET == 0:
                p1_for_ai = self._alive_p1()
                centre    = nc.FixedVec2.from_tile(self.map_size // 2, self.map_size // 2)
                for u in self._alive_p2():
                    if p1_for_ai:
                        ux = u.pos.x / nc.FixedVec2.SCALE
                        uy = u.pos.y / nc.FixedVec2.SCALE
                        target = min(
                            p1_for_ai,
                            key=lambda t: (t.pos.x / nc.FixedVec2.SCALE - ux) ** 2
                                        + (t.pos.y / nc.FixedVec2.SCALE - uy) ** 2,
                        )
                        batch.push_move(P2, u.id, target.pos)
                    else:
                        batch.push_move(P2, u.id, centre)

        return batch
