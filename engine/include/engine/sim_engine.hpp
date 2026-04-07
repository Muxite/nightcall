#pragma once
// ── SimEngine ─────────────────────────────────────────────────────────────────
// The deterministic simulation kernel. Advances GameState by applying
// validated commands and running systems in a fixed, documented order.
//
// System order (part of the simulation spec — do not reorder without updating
// determinism tests):
//   1. apply_commands     — stamp move/stop orders onto MoveComp
//   2. system_movement    — advance unit positions by velocity
//   3. system_collision   — push overlapping units apart (min separation = 1 tile)
//   4. system_combat      — deal damage between units in range (range = 1.8 tiles)
//   5. system_heal        — recover HP for out-of-combat units
//   6. system_win         — check win condition (including mutual-annihilation draw)
//   7. system_cleanup     — remove dead units and their components
//
// Invariant: given the same initial GameState and the same sequence of
// CommandBatches, tick() always produces bit-identical output on any platform.

#include "game_state.hpp"
#include "command.hpp"
#include <functional>

namespace nc {

class SimEngine {
public:
    SimEngine() = default;

    // Advance |state| by exactly one tick.
    // The batch is validated before application; invalid commands are dropped.
    void tick(GameState& state, const CommandBatch& batch);

    // ── Training helper ───────────────────────────────────────────────────────
    // Run a full game from |initial| until a winner is found or |max_ticks|
    // is reached. Policies are called once per tick per player.
    struct RolloutResult {
        GameState final_state;
        uint32_t  ticks;
        bool      timed_out;
    };

    RolloutResult rollout(
        GameState                                                      initial,
        std::function<CommandBatch(const GameState&, PlayerId)>        p1_policy,
        std::function<CommandBatch(const GameState&, PlayerId)>        p2_policy,
        uint32_t                                                       max_ticks);

private:
    // Systems called in order from tick(). Each is a pure transformation of
    // GameState — no I/O, no allocations outside of GameState's own vectors.
    void apply_commands    (GameState& state, const CommandBatch& batch);
    void system_movement   (GameState& state);
    void system_collision  (GameState& state);
    void system_combat     (GameState& state);
    void system_heal       (GameState& state);
    void system_win        (GameState& state);
    void system_cleanup    (GameState& state);
};

} // namespace nc
