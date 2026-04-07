#include "engine/sim_engine.hpp"
#include "engine/command_validator.hpp"
#include "systems/movement_system.hpp"
#include "systems/collision_system.hpp"
#include "systems/combat_system.hpp"
#include "systems/heal_system.hpp"
#include "systems/cleanup_system.hpp"
#include <variant>

namespace nc {

// ── apply_commands ────────────────────────────────────────────────────────────
// Validate the batch, then stamp each command's intent onto the relevant
// component. Commands do not move units directly — they configure components
// that the movement system will act on this same tick.

void SimEngine::apply_commands(GameState& state, const CommandBatch& batch) {
    CommandBatch valid = CommandValidator::filter(state, batch);

    for (const Command& cmd : valid.commands) {
        std::visit([&](const auto& c) {
            using T = std::decay_t<decltype(c)>;

            if constexpr (std::is_same_v<T, MoveCmd>) {
                MoveComp* mc = state.comps.move_of(c.unit);
                if (!mc) return;
                mc->target   = c.destination;
                mc->velocity = FixedVec2::velocity_toward(
                    state.find_unit(c.unit)->pos, c.destination, mc->speed);
                mc->moving   = !mc->velocity.is_zero();
            }
            else if constexpr (std::is_same_v<T, StopCmd>) {
                MoveComp* mc = state.comps.move_of(c.unit);
                if (!mc) return;
                mc->moving   = false;
                mc->velocity = {0, 0};
            }
            // NoOpCmd: nothing to do.
        }, cmd);
    }
}

// ── System pipeline ───────────────────────────────────────────────────────────
// Order is part of the specification. Do NOT reorder without updating tests.

void SimEngine::system_movement  (GameState& state) { systems::movement(state);  }
void SimEngine::system_collision (GameState& state) { systems::collision(state); }
void SimEngine::system_combat    (GameState& state) { systems::combat(state);    }
void SimEngine::system_heal      (GameState& state) { systems::heal(state);      }

void SimEngine::system_win(GameState& state) {
    if (state.winner) return;  // already decided

    // Check whether each player still has any alive units.
    bool p1_alive = false, p2_alive = false;
    for (const auto& u : state.units) {
        if (!u.alive) continue;
        if (u.owner == PlayerId::P1) p1_alive = true;
        if (u.owner == PlayerId::P2) p2_alive = true;
    }

    if (!p1_alive && !p2_alive) {
        state.winner = PlayerId::None;  // draw
    } else if (!p2_alive) {
        state.winner = PlayerId::P1;
    } else if (!p1_alive) {
        state.winner = PlayerId::P2;
    }
}

void SimEngine::system_cleanup(GameState& state) { systems::cleanup(state); }

// ── tick ──────────────────────────────────────────────────────────────────────

void SimEngine::tick(GameState& state, const CommandBatch& batch) {
    if (state.winner) return;  // game already over — no-op

    apply_commands   (state, batch);
    system_movement  (state);
    system_collision (state);
    system_combat    (state);
    system_heal      (state);
    system_win       (state);
    system_cleanup   (state);

    ++state.tick;
}

// ── rollout ───────────────────────────────────────────────────────────────────

SimEngine::RolloutResult SimEngine::rollout(
    GameState                                               initial,
    std::function<CommandBatch(const GameState&, PlayerId)> p1_policy,
    std::function<CommandBatch(const GameState&, PlayerId)> p2_policy,
    uint32_t                                                max_ticks)
{
    GameState state = std::move(initial);

    for (uint32_t t = 0; t < max_ticks; ++t) {
        if (state.winner) {
            return { std::move(state), t, false };
        }

        CommandBatch batch;
        // Merge both players' commands into one batch.
        for (auto& c : p1_policy(state, PlayerId::P1).commands) batch.push(c);
        for (auto& c : p2_policy(state, PlayerId::P2).commands) batch.push(c);

        tick(state, batch);
    }

    return { std::move(state), max_ticks, /*timed_out=*/true };
}

} // namespace nc
