#include "movement_system.hpp"
#include "engine/game_state.hpp"

namespace nc::systems {

// ── Movement system ───────────────────────────────────────────────────────────
// For each unit with an active MoveComp:
//   1. Add velocity to position.
//   2. Check if the unit has reached (or passed) its target.
//   3. If so, snap to target and stop.
//
// "Reached" is defined as: the unit was closer to the target before the step
// than after, OR the distance is now within one step (to avoid oscillation).

void movement(GameState& state) {
    for (auto& mc : state.comps.move) {
        if (!mc.moving) continue;

        Unit* u = state.find_unit(mc.unit_id);
        if (!u || !u->alive) {
            mc.moving = false;
            continue;
        }

        int64_t dist_before = u->pos.dist_sq(mc.target);

        // Advance position by velocity.
        u->pos += mc.velocity;

        int64_t dist_after = u->pos.dist_sq(mc.target);
        int64_t step_sq    = static_cast<int64_t>(mc.speed) * mc.speed;

        // Stop condition: passed target (dist increased) or close enough to stop.
        if (dist_after > dist_before || dist_after <= step_sq) {
            u->pos    = mc.target;
            mc.moving   = false;
            mc.velocity = {0, 0};
        }
    }
}

} // namespace nc::systems
