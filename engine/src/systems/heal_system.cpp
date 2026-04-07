#include "heal_system.hpp"
#include "engine/game_state.hpp"
#include <algorithm>

namespace nc::systems {

// ── Heal system ───────────────────────────────────────────────────────────────
// For each alive unit with a HealComp:
//   - Increment ticks_since_combat (the combat system resets this to 0 on hit).
//   - If ticks_since_combat >= combat_cooldown, recover heal_rate HP per tick.
//   - HP is capped at max_hp.

void heal(GameState& state) {
    for (auto& hc : state.comps.heal) {
        Unit* u = state.find_unit(hc.unit_id);
        if (!u || !u->alive) continue;
        if (u->hp <= 0) continue;  // dead units do not heal

        ++hc.ticks_since_combat;

        if (hc.ticks_since_combat >= hc.combat_cooldown) {
            u->hp = std::min(u->hp + hc.heal_rate, u->max_hp);
        }
    }
}

} // namespace nc::systems
