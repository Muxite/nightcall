#include "cleanup_system.hpp"
#include "engine/game_state.hpp"
#include <algorithm>

namespace nc::systems {

// ── Cleanup system ────────────────────────────────────────────────────────────
// Runs last every tick. Removes dead units from the units vector and their
// associated components from all component stores.
// Keeping dead units alive through the tick (killing them mid-tick but only
// removing them here) ensures that all systems within a tick see a consistent
// snapshot: a unit killed early in the tick pipeline is still visible to later
// systems but flagged alive=false.

void cleanup(GameState& state) {
    // Collect IDs of dead units first to avoid iterator invalidation.
    std::vector<UnitId> dead;
    for (const auto& u : state.units)
        if (!u.alive) dead.push_back(u.id);

    for (UnitId id : dead) {
        // Remove components before erasing the unit.
        state.comps.remove(id);

        state.units.erase(
            std::remove_if(state.units.begin(), state.units.end(),
                [id](const Unit& u) { return u.id == id; }),
            state.units.end());
    }
}

} // namespace nc::systems
