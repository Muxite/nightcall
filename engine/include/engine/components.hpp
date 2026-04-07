#pragma once
// ── Component definitions ────────────────────────────────────────────────────
// Each component is a plain data struct with a unit_id owner field.
// Components are stored in flat sorted vectors inside ComponentStore (game_state.hpp).
//
// Design: units acquire capabilities through composition — a unit type
// is defined by which components it carries. To add a new ability:
//   1. Define a new XxxComp struct here.
//   2. Add a std::vector<XxxComp> to ComponentStore in game_state.hpp.
//   3. Add an XxxSpec to UnitDef in unit_def.hpp and populate k_unit_defs.
//   4. Write a system that processes the component (engine/src/systems/).
//   5. Call the system from SimEngine::tick() in the correct order.
//
// No other changes are required to existing code.

#include "types.hpp"

namespace nc {

// ── MeleeAttackComp ──────────────────────────────────────────────────────────
// Unit deals damage to all enemy units within range each tick.
//
// Multi-enemy scaling rule (emergent strategy driver):
//   If N enemies are within range:
//     - This unit deals (damage / N) to each enemy  [damage splits, not amplifies]
//     - Each enemy independently deals its own damage to this unit [damage stacks]
//   Net effect: being outnumbered hurts — you take N× damage, deal 1/N× each.
//   This incentivises concentrating forces and protecting wounded units.
struct MeleeAttackComp {
    UnitId  unit_id{UnitId::Invalid};
    int32_t damage{10};                    // base damage per tick (HP units)
    int32_t range{1 * FixedVec2::SCALE};   // attack radius (FixedVec2 units)
};

// ── HealComp ─────────────────────────────────────────────────────────────────
// Unit slowly regenerates HP when it has not been in combat recently.
// "In combat" = this unit dealt or received damage within the last
// |combat_cooldown| ticks. The |ticks_since_combat| counter resets to 0
// whenever the unit attacks or is attacked; healing resumes once it exceeds
// |combat_cooldown|.
struct HealComp {
    UnitId  unit_id{UnitId::Invalid};
    int32_t heal_rate{1};           // HP recovered per out-of-combat tick
    int32_t combat_cooldown{20};    // ticks to wait after last combat
    int32_t ticks_since_combat{0};  // runtime counter — resets on any damage event
};

// ── MoveComp ─────────────────────────────────────────────────────────────────
// Unit moves at a constant velocity vector toward a destination.
// No pathfinding — the velocity is set directly by MoveCmd and applied
// each tick. The unit stops (velocity = 0) when it reaches or passes target.
// Speed is the magnitude of velocity in FixedVec2 units per tick.
struct MoveComp {
    UnitId    unit_id{UnitId::Invalid};
    int32_t   speed{FixedVec2::SCALE / 4};  // max speed (0.25 tiles/tick default)
    FixedVec2 velocity{0, 0};               // current per-tick displacement (0 = stopped)
    FixedVec2 target{0, 0};                 // destination (only meaningful when moving)
    bool      moving{false};                // false = unit is stationary
    uint8_t   _pad[3]{};                    // explicit padding — always zero
};

} // namespace nc
