#pragma once
// ── Unit type registry ───────────────────────────────────────────────────────
// A UnitDef describes a unit type's stats and which component specs it carries.
// To add a new unit type:
//   1. Add an enum value to UnitType in types.hpp.
//   2. Append a UnitDef entry to k_unit_defs below (order must match enum values).
//   3. Add any new XxxSpec fields to UnitDef if required.
// No system code changes needed — systems look up components at runtime.

#include "types.hpp"
#include "components.hpp"
#include <optional>
#include <array>

namespace nc {

// ── Component specification structs ─────────────────────────────────────────
// These are the "template" values stamped into component instances when a
// unit is spawned. Each maps to a corresponding XxxComp struct.

struct MeleeAttackSpec {
    int32_t damage;
    int32_t range;   // FixedVec2 units
};

struct HealSpec {
    int32_t heal_rate;
    int32_t combat_cooldown;
};

struct MoveSpec {
    int32_t speed;   // FixedVec2 units per tick
};

// ── UnitDef ──────────────────────────────────────────────────────────────────
struct UnitDef {
    UnitType    type;
    int32_t     max_hp;
    ResourceCost train_cost;

    // Presence of an optional means the unit type carries that component.
    // Absence means the unit type simply does not have that ability.
    std::optional<MeleeAttackSpec> melee_attack;
    std::optional<HealSpec>        heal;

    // All unit types can move; speed 0 would make a unit stationary.
    MoveSpec move;
};

// ── k_unit_defs — the single source of truth for unit parameters ──────────────
// Index must match UnitType enum value (Soldier = 0, etc.).
// Game balance lives here — tweak values without touching any system code.
inline const std::array<UnitDef, 1> k_unit_defs{{
    {   /* Soldier */
        .type       = UnitType::Soldier,
        .max_hp     = 100,
        .train_cost = { .gold = 50, .wood = 0 },

        .melee_attack = MeleeAttackSpec{
            .damage = 5,                             // 5 HP/tick vs single enemy
            .range  = 9 * FixedVec2::SCALE / 5,     // 1.8 tiles = 1.8 × collision radius
        },

        .heal = HealSpec{
            .heal_rate       = 1,   // 1 HP/tick while out of combat
            .combat_cooldown = 20,  // 2 seconds at 10 ticks/sec
        },

        .move = MoveSpec{
            .speed = FixedVec2::SCALE / 4,  // 0.25 tiles/tick = 2.5 tiles/sec @ 10Hz
        },
    },
}};

[[nodiscard]] inline const UnitDef& get_unit_def(UnitType t) noexcept {
    return k_unit_defs[static_cast<uint8_t>(t)];
}

} // namespace nc
