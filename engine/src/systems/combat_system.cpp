#include "combat_system.hpp"
#include "engine/game_state.hpp"
#include <map>
#include <vector>
#include <algorithm>

namespace nc::systems {

// ── Combat system ─────────────────────────────────────────────────────────────
// Multi-enemy damage rule (the core strategic driver):
//
//   For each attacker A with a MeleeAttackComp:
//     Let enemies = { alive enemy units within A.range }
//     Let N       = |enemies|
//
//     If N == 0: A is out of combat this tick.
//
//     If N >= 1:
//       A deals floor(A.damage / N) to each enemy.   [damage splits]
//       Each enemy E independently deals E.damage to A via its own pass.
//
//   Net effect on A surrounded by N enemies:
//     A takes:   sum of each enemy's damage (= N × base_damage for equal units)
//     A deals:   base_damage / N to each enemy
//
// This is implemented by accumulating all damage in a map<UnitId, int32_t>
// and applying it after all attackers have been processed. This avoids
// processing-order bias: whether A or B is processed first does not affect
// the damage they deal to each other.
//
// The map is ordered (std::map) so iteration is deterministic.

void combat(GameState& state) {
    // Accumulate damage dealt to each unit this tick.
    std::map<UnitId, int32_t> damage_taken;

    // Track which units were in combat so we can reset their heal cooldown.
    std::vector<UnitId> in_combat;

    for (const auto& attacker : state.units) {
        if (!attacker.alive) continue;
        const MeleeAttackComp* mac = state.comps.melee_of(attacker.id);
        if (!mac) continue;

        const int64_t range_sq = static_cast<int64_t>(mac->range) * mac->range;

        // Gather all living enemies within range.
        std::vector<UnitId> enemies_in_range;
        enemies_in_range.reserve(8);

        for (const auto& target : state.units) {
            if (!target.alive)               continue;
            if (target.owner == attacker.owner) continue;
            if (attacker.pos.dist_sq(target.pos) <= range_sq)
                enemies_in_range.push_back(target.id);
        }

        const int32_t N = static_cast<int32_t>(enemies_in_range.size());
        if (N == 0) continue;

        // Damage per enemy: floor(damage / N), minimum 1.
        const int32_t dmg_each = std::max(1, mac->damage / N);

        for (UnitId eid : enemies_in_range)
            damage_taken[eid] += dmg_each;

        in_combat.push_back(attacker.id);
    }

    // Apply accumulated damage and update combat timers.
    for (auto& [uid, dmg] : damage_taken) {
        Unit* u = state.find_unit(uid);
        if (!u || !u->alive) continue;

        u->hp -= dmg;
        if (u->hp <= 0) {
            u->hp    = 0;
            u->alive = false;
        }

        in_combat.push_back(uid);
    }

    // Reset ticks_since_combat for every unit involved in combat this tick.
    for (UnitId uid : in_combat) {
        if (HealComp* hc = state.comps.heal_of(uid))
            hc->ticks_since_combat = 0;
    }
}

} // namespace nc::systems
