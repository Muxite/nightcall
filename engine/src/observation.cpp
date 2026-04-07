#include "engine/observation.hpp"
#include <algorithm>
#include <cmath>

namespace nc {

Observation observe(const GameState& state, PlayerId player) {
    // Resolve the Player struct for the observer.
    const Player* self_player = nullptr;
    for (const auto& p : state.players)
        if (p.id == player) { self_player = &p; break; }

    Observation obs;
    obs.tick       = state.tick;
    obs.observer   = player;
    obs.self_gold  = self_player ? self_player->gold : 0;
    obs.self_wood  = self_player ? self_player->wood : 0;

    // Full visibility for now — add fog-of-war here in a later phase.
    for (const auto& u : state.units) {
        if (!u.alive) continue;
        const MoveComp* mc = state.comps.move_of(u.id);
        obs.units.push_back(UnitObservation{
            .id        = u.id,
            .owner     = u.owner,
            .type      = u.type,
            .pos       = u.pos,
            .hp        = u.hp,
            .max_hp    = u.max_hp,
            .moving    = mc ? mc->moving : false,
            .vel_x     = mc ? mc->velocity.x : 0,
            .vel_y     = mc ? mc->velocity.y : 0,
            .max_speed = mc ? mc->speed : 0,
        });
    }

    return obs;
}

// ── Feature vector encoding ───────────────────────────────────────────────────
// Flat float representation for neural-net input (Phase 2 / AI training).
// Format: [self_gold_norm, self_wood_norm, per-unit features...]
// Each unit: [owner, type, x, y, hp_frac, moving, vel_x_norm, vel_y_norm]
// vel_* = velocity components / max(unit max_speed, 1) in [-1, 1] when moving.
// Padded/truncated to a fixed size so the network has a constant input shape.
// MAX_UNITS must match the constant in the Python training env.

std::vector<float> Observation::to_feature_vector() const {
    constexpr int   MAX_UNITS   = 128;
    constexpr int   UNIT_FEATS  = 8;
    constexpr float MAP_SCALE   = 20.0f * static_cast<float>(FixedVec2::SCALE);
    constexpr float GOLD_SCALE  = 1000.0f;
    constexpr float WOOD_SCALE  = 500.0f;

    std::vector<float> v;
    v.reserve(2 + MAX_UNITS * UNIT_FEATS);

    v.push_back(static_cast<float>(self_gold) / GOLD_SCALE);
    v.push_back(static_cast<float>(self_wood) / WOOD_SCALE);

    int count = 0;
    for (const auto& u : units) {
        if (count >= MAX_UNITS) break;
        float owner_norm = (u.owner == observer) ? 1.0f : -1.0f;
        v.push_back(owner_norm);
        v.push_back(static_cast<float>(u.type) / 8.0f);
        v.push_back(static_cast<float>(u.pos.x) / MAP_SCALE);
        v.push_back(static_cast<float>(u.pos.y) / MAP_SCALE);
        v.push_back(u.max_hp > 0 ? static_cast<float>(u.hp) / static_cast<float>(u.max_hp)
                                  : 0.0f);
        v.push_back(u.moving ? 1.0f : 0.0f);
        {
            const float sp = static_cast<float>(std::max(u.max_speed, 1));
            v.push_back(static_cast<float>(u.vel_x) / sp);
            v.push_back(static_cast<float>(u.vel_y) / sp);
        }
        ++count;
    }

    // Pad remaining unit slots with zeros so size is always fixed.
    v.resize(2 + MAX_UNITS * UNIT_FEATS, 0.0f);
    return v;
}

} // namespace nc
