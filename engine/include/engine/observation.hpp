#pragma once
// ── Observation API ───────────────────────────────────────────────────────────
// The Observation is the only legal read path for decision-making code
// (the AI agent and, in principle, the human player's UI). It applies fog of
// war and presents a player-centric view of the world.
//
// Keeping the AI's input to Observation (not raw GameState) means:
//   - The trained policy learns to handle partial information.
//   - Cheating through omniscience is structurally impossible.
//   - Fog-of-war rules can be tightened later without touching AI code.

#include "types.hpp"
#include "game_state.hpp"
#include <vector>

namespace nc {

// Per-unit data visible to the observing player.
struct UnitObservation {
    UnitId    id;
    PlayerId  owner;
    UnitType  type;
    FixedVec2 pos;
    int32_t   hp;
    int32_t   max_hp;
    bool      moving;
    int32_t   vel_x{0};
    int32_t   vel_y{0};
    int32_t   max_speed{0};
};

// Everything a single player can legally see this tick.
struct Observation {
    uint32_t                     tick;
    PlayerId                     observer;
    int32_t                      self_gold;
    int32_t                      self_wood;
    std::vector<UnitObservation> units;   // all units visible to |observer|

    // Encode as a flat float vector for neural-net input.
    // Size is determined by FeatureEncoder::FEATURE_DIM (ai module, Phase 3).
    [[nodiscard]] std::vector<float> to_feature_vector() const;
};

// Build an Observation for |player| from the current |state|.
// Currently implements full visibility (no fog). Add fog-of-war logic here.
[[nodiscard]] Observation observe(const GameState& state, PlayerId player);

} // namespace nc
