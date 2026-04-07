#pragma once
// ── GameState ─────────────────────────────────────────────────────────────────
// The complete, authoritative snapshot of the game world at tick N.
// Constraints:
//   - Pure data: no pointers to external resources, no virtual functions,
//     no I/O handles, no std::function.
//   - Serialisable: can be turned into bytes and reconstructed identically.
//   - No platform dependencies: links against nothing except the C++ stdlib.

#include "types.hpp"
#include "components.hpp"
#include <vector>
#include <array>
#include <optional>
#include <span>
#include <cstddef>
#include <algorithm>

namespace nc {

// ── Tile ──────────────────────────────────────────────────────────────────────
struct Tile {
    TerrainType terrain{TerrainType::Plains};
};

// ── Unit ──────────────────────────────────────────────────────────────────────
struct Unit {
    UnitId    id{UnitId::Invalid};
    PlayerId  owner{PlayerId::None};
    UnitType  type{UnitType::Soldier};
    uint8_t   _pad[2]{};            // explicit padding — always zero, keeps serialisation deterministic
    FixedVec2 pos{};
    int32_t   hp{0};
    int32_t   max_hp{0};
    bool      alive{true};
    uint8_t   _pad2[3]{};           // explicit padding — always zero
};

// ── Player ────────────────────────────────────────────────────────────────────
struct Player {
    PlayerId id{PlayerId::None};
    uint8_t  _pad[3]{};             // explicit padding — always zero
    int32_t  gold{0};
    int32_t  wood{0};
};

// ── ComponentStore ────────────────────────────────────────────────────────────
// Holds flat vectors of each component type.
// Each vector is kept sorted by unit_id so iteration order is deterministic.
//
// To add a component type:
//   1. Add std::vector<XxxComp> here.
//   2. Add helper accessor methods (xxx_of / const overload).
//   3. Update remove() to erase from the new vector.
//   4. Update GameState::spawn_unit() in game_state.cpp to initialise the component.
struct ComponentStore {
    std::vector<MeleeAttackComp> melee_attack;
    std::vector<HealComp>        heal;
    std::vector<MoveComp>        move;

    // ── Accessors ─────────────────────────────────────────────────────────────
    // Linear scan — sufficient for unit counts in the hundreds.
    // Replace with binary search on sorted vectors if profiling demands it.
    template<typename T>
    [[nodiscard]] static T* find(std::vector<T>& vec, UnitId id) noexcept {
        for (auto& c : vec)
            if (c.unit_id == id) return &c;
        return nullptr;
    }
    template<typename T>
    [[nodiscard]] static const T* find(const std::vector<T>& vec, UnitId id) noexcept {
        for (const auto& c : vec)
            if (c.unit_id == id) return &c;
        return nullptr;
    }

    [[nodiscard]] MeleeAttackComp*       melee_of(UnitId id)       noexcept { return find(melee_attack, id); }
    [[nodiscard]] const MeleeAttackComp* melee_of(UnitId id) const noexcept { return find(melee_attack, id); }
    [[nodiscard]] HealComp*              heal_of(UnitId id)        noexcept { return find(heal, id); }
    [[nodiscard]] const HealComp*        heal_of(UnitId id)  const noexcept { return find(heal, id); }
    [[nodiscard]] MoveComp*              move_of(UnitId id)        noexcept { return find(move, id); }
    [[nodiscard]] const MoveComp*        move_of(UnitId id)  const noexcept { return find(move, id); }

    // Remove all components belonging to |id|.
    void remove(UnitId id) {
        auto erase_id = [id](auto& vec) {
            vec.erase(
                std::remove_if(vec.begin(), vec.end(),
                    [id](const auto& c) { return c.unit_id == id; }),
                vec.end());
        };
        erase_id(melee_attack);
        erase_id(heal);
        erase_id(move);
    }
};

// ── GameState ─────────────────────────────────────────────────────────────────
struct GameState {
    uint32_t tick{0};
    uint16_t map_width{0};
    uint16_t map_height{0};

    std::vector<Tile>         tiles;    // row-major: tiles[y * map_width + x]
    std::vector<Unit>         units;    // includes just-killed units until cleanup
    std::array<Player, 2>     players{};

    ComponentStore            comps;

    std::optional<PlayerId>   winner;
    uint32_t                  next_unit_id{1};  // monotonically increasing ID counter

    // ── Unit lookup ───────────────────────────────────────────────────────────
    [[nodiscard]] Unit*       find_unit(UnitId id)       noexcept;
    [[nodiscard]] const Unit* find_unit(UnitId id) const noexcept;

    // ── Tile access ───────────────────────────────────────────────────────────
    [[nodiscard]] const Tile& tile_at(int x, int y) const noexcept {
        return tiles[static_cast<size_t>(y) * map_width + x];
    }

    // ── Serialisation ─────────────────────────────────────────────────────────
    // Simple sequential binary format. Used for determinism tests and replays.
    [[nodiscard]] std::vector<std::byte> serialize()                           const;
    [[nodiscard]] static GameState       deserialize(std::span<const std::byte> data);

    // ── Factory ───────────────────────────────────────────────────────────────
    // Create a default starting state: flat map, units on opposite sides.
    // If spawn_seed != 0, each unit gets a small random tile offset (deterministic
    // from the seed) so training episodes vary; spawn_seed == 0 preserves the
    // legacy fixed layout for tests and tooling.
    [[nodiscard]] static GameState make_default(uint16_t w, uint16_t h,
                                                 int p1_units, int p2_units,
                                                 uint32_t spawn_seed = 0);

    // Spawn a single unit for |owner| at position |pos| of type |type|.
    // Initialises all components from the matching UnitDef.
    UnitId spawn_unit(PlayerId owner, UnitType type, FixedVec2 pos);
};

} // namespace nc
