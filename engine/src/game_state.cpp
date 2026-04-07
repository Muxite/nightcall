#include "engine/game_state.hpp"
#include "engine/unit_def.hpp"
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace nc {

// ── Unit lookup ───────────────────────────────────────────────────────────────

Unit* GameState::find_unit(UnitId id) noexcept {
    for (auto& u : units)
        if (u.id == id) return &u;
    return nullptr;
}

const Unit* GameState::find_unit(UnitId id) const noexcept {
    for (const auto& u : units)
        if (u.id == id) return &u;
    return nullptr;
}

// ── Spawn ─────────────────────────────────────────────────────────────────────

UnitId GameState::spawn_unit(PlayerId owner, UnitType type, FixedVec2 pos) {
    const UnitDef& def = get_unit_def(type);
    UnitId id = static_cast<UnitId>(next_unit_id++);

    // Base unit
    units.push_back(Unit{
        .id     = id,
        .owner  = owner,
        .type   = type,
        .pos    = pos,
        .hp     = def.max_hp,
        .max_hp = def.max_hp,
        .alive  = true,
    });

    // MeleeAttackComp — if the unit type has melee attack
    if (def.melee_attack) {
        comps.melee_attack.push_back(MeleeAttackComp{
            .unit_id = id,
            .damage  = def.melee_attack->damage,
            .range   = def.melee_attack->range,
        });
    }

    // HealComp — if the unit type heals
    if (def.heal) {
        comps.heal.push_back(HealComp{
            .unit_id          = id,
            .heal_rate        = def.heal->heal_rate,
            .combat_cooldown  = def.heal->combat_cooldown,
            .ticks_since_combat = def.heal->combat_cooldown, // start ready to heal
        });
    }

    // MoveComp — all units can move
    comps.move.push_back(MoveComp{
        .unit_id  = id,
        .speed    = def.move.speed,
        .velocity = {0, 0},
        .target   = pos,
        .moving   = false,
    });

    return id;
}

// ── Factory ───────────────────────────────────────────────────────────────────

namespace {
uint32_t splitmix32(uint64_t& g) noexcept {
    uint64_t z = (g += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<uint32_t>(z ^ (z >> 31));
}
} // namespace

GameState GameState::make_default(uint16_t w, uint16_t h,
                                   int p1_units, int p2_units,
                                   uint32_t spawn_seed) {
    GameState state;
    state.map_width  = w;
    state.map_height = h;
    state.tiles.resize(static_cast<size_t>(w) * h);  // all Plains by default

    state.players[0] = Player{ .id = PlayerId::P1, .gold = 200, .wood = 100 };
    state.players[1] = Player{ .id = PlayerId::P2, .gold = 200, .wood = 100 };

    const int left_x  = static_cast<int>(w) / 4;
    const int right_x = static_cast<int>(3 * w) / 4;

    uint64_t rng = static_cast<uint64_t>(spawn_seed) * 2u + 1u;

    const int span_x = std::max(4, static_cast<int>(w) / 6);
    const int span_y = std::max(4, static_cast<int>(h) / 6);

    auto place_units = [&](PlayerId owner, int count, int anchor_x) {
        int start_y = static_cast<int>(h) / 2 - count / 2;
        for (int i = 0; i < count; ++i) {
            int tile_x = anchor_x;
            int tile_y = start_y + i;
            if (spawn_seed != 0) {
                const uint32_t rx = splitmix32(rng);
                const uint32_t ry = splitmix32(rng);
                int jx = static_cast<int>(rx % (2u * static_cast<uint32_t>(span_x) + 1u))
                       - span_x;
                int jy = static_cast<int>(ry % (2u * static_cast<uint32_t>(span_y) + 1u))
                       - span_y;
                tile_x += jx;
                tile_y += jy;
            }
            if (tile_x < 1) tile_x = 1;
            if (tile_x >= static_cast<int>(w) - 1) tile_x = static_cast<int>(w) - 2;
            if (tile_y < 0) tile_y = 0;
            if (tile_y >= static_cast<int>(h)) tile_y = static_cast<int>(h) - 1;
            state.spawn_unit(owner, UnitType::Soldier,
                             FixedVec2::from_tile(tile_x, tile_y));
        }
    };

    place_units(PlayerId::P1, p1_units, left_x);
    place_units(PlayerId::P2, p2_units, right_x);

    return state;
}

// ── Serialisation ─────────────────────────────────────────────────────────────
// Simple sequential binary format — write each field in declaration order.
// Not designed for cross-version compatibility; used only for determinism tests
// and within-session replays. Version tag added at the front for future use.

namespace {
    constexpr uint32_t SERIAL_VERSION = 1;

    struct Writer {
        std::vector<std::byte>& buf;

        template<typename T>
        void write(const T& v) {
            const auto* p = reinterpret_cast<const std::byte*>(&v);
            buf.insert(buf.end(), p, p + sizeof(T));
        }
        template<typename T>
        void write_vec(const std::vector<T>& v) {
            write(static_cast<uint32_t>(v.size()));
            for (const auto& e : v) write(e);
        }
    };

    struct Reader {
        const std::byte* ptr;
        const std::byte* end;

        template<typename T>
        T read() {
            if (ptr + sizeof(T) > end)
                throw std::runtime_error("GameState::deserialize: buffer underflow");
            T v;
            std::memcpy(&v, ptr, sizeof(T));
            ptr += sizeof(T);
            return v;
        }
        template<typename T>
        std::vector<T> read_vec() {
            uint32_t n = read<uint32_t>();
            std::vector<T> v(n);
            for (auto& e : v) e = read<T>();
            return v;
        }
    };
}

std::vector<std::byte> GameState::serialize() const {
    std::vector<std::byte> buf;
    buf.reserve(1024);
    Writer w{buf};

    w.write(SERIAL_VERSION);
    w.write(tick);
    w.write(map_width);
    w.write(map_height);
    w.write_vec(tiles);
    w.write_vec(units);
    w.write(players);
    w.write_vec(comps.melee_attack);
    w.write_vec(comps.heal);
    w.write_vec(comps.move);
    // winner: write as uint8 (0 = none, 1 = P1, 2 = P2)
    uint8_t win = winner.has_value() ? static_cast<uint8_t>(*winner) : 0;
    w.write(win);
    w.write(next_unit_id);

    return buf;
}

GameState GameState::deserialize(std::span<const std::byte> data) {
    Reader r{data.data(), data.data() + data.size()};

    uint32_t ver = r.read<uint32_t>();
    if (ver != SERIAL_VERSION)
        throw std::runtime_error("GameState::deserialize: unsupported version");

    GameState s;
    s.tick        = r.read<uint32_t>();
    s.map_width   = r.read<uint16_t>();
    s.map_height  = r.read<uint16_t>();
    s.tiles       = r.read_vec<Tile>();
    s.units       = r.read_vec<Unit>();
    s.players     = r.read<std::array<Player, 2>>();
    s.comps.melee_attack = r.read_vec<MeleeAttackComp>();
    s.comps.heal         = r.read_vec<HealComp>();
    s.comps.move         = r.read_vec<MoveComp>();
    uint8_t win  = r.read<uint8_t>();
    s.winner = (win == 0) ? std::nullopt
                           : std::optional<PlayerId>(static_cast<PlayerId>(win));
    s.next_unit_id = r.read<uint32_t>();

    return s;
}

} // namespace nc
