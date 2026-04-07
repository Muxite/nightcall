#pragma once
#include <cstdint>
#include <cstddef>

namespace nc {

// ── Strongly-typed IDs ───────────────────────────────────────────────────────
// Using enum class prevents accidental mixing of IDs across types.
enum class UnitId   : uint32_t { Invalid = 0 };
enum class PlayerId : uint8_t  { None = 0, P1 = 1, P2 = 2 };

// ── Game enumerations ────────────────────────────────────────────────────────
// To add a new unit type: add an enum value here, then add a UnitDef entry
// in unit_def.hpp. The system code requires no other changes.
enum class UnitType    : uint8_t { Soldier = 0 /*, Archer, Siege, ... */ };
enum class TerrainType : uint8_t { Plains = 0, Forest, Water, Mountain };

// ── Fixed-point 2D vector ────────────────────────────────────────────────────
// 1 unit = 1 tile. SCALE subdivisions per tile (256 = 1/256-tile precision).
// All simulation positions, velocities, and ranges use this type.
// Integer arithmetic guarantees bit-identical results across all platforms,
// compilers, and optimisation levels — essential for training parity.
struct FixedVec2 {
    static constexpr int32_t SCALE = 256;

    int32_t x{0};
    int32_t y{0};

    constexpr FixedVec2() = default;
    constexpr FixedVec2(int32_t x_, int32_t y_) : x(x_), y(y_) {}

    constexpr FixedVec2 operator+(FixedVec2 o) const noexcept { return {x + o.x, y + o.y}; }
    constexpr FixedVec2 operator-(FixedVec2 o) const noexcept { return {x - o.x, y - o.y}; }
    constexpr FixedVec2& operator+=(FixedVec2 o) noexcept { x += o.x; y += o.y; return *this; }
    constexpr bool operator==(FixedVec2 o) const noexcept { return x == o.x && y == o.y; }
    constexpr bool operator!=(FixedVec2 o) const noexcept { return !(*this == o); }

    // Squared distance — avoids sqrt for fast range checks (compare dist_sq <= range*range).
    [[nodiscard]] constexpr int64_t dist_sq(FixedVec2 o) const noexcept {
        int64_t dx = x - o.x;
        int64_t dy = y - o.y;
        return dx * dx + dy * dy;
    }

    // Integer square root — deterministic on all platforms (no FP).
    // Uses the Babylonian method converging to floor(sqrt(n)).
    [[nodiscard]] static int32_t isqrt(int64_t n) noexcept {
        if (n <= 0) return 0;
        int64_t r = n;
        int64_t q = (r + 1) / 2;
        while (q < r) { r = q; q = (r + n / r) / 2; }
        return static_cast<int32_t>(r);
    }

    [[nodiscard]] int32_t dist(FixedVec2 o) const noexcept {
        return isqrt(dist_sq(o));
    }

    // Compute a velocity vector pointing from |from| toward |to| with magnitude |speed|.
    // Returns {0,0} if from == to.
    [[nodiscard]] static FixedVec2 velocity_toward(FixedVec2 from, FixedVec2 to,
                                                    int32_t speed) noexcept {
        int64_t dx = to.x - from.x;
        int64_t dy = to.y - from.y;
        int32_t d  = isqrt(dx * dx + dy * dy);
        if (d == 0) return {0, 0};
        return { static_cast<int32_t>(dx * speed / d),
                 static_cast<int32_t>(dy * speed / d) };
    }

    // Construct from whole-tile coordinates.
    [[nodiscard]] static constexpr FixedVec2 from_tile(int32_t tx, int32_t ty) noexcept {
        return { tx * SCALE, ty * SCALE };
    }

    [[nodiscard]] constexpr bool is_zero() const noexcept { return x == 0 && y == 0; }
};

// ── Tile coordinate (integer grid position) ───────────────────────────────────
struct TileCoord {
    int16_t x{0};
    int16_t y{0};
    constexpr bool operator==(TileCoord o) const noexcept { return x == o.x && y == o.y; }
};

// ── Resource costs ────────────────────────────────────────────────────────────
struct ResourceCost {
    int32_t gold{0};
    int32_t wood{0};
};

} // namespace nc
