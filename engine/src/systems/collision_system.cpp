#include "collision_system.hpp"
#include "engine/game_state.hpp"
#include <vector>

namespace nc::systems {

// Minimum centre-to-centre separation between any two alive units.
// This is "r" in the ruleset spec.  Attack range = 1.8 * r = 9*SCALE/5.
static constexpr int32_t COLLISION_RADIUS    = FixedVec2::SCALE;          // 256
static constexpr int64_t COLLISION_RADIUS_SQ = (int64_t)COLLISION_RADIUS
                                               * COLLISION_RADIUS;         // 65536

// ── Accumulate-then-apply collision resolution ────────────────────────────────
// All pairwise displacements are summed into per-unit accumulators first.
// Positions are only updated once at the end, so the result is independent of
// the pair iteration order.  Sequential (immediate) apply was biased toward
// pushing early-indexed units (top flank) much further than late-indexed ones,
// causing entire flanks to lock up while others advanced freely.

void collision(GameState& state) {
    const int n = static_cast<int>(state.units.size());
    if (n < 2) return;

    // Accumulate displacement for each unit index.
    std::vector<int32_t> acc_x(n, 0);
    std::vector<int32_t> acc_y(n, 0);

    for (int i = 0; i < n - 1; ++i) {
        const Unit& a = state.units[i];
        if (!a.alive) continue;

        for (int j = i + 1; j < n; ++j) {
            const Unit& b = state.units[j];
            if (!b.alive) continue;

            int64_t dist_sq = a.pos.dist_sq(b.pos);
            if (dist_sq >= COLLISION_RADIUS_SQ) continue;   // no overlap

            int32_t dx = b.pos.x - a.pos.x;
            int32_t dy = b.pos.y - a.pos.y;
            int32_t d  = FixedVec2::isqrt(dist_sq);

            int32_t push_x, push_y;
            if (d == 0) {
                // Exact coincidence: push along +x/-x with full half-radius each.
                push_x = COLLISION_RADIUS / 2;
                push_y = 0;
            } else {
                // Each unit receives half the overlap distance along the ab axis.
                const int32_t overlap = COLLISION_RADIUS - d;
                push_x = (overlap * dx) / (2 * d);
                push_y = (overlap * dy) / (2 * d);
                // Guarantee at least 1 fixed-unit of push to prevent rounding stalls.
                if (push_x == 0 && push_y == 0) {
                    push_x = (dx >= 0) ? 1 : -1;
                }
            }

            // Unit i pushed away from j (negative direction), j pushed away from i.
            acc_x[i] -= push_x;
            acc_y[i] -= push_y;
            acc_x[j] += push_x;
            acc_y[j] += push_y;
        }
    }

    // Apply accumulated displacements all at once — no ordering bias.
    for (int i = 0; i < n; ++i) {
        if (!state.units[i].alive) continue;
        state.units[i].pos.x += acc_x[i];
        state.units[i].pos.y += acc_y[i];
    }
}

} // namespace nc::systems
