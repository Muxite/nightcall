#pragma once
#include "engine/game_state.hpp"

namespace nc::systems {

// ── Collision system ──────────────────────────────────────────────────────────
// Pushes overlapping alive units apart so they maintain a minimum separation.
//
// Collision radius r = SCALE (1 tile).  Two units push apart whenever their
// centre-to-centre distance is less than r.  Each unit receives half the
// overlap displacement along the separation axis (equal and opposite impulses).
//
// Attack range in unit_def.hpp is set to 1.8 * r = 9*SCALE/5 so that units
// engage at arm's length rather than having to fully overlap to fight.
//
// Called after movement, before combat, so positions are stable when the
// combat range check runs.

void collision(GameState& state);

} // namespace nc::systems
