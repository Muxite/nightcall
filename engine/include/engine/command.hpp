#pragma once
// ── Command definitions ───────────────────────────────────────────────────────
// Commands are the ONLY way to mutate game state. Both the human player and the
// AI issue the same command types through the same CommandValidator path.
// This guarantees structural fairness and makes replay trivial (record batches).
//
// To add a new command type:
//   1. Define an XxxCmd struct here.
//   2. Add it to the Command variant alias.
//   3. Add a validate() overload in command_validator.hpp/.cpp.
//   4. Handle it in SimEngine::apply_commands() in sim_engine.cpp.

#include "types.hpp"
#include <variant>
#include <vector>

namespace nc {

// ── Command structs ───────────────────────────────────────────────────────────

// Order a unit to move toward a world-space destination at its defined speed.
// No pathfinding: the unit travels in a straight line.
struct MoveCmd {
    PlayerId  issuer;
    UnitId    unit;
    FixedVec2 destination;
};

// Order a unit to immediately halt movement.
struct StopCmd {
    PlayerId issuer;
    UnitId   unit;
};

// Explicit no-op — the player takes no action this tick.
struct NoOpCmd {
    PlayerId issuer;
};

// ── Command variant ───────────────────────────────────────────────────────────
// std::variant ensures exhaustive handling at compile time via std::visit.
// Serialisable without runtime type tagging — the index() is the type tag.
using Command = std::variant<MoveCmd, StopCmd, NoOpCmd>;

// ── CommandBatch ──────────────────────────────────────────────────────────────
// All commands for a single tick, potentially from both players.
// Commands are processed in order; within a tick all effects are accumulated
// before being applied (see SimEngine::apply_commands).
struct CommandBatch {
    std::vector<Command> commands;

    void push(Command c) { commands.push_back(std::move(c)); }
    [[nodiscard]] bool empty() const noexcept { return commands.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return commands.size(); }
};

} // namespace nc
