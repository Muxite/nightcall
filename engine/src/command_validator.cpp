#include "engine/command_validator.hpp"
#include <algorithm>

namespace nc {

// ── Shared helper ─────────────────────────────────────────────────────────────

ValidationResult CommandValidator::check_unit_owner(const GameState& state,
                                                     UnitId           unit,
                                                     PlayerId         issuer) {
    const Unit* u = state.find_unit(unit);
    if (!u || !u->alive)
        return { ValidationStatus::InvalidUnit };
    if (u->owner != issuer)
        return { ValidationStatus::WrongOwner };
    return { ValidationStatus::Ok };
}

// ── Per-command validation ────────────────────────────────────────────────────

ValidationResult CommandValidator::validate(const GameState& state, const MoveCmd& c) {
    if (auto r = check_unit_owner(state, c.unit, c.issuer); !r.ok()) return r;

    // Destination must be within map bounds (in tile space).
    int32_t tx = c.destination.x / FixedVec2::SCALE;
    int32_t ty = c.destination.y / FixedVec2::SCALE;
    if (tx < 0 || tx >= state.map_width || ty < 0 || ty >= state.map_height)
        return { ValidationStatus::OutOfBounds };

    // Unit must have a MoveComp to accept movement orders.
    if (!state.comps.move_of(c.unit))
        return { ValidationStatus::NoMoveComponent };

    return { ValidationStatus::Ok };
}

ValidationResult CommandValidator::validate(const GameState& state, const StopCmd& c) {
    return check_unit_owner(state, c.unit, c.issuer);
}

ValidationResult CommandValidator::validate(const GameState& /*state*/, const NoOpCmd&) {
    return { ValidationStatus::Ok };
}

// ── Public interface ──────────────────────────────────────────────────────────

ValidationResult CommandValidator::validate(const GameState& state, const Command& cmd) {
    return std::visit([&](const auto& c) { return validate(state, c); }, cmd);
}

CommandBatch CommandValidator::filter(const GameState& state, CommandBatch batch) {
    auto& cmds = batch.commands;
    cmds.erase(
        std::remove_if(cmds.begin(), cmds.end(),
            [&](const Command& c) { return !validate(state, c).ok(); }),
        cmds.end());
    return batch;
}

} // namespace nc
