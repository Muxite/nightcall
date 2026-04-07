#pragma once
#include "command.hpp"
#include "game_state.hpp"

namespace nc {

enum class ValidationStatus {
    Ok,
    InvalidUnit,     // unit does not exist or is dead
    WrongOwner,      // command issuer does not own the unit
    OutOfBounds,     // destination is outside the map
    NoMoveComponent, // unit cannot move (has no MoveComp)
};

struct ValidationResult {
    ValidationStatus status{ValidationStatus::Ok};
    [[nodiscard]] bool ok() const noexcept { return status == ValidationStatus::Ok; }
};

// All validation is pure: read-only access to GameState, no side effects.
class CommandValidator {
public:
    // Validate a single command. Returns Ok or the first violation found.
    [[nodiscard]] static ValidationResult validate(const GameState& state,
                                                   const Command&   cmd);

    // Validate an entire batch, dropping invalid commands in-place.
    // Returns the filtered batch (all remaining commands are valid).
    [[nodiscard]] static CommandBatch filter(const GameState& state,
                                             CommandBatch     batch);

private:
    [[nodiscard]] static ValidationResult validate(const GameState& state, const MoveCmd& c);
    [[nodiscard]] static ValidationResult validate(const GameState& state, const StopCmd& c);
    [[nodiscard]] static ValidationResult validate(const GameState& state, const NoOpCmd&);

    // Shared helper: check unit exists, is alive, and is owned by |issuer|.
    [[nodiscard]] static ValidationResult check_unit_owner(const GameState& state,
                                                           UnitId           unit,
                                                           PlayerId         issuer);
};

} // namespace nc
