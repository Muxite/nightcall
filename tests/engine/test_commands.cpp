#include <catch2/catch_test_macros.hpp>
#include "engine/game_state.hpp"
#include "engine/command.hpp"
#include "engine/command_validator.hpp"
#include "engine/sim_engine.hpp"

using namespace nc;

// ── Validation tests ──────────────────────────────────────────────────────────

TEST_CASE("MoveCmd: valid command accepted", "[validation]") {
    GameState state = GameState::make_default(20, 20, 1, 0);
    UnitId uid = state.units[0].id;

    Command cmd = MoveCmd{ PlayerId::P1, uid, FixedVec2::from_tile(10, 10) };
    auto result = CommandValidator::validate(state, cmd);
    REQUIRE(result.ok());
}

TEST_CASE("MoveCmd: wrong owner rejected", "[validation]") {
    GameState state = GameState::make_default(20, 20, 1, 0);
    UnitId uid = state.units[0].id;

    Command cmd = MoveCmd{ PlayerId::P2, uid, FixedVec2::from_tile(10, 10) };
    auto result = CommandValidator::validate(state, cmd);
    REQUIRE(result.status == ValidationStatus::WrongOwner);
}

TEST_CASE("MoveCmd: invalid unit rejected", "[validation]") {
    GameState state = GameState::make_default(20, 20, 1, 0);

    Command cmd = MoveCmd{ PlayerId::P1, UnitId{999}, FixedVec2::from_tile(5, 5) };
    auto result = CommandValidator::validate(state, cmd);
    REQUIRE(result.status == ValidationStatus::InvalidUnit);
}

TEST_CASE("MoveCmd: out-of-bounds destination rejected", "[validation]") {
    GameState state = GameState::make_default(20, 20, 1, 0);
    UnitId uid = state.units[0].id;

    Command cmd = MoveCmd{ PlayerId::P1, uid, FixedVec2::from_tile(50, 50) };
    auto result = CommandValidator::validate(state, cmd);
    REQUIRE(result.status == ValidationStatus::OutOfBounds);
}

TEST_CASE("StopCmd: valid stop command accepted", "[validation]") {
    GameState state = GameState::make_default(20, 20, 1, 0);
    UnitId uid = state.units[0].id;

    Command cmd = StopCmd{ PlayerId::P1, uid };
    REQUIRE(CommandValidator::validate(state, cmd).ok());
}

TEST_CASE("NoOpCmd: always valid", "[validation]") {
    GameState state = GameState::make_default(20, 20, 0, 0);
    Command cmd = NoOpCmd{ PlayerId::P1 };
    REQUIRE(CommandValidator::validate(state, cmd).ok());
}

TEST_CASE("filter: removes invalid commands from batch", "[validation]") {
    GameState state = GameState::make_default(20, 20, 1, 0);
    UnitId uid = state.units[0].id;

    CommandBatch batch;
    batch.push(MoveCmd{ PlayerId::P1, uid, FixedVec2::from_tile(10, 10) });   // valid
    batch.push(MoveCmd{ PlayerId::P1, UnitId{999}, FixedVec2::from_tile(5, 5) });  // invalid
    batch.push(NoOpCmd{ PlayerId::P2 });  // valid

    CommandBatch filtered = CommandValidator::filter(state, batch);
    REQUIRE(filtered.size() == 2);
}

// ── Gameplay effect tests ─────────────────────────────────────────────────────

TEST_CASE("MoveCmd: unit begins moving toward destination", "[commands]") {
    GameState state = GameState::make_default(20, 20, 1, 0);
    UnitId uid = state.units[0].id;
    FixedVec2 start = state.units[0].pos;

    CommandBatch batch;
    batch.push(MoveCmd{ PlayerId::P1, uid, FixedVec2::from_tile(10, 10) });

    SimEngine engine;
    engine.tick(state, batch);

    // Unit should have moved away from its start position.
    const Unit* u = state.find_unit(uid);
    REQUIRE(u != nullptr);
    REQUIRE(u->pos != start);
}

TEST_CASE("StopCmd: unit that is moving stops", "[commands]") {
    GameState state = GameState::make_default(20, 20, 1, 0);
    UnitId uid = state.units[0].id;

    SimEngine engine;
    CommandBatch move_batch;
    move_batch.push(MoveCmd{ PlayerId::P1, uid, FixedVec2::from_tile(15, 15) });
    engine.tick(state, move_batch);  // starts moving

    CommandBatch stop_batch;
    stop_batch.push(StopCmd{ PlayerId::P1, uid });
    engine.tick(state, stop_batch);  // stops

    FixedVec2 pos_after_stop = state.find_unit(uid)->pos;
    engine.tick(state, {});  // another tick with no commands
    FixedVec2 pos_after_idle = state.find_unit(uid)->pos;

    REQUIRE(pos_after_stop == pos_after_idle);  // did not move
}

TEST_CASE("Unit arrives at destination and stops", "[commands]") {
    GameState state = GameState::make_default(20, 20, 1, 1);  // P2 unit needed to prevent immediate win
    UnitId uid = state.units[0].id;
    FixedVec2 dest = FixedVec2::from_tile(5, 5);

    CommandBatch batch;
    batch.push(MoveCmd{ PlayerId::P1, uid, dest });

    SimEngine engine;
    engine.tick(state, batch);  // start moving

    // Run until arrived or 1000 ticks.
    CommandBatch empty;
    for (int i = 0; i < 1000; ++i) {
        if (state.find_unit(uid)->pos == dest) break;
        engine.tick(state, empty);
    }

    REQUIRE(state.find_unit(uid)->pos == dest);

    // After arriving, one more tick should not change position.
    engine.tick(state, empty);
    REQUIRE(state.find_unit(uid)->pos == dest);
}

// ── Combat tests ──────────────────────────────────────────────────────────────

TEST_CASE("Adjacent enemies take damage each tick", "[combat]") {
    GameState state = GameState::make_default(10, 10, 1, 1);
    // Place units adjacent (within 1 tile range).
    state.units[0].pos = FixedVec2::from_tile(4, 5);
    state.units[1].pos = FixedVec2::from_tile(5, 5);

    int hp0_before = state.units[0].hp;
    int hp1_before = state.units[1].hp;

    SimEngine engine;
    engine.tick(state, {});

    // Both units should have taken damage.
    REQUIRE(state.units[0].hp < hp0_before);
    REQUIRE(state.units[1].hp < hp1_before);
}

TEST_CASE("Unit outnumbered 2-to-1 takes double damage", "[combat]") {
    // 1 P1 unit vs 2 P2 units, all adjacent.
    GameState state = GameState::make_default(10, 10, 0, 0);
    state.map_width = 10; state.map_height = 10;
    state.tiles.resize(100);

    UnitId a = state.spawn_unit(PlayerId::P1, UnitType::Soldier, FixedVec2::from_tile(5, 5));
    /*UnitId b =*/ state.spawn_unit(PlayerId::P2, UnitType::Soldier, FixedVec2::from_tile(5, 4));
    /*UnitId c =*/ state.spawn_unit(PlayerId::P2, UnitType::Soldier, FixedVec2::from_tile(5, 6));

    int hp_solo_before = state.find_unit(a)->hp;

    SimEngine engine;
    engine.tick(state, {});

    int hp_solo_after = state.find_unit(a)->hp;
    int damage_taken  = hp_solo_before - hp_solo_after;

    // Each P2 unit has damage=5 → P1 unit should take 5+5=10 damage.
    REQUIRE(damage_taken == 10);
}

TEST_CASE("Unit facing 2 enemies deals half damage to each", "[combat]") {
    GameState state = GameState::make_default(10, 10, 0, 0);
    state.map_width = 10; state.map_height = 10;
    state.tiles.resize(100);

    /*UnitId a =*/ state.spawn_unit(PlayerId::P1, UnitType::Soldier, FixedVec2::from_tile(5, 5));
    UnitId b    = state.spawn_unit(PlayerId::P2, UnitType::Soldier, FixedVec2::from_tile(5, 4));
    UnitId c    = state.spawn_unit(PlayerId::P2, UnitType::Soldier, FixedVec2::from_tile(5, 6));

    int hp_b_before = state.find_unit(b)->hp;
    int hp_c_before = state.find_unit(c)->hp;

    SimEngine engine;
    engine.tick(state, {});

    // P1 unit has damage=5, N=2 enemies → deals floor(5/2)=2 to each.
    REQUIRE(state.find_unit(b)->hp == hp_b_before - 2);
    REQUIRE(state.find_unit(c)->hp == hp_c_before - 2);
}

// ── Heal tests ────────────────────────────────────────────────────────────────

TEST_CASE("Unit out of combat heals after cooldown", "[heal]") {
    GameState state = GameState::make_default(20, 20, 1, 0);
    UnitId uid = state.units[0].id;

    // Manually wound the unit.
    state.units[0].hp = 50;

    // Force ticks_since_combat past the cooldown immediately.
    HealComp* hc = state.comps.heal_of(uid);
    REQUIRE(hc != nullptr);
    hc->ticks_since_combat = hc->combat_cooldown;

    SimEngine engine;
    engine.tick(state, {});

    REQUIRE(state.find_unit(uid)->hp > 50);
}

TEST_CASE("Unit in combat does not heal", "[heal]") {
    GameState state = GameState::make_default(10, 10, 1, 1);
    state.units[0].pos = FixedVec2::from_tile(4, 5);
    state.units[1].pos = FixedVec2::from_tile(5, 5);

    state.units[0].hp = 80;
    // Set heal cooldown elapsed for P1 unit — it would heal if not in combat.
    HealComp* hc = state.comps.heal_of(state.units[0].id);
    hc->ticks_since_combat = hc->combat_cooldown + 10;

    SimEngine engine;
    engine.tick(state, {});

    // Combat system runs before heal system; combat resets the timer.
    // The unit took damage, so hp should be less than 80 (not healed despite timer).
    REQUIRE(state.find_unit(state.units[0].id)->hp < 80);
}
