#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "engine/game_state.hpp"
#include "engine/observation.hpp"
#include "engine/sim_engine.hpp"
#include "engine/command.hpp"

using namespace nc;

// ── Helpers ───────────────────────────────────────────────────────────────────

static constexpr int FEATURE_DIM  = 2 + 128 * 8;   // matches observation.cpp constants
static constexpr int UNIT_FEATS   = 8;

// ── observe() correctness ─────────────────────────────────────────────────────

TEST_CASE("observe: returns all alive units (full visibility)", "[observation]") {
    GameState state = GameState::make_default(20, 20, 2, 3);
    Observation obs = observe(state, PlayerId::P1);

    REQUIRE(obs.tick     == 0);
    REQUIRE(obs.observer == PlayerId::P1);
    REQUIRE(obs.units.size() == 5);   // 2 P1 + 3 P2
}

TEST_CASE("observe: excludes dead units", "[observation]") {
    GameState state = GameState::make_default(20, 20, 2, 2);
    state.units[0].alive = false;   // kill first P1 unit

    Observation obs = observe(state, PlayerId::P1);
    REQUIRE(obs.units.size() == 3);  // 1 alive P1 + 2 P2
}

TEST_CASE("observe: reports correct resources for observer", "[observation]") {
    GameState state = GameState::make_default(20, 20, 1, 1);
    // make_default gives each player 200 gold, 100 wood
    Observation obs = observe(state, PlayerId::P1);

    REQUIRE(obs.self_gold == 200);
    REQUIRE(obs.self_wood == 100);
}

TEST_CASE("observe: unit positions match game state", "[observation]") {
    GameState state = GameState::make_default(20, 20, 1, 1);
    Observation obs = observe(state, PlayerId::P1);

    for (const auto& uo : obs.units) {
        const Unit* u = state.find_unit(uo.id);
        REQUIRE(u != nullptr);
        REQUIRE(uo.pos.x == u->pos.x);
        REQUIRE(uo.pos.y == u->pos.y);
        REQUIRE(uo.hp    == u->hp);
        REQUIRE(uo.owner == u->owner);
    }
}

TEST_CASE("observe: moving flag tracks MoveComp state", "[observation]") {
    GameState state  = GameState::make_default(20, 20, 1, 1);
    SimEngine engine;

    Observation before = observe(state, PlayerId::P1);
    for (const auto& u : before.units)
        if (u.owner == PlayerId::P1)
            REQUIRE_FALSE(u.moving);   // not moving yet

    CommandBatch batch;
    batch.push(MoveCmd{ PlayerId::P1, state.units[0].id,
                        FixedVec2::from_tile(15, 10) });
    engine.tick(state, batch);

    Observation after = observe(state, PlayerId::P1);
    for (const auto& u : after.units)
        if (u.owner == PlayerId::P1)
            REQUIRE(u.moving);         // now moving
}

// ── Feature vector size and structure ─────────────────────────────────────────

TEST_CASE("feature vector: correct dimension", "[observation]") {
    GameState state = GameState::make_default(20, 20, 3, 3);
    auto fv = observe(state, PlayerId::P1).to_feature_vector();
    REQUIRE(static_cast<int>(fv.size()) == FEATURE_DIM);
}

TEST_CASE("feature vector: size is constant regardless of unit count", "[observation]") {
    auto fv0 = observe(GameState::make_default(20, 20, 0, 0), PlayerId::P1).to_feature_vector();
    auto fv4 = observe(GameState::make_default(20, 20, 4, 4), PlayerId::P1).to_feature_vector();
    REQUIRE(fv0.size() == fv4.size());
    REQUIRE(static_cast<int>(fv0.size()) == FEATURE_DIM);
}

TEST_CASE("feature vector: resources at indices 0 and 1", "[observation]") {
    GameState state = GameState::make_default(20, 20, 1, 1);
    auto fv = observe(state, PlayerId::P1).to_feature_vector();

    // gold=200/1000=0.2, wood=100/500=0.2
    REQUIRE(fv[0] == Catch::Approx(0.2f));
    REQUIRE(fv[1] == Catch::Approx(0.2f));
}

TEST_CASE("feature vector: ally owner_norm = +1, enemy = -1", "[observation]") {
    GameState state = GameState::make_default(20, 20, 1, 1);
    auto fv = observe(state, PlayerId::P1).to_feature_vector();

    bool found_ally  = false;
    bool found_enemy = false;
    for (int i = 0; i < 2; ++i) {
        float owner_norm = fv[2 + i * UNIT_FEATS];
        if (owner_norm ==  1.0f) found_ally  = true;
        if (owner_norm == -1.0f) found_enemy = true;
    }
    REQUIRE(found_ally);
    REQUIRE(found_enemy);
}

TEST_CASE("feature vector: hp_frac = 1.0 for full-health units", "[observation]") {
    GameState state = GameState::make_default(20, 20, 2, 2);
    auto fv = observe(state, PlayerId::P1).to_feature_vector();

    // HP frac is slot 4 in each unit block (0..3 pos, 4 hp, 5 moving, 6–7 vel)
    for (int i = 0; i < 4; ++i) {
        float hp_frac = fv[2 + i * UNIT_FEATS + 4];
        REQUIRE(hp_frac == Catch::Approx(1.0f));
    }
}

TEST_CASE("feature vector: unused slots are zero-padded", "[observation]") {
    // Build a state with exactly 1 unit so slots 1..63 must be zero.
    GameState state;
    state.map_width  = 10;
    state.map_height = 10;
    state.tiles.resize(100);
    state.players[0] = Player{ .id = PlayerId::P1, .gold = 0, .wood = 0 };
    state.players[1] = Player{ .id = PlayerId::P2, .gold = 0, .wood = 0 };
    state.spawn_unit(PlayerId::P1, UnitType::Soldier, FixedVec2::from_tile(5, 5));

    auto fv = observe(state, PlayerId::P1).to_feature_vector();

    for (int slot = 1; slot < 128; ++slot) {
        for (int f = 0; f < UNIT_FEATS; ++f) {
            INFO("slot=" << slot << " feat=" << f);
            REQUIRE(fv[2 + slot * UNIT_FEATS + f] == 0.0f);
        }
    }
}

TEST_CASE("feature vector: position normalised to [0,1]", "[observation]") {
    // Units placed at known tile coordinates; check normalised values are in range.
    GameState state = GameState::make_default(20, 20, 4, 4);
    auto fv = observe(state, PlayerId::P1).to_feature_vector();

    for (int i = 0; i < 8; ++i) {
        float x_norm = fv[2 + i * UNIT_FEATS + 2];
        float y_norm = fv[2 + i * UNIT_FEATS + 3];
        REQUIRE(x_norm >= 0.0f);
        REQUIRE(x_norm <= 1.0f);
        REQUIRE(y_norm >= 0.0f);
        REQUIRE(y_norm <= 1.0f);
    }
}

TEST_CASE("feature vector: moving flag updates after move command", "[observation]") {
    GameState state  = GameState::make_default(20, 20, 1, 1);
    SimEngine engine;

    auto fv_before = observe(state, PlayerId::P1).to_feature_vector();

    CommandBatch batch;
    batch.push(MoveCmd{ PlayerId::P1, state.units[0].id,
                        FixedVec2::from_tile(15, 10) });
    engine.tick(state, batch);

    auto fv_after = observe(state, PlayerId::P1).to_feature_vector();

    // Find the P1 unit slot and check moving flag (feature index 5)
    bool found_moving = false;
    for (int i = 0; i < 128; ++i) {
        float owner = fv_after[2 + i * UNIT_FEATS];
        float moving = fv_after[2 + i * UNIT_FEATS + 5];
        if (owner == 1.0f && moving == 1.0f) { found_moving = true; break; }
    }
    REQUIRE(found_moving);
}
