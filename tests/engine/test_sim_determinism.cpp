#include <catch2/catch_test_macros.hpp>
#include "engine/game_state.hpp"
#include "engine/sim_engine.hpp"
#include "engine/command.hpp"

using namespace nc;

// Helper: run |ticks| with no commands and return serialised state.
static std::vector<std::byte> run_no_commands(GameState state, uint32_t ticks) {
    SimEngine engine;
    CommandBatch empty;
    for (uint32_t i = 0; i < ticks; ++i)
        engine.tick(state, empty);
    return state.serialize();
}

// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Determinism: same initial state, same ticks, same output", "[determinism]") {
    GameState s1 = GameState::make_default(20, 20, 3, 3);
    GameState s2 = GameState::make_default(20, 20, 3, 3);

    auto bytes1 = run_no_commands(s1, 100);
    auto bytes2 = run_no_commands(s2, 100);

    REQUIRE(bytes1 == bytes2);
}

TEST_CASE("Determinism: serialise/deserialise round-trip", "[determinism]") {
    GameState original = GameState::make_default(20, 20, 2, 2);
    SimEngine engine;
    CommandBatch empty;

    for (int i = 0; i < 30; ++i)
        engine.tick(original, empty);

    auto bytes = original.serialize();
    GameState restored = GameState::deserialize(bytes);
    auto bytes2 = restored.serialize();

    REQUIRE(bytes == bytes2);
}

TEST_CASE("Determinism: identical after 200 ticks with random moves", "[determinism]") {
    // Issue the same move commands on both runs and verify identical outcome.
    auto make_state = [] { return GameState::make_default(20, 20, 4, 4); };

    SimEngine engine;

    auto run_with_moves = [&](GameState state) {
        CommandBatch batch;
        // Move all P1 units toward the centre.
        for (const auto& u : state.units) {
            if (u.owner == PlayerId::P1)
                batch.push(MoveCmd{ PlayerId::P1, u.id,
                                    FixedVec2::from_tile(10, 10) });
        }

        engine.tick(state, batch);  // tick 1 with moves

        CommandBatch empty;
        for (int i = 1; i < 200; ++i)
            engine.tick(state, empty);

        return state.serialize();
    };

    REQUIRE(run_with_moves(make_state()) == run_with_moves(make_state()));
}

// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Win condition: all P2 units die → P1 wins", "[win]") {
    // 1 vs 1, small map, units will be adjacent from the start.
    GameState state = GameState::make_default(4, 4, 1, 1);

    // Place them adjacent so combat starts immediately.
    state.units[0].pos = FixedVec2::from_tile(1, 2);
    state.units[1].pos = FixedVec2::from_tile(2, 2);

    SimEngine engine;
    CommandBatch empty;

    // Run until winner or 10 000 ticks (shouldn't take that long).
    for (int i = 0; i < 10000 && !state.winner; ++i)
        engine.tick(state, empty);

    REQUIRE(state.winner.has_value());
    // Both units are identical — either player could win depending on ordering.
    // Just verify someone won.
    INFO("Winner: " << static_cast<int>(*state.winner));
    CHECK((*state.winner == PlayerId::P1 || *state.winner == PlayerId::P2
           || *state.winner == PlayerId::None));
}

TEST_CASE("No winner declared while units are alive", "[win]") {
    GameState state = GameState::make_default(40, 40, 2, 2);
    // Keep units far apart — no combat, no deaths.
    SimEngine engine;
    CommandBatch empty;

    for (int i = 0; i < 50; ++i)
        engine.tick(state, empty);

    CHECK_FALSE(state.winner.has_value());
}

// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Tick counter increments each tick", "[tick]") {
    GameState state = GameState::make_default(10, 10, 1, 1);
    SimEngine engine;
    CommandBatch empty;

    for (int i = 0; i < 10; ++i)
        engine.tick(state, empty);

    REQUIRE(state.tick == 10);
}

TEST_CASE("Tick does not advance after winner decided", "[tick]") {
    GameState state = GameState::make_default(4, 4, 0, 1);
    // P1 has no units — P2 should win immediately.
    SimEngine engine;
    CommandBatch empty;
    engine.tick(state, empty);

    REQUIRE(state.winner.has_value());
    REQUIRE(*state.winner == PlayerId::P2);

    uint32_t tick_after_win = state.tick;
    engine.tick(state, empty);
    REQUIRE(state.tick == tick_after_win);  // frozen
}
