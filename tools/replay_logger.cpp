// replay_logger — runs a short simulation and writes a JSONL replay file.
//
// Each line is one tick:
//   {"tick":0,"map_w":20,"map_h":20,"winner":null,
//    "units":[{"id":1,"owner":1,"x":1.25,"y":10.0,"hp":100,"max_hp":100,"moving":false},...]}
//
// Usage:
//   replay_logger [output.jsonl]   (defaults to replay.jsonl)

#include "engine/game_state.hpp"
#include "engine/sim_engine.hpp"
#include "engine/command.hpp"

#include <fstream>
#include <iostream>
#include <string>

using namespace nc;

// ── Tiny JSON helpers ─────────────────────────────────────────────────────────

static std::string json_unit(const Unit& u, const MoveComp* mc) {
    float x = static_cast<float>(u.pos.x) / static_cast<float>(FixedVec2::SCALE);
    float y = static_cast<float>(u.pos.y) / static_cast<float>(FixedVec2::SCALE);
    bool  moving = mc && mc->moving;

    return std::string("{")
        + "\"id\":"     + std::to_string(static_cast<uint32_t>(u.id))    + ","
        + "\"owner\":"  + std::to_string(static_cast<uint8_t>(u.owner))  + ","
        + "\"x\":"      + std::to_string(x)                               + ","
        + "\"y\":"      + std::to_string(y)                               + ","
        + "\"hp\":"     + std::to_string(u.hp)                            + ","
        + "\"max_hp\":" + std::to_string(u.max_hp)                        + ","
        + "\"moving\":" + (moving ? "true" : "false")
        + "}";
}

static std::string json_frame(const GameState& state) {
    std::string s;
    s += "{\"tick\":" + std::to_string(state.tick);
    s += ",\"map_w\":" + std::to_string(state.map_width);
    s += ",\"map_h\":" + std::to_string(state.map_height);

    if (state.winner.has_value()) {
        s += ",\"winner\":" + std::to_string(static_cast<uint8_t>(*state.winner));
    } else {
        s += ",\"winner\":null";
    }

    s += ",\"units\":[";
    bool first = true;
    for (const auto& u : state.units) {
        if (!u.alive) continue;
        if (!first) s += ",";
        first = false;
        s += json_unit(u, state.comps.move_of(u.id));
    }
    s += "]}";
    return s;
}

// ── Scenario ─────────────────────────────────────────────────────────────────
// 4 P1 units vs 4 P2 units on a 30×30 map.
// P1 is ordered to rush the centre; P2 stands still.
// Runs until a winner is decided or 500 ticks.

int main(int argc, char** argv) {
    const std::string out_path = (argc >= 2) ? argv[1] : "replay.jsonl";

    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "replay_logger: cannot open '" << out_path << "' for writing\n";
        return 1;
    }

    constexpr uint16_t W = 30, H = 30;
    constexpr int      P1 = 4, P2 = 4;
    constexpr uint32_t MAX_TICKS = 500;

    GameState   state  = GameState::make_default(W, H, P1, P2);
    SimEngine   engine;

    // Both sides rush toward the map centre so they collide and fight.
    CommandBatch opening;
    FixedVec2 centre = FixedVec2::from_tile(W / 2, H / 2);
    for (const auto& u : state.units) {
        PlayerId owner = u.owner;
        opening.push(MoveCmd{ owner, u.id, centre });
    }

    // Log tick 0 (initial state before any movement).
    out << json_frame(state) << "\n";

    CommandBatch empty;
    bool first_tick = true;

    for (uint32_t t = 0; t < MAX_TICKS; ++t) {
        engine.tick(state, first_tick ? opening : empty);
        first_tick = false;
        out << json_frame(state) << "\n";
        if (state.winner.has_value()) break;
    }

    std::cout << "replay_logger: wrote " << out_path
              << "  (" << state.tick << " ticks";
    if (state.winner.has_value())
        std::cout << ", winner=" << static_cast<int>(*state.winner);
    std::cout << ")\n";
    return 0;
}
