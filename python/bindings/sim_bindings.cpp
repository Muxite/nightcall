// nightcall_sim — pybind11 module exposing SimEngine, GameState, CommandBatch,
// and Observation to Python. This is the sole C++↔Python bridge; the training
// loop runs the same C++ SimEngine as the game binary.
//
// Exposed API (Python):
//   nc.PlayerId.{P1, P2, None_}
//   nc.UnitType.Soldier
//   nc.UnitId(int)
//   nc.FixedVec2(x, y)  /  nc.FixedVec2.from_tile(tx, ty)  /  .SCALE
//   nc.Unit           — read-only view of a unit
//   nc.Player         — read-only view of a player
//   nc.GameState.make_default(w, h, p1, p2, spawn_seed=0)
//   nc.CommandBatch() / .push_move(pid, uid, dest) / .push_stop / .push_noop
//   nc.SimEngine()    / .tick(state, batch)
//   nc.UnitObservation
//   nc.Observation    / .to_feature_vector() -> np.ndarray float32
//   nc.observe(state, player_id) -> Observation

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "engine/types.hpp"
#include "engine/game_state.hpp"
#include "engine/command.hpp"
#include "engine/sim_engine.hpp"
#include "engine/observation.hpp"

#include <cstring>
#include <string>

namespace py = pybind11;
using namespace nc;

// ── UnitId helper ─────────────────────────────────────────────────────────────
// UnitId is enum class : uint32_t — make it feel like an int in Python while
// remaining type-safe at the C++ boundary.

PYBIND11_MODULE(nightcall_sim, m) {
    m.doc() = "nightcall simulation engine — Python bindings";

    // ── Enums ─────────────────────────────────────────────────────────────────

    py::enum_<PlayerId>(m, "PlayerId")
        .value("None_",  PlayerId::None)
        .value("P1",     PlayerId::P1)
        .value("P2",     PlayerId::P2)
        .export_values();

    py::enum_<UnitType>(m, "UnitType")
        .value("Soldier", UnitType::Soldier)
        .export_values();

    // ── UnitId ────────────────────────────────────────────────────────────────

    py::class_<UnitId>(m, "UnitId")
        .def(py::init([](uint32_t v) { return static_cast<UnitId>(v); }))
        .def("__int__",  [](UnitId id) { return static_cast<uint32_t>(id); })
        .def("__index__",[](UnitId id) { return static_cast<uint32_t>(id); })
        .def("__eq__",   [](UnitId a, UnitId b)  { return a == b; })
        .def("__hash__", [](UnitId id) {
            return std::hash<uint32_t>{}(static_cast<uint32_t>(id));
        })
        .def("__repr__", [](UnitId id) {
            return "UnitId(" + std::to_string(static_cast<uint32_t>(id)) + ")";
        });

    // ── FixedVec2 ─────────────────────────────────────────────────────────────

    py::class_<FixedVec2>(m, "FixedVec2")
        .def(py::init<int32_t, int32_t>(), py::arg("x"), py::arg("y"))
        .def_readwrite("x", &FixedVec2::x)
        .def_readwrite("y", &FixedVec2::y)
        .def_static("from_tile", &FixedVec2::from_tile, py::arg("tx"), py::arg("ty"))
        .def_property_readonly_static("SCALE",
            [](py::object) { return FixedVec2::SCALE; })
        .def("__eq__",   &FixedVec2::operator==)
        .def("__repr__", [](const FixedVec2& v) {
            return "FixedVec2(" + std::to_string(v.x)
                 + ", " + std::to_string(v.y) + ")";
        });

    // ── Unit (read-only view) ─────────────────────────────────────────────────

    py::class_<Unit>(m, "Unit")
        .def_readonly("id",     &Unit::id)
        .def_readonly("owner",  &Unit::owner)
        .def_readonly("type",   &Unit::type)
        .def_readonly("pos",    &Unit::pos)
        .def_readonly("hp",     &Unit::hp)
        .def_readonly("max_hp", &Unit::max_hp)
        .def_readonly("alive",  &Unit::alive);

    // ── Player (read-only view) ───────────────────────────────────────────────

    py::class_<Player>(m, "Player")
        .def_readonly("id",   &Player::id)
        .def_readonly("gold", &Player::gold)
        .def_readonly("wood", &Player::wood);

    // ── GameState ─────────────────────────────────────────────────────────────

    py::class_<GameState>(m, "GameState")
        .def_static("make_default", &GameState::make_default,
                    py::arg("w"), py::arg("h"),
                    py::arg("p1_units"), py::arg("p2_units"),
                    py::arg("spawn_seed") = 0)
        .def_readonly("tick",       &GameState::tick)
        .def_readonly("map_width",  &GameState::map_width)
        .def_readonly("map_height", &GameState::map_height)
        .def_readonly("units",      &GameState::units)
        .def_readonly("players",    &GameState::players)
        .def_property_readonly("winner", [](const GameState& s) -> py::object {
            if (!s.winner.has_value()) return py::none();
            return py::cast(*s.winner);
        })
        // Serialise to Python bytes (for determinism checks / checkpointing)
        .def("serialize", [](const GameState& s) -> py::bytes {
            auto v = s.serialize();
            return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
        })
        .def_static("deserialize", [](py::bytes data) {
            std::string raw = data;    // py::bytes → std::string copies bytes
            std::vector<std::byte> v(raw.size());
            std::memcpy(v.data(), raw.data(), raw.size());
            return GameState::deserialize(v);
        });

    // ── CommandBatch ──────────────────────────────────────────────────────────

    py::class_<CommandBatch>(m, "CommandBatch")
        .def(py::init<>())
        .def("push_move", [](CommandBatch& b, PlayerId p, UnitId u, FixedVec2 dest) {
            b.push(MoveCmd{ p, u, dest });
        }, py::arg("issuer"), py::arg("unit"), py::arg("dest"))
        .def("push_stop", [](CommandBatch& b, PlayerId p, UnitId u) {
            b.push(StopCmd{ p, u });
        }, py::arg("issuer"), py::arg("unit"))
        .def("push_noop", [](CommandBatch& b, PlayerId p) {
            b.push(NoOpCmd{ p });
        }, py::arg("issuer"))
        .def("__len__", &CommandBatch::size);

    // ── SimEngine ─────────────────────────────────────────────────────────────

    py::class_<SimEngine>(m, "SimEngine")
        .def(py::init<>())
        .def("tick", &SimEngine::tick,
             py::arg("state"), py::arg("batch"),
             "Advance state by one tick, applying batch. No-op if winner set.");

    // ── UnitObservation ───────────────────────────────────────────────────────

    py::class_<UnitObservation>(m, "UnitObservation")
        .def_readonly("id",         &UnitObservation::id)
        .def_readonly("owner",      &UnitObservation::owner)
        .def_readonly("type",       &UnitObservation::type)
        .def_readonly("pos",        &UnitObservation::pos)
        .def_readonly("hp",         &UnitObservation::hp)
        .def_readonly("max_hp",     &UnitObservation::max_hp)
        .def_readonly("moving",     &UnitObservation::moving)
        .def_readonly("vel_x",      &UnitObservation::vel_x)
        .def_readonly("vel_y",      &UnitObservation::vel_y)
        .def_readonly("max_speed",  &UnitObservation::max_speed);

    // ── Observation ───────────────────────────────────────────────────────────

    py::class_<Observation>(m, "Observation")
        .def_readonly("tick",      &Observation::tick)
        .def_readonly("observer",  &Observation::observer)
        .def_readonly("self_gold", &Observation::self_gold)
        .def_readonly("self_wood", &Observation::self_wood)
        .def_readonly("units",     &Observation::units)
        // Return as float32 numpy array directly — no intermediate Python list.
        .def("to_feature_vector", [](const Observation& obs) {
            auto fv = obs.to_feature_vector();
            py::array_t<float> arr(static_cast<py::ssize_t>(fv.size()));
            std::copy(fv.begin(), fv.end(), arr.mutable_data());
            return arr;
        });

    // ── observe() free function ───────────────────────────────────────────────

    m.def("observe", &observe,
          py::arg("state"), py::arg("player"),
          "Build a player-centric Observation from state (full visibility currently).");
}
