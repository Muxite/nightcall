cmake_minimum_required(VERSION 3.28)
project(nightcall CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Options ──────────────────────────────────────────────────────────────────
option(NIGHTCALL_BUILD_TESTS   "Build unit tests"         ON)
option(NIGHTCALL_BUILD_PYTHON  "Build Python bindings"    OFF)  # Phase 2
option(NIGHTCALL_BUILD_APP     "Build the game binary"    OFF)  # Phase 4+

# ── vcpkg packages (only what Phase 1 needs) ─────────────────────────────────
find_package(spdlog REQUIRED)

# ── Submodules ────────────────────────────────────────────────────────────────
add_subdirectory(engine)

add_subdirectory(tools)
add_subdirectory(python)

if(NIGHTCALL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
