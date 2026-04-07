add_library(nightcall_engine STATIC
    src/game_state.cpp
    src/command_validator.cpp
    src/observation.cpp
    src/sim_engine.cpp
    src/systems/movement_system.cpp
    src/systems/collision_system.cpp
    src/systems/combat_system.cpp
    src/systems/heal_system.cpp
    src/systems/cleanup_system.cpp
)

target_include_directories(nightcall_engine
    PUBLIC  include
    PRIVATE src          # so sim_engine.cpp can #include "systems/..."
)

# Zero platform or third-party dependencies — intentional.
# Adding SDL3/bgfx here would break Python training bindings and test isolation.
target_compile_features(nightcall_engine PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(nightcall_engine PRIVATE /W4 /WX)
else()
    target_compile_options(nightcall_engine PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
