find_package(Catch2 3 REQUIRED)

add_executable(test_engine
    engine/test_sim_determinism.cpp
    engine/test_commands.cpp
    engine/test_observation.cpp
)

target_link_libraries(test_engine
    PRIVATE
        nightcall_engine
        Catch2::Catch2WithMain
)

# Catch2 CTest integration — registers each TEST_CASE as a separate ctest entry.
include(Catch)
catch_discover_tests(test_engine)
