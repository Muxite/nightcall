if(NOT NIGHTCALL_BUILD_PYTHON)
    return()
endif()

# ── Find Python ───────────────────────────────────────────────────────────────
# Hint variables (Python_EXECUTABLE, Python_ROOT_DIR) are injected by
# CMakePresets.json so the right interpreter is picked on machines with
# multiple Python installations.

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

# ── Find pybind11 (via vcpkg) ─────────────────────────────────────────────────

find_package(pybind11 REQUIRED)

# ── nightcall_sim extension module ───────────────────────────────────────────
# pybind11_add_module handles suffix (.pyd on Windows), visibility flags, etc.

pybind11_add_module(nightcall_sim bindings/sim_bindings.cpp)

target_link_libraries(nightcall_sim PRIVATE nightcall_engine)

# MinGW: suppress -hypot redefinition warning from Python headers, and
# statically embed the MinGW runtimes so the .pyd has no external DLL deps.
if(MINGW)
    target_compile_definitions(nightcall_sim PRIVATE _hypot=hypot)
    target_link_options(nightcall_sim PRIVATE
        -static-libgcc
        -static-libstdc++
        -Wl,-Bstatic,--whole-archive -lwinpthread -Wl,--no-whole-archive,-Bdynamic
    )
endif()

# Output the .pyd/.so directly into build/python/ so the training scripts can
# add that path once and find the module without any install step.
set_target_properties(nightcall_sim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)
