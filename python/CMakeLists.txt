# Python.h must be found before pybind11 is configured.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.13.6)
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(pyonika
  module.cpp
  bind_scg.cpp
  bind_soatl.cpp
  bind_factory.cpp
  bind_app.cpp
)
target_link_libraries(pyonika PRIVATE onika)
set_target_properties(pyonika PROPERTIES
  CXX_STANDARD 20
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)
install(TARGETS pyonika DESTINATION lib)

if(ONIKA_ENABLE_TESTS)
  set(PYONIKA_TEST_ENV
    "PYTHONPATH=${PROJECT_BINARY_DIR}/lib:$ENV{PYTHONPATH}"
    "ONIKA_CONFIG_PATH=${PROJECT_SOURCE_DIR}/data/config"
    "ONIKA_PLUGIN_PATH=${PROJECT_BINARY_DIR}/plugins")
  add_test(NAME pyonika_import
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/exemples/pyonika_dryrun_test_import_pyonika.py)
  add_test(NAME pyonika_run_simulation
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/exemples/pyonika_run_simulation.py)
  set_tests_properties(pyonika_import pyonika_run_simulation PROPERTIES ENVIRONMENT "${PYONIKA_TEST_ENV}")
endif()
