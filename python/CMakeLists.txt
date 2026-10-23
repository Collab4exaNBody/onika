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

# Package layout, identical in the build tree and in the install tree:
#   <root>/pyonika/{__init__,env,mpi,_paths}.py + _core.<abi>.so
#   <root>/pyproject.toml   (pip install <root>)
set(PYONIKA_BUILD_ROOT ${PROJECT_BINARY_DIR}/python)
set(PYONIKA_INSTALL_ROOT python)

pybind11_add_module(_core
  module.cpp
  bind_scg.cpp
  bind_soatl.cpp
  bind_factory.cpp
  bind_app.cpp
)
target_link_libraries(_core PRIVATE onika)
target_include_directories(_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(_core PROPERTIES
  CXX_STANDARD 20
  LIBRARY_OUTPUT_DIRECTORY ${PYONIKA_BUILD_ROOT}/pyonika
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
)
install(TARGETS _core DESTINATION ${PYONIKA_INSTALL_ROOT}/pyonika)
# header for extension modules of downstream packages (slot extractors)
install(FILES include/onika/python/slot_extractor_api.h DESTINATION include/onika/python)

file(MAKE_DIRECTORY ${PYONIKA_BUILD_ROOT}/pyonika)
file(GLOB PYONIKA_PY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/pyonika/*.py)
foreach(f ${PYONIKA_PY_SOURCES})
  configure_file(${f} ${PYONIKA_BUILD_ROOT}/pyonika/ COPYONLY)
endforeach()
configure_file(pyonika/_paths.py.in ${PYONIKA_BUILD_ROOT}/pyonika/_paths.py @ONLY)
configure_file(pyproject.toml.in ${PYONIKA_BUILD_ROOT}/pyproject.toml @ONLY)
install(FILES ${PYONIKA_PY_SOURCES} ${PYONIKA_BUILD_ROOT}/pyonika/_paths.py DESTINATION ${PYONIKA_INSTALL_ROOT}/pyonika)
install(FILES ${PYONIKA_BUILD_ROOT}/pyproject.toml DESTINATION ${PYONIKA_INSTALL_ROOT})

if(ONIKA_ENABLE_TESTS)
  set(PYONIKA_TEST_ENV
    "PYTHONPATH=${PYONIKA_BUILD_ROOT}:$ENV{PYTHONPATH}"
    "ONIKA_CONFIG_PATH=${PROJECT_SOURCE_DIR}/data/config"
    "ONIKA_PLUGIN_PATH=${PROJECT_BINARY_DIR}/plugins")
  add_test(NAME pyonika_import
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/examples/pyonika_dryrun_test_import_pyonika.py)
  add_test(NAME pyonika_run_simulation
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/examples/pyonika_run_simulation.py)
  add_test(NAME pyonika_tests
           COMMAND ${Python3_EXECUTABLE} -m pytest -q -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  set_tests_properties(pyonika_import pyonika_run_simulation pyonika_tests PROPERTIES ENVIRONMENT "${PYONIKA_TEST_ENV}")
endif()
