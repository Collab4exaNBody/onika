# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#


cmake_minimum_required(VERSION 3.26)

# ========================================
# === Compiler toolchain configuration ===
# ========================================
# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# setup cmake package file to be used with find_package in end-user project
set(ONIKA_CMAKE_PACKAGE ${CMAKE_CURRENT_BINARY_DIR}/onika-config.cmake)
string(TIMESTAMP ONIKA_BUILD_DATE "%Y-%m-%d %Hh%M:%S")
file(WRITE ${ONIKA_CMAKE_PACKAGE} "# exaNBody CMake package (generated on ${ONIKA_BUILD_DATE})\n\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "# Main package configuration\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(CMAKE_CXX_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(CMAKE_CXX_EXTENSIONS ${CMAKE_CXX_EXTENSIONS})\n")

# ==========================
# === project definition ===
# ==========================
project(onika VERSION 1.2 LANGUAGES C CXX)

# =======================================
# === CMake & toolchain configuration ===
# =======================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(RunCommand)
include(DebugTests)
include(GenerateBenchmarkSOATL)
include(OnikaPlugin)

# export necessary cmake includes to onika-config.cmake
file(APPEND ${ONIKA_CMAKE_PACKAGE} "list(APPEND CMAKE_MODULE_PATH \"${CMAKE_INSTALL_PREFIX}/cmake\")\ninclude(RunCommand)\ninclude(OnikaPlugin)\ninclude(AddRegressionTestDir)\n")

# configure project build and link preferences
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(ONIKA_COMPILE_FEATURES cxx_std_20)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release RelWithDebInfo Debug)


# ============================
# === OpenMP configuration ===
# ============================

# OpenMP with task and ompt support is appreciated
# packages and tools
find_package(OpenMP REQUIRED)
list(APPEND ONIKA_LIBRARIES ${OpenMP_CXX_LIBRARIES})
list(APPEND ONIKA_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})

# for export to onika-config file)
file(APPEND ${ONIKA_CMAKE_PACKAGE} "find_package(OpenMP REQUIRED)\n")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(ONIKA_OMP_NUM_THREADS_WORKAROUND_DEFAULT ON)
else()
  set(ONIKA_OMP_NUM_THREADS_WORKAROUND_DEFAULT OFF)
endif()
option(ONIKA_OMP_NUM_THREADS_WORKAROUND "Enable OpenMP num_threads bug workaround" ${ONIKA_OMP_NUM_THREADS_WORKAROUND_DEFAULT})
if(ONIKA_OMP_NUM_THREADS_WORKAROUND)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_OMP_NUM_THREADS_WORKAROUND=1)
endif()

if(OpenMP_CXX_VERSION GREATER_EQUAL 5.0)
  option(ONIKA_HAVE_OPENMP_TOOLS "Eanble use of OMPT (OpenMP ver=${OpenMP_CXX_VERSION})" ON)
  option(ONIKA_HAVE_OPENMP_DETACH "Enable use of OMP 5.0 detach clause (OpenMP ver=${OpenMP_CXX_VERSION})" ON)
else()
  option(ONIKA_HAVE_OPENMP_TOOLS "Force use of OMPT (OpenMP ver=${OpenMP_CXX_VERSION})" OFF)
  option(ONIKA_HAVE_OPENMP_DETACH "Force use of OMP 5.0 detach clause (OpenMP ver=${OpenMP_CXX_VERSION})" OFF)
endif()

if(ONIKA_HAVE_OPENMP_TOOLS)
  message(STATUS "OpenMP Tools enabled")
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_HAVE_OPENMP_TOOLS=1)
  option(ONIKA_ENABLE_TASK_PROFILING "Enable Onika task profiling feature" ON)
  if(ONIKA_ENABLE_TASK_PROFILING)
    list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_ENABLE_TASK_PROFILING=1)
  endif()
endif()

if(ONIKA_HAVE_OPENMP_DETACH)
  message(STATUS "OpenMP detach enabled")
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_HAVE_OPENMP_DETACH=1)
endif()

# configure maximum number of openmp task dependences
if(NOT ONIKA_OMP_MAX_DEPENDS_DEFAULT)
  set(ONIKA_OMP_MAX_DEPENDS_DEFAULT 10)
endif()
set(ONIKA_OMP_MAX_DEPENDS ${ONIKA_OMP_MAX_DEPENDS_DEFAULT} CACHE STRING "Maximum number of OpenMP task dynamic dependences")
list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_OMP_MAX_DEPENDS=${ONIKA_OMP_MAX_DEPENDS})

#configure default OpenMP scheduling
set(ONIKA_PARFOR_OMPSCHED_DEFAULT OMP_SCHED_GUIDED CACHE STRING "Set Onika's parallel_for scheduling : OMP_SCHED_STATIC OMP_SCHED_GUIDED OMP_SCHED_DYNAMIC")
set(ONIKA_BLKPARFOR_OMPSCHED_DEFAULT OMP_SCHED_GUIDED CACHE STRING "Set Onika's block_parallel_for scheduling : OMP_SCHED_STATIC OMP_SCHED_GUIDED OMP_SCHED_DYNAMIC")
list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_BLKPARFOR_OMPSCHED_DEFAULT=${ONIKA_BLKPARFOR_OMPSCHED_DEFAULT} ONIKA_PARFOR_OMPSCHED_DEFAULT=${ONIKA_PARFOR_OMPSCHED_DEFAULT})


# ====================================
# === local SMP node configuration ===
# ====================================
find_program(LSCPU_EXECUTABLE lscpu)
if(LSCPU_EXECUTABLE)
  execute_process(COMMAND ${LSCPU_EXECUTABLE} --parse=CPU,CORE OUTPUT_VARIABLE LSCPU_DATA)
  string(REGEX REPLACE "#[^\n]*\n" "" LSCPU_DATA "{${LSCPU_DATA}}")
  string(REPLACE "," ";" LSCPU_DATA "${LSCPU_DATA}")
  string(REPLACE "\n" ";" LSCPU_DATA "${LSCPU_DATA}")
  string(REPLACE ";}" "" LSCPU_DATA "${LSCPU_DATA}")
  string(REPLACE "{" "" LSCPU_DATA "${LSCPU_DATA}")
  list(GET LSCPU_DATA -2 ONIKA_HOST_HW_THREADS)
  list(GET LSCPU_DATA -1 ONIKA_HOST_HW_CORES)
  math(EXPR ONIKA_HOST_HW_THREADS "1+${ONIKA_HOST_HW_THREADS}")
  math(EXPR ONIKA_HOST_HW_CORES "1+${ONIKA_HOST_HW_CORES}")
  set(ONIKA_ADVISED_HW_THREADS ${ONIKA_HOST_HW_THREADS})
  message(STATUS "lscpu : threads=${ONIKA_HOST_HW_THREADS} cores=${ONIKA_HOST_HW_CORES}")
else()
  set(ONIKA_HOST_HW_CORES 4)
  set(ONIKA_HOST_HW_THREADS 8)
  set(ONIKA_ADVISED_HW_THREADS 128)
endif()
list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_ADVISED_HW_THREADS=${ONIKA_ADVISED_HW_THREADS})
# for export to onika-config file)
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_HOST_HW_CORES ${ONIKA_HOST_HW_CORES})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_HOST_HW_THREADS ${ONIKA_HOST_HW_THREADS})\n")

# ===================================
# ============ Cuda =================
# ===================================
option(ONIKA_BUILD_CUDA "Enable GPU Acceleration" OFF)
option(ONIKA_ENABLE_HIP "Use HIP instead of Cuda" OFF)
if(ONIKA_ENABLE_HIP)
  set(ONIKA_USE_CUDA OFF)
  set(ONIKA_USE_HIP ${ONIKA_BUILD_CUDA})
  if(ONIKA_BUILD_CUDA)
    if(NOT ROCM_INSTALL_ROOT)
      set(ROCM_INSTALL_ROOT "/opt/rocm")
    endif()
    list(APPEND CMAKE_MODULE_PATH "${ROCM_INSTALL_ROOT}/share/rocm/cmake")
    find_package(ROCM REQUIRED)
    enable_language(HIP)
    file(REAL_PATH "${ROCM_INSTALL_ROOT}" XNB_ROCM_ROOT)
    string(REGEX MATCH "[0-9].[0-9].[0-9]" XNB_ROCM_VERSION "${XNB_ROCM_ROOT}")
    if(XNB_ROCM_VERSION)
      set(ONIKA_HIP_VERSION "${XNB_ROCM_VERSION}")
    else()
      set(ONIKA_HIP_VERSION "${CMAKE_HIP_COMPILER_VERSION}")
    endif()

    # common compile definitions and features
    set(ONIKA_GPU_COMPILE_DEFINITIONS ONIKA_CUDA_VERSION=${CMAKE_HIP_COMPILER_VERSION} ONIKA_HIP_VERSION=${ONIKA_HIP_VERSION})
    set(ONIKA_GPU_COMPILE_FEATURES hip_std_20)
    set(ONIKA_GPU_LIBRARIES roctx64)

    # HIP specific compile options
    set(ONIKA_HIP_COMPILE_FLAGS "-Werror=return-local-addr;-Werror=return-stack-address;-Werror=return-type" CACHE STRING "HIP extra flags")
    message(STATUS "Onika uses HIP ${CMAKE_HIP_COMPILER_VERSION}")

    # for export to onika-config file
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "list(APPEND CMAKE_MODULE_PATH \"${ROCM_INSTALL_ROOT}/share/rocm/cmake\")\n")
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(CMAKE_HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})\n")
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "find_package(ROCM REQUIRED)\n")
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "enable_language(HIP)\n")
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_HIP_COMPILE_FLAGS ${ONIKA_HIP_COMPILE_FLAGS})\n")
  endif()
else()
  set(ONIKA_USE_HIP OFF)
  set(ONIKA_USE_CUDA ${ONIKA_BUILD_CUDA})
  if(ONIKA_BUILD_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "Cuda target architecture(s)")
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
#    message(STATUS "CUDA_DIR=${CUDA_DIR} , CUDA_ROOT=${CUDA_ROOT}")
    if(NOT CUDA_SDK_ROOT_DIR)
      if(CMAKE_CUDA_COMPILER)
        get_filename_component(CUDA_BIN ${CMAKE_CUDA_COMPILER} DIRECTORY)
        set(CUDA_SDK_ROOT_DIR ${CUDA_BIN}/..)
        cmake_path(ABSOLUTE_PATH CUDA_SDK_ROOT_DIR NORMALIZE)
      else()
        message(FATAL_ERROR "Cannot recover Cuda SDK root directory")
      endif()
    endif()
#    message(STATUS "CUDA_SDK_ROOT_DIR = ${CUDA_SDK_ROOT_DIR}")

    # common compile definitions and features
    set(ONIKA_GPU_COMPILE_DEFINITIONS ONIKA_CUDA_VERSION=${CMAKE_CUDA_COMPILER_VERSION})
    set(ONIKA_GPU_COMPILE_FEATURES cuda_std_20)
    set(ONIKA_GPU_INCLUDE_DIRS ${CUDA_SDK_ROOT_DIR}/include)
    set(ONIKA_GPU_LIBRARY_DIRS ${CUDA_SDK_ROOT_DIR}/lib64)
    set(ONIKA_GPU_LIBRARIES cudart)

    # CUDA specific compile options
    set(ONIKA_CUDA_COMPILE_FLAGS "--Werror;cross-execution-space-call;--extended-lambda" CACHE STRING "Cuda extra flags")
    message(STATUS "Onika uses CUDA v${CMAKE_CUDA_COMPILER_VERSION} , arch=${CMAKE_CUDA_ARCHITECTURES} , root=${CUDA_SDK_ROOT_DIR}")

    # for export to onika-config file
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(CUDA_SDK_ROOT_DIR ${CUDA_SDK_ROOT_DIR})\n")
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER})\n")
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})\n")
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "find_package(CUDA REQUIRED)\n")
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "enable_language(CUDA)\n")
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_CUDA_COMPILE_FLAGS ${ONIKA_CUDA_COMPILE_FLAGS})\n")
  endif()
endif()

# add GPU flags and compile features
list(APPEND ONIKA_COMPILE_FEATURES ${ONIKA_GPU_COMPILE_FEATURES})
list(APPEND ONIKA_COMPILE_DEFINITIONS ${ONIKA_GPU_COMPILE_DEFINITIONS})
list(APPEND ONIKA_INCLUDE_DIRS ${ONIKA_GPU_INCLUDE_DIRS})
list(APPEND ONIKA_LIBRARIES ${ONIKA_GPU_LIBRARIES})
list(APPEND ONIKA_LIBRARY_DIRS ${ONIKA_GPU_LIBRARY_DIRS})

# Memory alignment settings
set(ONIKA_DEFAULT_ALIGNMENT "-1" CACHE STRING "Default field array alignment (-1 for autodetect)")
if(${ONIKA_DEFAULT_ALIGNMENT} GREATER 0)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_DEFAULT_ALIGNMENT=${ONIKA_DEFAULT_ALIGNMENT})
endif()
set(ONIKA_DEFAULT_CHUNK_SIZE "-1" CACHE STRING "Default field array vector size (-1 for autodetect)")
if(${ONIKA_DEFAULT_CHUNK_SIZE} GREATER 0)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_DEFAULT_CHUNK_SIZE=${ONIKA_DEFAULT_CHUNK_SIZE})
endif()
set(ONIKA_MINIMUM_CUDA_ALIGNMENT "-1" CACHE STRING "Default GPU memory alignment (-1 for autodetect)")
if(${ONIKA_MINIMUM_CUDA_ALIGNMENT} GREATER 0)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_MINIMUM_CUDA_ALIGNMENT=${ONIKA_MINIMUM_CUDA_ALIGNMENT})
endif()

# Parallel execution configuration
option(ONIKA_CU_ENABLE_KERNEL_BOUNDS "Use NVCC directive to restrict kernel launch bounds" OFF)
set(ONIKA_TASKS_PER_CORE "4" CACHE STRING "Number of OpenMP tasks per thread")
set(ONIKA_CU_MAX_THREADS_PER_BLOCK "256" CACHE STRING "Maximum number of threads per Cuda block")
set(ONIKA_CU_MIN_BLOCKS_PER_SM "6" CACHE STRING "Minimum number of blocks per SM")
if(ONIKA_CU_ENABLE_KERNEL_BOUNDS)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_CU_ENABLE_KERNEL_BOUNDS=1)
else()
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_CU_ENABLE_KERNEL_BOUNDS=0)
endif()
list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_CU_MAX_THREADS_PER_BLOCK=${ONIKA_CU_MAX_THREADS_PER_BLOCK} ONIKA_CU_MIN_BLOCKS_PER_SM=${ONIKA_CU_MIN_BLOCKS_PER_SM} ONIKA_TASKS_PER_CORE=${ONIKA_TASKS_PER_CORE})

# SOATL options
option(SOATL_SIZE_TYPE_32BITS "SOATL use 32bits array size (instead of 64bits)" ON)
if(SOATL_SIZE_TYPE_32BITS)
  message(STATUS "SOATL uses 32 bits size type")
  list(APPEND ONIKA_COMPILE_DEFINITIONS SOATL_SIZE_TYPE_32BITS=1)
else()
  message(STATUS "SOATL uses 64 bits size type")
endif()

# option to reset new allocations to zero
option(ONIKA_MEMORY_ZERO_ALLOC "Onika clears allocation to zero" OFF)
if(ONIKA_MEMORY_ZERO_ALLOC)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_MEMORY_ZERO_ALLOC=1)  
endif()


# ============================
# === MPI configuration ===
# ============================
if(MPI_CXX_INCLUDE_PATH AND MPI_CXX_LIBRARIES AND NOT MPI_FOUND)
  message(STATUS "MPI_CXX_INCLUDE_PATH and MPI_CXX_LIBRARIES forced, skip find_package(MPI REQUIRED)")
  set(MPI_FOUND ON)
  # for export to onika-config file
  set(ONIKA_CONFIG_IMPORT_MPI "set(MPI_CXX_INCLUDE_PATH ${MPI_CXX_INCLUDE_PATH})\nset(MPI_CXX_LIBRARIES ${MPI_CXX_LIBRARIES})\nset(MPI_FOUND ON)")
else()
  find_package(MPI REQUIRED)
  # for export to onika-config file
  set(ONIKA_CONFIG_IMPORT_MPI "find_package(MPI REQUIRED)")
endif()
#message(STATUS "MPI_CXX_INCLUDE_PATH = ${MPI_CXX_INCLUDE_PATH}")
#message(STATUS "MPI_CXX_LIBRARIES    = ${MPI_CXX_LIBRARIES}")
list(APPEND ONIKA_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})
list(APPEND ONIKA_LIBRARIES ${MPI_CXX_LIBRARIES})

# for export to onika-config file)
if(MPIEXEC_EXECUTABLE)
#  message(STATUS "MPIEXEC_EXECUTABLE = ${MPIEXEC_EXECUTABLE}")
  file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(MPIEXEC_EXECUTABLE ${MPIEXEC_EXECUTABLE})\n")
endif()
if(NOT ONIKA_MPIRUN_CMD)
  if(MPIEXEC_EXECUTABLE)
    set(ONIKA_MPIRUN_CMD ${MPIEXEC_EXECUTABLE})
  else()
    set(ONIKA_MPIRUN_CMD mpirun)
  endif()
endif()

file(APPEND ${ONIKA_CMAKE_PACKAGE} "${ONIKA_CONFIG_IMPORT_MPI}\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_MPIRUN_CMD ${ONIKA_MPIRUN_CMD})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(MPIEXEC_MAX_NUMPROCS ${MPIEXEC_MAX_NUMPROCS})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(MPIEXEC_PREFLAGS ${MPIEXEC_PREFLAGS})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(MPIEXEC_NUMCORE_FLAG ${MPIEXEC_NUMCORE_FLAG})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(MPIEXEC_NUMPROC_FLAG ${MPIEXEC_NUMPROC_FLAG})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(MPIEXEC_PREFLAGS_DBG ${MPIEXEC_PREFLAGS_DBG})\n")

# ===================================
# ====== Global parameters ==========
# ===================================
if(ONIKA_INTERNAL_UNIT_SYSTEM)
  list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_INTERNAL_UNIT_SYSTEM=${ONIKA_INTERNAL_UNIT_SYSTEM})
endif()
set(ONIKA_DEFAULT_CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/share/config")
set(ONIKA_DEFAULT_PLUGIN_DIR "${CMAKE_INSTALL_PREFIX}/plugins")
list(APPEND ONIKA_COMPILE_DEFINITIONS ONIKA_VERSION="${CMAKE_PROJECT_VERSION}" ONIKA_DEFAULT_CONFIG_DIR="${ONIKA_DEFAULT_CONFIG_DIR}" ONIKA_DEFAULT_DATA_DIRS=".:${CMAKE_INSTALL_PREFIX}/share/data" ONIKA_DEFAULT_PLUGIN_DIR="${ONIKA_DEFAULT_PLUGIN_DIR}")


# =====================================
# === third party and external libs ===
# =====================================

# tinyexpr
add_library(tinyexpr STATIC thirdparty/tinyexpr/tinyexpr.c)
target_compile_options(tinyexpr PRIVATE -fPIC)
target_include_directories(tinyexpr PUBLIC thirdparty/tinyexpr)
list(APPEND ONIKA_LIBRARIES tinyexpr)
install(TARGETS tinyexpr DESTINATION lib)
install(FILES thirdparty/tinyexpr/tinyexpr.h DESTINATION thirdparty/tinyexpr)
file(APPEND ${ONIKA_CMAKE_PACKAGE} "\n# Thirdparty library tinyexpr is redistributed and imported\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "add_library(tinyexpr STATIC IMPORTED GLOBAL)\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set_target_properties(tinyexpr PROPERTIES IMPORTED_CONFIGURATIONS Release)\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set_target_properties(tinyexpr PROPERTIES IMPORTED_LOCATION_RELEASE ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}tinyexpr${CMAKE_STATIC_LIBRARY_SUFFIX})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "target_include_directories(tinyexpr INTERFACE ${CMAKE_INSTALL_PREFIX}/thirdparty/tinyexpr)\n")

# YAML
if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARIES AND NOT yaml-cpp_FOUND)
  message(STATUS "YAML manually configured :")
  message(STATUS "\tYAML_CPP_INCLUDE_DIR = ${YAML_CPP_INCLUDE_DIR}")
  message(STATUS "\tYAML_CPP_LIBRARIES   = ${YAML_CPP_LIBRARIES}")
  file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(YAML_CPP_INCLUDE_DIR ${YAML_CPP_INCLUDE_DIR})\n")
  file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(YAML_CPP_LIBRARIES ${YAML_CPP_LIBRARIES})\n")
  file(APPEND ${ONIKA_CMAKE_PACKAGE} "list(APPEND ONIKA_INCLUDE_DIRS \${YAML_CPP_INCLUDE_DIR})\n")
else()
  find_package(yaml-cpp REQUIRED)  
  if(yaml-cpp_DIR)
    message(STATUS "yaml-cpp_DIR = ${yaml-cpp_DIR}")
    file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(yaml-cpp_DIR ${yaml-cpp_DIR})\n")
  endif()
  file(APPEND ${ONIKA_CMAKE_PACKAGE} "find_package(yaml-cpp REQUIRED)\n")    
endif()
list(APPEND ONIKA_LIBRARIES ${YAML_CPP_LIBRARIES})
file(APPEND ${ONIKA_CMAKE_PACKAGE} "list(APPEND ONIKA_LIBRARIES \${YAML_CPP_LIBRARIES})\n")


# =====================================
# === Onika lib build configuration ===
# =====================================

# finally, add overall invariant compile options and dependent libraries
list(APPEND ONIKA_LIBRARIES dl)
list(APPEND ONIKA_COMPILE_OPTIONS -Wall)

# compile main lib
file(GLOB_RECURSE onika_CXX_SRCS src/*.cpp)
file(GLOB_RECURSE onika_CU_SRCS src/*.cu)
if(NOT ONIKA_BUILD_CUDA)
    set_source_files_properties(${onika_CU_SRCS} PROPERTIES LANGUAGE CXX)
endif()
set(onika_SRCS ${onika_CXX_SRCS} ${onika_CU_SRCS})
add_library(onika SHARED ${onika_SRCS})
target_include_directories(onika PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${ONIKA_INCLUDE_DIRS})
target_compile_definitions(onika PUBLIC ${ONIKA_COMPILE_DEFINITIONS})
target_compile_options(onika PUBLIC ${ONIKA_COMPILE_OPTIONS} $<$<COMPILE_LANGUAGE:CUDA>:${ONIKA_CUDA_COMPILE_FLAGS}> $<$<COMPILE_LANGUAGE:HIP>:${ONIKA_HIP_COMPILE_FLAGS}>)
target_compile_features(onika PUBLIC ${ONIKA_COMPILE_FEATURES})
target_link_directories(onika PUBLIC ${ONIKA_LIBRARY_DIRS})
target_link_libraries(onika PUBLIC ${ONIKA_LIBRARIES})
set_target_properties(onika PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
install(TARGETS onika DESTINATION lib)

# finalize cmake package file and install it
file(APPEND ${ONIKA_CMAKE_PACKAGE} "\n# Setup user build directory to hold plugin shared libraries in '' directory and built plugins in ''\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "file(MAKE_DIRECTORY \${PROJECT_BINARY_DIR}/lib)\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "file(MAKE_DIRECTORY \${PROJECT_BINARY_DIR}/plugins)\n")

file(APPEND ${ONIKA_CMAKE_PACKAGE} "\n# Configure and import pre-compiled onika library with all its public properties\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_COMPILE_OPTIONS ${ONIKA_COMPILE_OPTIONS})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_COMPILE_FEATURES ${ONIKA_COMPILE_FEATURES})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_COMPILE_DEFINITIONS ${ONIKA_COMPILE_DEFINITIONS})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include ${ONIKA_INCLUDE_DIRS})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_LIBRARY_DIRS ${CMAKE_INSTALL_PREFIX}/lib ${ONIKA_LIBRARY_DIRS})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_LINKED_LIBRARIES ${ONIKA_LIBRARIES})\n")

file(APPEND ${ONIKA_CMAKE_PACKAGE} "add_library(onika SHARED IMPORTED GLOBAL)\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set_target_properties(onika PROPERTIES IMPORTED_CONFIGURATIONS Release)\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set_target_properties(onika PROPERTIES IMPORTED_LOCATION_RELEASE ${CMAKE_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}onika${CMAKE_SHARED_LIBRARY_SUFFIX})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "target_compile_options(onika INTERFACE \${ONIKA_COMPILE_OPTIONS})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "target_compile_features(onika INTERFACE \${ONIKA_COMPILE_FEATURES})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "target_compile_definitions(onika INTERFACE \${ONIKA_COMPILE_DEFINITIONS})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "target_include_directories(onika INTERFACE \${ONIKA_INCLUDE_DIRS})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "target_link_directories(onika INTERFACE \${ONIKA_LIBRARY_DIRS})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "target_link_libraries(onika INTERFACE \${ONIKA_LINKED_LIBRARIES})\n")

file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_LIBRARIES onika)\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_BUILD_CUDA ${ONIKA_BUILD_CUDA})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_ENABLE_HIP ${ONIKA_ENABLE_HIP})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(ONIKA_RUN ${CMAKE_INSTALL_PREFIX}/bin/onikarun)\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(onika_DIR ${CMAKE_INSTALL_PREFIX})\n")
file(APPEND ${ONIKA_CMAKE_PACKAGE} "set(onika_FOUND TRUE)\n")

install(FILES ${ONIKA_CMAKE_PACKAGE} DESTINATION ${CMAKE_INSTALL_PREFIX})

# generate helper script to setup environment from installed location
set(ONIKA_SETUP_ENV_COMMANDS "${ONIKA_SETUP_ENV_COMMANDS}" CACHE STRING "Environment setup commands")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh
  "# Onika environment setup, as it was set during installation.\n"
  "# Please run this script before compiling your plugins or running your app.\n"
  "${ONIKA_SETUP_ENV_COMMANDS}\n"
  "[ \$LD_LIBRARY_PATH ] && LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:\$LD_LIBRARY_PATH\n"
  "[ \$LD_LIBRARY_PATH ] || LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib\n"
  "export LD_LIBRARY_PATH\n")
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh DESTINATION bin)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/bin/onikarun
  "source ${CMAKE_INSTALL_PREFIX}/bin/setup-env.sh\n"
  "[ \$ONIKA_CONFIG_PATH ] || ONIKA_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/data/config\n"
  "[ \$ONIKA_PLUGIN_PATH ] || ONIKA_PLUGIN_PATH=${CMAKE_INSTALL_PREFIX}/plugins\n"
  "export ONIKA_CONFIG_PATH\n"
  "export ONIKA_PLUGIN_PATH\n"
  "${CMAKE_INSTALL_PREFIX}/bin/onika-exec \$*\n")
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bin/onikarun DESTINATION bin)

# generate helper script to setup environment to allow running from build directory without install
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/onikarun
  "${ONIKA_SETUP_ENV_COMMANDS}\n"
  "[ \$LD_LIBRARY_PATH ] && LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib:\$LD_LIBRARY_PATH\n"
  "[ \$LD_LIBRARY_PATH ] || LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib\n"
  "[ \$ONIKA_CONFIG_PATH ] || ONIKA_CONFIG_PATH=${CMAKE_CURRENT_SOURCE_DIR}/data/config\n"
  "[ \$ONIKA_PLUGIN_PATH ] || ONIKA_PLUGIN_PATH=${PROJECT_BINARY_DIR}/plugins\n"
  "export LD_LIBRARY_PATH\n"
  "export ONIKA_CONFIG_PATH\n"
  "export ONIKA_PLUGIN_PATH\n"
  "${PROJECT_BINARY_DIR}/bin/onika-exec \$*\n")
file(CHMOD ${CMAKE_CURRENT_BINARY_DIR}/onikarun FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)

# compile tools and apps
file(GLOB app_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/apps ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
foreach(APPSRC ${app_sources})
  get_filename_component(APPNAME ${APPSRC} NAME_WLE)
  message(STATUS "Found application ${APPNAME}")
  add_executable(${APPNAME} apps/${APPNAME}.cpp)
  target_link_libraries(${APPNAME} PRIVATE onika)
  set_target_properties(${APPNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
  install(TARGETS ${APPNAME} DESTINATION bin)
endforeach()

# compile contributed plugins
add_subdirectory(plugins)

# install other files and directories directly copied from sources
install(DIRECTORY cmake DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY include/onika DESTINATION include)
install(DIRECTORY data DESTINATION ${CMAKE_INSTALL_PREFIX})

# ==================
# === test suite ===
# ==================
option(ONIKA_ENABLE_TESTS "ONIKA enable tests" ON)
if(ONIKA_ENABLE_TESTS)

  file(GLOB test_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
  foreach(TESTSRC ${test_sources})
    get_filename_component(TESTNAME ${TESTSRC} NAME_WLE)
    message(STATUS "Found test ${TESTNAME}")
    add_executable(${TESTNAME} tests/${TESTNAME}.cpp)
    target_link_libraries(${TESTNAME} PRIVATE onika)
  endforeach()

  enable_testing()

  ######### OpenMP tests ##################
  foreach(tn 1;2;3;4;5;6;7;8;9;10;11)
    #message(STATUS "omp_tasks test #${tn}")
    onika_add_test(omp_tasks_${tn} ${CMAKE_CURRENT_BINARY_DIR}/omp_tasks ${tn})
  endforeach()



  ############ SOATL tests ##################
#  onika_add_test(soatl_fa_pfa_aliasing ${CMAKE_CURRENT_BINARY_DIR}/soatl_fa_pfa_aliasing)
  onika_add_test(soatl_test1 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 0)
  onika_add_test(soatl_test2 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 34523452)
  onika_add_test(soatl_test3 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 1976)
  onika_add_test(soatl_test4 ${CMAKE_CURRENT_BINARY_DIR}/soatltest 1000 234234234)
#  onika_add_test(soatl_compute1 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 0)
#  onika_add_test(soatl_compute2 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 34523452)
#  onika_add_test(soatl_compute3 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 1976)
#  onika_add_test(soatl_compute4 ${CMAKE_CURRENT_BINARY_DIR}/soatlcomputetest 1000 234234234)
  onika_add_test(soatl_serialize ${CMAKE_CURRENT_BINARY_DIR}/soatlserializetest 10000)
  onika_add_test(soatl_tuple ${CMAKE_CURRENT_BINARY_DIR}/soatupletest 10)

  option(SOATL_ENABLE_BENCHMARKS "SOATL benchmarks" ON)
  if(SOATL_ENABLE_BENCHMARKS)
    set(SOATL_BENCHMARKS_SIZE 10000000 CACHE STRING "SOATL benchmarks number of elements")

    # find objdump
    get_filename_component(BINUTILS_DIR ${CMAKE_LINKER} DIRECTORY)
    find_file(SOATL_OBJDUMP objdump HINTS ${BINUTILS_DIR})

    # create a vecreport target
    if(SOATL_OBJDUMP)
      add_custom_target(vecreport)
    endif()

    GenerateBenchmark(64 16 d ON OFF)
    GenerateBenchmark(64 16 d ON ON)
    GenerateBenchmark(64 16 f ON OFF)
    GenerateBenchmark(64 16 f ON ON)
    GenerateBenchmark(32 8 d ON OFF)
    GenerateBenchmark(32 8 f ON OFF)
    GenerateBenchmark(16 4 d ON OFF)
    GenerateBenchmark(16 4 f ON OFF)
    GenerateBenchmark(1 1 d OFF OFF)
    GenerateBenchmark(1 1 d OFF ON)
    GenerateBenchmark(1 1 f OFF OFF)
    GenerateBenchmark(1 1 f OFF ON)
  endif()



  ###### xs data move algorithm tests ########
  set(NUMPROCS_LIST 1 7 8)
  set(IDROTATION_LIST 0 1 77)
  set(IDSTART_LIST 0 1 1023)
  set(IDCOUNT_LIST 10 110 100000)
  set(NPERMS_LIST 10 1000)
  set(SEED_LIST 0 1976)

  foreach(NUMPROCS ${NUMPROCS_LIST})
      foreach(IDROTATION ${IDROTATION_LIST})
        foreach(IDSTART ${IDSTART_LIST})
          foreach(IDCOUNT ${IDCOUNT_LIST})
            math(EXPR IDEND ${IDSTART}+${IDCOUNT})
            onika_add_test_par(xsdatamove_simple_np${NUMPROCS}_rot${IDROTATION}_st${IDSTART}_ic${IDCOUNT} ${NUMPROCS} 1 ${CMAKE_CURRENT_BINARY_DIR}/xs_data_move_simple ${MPIEXEC_POSTFLAGS} ${IDSTART} ${IDEND} ${IDROTATION})
          endforeach()
        endforeach()
      endforeach()
  endforeach()

  onika_add_test_par(xsdatamove_simple_adhoc1 3 1 ${CMAKE_CURRENT_BINARY_DIR}/xs_data_move_simple ${MPIEXEC_POSTFLAGS} 0 20 3)
  onika_add_test_par(xsdatamove_simple_adhoc2 3 1 ${CMAKE_CURRENT_BINARY_DIR}/xs_data_move_simple ${MPIEXEC_POSTFLAGS} 0 107 3)

  foreach(NUMPROCS ${NUMPROCS_LIST})
    foreach(SEED ${SEED_LIST})
      foreach(IDCOUNT ${IDCOUNT_LIST})
        foreach(NPERMS ${NPERMS_LIST})
          onika_add_test_par(xsdatamove_random_s${SEED}_ic${IDCOUNT}_permm${NPERMS}_np${NUMPROCS} ${NUMPROCS} 1 ${CMAKE_CURRENT_BINARY_DIR}/xs_data_move_random ${MPIEXEC_POSTFLAGS} ${SEED} ${IDCOUNT} ${NPERMS})
        endforeach()
      endforeach()
    endforeach()
  endforeach()

endif()


