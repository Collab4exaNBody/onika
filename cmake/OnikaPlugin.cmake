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

function(onika_add_plugin PluginName dirname)

  # allow optional building for every individual plugin
  if(NOT DEFINED ONIKA_BUILD_${PluginName}_DEFAULT)
    set(ONIKA_BUILD_${PluginName}_DEFAULT ON)
  endif()
  option(ONIKA_BUILD_${PluginName} "Build ${PluginName}" ${ONIKA_BUILD_${PluginName}_DEFAULT})

  if(ONIKA_BUILD_${PluginName})
    file(GLOB ${PluginName}_SCRS_CXX ${dirname}/*.cpp)
    file(GLOB ${PluginName}_SCRS_CU ${dirname}/*.cu)
    set(${PluginName}_SCRS ${${PluginName}_SCRS_CXX} ${${PluginName}_SCRS_CU})

    if(IS_DIRECTORY ${dirname}/lib)
      file(GLOB ${PluginName}_SCRS_LIB_CXX ${dirname}/lib/*.cpp)
      file(GLOB ${PluginName}_SCRS_LIB_CU ${dirname}/lib/*.cu)
      set(${PluginName}_SCRS_LIB ${${PluginName}_SCRS_LIB_CXX} ${${PluginName}_SCRS_LIB_CU})
    endif()

    if(NOT ONIKA_BUILD_CUDA)
      set_source_files_properties(${${PluginName}_SCRS_CU} PROPERTIES LANGUAGE CXX)
      set_source_files_properties(${${PluginName}_SCRS_LIB_CU} PROPERTIES LANGUAGE CXX)
    elseif(ONIKA_USE_HIP)
      set_source_files_properties(${${PluginName}_SCRS_CU} PROPERTIES LANGUAGE HIP)
      set_source_files_properties(${${PluginName}_SCRS_LIB_CU} PROPERTIES LANGUAGE HIP)
    endif()
    
    list(APPEND ${PluginName}_SCRS  ${${PluginName}_EXTERNAL_SRCS})
    include(${dirname}/${PluginName}.cmake OPTIONAL RESULT_VARIABLE ${PluginName}_CUSTOM_CMAKE)

    set(${PluginName}_SHARED_LIB ${PluginName})
    if(${PluginName}_SCRS_LIB)
#      message(STATUS "Plugin ${PluginName} contains a lib : ${${PluginName}_SCRS_LIB}")
      add_library(${${PluginName}_SHARED_LIB} SHARED ${${PluginName}_SCRS_LIB})
      target_include_directories(${${PluginName}_SHARED_LIB} PRIVATE ${dirname}/lib ${ONIKA_INCLUDE_DIRS} PUBLIC ${${PluginName}_INCLUDE_DIRS} ${dirname}/include)
      target_compile_definitions(${${PluginName}_SHARED_LIB} PRIVATE ${ONIKA_COMPILE_DEFINITIONS} PUBLIC ${${PluginName}_COMPILE_DEFINITIONS})
      target_compile_options(${${PluginName}_SHARED_LIB} PRIVATE ${ONIKA_COMPILE_OPTIONS} $<$<COMPILE_LANGUAGE:CUDA>:${ONIKA_CUDA_COMPILE_FLAGS}> $<$<COMPILE_LANGUAGE:HIP>:${ONIKA_HIP_COMPILE_FLAGS}> ${${PluginName}_COMPILE_OPTIONS})
      target_compile_features(${${PluginName}_SHARED_LIB} PRIVATE ${ONIKA_COMPILE_FEATURES} ${${PluginName}_COMPILE_FEATURES})
      target_link_directories(${${PluginName}_SHARED_LIB} PUBLIC ${ONIKA_LIBRARY_DIRS} ${${PluginName}_LINK_DIRECTORIES})
      target_link_libraries(${${PluginName}_SHARED_LIB} ${ONIKA_LIBRARIES} ${${PluginName}_LINK_LIBRARIES})
      install(TARGETS ${${PluginName}_SHARED_LIB} DESTINATION lib)
    elseif(EXISTS ${dirname}/include)
#      message(STATUS "plugin ${PluginName} has no lib sources but has includes, create interface lib")
      add_library(${${PluginName}_SHARED_LIB} INTERFACE)
      target_include_directories(${${PluginName}_SHARED_LIB} INTERFACE ${dirname}/include)
      target_compile_definitions(${${PluginName}_SHARED_LIB} INTERFACE ${${PluginName}_COMPILE_DEFINITIONS})
      target_compile_features(${${PluginName}_SHARED_LIB} INTERFACE ${${PluginName}_COMPILE_FEATURES})
    else()
      unset(${PluginName}_SHARED_LIB)
    endif()
        
#    message(STATUS "Plugin ${PluginName} :\n\tsources : ${${PluginName}_SCRS}\n\tdevice sources ${${PluginName}_SCRS_CU}")
    if(${PluginName}_SCRS)
      set(${PluginName}_PLUGIN_LIB ${PluginName}Plugin)
      add_library(${${PluginName}_PLUGIN_LIB} SHARED ${${PluginName}_SCRS})
      target_include_directories(${${PluginName}_PLUGIN_LIB} PRIVATE ${dirname} ${dirname}/include ${ONIKA_INCLUDE_DIRS} ${${PluginName}_INCLUDE_DIRS}) # ${CUDA_SAMPLES_INCLUDE_DIR}
      target_compile_definitions(${${PluginName}_PLUGIN_LIB} PRIVATE ${ONIKA_COMPILE_DEFINITIONS} ${${PluginName}_COMPILE_DEFINITIONS})        
    	target_compile_options(${${PluginName}_PLUGIN_LIB} PRIVATE ${ONIKA_COMPILE_OPTIONS} $<$<COMPILE_LANGUAGE:CUDA>:${ONIKA_CUDA_COMPILE_FLAGS}> $<$<COMPILE_LANGUAGE:HIP>:${ONIKA_HIP_COMPILE_FLAGS}> ${${PluginName}_COMPILE_OPTIONS})
      target_compile_features(${${PluginName}_PLUGIN_LIB} PRIVATE ${ONIKA_COMPILE_FEATURES} ${${PluginName}_COMPILE_FEATURES})
      target_link_directories(${${PluginName}_PLUGIN_LIB} PRIVATE ${ONIKA_LIBRARY_DIRS} ${${PluginName}_LINK_DIRECTORIES})
      target_link_libraries(${${PluginName}_PLUGIN_LIB} PRIVATE ${ONIKA_LIBRARIES} ${${PluginName}_LINK_LIBRARIES} ${${PluginName}_SHARED_LIB})
      install(TARGETS ${${PluginName}_PLUGIN_LIB} DESTINATION plugins)
    else()
      unset(${PluginName}_PLUGIN_LIB)
    endif()
    
    if(EXISTS ${dirname}/regression)
      message(STATUS "Plugin ${PluginName} has regression tests")
      # AddRegressionTestDir(${dirname}/regression)
    endif()

    if(EXISTS ${dirname}/tests)
      message(STATUS "Plugin ${PluginName} has compiled tests")
      add_subdirectory(${dirname}/tests)
    endif()

  endif()
endfunction()

