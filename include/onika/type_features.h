/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/

#pragma once

#include <type_traits>

namespace onika
{
  template<class T>
  struct supported_features
  {
    static inline constexpr bool gpu_default_construct = true;
    static inline constexpr bool gpu_non_default_construct = true; // if this is true and T has a copy constructor, then gpu_copy_construct must be true
    static inline constexpr bool gpu_copy_construct = true;
    static inline constexpr bool gpu_destruct = true;
    static inline constexpr bool gpu_copy_assign = true;
    static inline constexpr bool gpu_move_construct = true;
    static inline constexpr bool gpu_move_assign = true;
  };
}

