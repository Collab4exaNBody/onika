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

#include <functional>
#include <string_view>
#include <cstdlib>
#include <onika/bit_rotl.h>

namespace onika
{
  template<class T> struct IsHashedValueIgnored : public std::false_type {};
  template<class T> static inline constexpr bool is_hashed_value_ignored_v = IsHashedValueIgnored<T>::value;
  template<class T> concept null_hashed_type = is_hashed_value_ignored_v<T>;
}

namespace std
{
  template<onika::null_hashed_type T> struct hash< T >
  {
    constexpr inline size_t operator () ( const T & ) const { return 0; }
  };
}

namespace onika
{
  template<class... Args>
  static inline size_t multi_hash(const Args& ... args)
  {
    size_t H = 0;
    unsigned int i = 0;
    ( ... , ( H = H ^ onika::bit_rotl(typeid(Args).hash_code(),i) ^ onika::bit_rotl(std::hash<Args>{}(args),i+1) , ++i ) );
    return H;
  }
}
