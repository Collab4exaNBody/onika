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

#include <onika/cuda/cuda.h>

namespace onika
{
  namespace cuda
  {
    /* helper place holders to help declare variables in shared memory,
     * for which initialization at declaration is forbidden
     */
    template<class T>
    struct UnitializedPlaceHolder
    {
      static_assert( sizeof(unsigned char) == 1 , "expected char to be 1 byte" );

      using value_type = T;
      alignas( alignof(value_type) ) unsigned char m_bytes[ sizeof(value_type) ];
            
      ONIKA_HOST_DEVICE_FUNC inline       value_type& get_ref()       { return * reinterpret_cast<value_type*>(m_bytes); }
      ONIKA_HOST_DEVICE_FUNC inline const value_type& get_ref() const { return * reinterpret_cast<value_type*>(m_bytes); }      
    };

    template<class T, size_t N>
    struct UnitializedArrayPlaceHolder
    {
      static_assert( sizeof(unsigned char) == 1 , "expected char to be 1 byte" );

      using value_type = T;
      static inline constexpr size_t ArraySize = N;
      alignas( alignof(value_type) ) unsigned char m_bytes[ sizeof(value_type) * ArraySize ];
      
      ONIKA_HOST_DEVICE_FUNC inline       value_type* get_array()       { return reinterpret_cast<value_type*>(m_bytes); }
      ONIKA_HOST_DEVICE_FUNC inline const value_type* get_array() const { return reinterpret_cast<const value_type*>(m_bytes); }

      ONIKA_HOST_DEVICE_FUNC inline operator       value_type* ()       { return get_array(); }
      ONIKA_HOST_DEVICE_FUNC inline operator const value_type* () const { return get_array(); }

      ONIKA_HOST_DEVICE_FUNC inline       value_type& operator [] (size_t i)       { return get_array()[i]; }
      ONIKA_HOST_DEVICE_FUNC inline const value_type& operator [] (size_t i) const { return get_array()[i]; }
    };

  }
}
