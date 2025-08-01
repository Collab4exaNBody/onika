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
#include <cstdint>
#include <cstdlib>

namespace onika
{
  
  // Cuda compatible std::array replacement
  template<class T, size_t N>
  struct oarray_t
  {
    static inline constexpr size_t array_size = N;
    using value_type = T;
    T m_data[N] = {};
    
    ONIKA_HOST_DEVICE_FUNC inline T& operator [] (size_t i) { return m_data[i]; }
    ONIKA_HOST_DEVICE_FUNC inline const T& operator [] (size_t i) const { return m_data[i]; }
    ONIKA_HOST_DEVICE_FUNC inline T * data() { return m_data; }
    ONIKA_HOST_DEVICE_FUNC inline const T * data() const { return m_data; }
    static inline constexpr size_t size() { return N; }

    ONIKA_HOST_DEVICE_FUNC inline const T* begin() const { return m_data; }
    ONIKA_HOST_DEVICE_FUNC inline T* begin() { return m_data; }
    ONIKA_HOST_DEVICE_FUNC inline const T* end() const { return m_data+N; }
    ONIKA_HOST_DEVICE_FUNC inline T* end() { return m_data+N; }
  };

  // Cuda compatible std::inplace_vector replacement
  template<class T,uint32_t MaxSize>
  struct inplace_vector
  {
    static inline constexpr uint32_t max_size = MaxSize;
    using value_type = T;
    T m_data[max_size] = {};
    uint32_t m_size = 0;
    
    inplace_vector() = default;
    inplace_vector(const inplace_vector&) = default;
    inplace_vector(inplace_vector &&) = default;
    
    inline inplace_vector(std::initializer_list<T> l)
    {
      std::copy(l.begin(),l.end(),data());
      m_size = l.size();
    }
    
    inplace_vector & operator = (const inplace_vector &) = default;
    inplace_vector & operator = (inplace_vector &&) = default;
    
    ONIKA_HOST_DEVICE_FUNC inline T& operator [] (size_t i) { return m_data[i]; }
    ONIKA_HOST_DEVICE_FUNC inline const T& operator [] (size_t i) const { return m_data[i]; }
    ONIKA_HOST_DEVICE_FUNC inline T * data() { return m_data; }
    ONIKA_HOST_DEVICE_FUNC inline const T * data() const { return m_data; }
    
    ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return m_size; }
    ONIKA_HOST_DEVICE_FUNC inline bool empty() const { return m_size==0; }

    ONIKA_HOST_DEVICE_FUNC inline void push_back(const T & val) { m_data[m_size++] = val; }
    ONIKA_HOST_DEVICE_FUNC inline void emplace_back(T && val) { m_data[m_size++] = std::move(val); }
    ONIKA_HOST_DEVICE_FUNC inline void pop_back() { --m_size; }
    ONIKA_HOST_DEVICE_FUNC inline void clear() { m_size = 0; }

    ONIKA_HOST_DEVICE_FUNC inline const T* begin() const { return m_data; }
    ONIKA_HOST_DEVICE_FUNC inline T* begin() { return m_data; }
    ONIKA_HOST_DEVICE_FUNC inline const T* end() const { return m_data+m_size; }
    ONIKA_HOST_DEVICE_FUNC inline T* end() { return m_data+m_size; }
  };

}
