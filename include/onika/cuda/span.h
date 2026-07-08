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

#include <vector>
#include <ranges>
#include <onika/cuda/cuda.h>
#include <onika/type_utils.h>

namespace onika
{

  namespace cuda
  {
    template<class T>
    struct span
    {
      using value_type = T;
      T * m_start;
      size_t m_size;
      ONIKA_HOST_DEVICE_FUNC inline T * data() { return m_start; }
      ONIKA_HOST_DEVICE_FUNC inline const T * data() const { return m_start; }
      ONIKA_HOST_DEVICE_FUNC inline T& operator [] (size_t i) { return m_start[i]; }
      ONIKA_HOST_DEVICE_FUNC inline const T& operator [] (size_t i) const { return m_start[i]; }
      ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return m_size; }
      ONIKA_HOST_DEVICE_FUNC inline bool empty() const { return size() == 0; }
      ONIKA_HOST_DEVICE_FUNC inline auto begin() const { return m_start; }
      ONIKA_HOST_DEVICE_FUNC inline auto end() const { return m_start + m_size; }
    };
  }

  // partial specialization to accept onika::cuda::span as span in implementation specializations
  template<class T> struct is_span_t< ::onika::cuda::span<T> > : public std::true_type {};

  namespace cuda
  {
    template< std::ranges::contiguous_range T >
    inline span<const typename T::value_type> make_const_span(const T& r)
    {
      return { r.data() , r.size() };
    }

    template<class T>
    inline span<const T> make_const_span(const T* b, const T* e)
    {
      return { b , e - b };
    }

    template< std::ranges::contiguous_range T >
    inline span<typename T::value_type> make_span(T& r)
    {
      return { r.data() , r.size() };
    }

    template<class T>
    inline span<T> make_span(T* b, T* e)
    {
      return { b , e - b };
    }

  }

}

