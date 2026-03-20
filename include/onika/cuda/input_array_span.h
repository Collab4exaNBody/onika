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
#include <onika/cuda/uninitialized_place_holder.h>
#include <ranges>
#include <type_traits>

namespace onika
{
  namespace cuda
  {

    /* span for read only, dynamic sized, input array passed to compute kernels.
     * it may hold an internal copy of the input array if it is small enough so that
     * the kernel execution avoids latency of managed memory access through the original
     * array's pointer.
     * when the span is copied into the kernel functor, the internal array copy is also copied,
     * so if the original array fits into internal copy space, access from the kernel to the array
     * actually acces the internal array already copied during kernel launch phase rather than
     * triggering an eventual page default in managed memory.
     */
    template<class T, size_t N=1>
    struct InputArraySpan
    {
      static_assert( std::is_trivially_copyable_v<T> || std::is_copy_constructible_v<T> );
      
      using value_type = T;
      static inline constexpr size_t InternalArraySize = N;
      
      const value_type * m_data_pointer = nullptr;
      size_t m_size = 0;
      UnitializedArrayPlaceHolder<value_type,InternalArraySize> m_internal_copy;
      
      InputArraySpan() = default;
      InputArraySpan(const InputArraySpan&) = default;
      InputArraySpan(InputArraySpan&&) = default;

      InputArraySpan& operator = (InputArraySpan&&) = default;
      InputArraySpan& operator = (const InputArraySpan&) = default;
      
      template< std::ranges::contiguous_range R >
      ONIKA_HOST_DEVICE_FUNC inline explicit InputArraySpan(const R& r)
      {
        m_size = r.size();
        const auto * src = r.data();
        if( m_size <= InternalArraySize )
        {
          m_data_pointer = nullptr;
          auto * dst = m_internal_copy.get_array();
               if constexpr ( std::is_trivially_copyable_v<value_type>    ) memcpy( dst , src , sizeof(value_type) * m_size ) ;
          else if constexpr ( std::is_copy_constructible_v<value_type>    ) for(size_t i=0;i<m_size;i++) new (dst+i) value_type ( src[i] );
        }
        else
        {
          m_data_pointer = src;
        }
      }

      ONIKA_HOST_DEVICE_FUNC inline void reset()
      {
        if( m_size > 0 && m_size <= InternalArraySize )
        {
          for(size_t i=0;i<m_size;i++) m_internal_copy[i] . ~ value_type ();
          m_size = 0;
        }
      }
      
      ONIKA_HOST_DEVICE_FUNC inline ~InputArraySpan()
      {
        reset();
      }
      
      ONIKA_HOST_DEVICE_FUNC inline       value_type * data()       { return ( m_size <= InternalArraySize ) ? m_internal_copy.get_array() : m_data_pointer; }
      ONIKA_HOST_DEVICE_FUNC inline const value_type * data() const { return ( m_size <= InternalArraySize ) ? m_internal_copy.get_array() : m_data_pointer; }

      ONIKA_HOST_DEVICE_FUNC inline       value_type& operator [] (size_t i)       { return data()[i]; }
      ONIKA_HOST_DEVICE_FUNC inline const value_type& operator [] (size_t i) const { return data()[i]; }
      
      ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return m_size; }
      ONIKA_HOST_DEVICE_FUNC inline bool empty() const { return size() == 0; }
      ONIKA_HOST_DEVICE_FUNC inline auto begin() const { return data(); }
      ONIKA_HOST_DEVICE_FUNC inline auto end() const { return data() + m_size; }
    };

    template< std::ranges::contiguous_range T , size_t N=1>
    inline InputArraySpan< typename T::value_type , N > make_input_array_span( const T& r , std::integral_constant<size_t,N> = {} )
    {
      return InputArraySpan< typename T::value_type , N > ( r );
    }

  }
  
  // partial specialization to accept onika::cuda::span as span in implementation specializations
  template<class T, size_t N> struct is_span_t< ::onika::cuda::InputArraySpan<T,N> > : public std::true_type {};
  
}
