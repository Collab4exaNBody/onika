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
#include <onika/cuda/span.h>

namespace onika
{

  namespace cuda
  {

    template<class T, class U>
    struct pair
    {
      T first;
      U second;
      ONIKA_HOST_DEVICE_FUNC inline bool operator != ( const pair& other ) const { return first!=other.first || second!=other.second; }
      ONIKA_HOST_DEVICE_FUNC inline bool operator == ( const pair& other ) const { return first==other.first || second==other.second; }
      ONIKA_HOST_DEVICE_FUNC inline bool operator < ( const pair& other ) const { return first<other.first || ( first==other.first && second<other.second ); }
      ONIKA_HOST_DEVICE_FUNC inline bool operator >= ( const pair& other ) const { return first>other.first || ( first==other.first && second>=other.second ); }
    };

    ONIKA_HOST_DEVICE_FUNC inline void memmove( void* dst , const void* src , unsigned int n )
    {
      static_assert( sizeof(uint64_t) == 8 && sizeof(uint8_t) == 1 , "Wrong standard type sizes" );
      if( dst < src )
      {
        volatile uint8_t* dst1 = (uint8_t*) dst;
        const volatile uint8_t* src1 = (const uint8_t*) src;

        volatile uint64_t* dst8 = (uint64_t*) dst;
        const volatile uint64_t* src8 = (const uint64_t*) src;
        unsigned int n8 = n / 8;
        
        unsigned int i;
        for(i=0;i<n8;i++) dst8[i] = src8[i];
        i *= 8;
        for(;i<n;i++) dst1[i] = src1[i];
      }
      else if( dst > src )
      {
        // not implemented yet
        ONIKA_CU_ABORT();
      }
      // else { /* nothing to do */ }
    }

    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline const T * vector_data( const onika::memory::CudaMMVector<T> & v ) { return v.data(); }
    
    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline T * vector_data( onika::memory::CudaMMVector<T> & v ) { return v.data(); }

    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline T * vector_data( onika::cuda::span<T> v ) { return v.data(); }
    
    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline size_t vector_size( const onika::memory::CudaMMVector<T> & v ) { return v.size(); }

    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline size_t vector_size( onika::cuda::span<T> v ) { return v.size(); }

    template<class Iterator, class T>
    ONIKA_HOST_DEVICE_FUNC inline Iterator lower_bound( Iterator begin , Iterator end , const T& x )
    {
      while( begin != end )
      {
        if( ! ( (*begin) < x ) ) return begin;
        ++begin;
      }
      return end;
    }
    
    struct PrintfBaseStdOutStream
    {
      ONIKA_HOST_DEVICE_FUNC inline PrintfBaseStdOutStream operator << ( const char* s ) const { printf("%s",s); return {}; }
      ONIKA_HOST_DEVICE_FUNC inline PrintfBaseStdOutStream operator << ( uint64_t i ) const { printf("%llu",static_cast<unsigned long long>(i)); return {}; }
      ONIKA_HOST_DEVICE_FUNC inline PrintfBaseStdOutStream operator << ( int64_t i ) const { printf("%lld",static_cast<long long>(i)); return {}; }
      ONIKA_HOST_DEVICE_FUNC inline PrintfBaseStdOutStream operator << ( uint32_t i ) const { printf("%llu",static_cast<unsigned long long>(i)); return {}; }
      ONIKA_HOST_DEVICE_FUNC inline PrintfBaseStdOutStream operator << ( int32_t i ) const { printf("%lld",static_cast<long long>(i)); return {}; }
      ONIKA_HOST_DEVICE_FUNC inline PrintfBaseStdOutStream operator << ( uint16_t i ) const { printf("%llu",static_cast<unsigned long long>(i)); return {}; }
      ONIKA_HOST_DEVICE_FUNC inline PrintfBaseStdOutStream operator << ( int16_t i ) const { printf("%lld",static_cast<long long>(i)); return {}; }
      ONIKA_HOST_DEVICE_FUNC inline PrintfBaseStdOutStream operator << ( double d ) const { printf("% .9e",d); return {}; }
      ONIKA_HOST_DEVICE_FUNC inline PrintfBaseStdOutStream operator << ( float d ) const { printf("% .9e",double(d)); return {}; }
      ONIKA_HOST_DEVICE_FUNC static inline void flush() {}
    };

    static inline constexpr PrintfBaseStdOutStream cout = {};
  }

}

