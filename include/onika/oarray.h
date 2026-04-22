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
#include <compare>

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

    template<class U>
    ONIKA_HOST_DEVICE_FUNC
    inline std::strong_ordering operator <=> (const oarray_t<U,N>& rhs) const requires std::totally_ordered_with<T,U> 
    {
      for(auto i=0;i<N;i++)
      {
        if( m_data[i] < rhs[i] ) return std::strong_ordering::less;
        else if( m_data[i] > rhs[i] ) return std::strong_ordering::greater;
      }
      return std::strong_ordering::equal;
    }

    template<class U>
    ONIKA_HOST_DEVICE_FUNC
    inline bool operator == (const oarray_t<U,N>& rhs) const requires std::equality_comparable_with<T,U>
    {
      for(auto i=0;i<N;i++)
      {
        if( ! ( m_data[i] == rhs[i] ) ) return false;
      }
      return true;
    }

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

  template<class T> struct IsOArray : public std::false_type {};
  template<class T, size_t N> struct IsOArray< oarray_t<T,N> > : public std::true_type {};
  template<class T> static inline constexpr bool is_oarray_v = IsOArray<T>::value;
  template<class T> static inline constexpr bool is_integral_oarray_v = IsOArray<T>::value;
  template<class T> concept some_oarray_t = is_oarray_v<T>;
  template<class T> concept some_integral_oarray_t = requires(T a) { { std::integral_constant< bool , is_oarray_v<T> && std::is_integral_v<typename T::value_type> >{} } -> std::same_as<std::true_type>; };  
}

namespace std
{
  template<onika::some_integral_oarray_t ArrayT> struct hash< ArrayT >
  {
    inline size_t operator () ( const ArrayT & a ) const
    {
       return std::hash<std::string_view>{}( std::string_view( (const char*) a.m_data , ArrayT::array_size * sizeof(typename ArrayT::value_type) ) );
    }
  };
}
