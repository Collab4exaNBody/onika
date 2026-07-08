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

#include <onika/type_features.h>
#include <onika/memory/mm_vector.h>

#include <onika/soatl/field_id.h>
#include <onika/soatl/field_arrays.h>
#include "declare_fields.h"

#include <iostream>

#include <random>


std::default_random_engine rng;
std::uniform_int_distribution<size_t> size_random_distribution(0,1000);
size_t random_size() { return size_random_distribution(rng); }


struct TrivialAggregateWithInitializedMembers
{
  double * m_data = nullptr;
  size_t m_size = 0;
  unsigned long m_flags = 0x0F;
};

struct DefaultImplIsGPUCompatible
{
  double * m_data = nullptr;
  size_t m_size = 0;
  unsigned long m_flags = 0x0F;
  
  DefaultImplIsGPUCompatible() = default;
  DefaultImplIsGPUCompatible(DefaultImplIsGPUCompatible &&) = default;
  DefaultImplIsGPUCompatible(const DefaultImplIsGPUCompatible &) = default;

  DefaultImplIsGPUCompatible& operator = (DefaultImplIsGPUCompatible &&) = default;
  DefaultImplIsGPUCompatible& operator = (const DefaultImplIsGPUCompatible &) = default;
};

template<bool TypeFeature=false>
struct NotGPUCopyable
{
  double * m_data = nullptr;
  size_t m_size = 0;
  unsigned long m_flags = 0x0F;
  
  NotGPUCopyable() = default;
  NotGPUCopyable(NotGPUCopyable &&) = default;

  NotGPUCopyable& operator = (NotGPUCopyable &&) = default;
  NotGPUCopyable& operator = (const NotGPUCopyable & other) { m_data=other.m_data; m_size=other.m_size; m_flags=other.m_flags; return *this; }
  
  inline NotGPUCopyable(const NotGPUCopyable& other)
    : m_data(other.m_data)
    , m_size(other.m_size)
    , m_flags(other.m_flags)
    {}
};

namespace onika
{
  template<>
  struct supported_features< NotGPUCopyable<true> >
  {
    static inline constexpr bool gpu_default_construct = true;
    static inline constexpr bool gpu_non_default_construct = false; // if this is true and T has a copy constructor, then gpu_copy_construct must be true
    static inline constexpr bool gpu_copy_construct = false;
    static inline constexpr bool gpu_destruct = true;
    static inline constexpr bool gpu_copy_assign = false;
    static inline constexpr bool gpu_move_construct = true;
    static inline constexpr bool gpu_move_assign = true;
  };
}

using NotGPUCopyableNoTypeFeature = NotGPUCopyable<false>;
using NotGPUCopyableTypeFeature = NotGPUCopyable<true>;

template<class T>
void test_mm_vector( onika::memory::CudaMMVector<T> & vec1 , const onika::memory::CudaMMVector<T> & vec2 )
{
  for(int i=0;i<10;i++)
  {
    vec1.resize( random_size() );
    if( random_size() < 300 ) vec1.clear();
    vec1.assign( random_size() , T{} );
    vec1 = vec2;
  }
}

template<class T>
void test_mm_vector_nocopy( onika::memory::CudaMMVector<T> & vec1 , onika::memory::CudaMMVector<T> && vec2 )
{
  for(int i=0;i<10;i++)
  {
    vec2.resize( random_size() );
    vec1.resize( random_size() );
    if( random_size() < 300 ) vec1.clear();
    vec1.assign(random_size() /* ,T{} */ ); // uses only default constructor and not copy constructor
    vec1 = std::move(vec2);
  }
}

int main(int argc, char* argv[])
{ 
  long seed = 26101976;
  if(argc>1) seed = std::atol( argv[1] );
  rng.seed( seed );
  
  {
    onika::memory::CudaMMVector<TrivialAggregateWithInitializedMembers> vec1;
    onika::memory::CudaMMVector<TrivialAggregateWithInitializedMembers> vec2(5);
    test_mm_vector(vec1,vec2);
  }

  {
    onika::memory::CudaMMVector<DefaultImplIsGPUCompatible> vec1;
    onika::memory::CudaMMVector<DefaultImplIsGPUCompatible> vec2(5);
    test_mm_vector(vec1,vec2);
  }

  {
   onika::memory::CudaMMVector<NotGPUCopyableNoTypeFeature> vec1;
   onika::memory::CudaMMVector<NotGPUCopyableNoTypeFeature> vec2(5);
   test_mm_vector_nocopy(vec1, std::move(vec2) );
  }

  {
   onika::memory::CudaMMVector<NotGPUCopyableTypeFeature> vec1;
   onika::memory::CudaMMVector<NotGPUCopyableTypeFeature> vec2(5);
   test_mm_vector(vec1, vec2 );
  }
  
  {
    static constexpr size_t ALIGN = 64;
    static constexpr size_t CHUNK = 8;
    using AllocatorT = onika::soatl::PackedFieldArraysAllocatorImpl< onika::memory::DefaultAllocator, ALIGN, CHUNK, field::_rx,field::_ry,field::_rz, particle_field_ids... > ;
    using FArraysT = onika::soatl::FieldArraysWithAllocator< ALIGN, CHUNK, AllocatorT, StoredPointerCount , __particle_rx, __particle_ry, __particle_rz, __particle_e, __particle_atype >;
  }
  
  return 0;
}

