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
#include <iostream>

struct TrivialAggregateWithInitializedMembers
{
  double * m_data = nullptr;
  size_t m_size = 0;
  uint64_t m_flags = 0x0F;
};


struct DefaultImplIsGPUCompatible
{
  double * m_data = nullptr;
  size_t m_size = 0;
  uint64_t m_flags = 0x0F;
  
  DefaultImplIsGPUCompatible() = default;
  DefaultImplIsGPUCompatible(DefaultImplIsGPUCompatible &&) = default;
  DefaultImplIsGPUCompatible(const DefaultImplIsGPUCompatible &) = default;

  DefaultImplIsGPUCompatible& operator = (DefaultImplIsGPUCompatible &&) = default;
  DefaultImplIsGPUCompatible& operator = (const DefaultImplIsGPUCompatible &) = default;
};

struct NotGPUCopyable
{
  double * m_data = nullptr;
  size_t m_size = 0;
  uint64_t m_flags = 0x0F;
  
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

template<class T>
void test_mm_vector( onika::memory::CudaMMVector<T> & vec1 , const onika::memory::CudaMMVector<T> & vec2 )
{
  vec1.resize(10);
  vec1.clear();
  vec1.assign(15, T{} );
  vec1 = vec2;
}

template<class T>
void test_mm_vector_nocopy( onika::memory::CudaMMVector<T> & vec1 , const onika::memory::CudaMMVector<T> & vec2 )
{
  vec1.resize(10);
  vec1.clear();
  vec1.assign(15 /* ,T{} */ ); // uses only default constructor and not copy constructor
  vec1 = std::move(vec2);
}

int main(int argc, char* argv[])
{  
   std::cout << "NotGPUCopyable supports GPU copy = "<< ( onika::supported_features<NotGPUCopyable>::gpu_copy_construct && onika::supported_features<NotGPUCopyable>::gpu_copy_assign ) << std::endl;

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
   onika::memory::CudaMMVector<NotGPUCopyable> vec1;
   onika::memory::CudaMMVector<NotGPUCopyable> vec2(5);
   test_mm_vector_nocopy(vec1,vec2);
  }
  
  return 0;
}

