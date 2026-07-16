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

#include <onika/memory/simd.h>
#include <onika/cuda/cuda.h>
#include <yaml-cpp/yaml.h>
#include <cstdlib>
#include <vector>
#include <algorithm>

namespace onika
{
namespace memory
{
    
  struct UniquePointerPool
  {
    static inline constexpr size_t MAX_ARRAY_SIZE = 1024;
    void ** m_unique_ptrs[MAX_ARRAY_SIZE] = { nullptr, };
  };
  
  struct DelayedMemoryOperations
  {
    static inline constexpr size_t MAX_POOL_SIZE = 1024*1024;
    static inline constexpr unsigned long long LOCKED_COUNTER_VALUE = ( 1ull << 32 ) - 1;
    unsigned long long m_op_count = 0;
    UniquePointerPool * m_delayed_operations[MAX_POOL_SIZE] = { nullptr, };
  };

  /*
    Host allocation kinds
  */
  enum class HostAllocationPolicy
  {
    MALLOC    = 0x00 ,
    CUDA_HOST = 0x01
  };

# ifdef ONIKA_CUDA_VERSION
  static inline constexpr HostAllocationPolicy CUDA_FALLBACK_ALLOC_POLICY = HostAllocationPolicy::CUDA_HOST;
# else
  static inline constexpr HostAllocationPolicy CUDA_FALLBACK_ALLOC_POLICY = HostAllocationPolicy::MALLOC;
# endif

  // prefered default alignment and chunk size on this machine
# ifndef ONIKA_DEFAULT_ALIGNMENT
  static inline constexpr size_t DEFAULT_ALIGNMENT = SimdRequirements<double>::alignment;
# else
  static inline constexpr size_t DEFAULT_ALIGNMENT = ONIKA_DEFAULT_ALIGNMENT;
# endif

# ifndef ONIKA_DEFAULT_CHUNK_SIZE
  static inline constexpr size_t DEFAULT_CHUNK_SIZE = SimdRequirements<float>::chunksize;
# else
  static inline constexpr size_t DEFAULT_CHUNK_SIZE = ONIKA_DEFAULT_CHUNK_SIZE;
# endif

#ifndef ONIKA_MINIMUM_CUDA_ALIGNMENT
  static inline constexpr size_t MINIMUM_CUDA_ALIGNMENT = 256;
#else
  static inline constexpr size_t MINIMUM_CUDA_ALIGNMENT = ONIKA_MINIMUM_CUDA_ALIGNMENT;
#endif

  struct MemoryChunkInfo
  {
    static inline constexpr uint32_t MEM_FLAG_NONE = 0x00;
    static inline constexpr uint32_t MEM_FLAG_PENDING_DEALLOCATE = 0x01;
    static inline constexpr uint32_t MEM_FLAG_PENDING_DEVICE_TO_MANAGED = 0x02;
    static inline constexpr uint32_t MEM_FLAG_ZERO_INITIALIZED = 0x04;
    static inline constexpr uint64_t MEM_INFO_VALUE16_MASK = (1ull << 16) - 1ull;
    
    void * m_alloc_base = nullptr;
    uint64_t m_alloc_size = 0;
    uint64_t m_info = 0;

    ONIKA_HOST_DEVICE_FUNC inline void * base_ptr() const { return m_alloc_base; }
    ONIKA_HOST_DEVICE_FUNC inline uint64_t size() const { return m_alloc_size; }

    ONIKA_HOST_DEVICE_FUNC inline uint16_t alignment() const { return m_info & MEM_INFO_VALUE16_MASK; } // bits 0-15
    ONIKA_HOST_DEVICE_FUNC inline HostAllocationPolicy mem_type() const { return static_cast<HostAllocationPolicy>( (m_info>>16) & MEM_INFO_VALUE16_MASK ); } // bits 16-31
    ONIKA_HOST_DEVICE_FUNC inline uint16_t flags() const { return (m_info>>32) & MEM_INFO_VALUE16_MASK; } // bits 32-47
    ONIKA_HOST_DEVICE_FUNC inline uint16_t reserved() const { return (m_info>>48) & MEM_INFO_VALUE16_MASK; } // bits 47-63, must be zero
    
    static inline MemoryChunkInfo make(void* ptr, uint64_t sz, uint64_t alignment, HostAllocationPolicy mem_type_e, uint64_t flags=0 )
    {
      uint64_t mem_type = uint64_t(mem_type_e);
      return { ptr , sz, ( alignment & MEM_INFO_VALUE16_MASK ) | ( ( mem_type & MEM_INFO_VALUE16_MASK ) << 16 ) | ( ( flags & MEM_INFO_VALUE16_MASK ) << 32 ) };
    }

    ONIKA_HOST_DEVICE_FUNC static inline constexpr size_t allocation_effective_alignment(size_t a)
    {
      constexpr size_t mem_info_al = alignof(MemoryChunkInfo);
      if( mem_info_al > a ) a = mem_info_al;
      if( sizeof(void*) > a ) a = sizeof(void*); // mandatory for posix_memalign
      return a;
    }

    ONIKA_HOST_DEVICE_FUNC static inline constexpr size_t allocation_size_for_payload(size_t s)
    {
      constexpr size_t mem_info_al = alignof(MemoryChunkInfo);
      const size_t al_sz = (s+mem_info_al-1) & ( ~ (mem_info_al-1) );
      const size_t tot_sz = al_sz + sizeof(MemoryChunkInfo);
      return tot_sz;
    }

    ONIKA_HOST_DEVICE_FUNC inline void read(void* ptr, size_t s)
    {
      constexpr size_t mem_info_al = alignof(MemoryChunkInfo);
      const size_t al_sz = (s+mem_info_al-1) & ( ~ (mem_info_al-1) );
      MemoryChunkInfo * mem_info_ptr = reinterpret_cast<MemoryChunkInfo*>( reinterpret_cast<uint8_t*>(ptr) + al_sz );
      assert( ptr == mem_info_ptr->m_alloc_base );
      m_alloc_base = mem_info_ptr->m_alloc_base;
      m_alloc_size = mem_info_ptr->m_alloc_size;
      m_info = mem_info_ptr->m_info;
    }

    ONIKA_HOST_DEVICE_FUNC inline void write() const
    {
      constexpr size_t mem_info_al = alignof(MemoryChunkInfo);
      const size_t al_sz = (m_alloc_size+mem_info_al-1) & ( ~ (mem_info_al-1) );
      MemoryChunkInfo * mem_info_ptr = reinterpret_cast<MemoryChunkInfo*>( reinterpret_cast<uint8_t*>(m_alloc_base) + al_sz );
      mem_info_ptr->m_alloc_base = m_alloc_base;
      mem_info_ptr->m_alloc_size = m_alloc_size;
      mem_info_ptr->m_info = m_info;
    }
    
    ONIKA_HOST_DEVICE_FUNC inline bool check_consistency(void* ptr, size_t expected_size) const
    {
      unsigned int al = alignment();
      if( al == 0 ) return false;
      const uint64_t ar = ( ( (const uint8_t*)m_alloc_base ) - ( (const uint8_t*)nullptr ) ) % al;
      return ptr==base_ptr() && size()==expected_size && ar==0 && ( mem_type()==HostAllocationPolicy::MALLOC || mem_type()==HostAllocationPolicy::CUDA_HOST ) && reserved()==0 ;
    }
  };

  /*
    Configurable host allocator, either based on malloc or cudaMallocManaged
  */
  struct GenericHostAllocator
  {
    static inline constexpr size_t DefaultAlignBytes = std::max( MINIMUM_CUDA_ALIGNMENT , DEFAULT_ALIGNMENT );
    
    static bool s_enable_debug_log;
    static void set_debug_log(bool b);

    // CPU/GPU compatible methods
    ONIKA_HOST_DEVICE_FUNC static inline MemoryChunkInfo memory_info( void* ptr , size_t s );
    ONIKA_HOST_DEVICE_FUNC static inline bool is_gpu_addressable( void* ptr , size_t s );
    ONIKA_HOST_DEVICE_FUNC void deallocate_async( void* ptr , size_t s ) const;
    
    // CPU only methods
    void deallocate( void* ptr , size_t s ) const;
    void* allocate( size_t s , size_t a ) const;
    bool operator == (const GenericHostAllocator& other) const;
    HostAllocationPolicy get_policy() const;
    bool allocates_gpu_addressable() const;
    void set_gpu_addressable_allocation(bool yn );
    
    // members
    HostAllocationPolicy m_alloc_policy = HostAllocationPolicy::MALLOC;

    static DelayedMemoryOperations * s_device_delayed_memory_operations;
    static DelayedMemoryOperations s_host_delayed_memory_operations;

#   ifdef ONIKA_CUDA_VERSION
    static bool s_enable_cuda;
    static bool cuda_enabled();
    static void set_cuda_enabled(bool yn);
#   else
    static inline constexpr bool cuda_enabled() { return false; }
    static inline constexpr void set_cuda_enabled(bool) {}
#   endif
  };

  // STL compatible host memory allocator.
  template <class T>
  struct CudaManagedAllocator
  {
    typedef T value_type;

    static inline T* allocate (std::size_t n)
    {
      constexpr size_t al = (CUDA_FALLBACK_ALLOC_POLICY==HostAllocationPolicy::CUDA_HOST) ? std::max( alignof(T) , MINIMUM_CUDA_ALIGNMENT ) : alignof(T);
      return static_cast<T*>( GenericHostAllocator{CUDA_FALLBACK_ALLOC_POLICY} .allocate( sizeof(T) * n , al ) );
    }

    static inline void deallocate (T* p, std::size_t n)
    {
      GenericHostAllocator{CUDA_FALLBACK_ALLOC_POLICY} .deallocate( p , sizeof(T) * n );
    }

    template<class U> inline bool operator == (const U&) const { return false; }
    inline bool operator == (const CudaManagedAllocator<T>&) const { return true; }

    template<class U> inline bool operator != (const U& other) const { return ! (*this == other); }
  };

  // default allocator used if none provided
  using DefaultAllocator = GenericHostAllocator;

  template <class T>
  struct NullAllocator
  {
    typedef T value_type;
    ONIKA_HOST_DEVICE_FUNC static inline T* allocate (std::size_t n) { return nullptr; }
    ONIKA_HOST_DEVICE_FUNC static inline void deallocate (T* p, std::size_t n) { }

    template<class U> ONIKA_HOST_DEVICE_FUNC inline bool operator == (const U&) const { return false; }
    ONIKA_HOST_DEVICE_FUNC inline bool operator == (const NullAllocator<T>&) const { return true; }

    template<class U> ONIKA_HOST_DEVICE_FUNC inline bool operator != (const U& other) const { return ! (*this == other); }    
  };

  // ========= inline implementations for device compatible functions =================
  ONIKA_HOST_DEVICE_FUNC
  inline bool GenericHostAllocator::is_gpu_addressable( void* ptr , size_t s )
  {
    if( ptr == nullptr ) { return true; }
    auto mem_type = memory_info(ptr,s).mem_type();
    return (mem_type  == HostAllocationPolicy::CUDA_HOST );
  }

  ONIKA_HOST_DEVICE_FUNC
  inline MemoryChunkInfo GenericHostAllocator::memory_info( void* ptr , size_t s )
  {
    MemoryChunkInfo info = { nullptr, 0, 0 };
    info.read(ptr,s);
#   ifndef NDEBUG
    if( ! info.check_consistency(ptr,s) )
    {
      printf("Corrupted memory allocation : size=%ld/%ld alignment=%d type=%d flags=%04X reserved=%d\n", long(s), long(info.size()), int(info.alignment()), int(info.mem_type()), int(info.flags()), int(info.reserved()) );
      ONIKA_CU_ABORT();
    }
#   endif
    return info;
  }
  // ==============================================================================

  // useful macro to indicate the compiler a pointer is aligned
# ifndef __CUDACC__
# define ONIKA_ASSUME_ALIGNED(x) x = ( decltype(x) __restrict__ ) __builtin_assume_aligned( x , ::onika::memory::DEFAULT_ALIGNMENT )
# define ONIKA__bultin_assume_aligned(x,a) __builtin_assume_aligned( x , a )
# else
# define ONIKA_ASSUME_ALIGNED(x) while(false)
# define ONIKA__bultin_assume_aligned(x,a) (x)
# endif

} // onika::memory

} // onika

// inlcuded here for backward compatibility
#include <onika/memory/mm_vector.h>
