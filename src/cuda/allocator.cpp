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
#include <iostream>
#include <malloc.h>

#ifdef ONIKA_CUDA_VERSION
#include <onika/cuda/cuda_context.h>
#include <onika/cuda/cuda_error.h>
#endif

#include <onika/memory/allocator.h>
#include <onika/debug.h>

#include <cstring>

namespace onika
{
  namespace memory
  {

#   ifdef ONIKA_CUDA_VERSION
    bool GenericHostAllocator::s_enable_cuda = true;
    bool GenericHostAllocator::cuda_enabled()
    {
      return s_enable_cuda;
    }
    void GenericHostAllocator::set_cuda_enabled(bool yn)
    {
      s_enable_cuda = yn;
    }
#   endif

    bool GenericHostAllocator::s_enable_debug_log = false;
    void GenericHostAllocator::set_debug_log(bool b) { s_enable_debug_log = b; }

    bool GenericHostAllocator::operator == (const GenericHostAllocator& other) const
    {
      return m_alloc_policy == other.m_alloc_policy;
    }
    
    HostAllocationPolicy GenericHostAllocator::get_policy() const
    {
      return cuda_enabled() ? m_alloc_policy : HostAllocationPolicy::MALLOC;
    }

    bool GenericHostAllocator::allocates_gpu_addressable() const
    {
      return get_policy() == HostAllocationPolicy::CUDA_HOST;
    }
    
    void GenericHostAllocator::set_gpu_addressable_allocation(bool yn )
    {
      m_alloc_policy = ( yn ? HostAllocationPolicy::CUDA_HOST : HostAllocationPolicy::MALLOC );
    }
  
    void* GenericHostAllocator::allocate(size_t s, size_t a) const
    {
      void* ptr = nullptr;

      a = MemoryChunkInfo::allocation_effective_alignment(a);
      const size_t alloc_size = MemoryChunkInfo::allocation_size_for_payload(s);

      auto alloc_pol = get_policy();
      switch( alloc_pol )
      {
        case HostAllocationPolicy::MALLOC :
        {
          int r = posix_memalign( & ptr, a, alloc_size );
          if( r != 0 ) { printf("Allocation failed. aborting.\n"); ONIKA_CU_ABORT(); }
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") printf("MALLOC: alloc %ld (payload=%ld) bytes @%p , align=%d\n",long(alloc_size),long(s),ptr,int(a)); }
        }
        break;

        case HostAllocationPolicy::CUDA_HOST :
        {
#         if defined(ONIKA_CUDA_VERSION)
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MALLOC_MANAGED( & ptr, alloc_size ) );
          auto pa = reinterpret_cast<uint8_t*>(ptr) - (uint8_t*)nullptr;
          if( ( pa % a ) != 0 )
          {
            printf("cudaMallocManaged returned a pointer that is not aligned on a %d bytes boundary\n",int(a));
            ONIKA_CU_ABORT();
          }          
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") printf("CUDA: alloc %ld (payload=%ld) bytes @%p , align=%d", long(alloc_size),long(s),ptr,int(a)); }
#         else
          printf("Cuda is disabled, no support for CUDA_HOST allocation policy\n";
          ptr = nullptr;
          ONIKA_CU_ABORT();
#         endif
        }
        break;
        
        default:
        {
          printf("Corrupted allocation mode (unknown value %d)\n",int(alloc_pol));
          ONIKA_CU_ABORT();
        }
        break;
      }

      if( s>0 && ptr==nullptr )
      {
        printf("onika::memory::GenericHostAllocator::allocate(%ld,%d) : Allocation failed (cuda_enabled=%d)\n",long(s),int(a),int(cuda_enabled()));
        ONIKA_CU_ABORT();
      }

#     ifdef ONIKA_MEMORY_ZERO_ALLOC
      MemoryChunkInfo mem_info = MemoryChunkInfo::make(ptr,s,a,alloc_pol,MemoryChunkInfo::MEM_FLAG_ZERO_INITIALIZED);
      if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") printf("zero %ld (%ld+%ld) bytes @%p\n",long(alloc_size),long(s),mem_info.m_alloc_base); }
      if( mem_info.m_alloc_base != nullptr ) { ONIKA_CU_MEMSET( mem_info.m_alloc_base , 0 , s ); }
#     else
      MemoryChunkInfo mem_info = MemoryChunkInfo::make(ptr,s,a,alloc_pol,MemoryChunkInfo::MEM_FLAG_NONE);
#     endif
      mem_info.write();

#     ifndef NDEBUG
      mem_info.read(ptr,s);
      assert( mem_info.check_consistency(ptr,s) );
#     endif

      return ptr;
    }

    void GenericHostAllocator::deallocate( void* ptr , size_t s ) const
    {
      if( ptr == nullptr )
      {
        assert( s == 0 );
        return;
      }
      assert( s > 0 );
      // general case, allocated size is known
      auto info = memory_info(ptr,s);
      assert( info.check_consistency(ptr,s) );
      switch( info.mem_type() )
      {
        case HostAllocationPolicy::MALLOC :
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") printf("MALLOC: free %ld bytes @%p align=%d\n",long(info.size()),info.base_ptr(),int(info.alignment())); }
          free(info.base_ptr());
          break;
        case HostAllocationPolicy::CUDA_HOST :
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") printf("CUDA: free %ld bytes @%p align=%d\n",long(info.size()),info.base_ptr(),int(info.alignment())); }
#         ifdef ONIKA_CUDA_VERSION
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_FREE(info.base_ptr()) );
#         else
          printf("Free memory with type CUDA_HOST but cuda is not available\n");
          ONIKA_CU_ABORT();
#         endif
          break;
        default:
          printf("Corrupted memory flags (mem_type=%d)\n",(int)info.mem_type());
          ONIKA_CU_ABORT();
          break;
      }
    }


    static DelayedMemoryOperations * make_delayed_memory_operations()
    {
      DelayedMemoryOperations * mem_ops = nullptr;
#   ifdef ONIKA_CUDA_VERSION
      if( onika::cuda::get_default_cuda_ctx() != nullptr && onika::cuda::get_global_gpu_enable() )
      {
        ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MALLOC( & mem_ops, sizeof(DelayedMemoryOperations) ) );
        ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMSET( mem_ops , 0 , sizeof(DelayedMemoryOperations) ) );
      }
#   endif
      return mem_ops;
    }

    DelayedMemoryOperations * GenericHostAllocator::s_device_delayed_memory_operations = make_delayed_memory_operations();
    DelayedMemoryOperations GenericHostAllocator::s_host_delayed_memory_operations = {};
  }
}

