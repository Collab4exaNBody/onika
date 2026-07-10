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
      auto alloc_pol = get_policy();
      switch( alloc_pol )
      {
        case HostAllocationPolicy::MALLOC :
        {
          a = std::max( a , sizeof(void*) ); // this is required by posix_memalign.
          int r = posix_memalign( &ptr, a, s + add_info_size );
          if( r != 0 ) { std::cerr<<"Allocation failed. aborting.\n"; std::abort(); }
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") std::cout<<"MALLOC: alloc "<<s + add_info_size<<" ("<<s<<"+"<<add_info_size<<") bytes @"<<ptr<< " , align="<<a<<std::endl; }
        }
        break;

        case HostAllocationPolicy::CUDA_HOST :
        {
#         if defined(ONIKA_CUDA_VERSION)
          ptr = nullptr;
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MALLOC_MANAGED( &ptr, s + add_info_size ) );
          auto pa = reinterpret_cast<uint8_t*>(ptr) - (uint8_t*)nullptr;
          if( ( pa % a ) != 0 )
          {
            std::cerr << "cudaMallocManaged returned a pointer that is not aligned on a "<<a<<" bytes boundary"<<std::endl;
            std::abort();
          }          
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") std::cout<<"CUDA: alloc "<<s + add_info_size<<" ("<<s<<"+"<<add_info_size<<") bytes @"<<ptr<< " , align="<<a<<std::endl; }
          // lout << "cudaMallocManaged("<<s<<","<<a<<") -> @"<<ptr<<std::endl;
#         else
          std::cerr << "Cuda is disabled, no support for CUDA_HOST allocation policy"<<std::endl;
          ptr = nullptr;
          std::abort();
#         endif
        }
        break;
        
        default:
        {
          std::cerr << "Corrupted allocation flag (unknown value "<<static_cast<uint32_t>(alloc_pol)<<")"<<std::endl;
          std::abort();
        }
        break;
      }

      if( s>0 && ptr==nullptr )
      {
        std::cerr<< "onika::memory::GenericHostAllocator::allocate("<<s<<","<<a<<") : Allocation failed (cuda_enabled="<<std::boolalpha<<cuda_enabled()<<")"<<std::endl<<std::flush;
        std::abort();
      }

#     ifdef ONIKA_MEMORY_ZERO_ALLOC
      if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") std::cout<<"zero "<<s + add_info_size<<" ("<<s<<"+"<<add_info_size<<") bytes @" << ptr << std::endl; }
      if( ptr != nullptr ) { std::memset( ptr , 0 , s + add_info_size ); }
#     endif

      // final check of memory chunk with verification flags and size markers
      uint32_t alloc_flags = static_cast<uint32_t>(alloc_pol) | ( static_cast<uint32_t>(a) << 8 );
      * reinterpret_cast<size_t*>( reinterpret_cast<uint8_t*>(ptr) + s ) = s;
      * reinterpret_cast<uint32_t*>( reinterpret_cast<uint8_t*>(ptr) + s + sizeof(size_t) ) = alloc_flags;
      const auto minfo = memory_info(ptr,s);
      if( minfo.alloc_size != s || minfo.alloc_flags != alloc_flags )
      {
        std::cerr<< "onika::memory::GenericHostAllocator::allocate("<<s<<","<<a<<") : Inernal error : created memory block is corrupted : alloc_size "<<minfo.alloc_size<<"/"<<s<<" , alloc_flags "<<minfo.alloc_flags<<"/"<<alloc_flags<<std::endl;
        std::abort();
      }
      
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
      switch( info.mem_type() )
      {
        case HostAllocationPolicy::MALLOC :
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") printf("MALLOC: free %ld bytes @%p align=%d\n",long(info.size()),info.base_ptr(),int(info.alignment())); }
          free(info.alloc_base);
          break;
        case HostAllocationPolicy::CUDA_HOST :
          if( s_enable_debug_log ) { _Pragma("omp critical(dbg_mesg)") printf("CUDA: free %ld bytes @%p align=%d\n",long(info.size()),info.base_ptr(),int(info.alignment())); }
#         ifdef ONIKA_CUDA_VERSION
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_FREE(info.alloc_base) );
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


  }
}

