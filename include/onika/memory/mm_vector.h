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
#include <onika/flat_tuple.h>
#include <yaml-cpp/yaml.h>
#include <cstdlib>
#include <vector>

namespace onika
{

namespace memory
{  
  // Host-Device compatible STL vectors
# ifdef ONIKA_STL_BASED_MM_VECTOR

  template<class T> using CudaMMVector = std::vector< T , CudaManagedAllocator<T> >;

# else

  template<class T, class... CTorArgs> struct GPUDataInitFunctor
  {
    T * __restrict__ const m_src_pointer = nullptr;
    T * __restrict__ const m_dst_pointer = nullptr;

    const size_t m_src_size = 0;
    const size_t m_dst_prev_size = 0;
    const size_t m_dst_size = 0;

    const bool m_move_src = true;
    const bool m_del_src = true;
    
    const FlatTuple<CTorArgs...> m_ctor_args;
    
    template<size_t... Ints>
    ONIKA_HOST_DEVICE_FUNC inline void ctor_with_args( size_t i, std::index_sequence<Ints...> )
    {
      new(m_dst_pointer+i) T ( m_ctor_args.get(tuple_index<Ints>) ... );
    }
    
    ONIKA_HOST_DEVICE_FUNC inline void move_item( size_t i )
    {
      new(m_dst_pointer+i) T ( std::move( m_src_pointer[i] ) );
    }

    ONIKA_HOST_DEVICE_FUNC inline void copy_item( size_t i )
    {
      if constexpr ( std::is_trivially_copyable_v<T> ) m_dst_pointer[i] = m_src_pointer[i];
      else new(m_dst_pointer+i) T ( m_src_pointer[i] );
    }
    
    ONIKA_HOST_DEVICE_FUNC inline void del_item( size_t i )
    {
      m_dst_pointer[i].~T();
    }
    
    ONIKA_HOST_DEVICE_FUNC inline void operator () (size_t i) const
    {
      if( m_src_pointer != m_dst_pointer )
      {
        if( i < m_src_size && i < m_dst_size )
        {
          if( m_move_src )
          {
            if( i < m_dst_prev_size ) m_dst_pointer[i] = std::move( m_src_pointer[i] ); // move copy
            else new(m_dst_pointer+i) T ( std::move( m_src_pointer[i] ) ); // move construct
          }
          else
          {
            if( i < m_dst_prev_size ) m_dst_pointer[i] = m_src_pointer[i]; // copy
            else new(m_dst_pointer+i) T ( m_src_pointer[i] ); // copy constructor
          }
        }
      }
      if( i >= m_dst_size && i < m_dst_prev_size ) m_dst_pointer[i].T::~T();
      if( i < m_src_size && m_del_src ) m_src_pointer[i].T::~T();
    }
  };

  template<class InitFunctorT>
  ONIKA_DEVICE_KERNEL_FUNC
  ONIKA_STATIC_INLINE_KERNEL
  void initialize_array_gpu_kernel( const __grid_constant__ size_t init_start, const __grid_constant__ size_t init_elements, const __grid_constant__ InitFunctorT init_func )
  {
    const size_t i = ONIKA_CU_BLOCK_IDX * ONIKA_CU_BLOCK_SIZE + ONIKA_CU_THREAD_IDX;
    if( i < init_elements )
    {
      init_func( init_start + i );
    }
  }

  /*
   * Simple array with managed memory allocation.
   * WARNING: this is not a std::vector, resize fully deallocates and reallocates memory at each call,
   * and elements are NOT conserved across resize
   */
  template<class T>
  struct CudaMMVector
  {
    T * __restrict__ m_data_pointer = nullptr;
    size_t m_size = 0;
    size_t m_capacity = 0;
    
    ONIKA_HOST_DEVICE_FUNC inline const T & operator [] (size_t i) const { return m_data_pointer[i]; }
    ONIKA_HOST_DEVICE_FUNC inline T & operator [] (size_t i) { return m_data_pointer[i]; }
    ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return m_size; }
    ONIKA_HOST_DEVICE_FUNC inline T * data() const { return m_data_pointer; }
    ONIKA_HOST_DEVICE_FUNC inline T * begin() const { return m_data_pointer; }
    ONIKA_HOST_DEVICE_FUNC inline T * end() const { return m_data_pointer + m_size; }

    ONIKA_HOST_DEVICE_FUNC inline CudaMMVector() {}
    
    ONIKA_HOST_DEVICE_FUNC inline CudaMMVector(CudaMMVector && other)
    {
      m_data_pointer = other.m_data_pointer;
      m_size = other.m_size;
      m_capacity = other.m_capacity;
      other.m_data_pointer = nullptr;
      other.m_size = 0;
      other.m_capacity = 0;
    }

    inline CudaMMVector(size_t sz, const T& defval = {})
    {
      resize(sz,defval);
    }
    
    inline CudaMMVector(const CudaMMVector& other)
    {
    }

    template<class InitFuncT>
    static inline void apply_init_func( size_t init_start, size_t init_end, const InitFuncT& init_func )
    {
      if( init_end <= init_start ) return;
      const size_t init_elements = init_end - init_start;
      bool cpu_init = true;
      if constexpr( ::onika::cuda::gpu_frontend_compiler() )
      {
        if( onika::cuda::CudaContext::default_cuda_ctx()!=nullptr && onika::cuda::CudaContext::global_gpu_enable() )
        {
          static constexpr size_t bsize = 64;
          ONIKA_CU_LAUNCH_KERNEL( (init_elements+bsize-1)/bsize,bsize,0,0,initialize_array_gpu_kernel,init_start,init_elements,init_func);
          cpu_init = false;
        }
      }
      if ( cpu_init )
      {
#       pragma omp parallel for schedule(static)
        for(size_t i=init_start;i<init_end;i++) init_func(i);
      }
    }

    template<class... ConstructorArgs>
    static inline void do_transfer_init(T * const src_pointer, size_t src_size, T * const dst_pointer, size_t dst_size, bool copy_flag, const ConstructorArgs & ... ctor_args )
    {
      const size_t transfer_count = std::min(src_size,dst_size);
      const size_t init_start = ( src_pointer != dst_pointer ) ? 0 : transfer_count;
      const size_t init_end = std::max(src_size,dst_size);
      if( init_end > init_start )
      {
        const size_t init_elements = init_end - init_start;
        const size_t ctor_count = ( dst_size > src_size ) ? dst_size : transfer_count;
        const size_t dtor_count = ( dst_size < src_size ) ? src_size : ctor_count;
        GPUDataInitFunctor<T,ConstructorArgs...> init_func = { old_ptr, m_data_pointer, transfer_count, ctor_count, dtor_count, copy_flag, onika::FlatTuple<ConstructorArgs...>{ctor_args...} };
        apply_init(init_start,init_end,init_func);
      }
    }

    inline void copy_from(const CudaMMVector& other)
    {
      if( m_capacity < other.size() )
      GPUDataInitFunctor<T> array_init_func = { old_ptr, m_data_pointer, transfer_count, ctor_count, dtor_count, true };
    }

    template<class... ConstructorArgs>
    inline void resize(size_t sz, const ConstructorArgs & ... ctor_args )
    {
      T * const old_ptr = m_data_pointer;
      const size_t old_capacity = m_capacity;
      const size_t old_size = m_size;

      if( sz > m_capacity )
      {
        if( m_capacity*2 >= m_size ) m_capacity *= 2;
        else m_capacity = m_size;
        m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
      }
      m_size = sz;

      ...

      if( m_data_pointer != old_ptr ) CudaManagedAllocator<T>::deallocate( old_ptr , old_capacity );
    }

    inline void clear() { resize(0); }
    
    inline ~CudaMMVector()
    {
      clear();
      if( m_data_pointer != nullptr ) CudaManagedAllocator<T>::deallocate( m_data_pointer , m_capacity );
      m_data_pointer = nullptr;
      m_capacity = 0;
    }
  };

# endif // ONIKA_STL_BASED_MM_VECTOR

} // onika::memory

} // onika

namespace YAML
{

  template<class T> struct convert< ::onika::memory::CudaMMVector<T> >
  {
    static inline Node encode(const ::onika::memory::CudaMMVector<T>& v)
    {
      Node node;
      for(const auto & x : v) node.push_back(x);
      return node;
    }
    static inline bool decode(const Node& node, ::onika::memory::CudaMMVector<T>& v)
    {
      if( ! node.IsSequence() ) { return false; }
      const size_t sz = node.size();
      v.clear();
      v.reserve( sz );
      for(size_t i=0;i<sz;i++) v.push_back( node[i].as<T>() );
      return true;
    }
  };

}

