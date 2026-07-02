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

  /*
   * Simple array with managed memory allocation.
   * WARNING: this is not a std::vector, resize fully deallocates and reallocates memory at each call,
   * and elements are NOT conserved across resize
   */
  template<class T>
  struct CudaMMArray
  {
    T * m_data_pointer = nullptr;
    size_t m_size = 0;
    ONIKA_HOST_DEVICE_FUNC inline const T & operator [] (size_t i) const { return m_data_pointer[i]; }
    ONIKA_HOST_DEVICE_FUNC inline T & operator [] (size_t i) { return m_data_pointer[i]; }
    ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return m_size; }
    inline void resize(size_t sz)
    {
      if( m_data_pointer != nullptr ) CudaManagedAllocator<T>::deallocate( m_data_pointer , m_size );
      m_size = sz;
      if( m_size > 0 ) m_data_pointer = CudaManagedAllocator<T>::allocate( m_size );
      else m_data_pointer = nullptr;
    }
    inline void clear() { resize(0); }
    inline T * begin() const { return m_data_pointer; }
    inline T * end() const { return m_data_pointer + m_size; }
    inline ~CudaMMArray() { clear(); }
  };

# else

  template<class T, class... CTorArgs> struct GPUDataInitFunctor
  {
    T * const m_src_pointer = nullptr;
    T * const m_dst_pointer = nullptr;

    const size_t m_src_size = 0; // number of items to be transfered from src to dst
    // (if m_src_pointer is null however, elements are initialized rather than transfered
    const size_t m_dst_prev_size = 0; // number of constructed items in dst buffer
    const size_t m_dst_size = 0; // total number of items in dst buffer

    const bool m_move_src = true; // use move instead of copy
    const bool m_del_src = true; // destruct src elements after they are transfered
    
    // constructor parameters to intialized newly constructed elements in dst
    const FlatTuple<CTorArgs...> m_ctor_args = {};
    
    template<size_t... Ints>
    ONIKA_HOST_DEVICE_FUNC inline void init_with_args( size_t i, std::index_sequence<Ints...> ) const
    {
      if( i < m_dst_prev_size ) m_dst_pointer[i] = T ( m_ctor_args.get(tuple_index<Ints>) ... );
      else new(m_dst_pointer+i) T ( m_ctor_args.get(tuple_index<Ints>) ... );
    }

    ONIKA_HOST_DEVICE_FUNC inline void move_src_to_dst( size_t i ) const
    {
      if( i < m_dst_prev_size ) m_dst_pointer[i] = std::move( m_src_pointer[i] ); // move assign
      else new(m_dst_pointer+i) T ( std::move( m_src_pointer[i] ) ); // move construct
    }

    ONIKA_HOST_DEVICE_FUNC inline void copy_src_to_dst( size_t i ) const
    {
      if( i < m_dst_prev_size ) m_dst_pointer[i] = m_src_pointer[i]; // copy assign
      else new(m_dst_pointer+i) T ( m_src_pointer[i] ); // copy constructor
    }

    ONIKA_HOST_DEVICE_FUNC inline void operator () (size_t i) const
    {
      if( i < m_src_size && i < m_dst_size )
      {
        if( m_src_pointer == nullptr ) init_with_args(i,std::make_index_sequence<m_ctor_args.size()>{});
        else if( m_move_src ) move_src_to_dst(i);
        else copy_src_to_dst(i);
      }
      else
      {
        if( i >= m_dst_size && i < m_dst_prev_size ) m_dst_pointer[i].T::~T();
        if( i >= m_dst_prev_size && i < m_dst_size ) init_with_args(i,std::make_index_sequence<m_ctor_args.size()>{});
      }
      if( m_del_src && i < m_src_size && m_src_pointer != nullptr ) m_src_pointer[i].T::~T();
    }
  };

  template<class InitFunctorT>
  ONIKA_DEVICE_KERNEL_FUNC
  ONIKA_STATIC_INLINE_KERNEL
  void initialize_array_gpu_kernel( const ONIKA_CU_GRID_CONSTANT size_t init_start
                                  , const ONIKA_CU_GRID_CONSTANT size_t init_elements
                                  , const ONIKA_CU_GRID_CONSTANT InitFunctorT init_func )
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
  struct alignas(32) CudaMMVector
  {
    T * __restrict__ m_data_pointer = nullptr;
    size_t m_size = 0;
    size_t m_capacity = 0;
    bool m_host_access_hint = false; // if true, will always use cpu to initialize/write buffer
    
    ONIKA_HOST_DEVICE_FUNC inline const T & operator [] (size_t i) const { return m_data_pointer[i]; }
    ONIKA_HOST_DEVICE_FUNC inline T & operator [] (size_t i) { return m_data_pointer[i]; }
    ONIKA_HOST_DEVICE_FUNC inline const T & at (size_t i) const { assert(i<m_size); return m_data_pointer[i]; }
    ONIKA_HOST_DEVICE_FUNC inline T & at (size_t i) { assert(i<m_size); return m_data_pointer[i]; }
    ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return m_size; }
    ONIKA_HOST_DEVICE_FUNC inline size_t capacity() const { return m_capacity; }
    ONIKA_HOST_DEVICE_FUNC inline T * data() const { return m_data_pointer; }
    ONIKA_HOST_DEVICE_FUNC inline T * begin() const { return m_data_pointer; }
    ONIKA_HOST_DEVICE_FUNC inline T * end() const { return m_data_pointer + m_size; }

    ONIKA_HOST_DEVICE_FUNC inline CudaMMVector() {}
    
    ONIKA_HOST_DEVICE_FUNC inline CudaMMVector(CudaMMVector && other)
    {
      move_from( std::move(other) );
    }

    inline CudaMMVector(size_t sz, const T& defval)
    {
      resize( sz , defval );
    }

    inline CudaMMVector(size_t sz)
    {
      resize( sz );
    }
    
    inline CudaMMVector(const CudaMMVector& other)
    {
      copy_from(other);
    }

    inline CudaMMVector& operator = (const CudaMMVector& other)
    {
      copy_from(other);
    }

    ONIKA_HOST_DEVICE_FUNC inline CudaMMVector& operator = (CudaMMVector&& other)
    {
      move_from( std::move(other) );
    }

    inline void set_host_access_hint(bool h)
    {
      m_host_access_hint = h;
    }

    inline bool host_access_hint(bool h) const
    {
      return m_host_access_hint;
    }

    inline void push_back(const T& item)
    {
      resize( size()+1 , item );
    }

    ONIKA_HOST_DEVICE_FUNC inline void move_from(CudaMMVector && other)
    {
      m_data_pointer = other.m_data_pointer;
      m_size = other.m_size;
      m_capacity = other.m_capacity;
      other.m_data_pointer = nullptr;
      other.m_size = 0;
      other.m_capacity = 0;
    }

    inline void copy_from(const CudaMMVector& other)
    {
      using InitFuncT = GPUDataInitFunctor<T>;
      if( other.size() > capacity() )
      {
        // calls destructor before deallocate
        apply_init( 0, size(), InitFuncT{ nullptr, m_data_pointer, 0, size(), 0, false, false, onika::FlatTuple<>{} } );
        CudaManagedAllocator<T>::deallocate( m_data_pointer , m_capacity );
        m_capacity = other.size();
        m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
        m_size = 0;
      }
      apply_init( 0, std::max(size(),other.size())
                , InitFuncT{ other.m_data_pointer, m_data_pointer, other.size(), size(), other.size(), false, false, onika::FlatTuple<>{} }
                );
      m_size = other.size();
    }

    inline void shrink_to_fit()
    {
      using InitFuncT = GPUDataInitFunctor<T>;
      T * const old_ptr = m_data_pointer;
      const size_t old_capacity = m_capacity;
      if( m_size != m_capacity )
      {
        m_capacity = m_size;
        if(m_capacity>0) m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
        else m_data_pointer = nullptr;
      }
      if( old_ptr != m_data_pointer )
      {
        const size_t init_start = 0;
        const size_t init_end = m_size;
        apply_init( init_start, init_end, InitFuncT{ old_ptr, m_data_pointer, m_size, 0, m_size, true, true } );
        CudaManagedAllocator<T>::deallocate( old_ptr , old_capacity );
      }
    }

    template<class... ConstructorArgs>
    inline void resize(size_t sz, const ConstructorArgs & ... ctor_args )
    {
      if( sz == m_size ) return;
      using ArgsTupleT = onika::FlatTuple<ConstructorArgs...>;
      using InitFuncT = GPUDataInitFunctor<T,ConstructorArgs...>;
      T * const old_ptr = m_data_pointer;
      const size_t old_capacity = m_capacity;
      const size_t old_size = m_size;
      if( sz > m_capacity )
      {
        if( m_capacity*2 >= sz ) m_capacity *= 2;
        else m_capacity = sz;
        m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
      }
      m_size = sz;
      const bool reallocd = ( old_ptr != m_data_pointer );
      const size_t prev_size = reallocd ? 0 : old_size;
      const size_t init_start = reallocd ? 0 : std::min(old_size,m_size);
      const size_t init_end = std::max(old_size,m_size);
      apply_init( init_start, init_end, InitFuncT{ reallocd ? old_ptr : nullptr , m_data_pointer, reallocd ? old_size : 0, prev_size, m_size, reallocd, reallocd, ArgsTupleT{ctor_args...} } );
      if( reallocd ) CudaManagedAllocator<T>::deallocate( old_ptr , old_capacity );
    }

    template<class... ConstructorArgs>
    inline void assign(size_t sz, const ConstructorArgs & ... ctor_args )
    {
      using ArgsTupleT = onika::FlatTuple<ConstructorArgs...>;
      using InitFuncT = GPUDataInitFunctor<T,ConstructorArgs...>;
      T * const old_ptr = m_data_pointer;
      const size_t old_capacity = m_capacity;
      const size_t old_size = m_size;
      if( sz > m_capacity )
      {
        m_capacity = sz;
        m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
      }
      m_size = sz;
      const size_t init_start = 0;
      const size_t init_end = std::max(old_size,m_size);
      apply_init( init_start, init_end, InitFuncT{ nullptr, m_data_pointer, m_size, old_size, m_size, false, false, ArgsTupleT{ctor_args...} } );
      if( m_data_pointer != old_ptr ) CudaManagedAllocator<T>::deallocate( old_ptr , old_capacity );
    }
    
    inline void reserve(size_t ncap)
    {
      using InitFuncT = GPUDataInitFunctor<T>;
      T * const old_ptr = m_data_pointer;
      const size_t old_capacity = m_capacity;
      if( ncap > m_capacity )
      {
        m_capacity = ncap;
        m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
      }
      if( old_ptr != m_data_pointer )
      {
        const size_t init_start = 0;
        const size_t init_end = m_size;
        apply_init( init_start, init_end, InitFuncT{ old_ptr, m_data_pointer, m_size, 0, m_size, true, true } );
        CudaManagedAllocator<T>::deallocate( old_ptr , old_capacity );
      }
    }

    template<class InitFuncT>
    inline void apply_init( size_t init_start, size_t init_end, const InitFuncT& init_func )
    {
      if( init_end <= init_start ) return;
      const size_t init_elements = init_end - init_start;
      bool cpu_init = true;
      if constexpr( gpu_frontend_compiler() )
      {
        if( !m_host_access_hint && onika::cuda::get_default_cuda_ctx()!=nullptr && onika::cuda::get_global_gpu_enable() )
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

    inline void clear() { resize(0); }
    
    inline ~CudaMMVector()
    {
      clear();
      if( m_data_pointer != nullptr ) CudaManagedAllocator<T>::deallocate( m_data_pointer , m_capacity );
      m_data_pointer = nullptr;
      m_capacity = 0;
    }
  };

  template<class T> using CudaMMArray = CudaMMVector<T>;

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

