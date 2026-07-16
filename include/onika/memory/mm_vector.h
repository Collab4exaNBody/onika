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
#include <onika/cuda/span.h>
#include <onika/memory/allocator.h>
#include <onika/cuda/cuda_context.h>
#include <onika/flat_tuple.h>
#include <onika/type_utils.h>
#include <yaml-cpp/yaml.h>
#include <cstdlib>

namespace onika
{

namespace memory
{

#   if defined(ONIKA_CUDA_VERSION) && defined(ONIKA_CUDAMMVECTOR_GPU_OPERATIONS)

    template<class T>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_STATIC_INLINE_KERNEL
    void call_destructor_kernel( T * data_ptr , ONIKA_CU_GRID_CONSTANT const size_t el_start, ONIKA_CU_GRID_CONSTANT const size_t el_end )
    {
      const size_t i = el_start + ( ONIKA_CU_BLOCK_DIMS.x * ONIKA_CU_BLOCK_COORD.x ) + ONIKA_CU_THREAD_COORD.x;
      if( i >= el_start && i < el_end ) ( data_ptr + i ) -> T::~T();
    }

    template<class T, class CTorArgsTupleT, size_t... Ints>
    ONIKA_DEVICE_FUNC
    void call_constructor_for_element( T * data_ptr , const size_t i, const CTorArgsTupleT& ctor_args_tuple, std::index_sequence<Ints...> IS )
    {
      new( data_ptr + i ) T ( ctor_args_tuple.get(onika::tuple_index<Ints>) ... );
    }

    template<class T, class CTorArgsTupleT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_STATIC_INLINE_KERNEL
    void call_constructor_kernel( T * data_ptr , ONIKA_CU_GRID_CONSTANT const size_t el_start, ONIKA_CU_GRID_CONSTANT const size_t el_end, ONIKA_CU_GRID_CONSTANT const CTorArgsTupleT ctor_args_tuple )
    {
      const size_t i = el_start + ( ONIKA_CU_BLOCK_DIMS.x * ONIKA_CU_BLOCK_COORD.x ) + ONIKA_CU_THREAD_COORD.x;
      if( i >= el_start && i < el_end ) call_constructor_for_element( data_ptr, i, ctor_args_tuple, std::make_index_sequence< onika::tuple_size_const_v<CTorArgsTupleT> >{} );      
    }

#   endif

  /*
   * Simple array with managed memory allocation.
   * WARNING: this is not a std::vector, resize fully deallocates and reallocates memory at each call,
   * and elements are NOT conserved across resize
   */
  template<class T>
  struct CudaMMVector
  {    
    T * m_data_pointer = nullptr;
    size_t m_size = 0;
    size_t m_capacity = 0;
    
    ONIKA_HOST_DEVICE_FUNC inline CudaMMVector() {}
    
    inline CudaMMVector(CudaMMVector && other)
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
      assign( other.const_span() );
    }
    
    inline CudaMMVector( std::initializer_list<T> other)
    {
      assign( { other.begin() , other.size() } );
    }

    inline CudaMMVector& operator = (const CudaMMVector& other)
    {
      assign( other.const_span() );
      return *this;
    }

    inline CudaMMVector& operator = ( std::span<const T> other)
    {
      assign( onika::cuda::span<const T>{ other.data() , other.size() } );
      return *this;
    }

    inline CudaMMVector& operator = ( std::span<T> other)
    {
      assign( onika::cuda::span<const T>{ other.data() , other.size() } );
      return *this;
    }

    inline CudaMMVector& operator = (onika::cuda::span<const T> other)
    {
      assign( other );
      return *this;
    }

    inline CudaMMVector& operator = (CudaMMVector&& other)
    {
      move_from( std::move(other) );
      return *this;
    }

    ONIKA_HOST_DEVICE_FUNC inline const T & operator [] (size_t i) const { return m_data_pointer[i]; }
    ONIKA_HOST_DEVICE_FUNC inline T & operator [] (size_t i) { return m_data_pointer[i]; }

    ONIKA_HOST_DEVICE_FUNC inline const T & at (size_t i) const { assert(i<m_size); return m_data_pointer[i]; }
    ONIKA_HOST_DEVICE_FUNC inline T & at (size_t i) { assert(i<m_size); return m_data_pointer[i]; }

    ONIKA_HOST_DEVICE_FUNC inline T& front() { return *m_data_pointer; }
    ONIKA_HOST_DEVICE_FUNC inline const T& front() const { return *m_data_pointer; }

    ONIKA_HOST_DEVICE_FUNC inline T& back() { return *(m_data_pointer+m_size-1); }
    ONIKA_HOST_DEVICE_FUNC inline const T& back() const { return *(m_data_pointer+m_size-1); }

    ONIKA_HOST_DEVICE_FUNC inline size_t size() const { return m_size; }
    ONIKA_HOST_DEVICE_FUNC inline bool empty() const { return size()==0; }

    ONIKA_HOST_DEVICE_FUNC inline size_t capacity() const { return m_capacity; }

    ONIKA_HOST_DEVICE_FUNC inline T * data() const { return m_data_pointer; }

    ONIKA_HOST_DEVICE_FUNC inline T * begin() const { return m_data_pointer; }
    ONIKA_HOST_DEVICE_FUNC inline T * end() const { return m_data_pointer + m_size; }

    ONIKA_HOST_DEVICE_FUNC operator onika::cuda::span<T> () const { return { data() , size() }; }
    ONIKA_HOST_DEVICE_FUNC operator onika::cuda::span<const T> () const { return { data() , size() }; }
    ONIKA_HOST_DEVICE_FUNC onika::cuda::span<T> span() const { return { data() , size() }; }
    ONIKA_HOST_DEVICE_FUNC onika::cuda::span<const T> const_span() const { return { data() , size() }; }

    inline void push_back(const T& item)
    {
      resize( size()+1 , item );
    }

    inline void move_from(CudaMMVector && other)
    {
      if( m_capacity>0 && m_data_pointer!=nullptr )
      {
        CudaManagedAllocator<T>::deallocate( m_data_pointer , m_capacity );
        m_capacity = 0;
        m_data_pointer = nullptr;
      }
      assert( m_capacity == 0 && m_data_pointer == nullptr );
      m_data_pointer = other.m_data_pointer;
      m_size = other.m_size;
      m_capacity = other.m_capacity;
      other.m_data_pointer = nullptr;
      other.m_size = 0;
      other.m_capacity = 0;
    }

    inline void assign( std::vector<T>::const_iterator other_begin, std::vector<T>::const_iterator other_end )
    {
      const T * data_ptr = & *(other_begin);
      const size_t sz = std::distance( other_begin , other_end );
      assign( onika::cuda::span<const T> { data_ptr, sz} );
    }

    inline void assign(onika::cuda::span<const T> other)
    {
      if( other.size() > capacity() )
      {
        // calls destructor before deallocate
        CudaManagedAllocator<T>::deallocate( m_data_pointer , m_capacity );
        m_capacity = other.size();
        m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
      }
      m_size = other.size();
      ONIKA_CU_MEMCPY( m_data_pointer , other.data() , m_size * sizeof(T) );
    }

    inline void realloc(size_t new_capacity)
    {
      T * const old_ptr = m_data_pointer;
      const size_t old_capacity = m_capacity;
      if( new_capacity != m_capacity )
      {
        m_capacity = new_capacity;
        if(m_capacity>0) m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
        else m_data_pointer = nullptr;
      }
      if( old_ptr != m_data_pointer )
      {
        const size_t elements_to_copy = std::min( m_size , m_capacity );
        if( elements_to_copy > 0 ) { ONIKA_CU_MEMCPY( m_data_pointer , old_ptr , elements_to_copy * sizeof(T) ); }
        if( old_ptr != nullptr ) { CudaManagedAllocator<T>::deallocate( old_ptr , old_capacity ); }
      }
    }

    inline void shrink_to_fit()
    {
      realloc( size() );
    }

    inline void resizeNoInit(size_t sz)
    {
      if( sz > m_capacity ) realloc( (m_capacity*2>=sz) ? (m_capacity*2) : sz );
      m_size = sz;
    }

    inline void resizeZeroInit(size_t sz)
    {
      if( sz > m_capacity ) realloc( (m_capacity*2>=sz) ? (m_capacity*2) : sz );
      if( sz > m_size ) { ONIKA_CU_MEMSET( m_data_pointer + m_size , 0 , (sz-m_size)*sizeof(T) ); }
      m_size = sz;
    }

    template<class... CtorArgs>
    inline void resize(size_t sz , const CtorArgs& ... init_val_ctor)
#   if defined(ONIKA_CUDA_VERSION) && defined(ONIKA_CUDAMMVECTOR_GPU_OPERATIONS)
      requires( ( std::is_trivially_destructible_v<T> && std::is_trivially_constructible_v<T> && sizeof...(CtorArgs)==0 ) || ( ! onika::is_gpu_destructible_v<T> && ! onika::is_gpu_constructible_v<T> ) || ONIKA_GPU_FRONTEND_COMPILER::value )
#   endif
    {
      if constexpr ( ! std::is_trivially_destructible_v<T> ) if( sz < m_size )
      {
        bool cpu_call_dtor = true;
#       if defined(ONIKA_CUDA_VERSION) && defined(ONIKA_CUDAMMVECTOR_GPU_OPERATIONS)
        if constexpr ( onika::is_gpu_destructible_v<T> ) if( onika::cuda::CudaContext::default_cuda_ctx()!=nullptr && onika::cuda::CudaContext::global_gpu_enable() )
        {
          static constexpr size_t BLOCK_SIZE = 64;
          const size_t N = m_size - sz;
          ONIKA_CU_LAUNCH_KERNEL(N,BLOCK_SIZE,0,0, call_destructor_kernel, m_data_pointer, sz, m_size );
          cpu_call_dtor = false;
        }
#       endif
        if( cpu_call_dtor )
        {
//#       pragma omp parallel for schedule(static)
          for(size_t i=sz ; i<m_size ; i++) (m_data_pointer+i) -> T::~T();
        }        
      }
      m_size = std::min(sz,m_size);

      if( sz > m_capacity ) realloc( (m_capacity*2>=sz) ? (m_capacity*2) : sz );
      
      if constexpr ( !std::is_trivially_constructible_v<T> || sizeof...(CtorArgs)>0 ) if( sz > m_size )
      {
        bool cpu_call_ctor = true;
#       if defined(ONIKA_CUDA_VERSION) && defined(ONIKA_CUDAMMVECTOR_GPU_OPERATIONS)
        if constexpr ( onika::is_gpu_constructible_v<T> ) if( onika::cuda::CudaContext::default_cuda_ctx()!=nullptr && onika::cuda::CudaContext::global_gpu_enable() )
        {
          static constexpr size_t BLOCK_SIZE = 64;
          const size_t N = sz - m_size;
          const onika::FlatTuple<CtorArgs...> ctor_args {init_val_ctor...};
          ONIKA_CU_LAUNCH_KERNEL(N,BLOCK_SIZE,0,0, call_constructor_kernel, m_data_pointer, m_size, sz, ctor_args );
          cpu_call_ctor = false;
        }
#       endif
        if( cpu_call_ctor )
        {
//#       pragma omp parallel for schedule(static)
          for(size_t i=m_size ; i<sz ; i++) new(m_data_pointer+i) T ( init_val_ctor ... );
        }
      }
      m_size = std::max(sz,m_size);

      assert( m_size == sz );
    }

    template<class... CtorArgs>
    inline void assign(size_t sz , const CtorArgs& ... init_val_ctor )
    {
      clear();
      resize(sz,init_val_ctor...);
    }
    
    inline void reserve(size_t ncap)
    {
      if( ncap > m_capacity ) realloc( ncap );
    }

    inline void clear() { resize(0); }
    
    ONIKA_HOST_DEVICE_FUNC
    inline ~CudaMMVector()
    {
      if constexpr ( gpu_device_execution() )
      {
        //assert( m_capacity == 0 && m_data_pointer == nullptr );
        if( m_capacity>0 && m_data_pointer!=nullptr )
        {
          printf("GPU: call to ~CudaMMVector() with non empty capacity => %ldx%ld bytes leaked\n",long(m_capacity),long(sizeof(T)));
        }
      }
      else
      {
        clear();
        realloc(0);
      }
      m_size = 0;
      m_data_pointer = nullptr;
      m_capacity = 0;
    }
  };

  template<class T> using CudaMMArray = CudaMMVector<T>;

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

