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
#include <onika/flat_tuple.h>
#include <onika/memory/allocator.h>
#include <onika/type_features.h>
#include <yaml-cpp/yaml.h>
#include <cstdlib>
#include <vector>

namespace onika
{

namespace memory
{
  template<class T, bool _TransferFromSource, bool _MoveSource, bool _DestructSource, class... CTorArgs> struct GPUDataInitFunctor
  {
    static inline constexpr bool TransferFromSource = _TransferFromSource;
    static inline constexpr bool MoveSource = _MoveSource;
    static inline constexpr bool DestructSource = _DestructSource;
    using SrcPtrT = std::conditional_t< TransferFromSource , std::conditional_t< MoveSource || DestructSource , T * , const T * > , nullptr_t >;
    
    SrcPtrT const m_src_pointer = nullptr;
    T * const m_dst_pointer = nullptr;

    const size_t m_src_size = 0; // number of items to be transfered from src to dst
    // (if m_src_pointer is null however, elements are initialized rather than transfered

    const size_t m_dst_prev_size = 0; // number of constructed items in dst buffer
    const size_t m_dst_size = 0; // total number of items in dst buffer
    
    // constructor parameters to intialized newly constructed elements in dst
    const FlatTuple<CTorArgs...> m_ctor_args = {};

    template<size_t... Ints>
    ONIKA_HOST_DEVICE_FUNC inline void init_value_ctor_args( size_t i , std::index_sequence<Ints...> ) const
    {
      if( i < m_dst_prev_size ) destruct_dst(i);
      new(m_dst_pointer+i) T( m_ctor_args.get(tuple_index<Ints>) ... );
    }

    ONIKA_HOST_DEVICE_FUNC inline void init_value( size_t i ) const
    {
      init_value_ctor_args(i,std::make_index_sequence<sizeof...(CTorArgs)>{});
    }

    ONIKA_HOST_DEVICE_FUNC inline void destruct_src( size_t i ) const
    {
      m_src_pointer[i].T::~T();
    }

    ONIKA_HOST_DEVICE_FUNC inline void destruct_dst( size_t i ) const
    {
      m_dst_pointer[i].T::~T();
    }

    ONIKA_HOST_DEVICE_FUNC inline void move_src_to_dst( size_t i ) const requires(TransferFromSource && MoveSource)
    {
      if( i < m_dst_prev_size ) m_dst_pointer[i] = std::move( m_src_pointer[i] ); // move assign
      else new(m_dst_pointer+i) T ( std::move( m_src_pointer[i] ) ); // move construct
    }

    ONIKA_HOST_DEVICE_FUNC inline void copy_src_to_dst( size_t i ) const requires(TransferFromSource)
    {
      if( i < m_dst_prev_size ) m_dst_pointer[i] = m_src_pointer[i]; // copy assign
      else new(m_dst_pointer+i) T ( m_src_pointer[i] ); // copy constructor
    }    

    ONIKA_HOST_DEVICE_FUNC inline void operator () (size_t i) const
    {
      if( i < m_src_size && i < m_dst_size )
      {
        if constexpr ( TransferFromSource )
        {
          if constexpr ( MoveSource ) move_src_to_dst(i);
          else copy_src_to_dst(i);
        }
        else
        {
          init_value(i);
        }
      }
      else
      {
        if( i >= m_dst_size && i < m_dst_prev_size ) destruct_dst(i);        
        if( i >= m_dst_prev_size && i < m_dst_size ) init_value(i);
      }
      if constexpr ( TransferFromSource && DestructSource ) if( i < m_src_size ) destruct_src(i);
    }
  };

  template<class T> struct IsAGPUDataInitFunctor : public std::false_type {};
  template<class T, bool _TransferFromSource, bool _MoveSource, bool _DestructSource, class... CTorArgs> struct IsAGPUDataInitFunctor< GPUDataInitFunctor<T,_TransferFromSource,_MoveSource,_DestructSource,CTorArgs...> > : public std::true_type {};
  template<class T> static inline constexpr bool is_a_gpu_data_init_functor_v = IsAGPUDataInitFunctor<T>::value ;
  template<class T> concept SomeGPUDataInitFunctor = is_a_gpu_data_init_functor_v<T>;

  template<SomeGPUDataInitFunctor InitFunctorT>
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
      copy_from( other );
    }
    inline CudaMMVector( std::initializer_list<T> other)
    {
      copy_from( { other.begin() , other.size() } );
    }

    inline CudaMMVector& operator = (const CudaMMVector& other)
    {
      copy_from(other);
      return *this;
    }

    ONIKA_HOST_DEVICE_FUNC inline CudaMMVector& operator = (CudaMMVector&& other)
    {
      move_from( std::move(other) );
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

    operator onika::cuda::span<T> () const { return { data() , size() }; }
    operator onika::cuda::span<const T> () const { return { data() , size() }; }

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

    inline void copy_from(onika::cuda::span<const T> other)
    {
      if( other.size() > capacity() )
      {
        // calls destructor before deallocate
        GPUDataInitFunctor<T,false,false,false> deinit_func = { {}, m_data_pointer, 0, size(), 0 };
        apply_init( 0, size(), std::move(deinit_func) );
        CudaManagedAllocator<T>::deallocate( m_data_pointer , m_capacity );
        m_capacity = other.size();
        m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
        m_size = 0;
      }
      GPUDataInitFunctor<T,true,false,false> copy_init_func = { other.data(), m_data_pointer, other.size(), size(), other.size() };
      apply_init( 0, std::max(size(),other.size()) , std::move(copy_init_func) );
      m_size = other.size();
    }

    inline void shrink_to_fit()
    {
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
        GPUDataInitFunctor<T,true,true,true> move_init_func = { old_ptr, m_data_pointer, m_size, 0, m_size };
        apply_init( init_start, init_end, std::move(move_init_func) );
        CudaManagedAllocator<T>::deallocate( old_ptr , old_capacity );
      }
    }

    template<class... CTorArgs>
    inline void resize(size_t sz, FlatTuple<CTorArgs...> && init_ctor_args )
    {
      if( sz == m_size ) return;
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
      if( old_ptr != m_data_pointer )
      {
        GPUDataInitFunctor<T,true,true,true,CTorArgs...> move_init_func = { old_ptr, m_data_pointer, old_size, 0, m_size, std::move( init_ctor_args ) };
        apply_init( 0, std::max(old_size,m_size), std::move(move_init_func) );
        CudaManagedAllocator<T>::deallocate( old_ptr , old_capacity );
      }
      else
      {
        GPUDataInitFunctor<T,false,false,false,CTorArgs...> init_func = { nullptr, m_data_pointer, 0, old_size, m_size, std::move( init_ctor_args ) };
        const size_t init_start = std::min(old_size,m_size);
        const size_t init_elements = std::max(old_size,m_size) - init_start;
        apply_init( init_start, init_elements, std::move(init_func) );
      }
    }

    inline void resize(size_t sz)
    {
      resize( sz, FlatTuple<>{} );
    }

    inline void resize(size_t sz, const T& init_val)
    {
      resize( sz, FlatTuple<T>{init_val} );
    }

    template<class... CTorArgs>
    inline void assign_ctor_args(size_t sz, FlatTuple<CTorArgs...> && init_ctor_args )
    {
      T * const old_ptr = m_data_pointer;
      const size_t old_capacity = m_capacity;
      const size_t old_size = m_size;
      if( sz > m_capacity )
      {
        m_capacity = sz;
        m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
      }
      m_size = sz;      
      if( old_ptr != m_data_pointer )
      {
        GPUDataInitFunctor<T,false,false,false> deinit_func = { nullptr, old_ptr, 0, old_size, 0, {} };
        apply_init( 0, old_size, std::move(deinit_func) );
        CudaManagedAllocator<T>::deallocate( old_ptr , old_capacity );
      }
      size_t reset_size = ( old_ptr != m_data_pointer ) ? 0 : old_size ;
      GPUDataInitFunctor<T,false,false,false,CTorArgs...> init_func = { nullptr, m_data_pointer, reset_size, reset_size, m_size, std::move(init_ctor_args) };
      apply_init( 0, std::max(old_size,m_size), std::move(init_func) );
    }

    inline void assign(size_t sz, const T & init_val )
    {
      assign_ctor_args( sz, FlatTuple<T>{init_val} );
    }

    inline void assign(size_t sz )
    {
      assign_ctor_args( sz, FlatTuple<>{} );
    }

    inline void assign(const T* it1, const T* it2)
    {
      copy_from( onika::cuda::make_const_span(it1,it2) );
    }

    inline void assign(std::vector<T>::const_iterator it1, std::vector<T>::const_iterator it2)
    {
      copy_from( onika::cuda::span<const T>{ & (*it1) , std::distance(it1,it2) } );
    }
    
    inline void reserve(size_t ncap)
    {
      T * const old_ptr = m_data_pointer;
      const size_t old_capacity = m_capacity;
      if( ncap > m_capacity )
      {
        m_capacity = ncap;
        m_data_pointer = CudaManagedAllocator<T>::allocate( m_capacity );
      }
      if( old_ptr != m_data_pointer )
      {
        GPUDataInitFunctor<T,true,true,true> move_func = { old_ptr, m_data_pointer, m_size, 0, m_size };
        apply_init( 0, m_size, std::move(move_func) );
        CudaManagedAllocator<T>::deallocate( old_ptr , old_capacity );
      }
    }

    template<SomeGPUDataInitFunctor InitFuncT>
    inline void apply_init( size_t init_start, size_t init_end, InitFuncT && init_func )
    {
      static constexpr bool gpu_compatible_operation =
        supported_features<T>::gpu_destruct
        && 
        (
          ( init_func.m_ctor_args.size()==0 && supported_features<T>::gpu_default_construct )
          ||
          ( supported_features<T>::gpu_non_default_construct && supported_features<T>::gpu_copy_construct )
        )
        &&
        (
          ( InitFuncT::MoveSource && supported_features<T>::gpu_move_construct && supported_features<T>::gpu_move_assign )
          ||
          ( !InitFuncT::MoveSource && supported_features<T>::gpu_copy_construct && supported_features<T>::gpu_copy_assign )
        );
      
      if( init_end <= init_start ) return;
      const size_t init_elements = init_end - init_start;
      bool cpu_init = true;
      if constexpr( gpu_compatible_operation && gpu_frontend_compiler() )
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

