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

#include <type_traits>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <thread>
#include <onika/macro_utils.h>
#include <onika/atomic.h>
#include <omp.h>

#ifdef ONIKA_CUDA_VERSION
#ifdef ONIKA_HIP_VERSION
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#endif

#ifndef ONIKA_CU_MAX_THREADS_PER_BLOCK
#define ONIKA_CU_MAX_THREADS_PER_BLOCK 128
#endif

#ifndef ONIKA_CU_MIN_BLOCKS_PER_SM
#define ONIKA_CU_MIN_BLOCKS_PER_SM 6
#endif

#ifndef ONIKA_CU_ENABLE_KERNEL_BOUNDS
#define ONIKA_CU_ENABLE_KERNEL_BOUNDS 1
#endif

#define ONIKA_ALWAYS_INLINE inline __attribute__((always_inline))

namespace onika
{
  namespace cuda
  {

    struct onika_cu_atomic_flag_t
    {
      std::atomic_flag m_flag = ATOMIC_FLAG_INIT;
      uint8_t m_pad[3] = { 0, 0, 0 };
    };
    static_assert( sizeof(onika_cu_atomic_flag_t) == sizeof(uint32_t) , "Atomic flag type must be 4 bytes large" );
#   define ONIKA_CU_ATOMIC_FLAG_INIT ::onika::cuda::onika_cu_atomic_flag_t{ ATOMIC_FLAG_INIT , {0,0,0} }

    using onika_cu_memory_order_t = std::memory_order;

    /************** start of Cuda code definitions ***************/
#   if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

#   define ONIKA_GPU_DEVICE_EXECUTION_TYPE onika::TrueType

    [[ noreturn ]] __host__ __device__ inline void __onika_cu_abort()
    {
#     if defined(__HIP_DEVICE_COMPILE__)
      abort();
#     else
      __threadfence(); __trap();
#     endif
      __builtin_unreachable();
    }

#   ifdef __HIP_DEVICE_COMPILE__
#   define ONIKA_CU_GRID_CONSTANT       /**/
#   else
#   define ONIKA_CU_GRID_CONSTANT       __grid_constant__
#   endif

#   define ONIKA_DEVICE_CONSTANT_MEMORY __constant__
#   define ONIKA_CU_THREAD_LOCAL        __local__
#   define ONIKA_CU_BLOCK_SHARED        __shared__
#   define ONIKA_CU_GLOBAL_VARIABLE     __managed__

#   define ONIKA_CU_BLOCK_SIMD_FOR(T,i,s,e)           for(T i=s+ONIKA_CU_THREAD_IDX ; i<e ; i+=ONIKA_CU_BLOCK_SIZE )
#   define ONIKA_CU_BLOCK_SIMD_FOR_UNGUARDED(T,i,s,e) for(T _onika_tmp_j=s , i=s+ONIKA_CU_THREAD_IDX ; _onika_tmp_j<e ; _onika_tmp_j+=ONIKA_CU_BLOCK_SIZE, i+=ONIKA_CU_BLOCK_SIZE )

#   define ONIKA_CU_BLOCK_SYNC()      __syncthreads()
#   define ONIKA_CU_BLOCK_SYNC_OR(p)  __syncthreads_or(p)
#   define ONIKA_CU_BLOCK_SYNC_AND(p) __syncthreads_and(p)
#   define ONIKA_CU_BLOCK_FENCE()     __threadfence_block()

#   define ONIKA_CU_DEVICE_FENCE() __threadfence()
#   define ONIKA_CU_SYSTEM_FENCE() __threadfence_system()

#   define ONIKA_CU_WARP_SYNC()    __syncwarp()
#   define ONIKA_CU_WARP_SHFL_DOWN_SYNC(mask,var,delta,width) __shfl_down_sync(mask,var,delta,width)
#   define ONIKA_CU_BLOCK_ACTIVE_MASK() __activemask()

#   define ONIKA_CU_MEM_ORDER_RELEASE std::memory_order_release
#   define ONIKA_CU_MEM_ORDER_RELAXED std::memory_order_relaxed
#   define ONIKA_CU_MEM_ORDER_ACQUIRE std::memory_order_acquire
#   define ONIKA_CU_MEM_ORDER_SEQ_CST std::memory_order_seq_cst

#   define ONIKA_CU_ATOMIC_STORE(x,a,...) ( * (volatile std::remove_reference_t<decltype(x)> *) &(x) ) = (a)
#   define ONIKA_CU_ATOMIC_LOAD(x,...)   ( * (volatile const std::remove_reference_t<decltype(x)> *) &(x) )
#   define ONIKA_CU_ATOMIC_ADD(x,a,...)  atomicAdd( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_ATOMIC_SUB(x,a,...)  atomicSub( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_ATOMIC_MIN(x,a,...)  atomicMin( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_ATOMIC_MAX(x,a,...)  atomicMax( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_BLOCK_ATOMIC_ADD(x,a) atomicAdd( &(x) , static_cast<std::remove_reference_t<decltype(x)> >(a) )

#   define ONIKA_CU_ATOMIC_FLAG_TEST_AND_SET(f) ( ! atomicCAS((uint32_t*)&(f),0,1) )
#   define ONIKA_CU_ATOMIC_FLAG_CLEAR(f) * (volatile uint32_t *) &(f) = 0

#   define ONIKA_CU_WARP_ACTIVE_THREADS_ALL(pred) __all_sync( __activemask(), int(pred) )
#   define ONIKA_CU_WARP_ACTIVE_THREADS_ANY(pred) __any_sync( __activemask(), int(pred) )

#   if defined(__HIP_DEVICE_COMPILE__)
    __device__ inline void _hip_nanosleep(long long nanosecs)
    {
      static constexpr long long ticks_per_nanosec = 1;
      const long long start = clock64();
      while( ( clock64() - start ) < (nanosecs*ticks_per_nanosec) );
    }
#   define ONIKA_CU_NANOSLEEP(ns) ::onika::cuda::_hip_nanosleep(ns)
#   else
#   define ONIKA_CU_NANOSLEEP(ns) __nanosleep(ns)
#   endif

#   define ONIKA_CU_CLOCK() clock64()
#   define ONIKA_CU_CLOCK_ELAPSED(a,b) ((b)-(a))
    using onika_cu_clock_t = long long int;

#   define ONIKA_CU_GRID_DIMS    gridDim
#   define ONIKA_CU_GRID_SIZE    (gridDim.x*gridDim.y*gridDim.z)
#   define ONIKA_CU_BLOCK_COORD  blockIdx
#   define ONIKA_CU_BLOCK_IDX    ( ( ( (blockIdx.z*gridDim.y) + blockIdx.y ) * gridDim.x ) + blockIdx.x )
#   define ONIKA_CU_BLOCK_DIMS   blockDim
#   define ONIKA_CU_BLOCK_SIZE   (blockDim.x*blockDim.y*blockDim.z)
#   define ONIKA_CU_THREAD_COORD threadIdx
#   define ONIKA_CU_THREAD_IDX   ( ( ( (threadIdx.z*blockDim.y) + threadIdx.y ) * blockDim.x ) + threadIdx.x )

#   define ONIKA_CU_VALUE_IF_CUDA(a,b) (a)

#   define ONIKA_CU_ABORT() ::onika::cuda::__onika_cu_abort()

/************** end of Device code definitions ***************/
#else 
/************** start of HOST code definitions ***************/

#   define ONIKA_GPU_DEVICE_EXECUTION_TYPE onika::FalseType

#   define ONIKA_CU_GRID_CONSTANT       /**/
#   define ONIKA_DEVICE_CONSTANT_MEMORY /**/
#   define ONIKA_CU_BLOCK_SHARED        /**/
#   define ONIKA_CU_THREAD_LOCAL        /**/
#   define ONIKA_CU_GLOBAL_VARIABLE     /**/

#   define ONIKA_CU_BLOCK_SIMD_FOR_UNGUARDED ONIKA_CU_BLOCK_SIMD_FOR
#   define ONIKA_CU_BLOCK_SIMD_FOR(T,i,s,e) _Pragma("omp simd") for(T i=s ; i<e ; ++i)

#   define ONIKA_CU_BLOCK_SYNC()      (void)0
#   define ONIKA_CU_BLOCK_FENCE()     (void)0
#   define ONIKA_CU_BLOCK_WARP_SYNC() (void)0

#   define ONIKA_CU_DEVICE_FENCE() std::atomic_thread_fence(ONIKA_CU_MEM_ORDER_SEQ_CST)
#   define ONIKA_CU_SYSTEM_FENCE() std::atomic_thread_fence(ONIKA_CU_MEM_ORDER_SEQ_CST)

#   define ONIKA_CU_WARP_SHFL_DOWN_SYNC(mask,var,delta,width) (var)
#   define ONIKA_CU_BLOCK_ACTIVE_MASK() 1u

#   define ONIKA_CU_MEM_ORDER_RELEASE std::memory_order_release
#   define ONIKA_CU_MEM_ORDER_RELAXED std::memory_order_relaxed
#   define ONIKA_CU_MEM_ORDER_ACQUIRE std::memory_order_acquire
#   define ONIKA_CU_MEM_ORDER_SEQ_CST std::memory_order_seq_cst

#   define ONIKA_CU_ATOMIC_STORE(x,a,...) reinterpret_cast<std::atomic<std::remove_reference_t<decltype(x)> >*>(&(x))->store(a OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#   define ONIKA_CU_ATOMIC_LOAD(x,...) reinterpret_cast< const std::atomic<std::remove_cv_t<std::remove_reference_t<decltype(x)> > > * >(&(x))->load( __VA_ARGS__ )
#   define ONIKA_CU_ATOMIC_ADD(x,a,...) ::onika::capture_atomic_add( x , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_ATOMIC_SUB(x,a,...) ::onika::capture_atomic_sub( x , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_ATOMIC_MIN(x,a,...) ::onika::capture_atomic_min( x , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_ATOMIC_MAX(x,a,...) ::onika::capture_atomic_max( x , static_cast<std::remove_reference_t<decltype(x)> >(a) )
#   define ONIKA_CU_BLOCK_ATOMIC_ADD(x,a) (x) += (a)

#   define ONIKA_CU_ATOMIC_FLAG_TEST_AND_SET(f) ( ! (f).m_flag.test_and_set(std::memory_order_acquire) )
#   define ONIKA_CU_ATOMIC_FLAG_CLEAR(f) (f).m_flag.clear(std::memory_order_release)

#   define ONIKA_CU_WARP_ACTIVE_THREADS_ALL(pred) (pred)
#   define ONIKA_CU_WARP_ACTIVE_THREADS_ANY(pred) (pred)

#   define ONIKA_CU_NANOSLEEP(ns) std::this_thread::sleep_for(std::chrono::nanoseconds(ns))

#   define ONIKA_CU_CLOCK() std::chrono::high_resolution_clock::now()
#   define ONIKA_CU_CLOCK_ELAPSED(a,b) std::chrono::duration<double,std::micro>((b)-(a)).count()
    using onika_cu_clock_t = decltype(std::chrono::high_resolution_clock::now());

#   define ONIKA_CU_GRID_SIZE    omp_get_num_threads()
#   define ONIKA_CU_GRID_DIMS    onikaDim3_t{ static_cast<unsigned int>(omp_get_num_threads()),1,1} // should never be used
#   define ONIKA_CU_BLOCK_IDX    static_cast<unsigned int>(omp_get_thread_num())
#   define ONIKA_CU_BLOCK_COORD  onikaDim3_t{static_cast<unsigned int>(omp_get_thread_num()),0,0}  // should never be used
#   define ONIKA_CU_BLOCK_SIZE   1
#   define ONIKA_CU_BLOCK_DIMS   onikaDim3_t{1,1,1}
#   define ONIKA_CU_THREAD_IDX   0
#   define ONIKA_CU_THREAD_COORD onikaDim3_t{0,0,0}

#   define ONIKA_CU_VALUE_IF_CUDA(a,b) (b)

#   define ONIKA_CU_ABORT() std::abort()

/************** end of HOST code definitions ***************/
#endif 


/***************************************************************/
/******************** Cuda language support ********************/
/***************************************************************/


/************** begin cuda-c code definitions ***************/
#   if defined(__CUDACC__) || defined(__HIPCC__)

#   define ONIKA_GPU_FRONTEND_COMPILER onika::TrueType // code being compiled uses GPU fronted compiler, thus supports GPU Kernel execution

#   define ONIKA_DEVICE_KERNEL_FUNC __global__
#   ifdef __HIPCC__
#   define ONIKA_STATIC_INLINE_KERNEL static
#   else
#   define ONIKA_STATIC_INLINE_KERNEL [[maybe_unused]] static
#   endif

#   if ONIKA_CU_ENABLE_KERNEL_BOUNDS == 1
#   define ONIKA_DEVICE_KERNEL_BOUNDS(MaxThreadsPerBlock,MinBlocksPerSM) __launch_bounds__(MaxThreadsPerBlock,MinBlocksPerSM)
#   else
#   define ONIKA_DEVICE_KERNEL_BOUNDS(MaxThreadsPerBlock,MinBlocksPerSM) /**/
#   endif
#   define ONIKA_HOST_FUNC __host__ 
#   define ONIKA_DEVICE_FUNC __device__
#   define ONIKA_HOST_DEVICE_FUNC __host__ __device__
#   define ONIKA_BUILTIN_ASSUME_ALIGNED(p,a) p
#   define ONIKA_CU_LAUNCH_KERNEL(GDIM,BDIM,SHMEM,STREAM,FUNC, ... ) FUNC <<< GDIM , BDIM , SHMEM , STREAM >>> ( __VA_ARGS__ )

/************** end cuda-c code definitions ***************/
#   else
/************** begin of HOST code definitions ***************/

#   define ONIKA_GPU_FRONTEND_COMPILER onika::FalseType // host compiler (not nvccc nor hipcc) is used, thus it is not possible to run GPU Kernels

#   define ONIKA_HOST_FUNC /**/
#   define ONIKA_DEVICE_FUNC /**/
#   define ONIKA_HOST_DEVICE_FUNC /**/
#   define ONIKA_STATIC_INLINE_KERNEL static inline
#   define ONIKA_DEVICE_KERNEL_FUNC /**/
#   define ONIKA_DEVICE_KERNEL_BOUNDS(MaxThreadsPerBlock,MinBlocksPerSM) /**/
#   define ONIKA_BUILTIN_ASSUME_ALIGNED(p,a) __builtin_assume_aligned(p,a)
#   define ONIKA_CU_LAUNCH_KERNEL(GDIM,BDIM,SHMEM,STREAM,FUNC, ... ) do{ \
	FAKE_USE_OF_VARIABLES(__VA_ARGS__) \
	printf("Illegal call to Kernel %s<<<%d,%d,%d,%ld>>>(...) with no GPU support, in %s at %s:%d\n",#FUNC,int(GDIM),int(BDIM),int(SHMEM),long(STREAM),__FUNCTION__,__FILE__,__LINE__); \
	std::abort(); \
    }while(false)
#   endif
/************** end host code definitions ***************/



/***************************************************************/
/****************** Cuda hardware intrinsics *******************/
/***************************************************************/

#   if defined(__CUDA_ARCH__)
    __noinline__  __device__ inline unsigned int get_smid(void)
    {
      unsigned int ret;
      asm("mov.u32 %0, %smid;" : "=r"(ret) );
      return ret;
    }
    
    ONIKA_DEVICE_FUNC inline long long int onika_double_as_longlong(double x) { return __double_as_longlong(x); }
    ONIKA_DEVICE_FUNC inline double onika_longlong_as_double(long long int x) { return __longlong_as_double(x); }
    
#   else

    inline constexpr unsigned int get_smid(void) { return 0; }
    
    // this might be unefficient as hell, but it doesn't break strict aliasing rules
    ONIKA_DEVICE_FUNC inline long long int onika_double_as_longlong(double x)
    {
      long long int x_ll = 0;
      static_assert( sizeof(x) == sizeof(x_ll) );
      memcpy( & x_ll , & x , sizeof(x) );
      return x_ll;
    }
    ONIKA_DEVICE_FUNC inline double onika_longlong_as_double(long long int x_ll)
    {
      double x = 0.0;
      static_assert( sizeof(x) == sizeof(x_ll) );
      memcpy( & x , & x_ll , sizeof(x_ll) );
      return x;
    }
    
#   endif

    template<class T>
    struct alignas( alignof(T) ) UnitializedPlaceHolder
    {
      static_assert( sizeof(unsigned char) == 1 , "expected char to be 1 byte" );
      unsigned char byte[sizeof(T)];
      ONIKA_HOST_DEVICE_FUNC inline T& get_ref() { return * reinterpret_cast<T*>(byte); }
    };
  }
}

/***************************************************************/
/************** Cuda extended atomic operations ****************/
/***************************************************************/
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifndef ONIKA_HAS_GPU_ATOMIC_MIN_MAX_DOUBLE
ONIKA_DEVICE_FUNC inline double atomicMin(double* xp , double y)
{
  using onika::cuda::onika_double_as_longlong;
  using onika::cuda::onika_longlong_as_double;
  const unsigned long long int y_ull = onika_double_as_longlong( y );
  unsigned long long int x_ull = atomicAdd( (unsigned long long int *) xp , 0ull ); // this ensures a portable way to perform an atomic fetch of value at address xp
  double x = onika_longlong_as_double( x_ull );
  while( x > y )
  {
    x_ull = atomicCAS( (unsigned long long int *) xp , x_ull , y_ull );
    x = onika_longlong_as_double( x_ull );
  }
  return x;
}
ONIKA_DEVICE_FUNC inline double atomicMax(double* xp , double y)
{
  using onika::cuda::onika_double_as_longlong;
  using onika::cuda::onika_longlong_as_double;
  const unsigned long long int y_ull = onika_double_as_longlong( y );
  unsigned long long int x_ull = atomicAdd( (unsigned long long int *) xp , 0ull ); // this ensures a portable way to perform an atomic fetch of value at address xp
  double x = onika_longlong_as_double( x_ull );
  while( x < y )
  {
    x_ull = atomicCAS( (unsigned long long int *) xp , x_ull , y_ull );
    x = onika_longlong_as_double( x_ull );
  }
  return x;
}
#endif // if hardware atomicMin / atomicMax are missing
#endif // device code only



// user helpers to select implementations depending on Cuda or Host execution space
#include <onika/integral_constant.h>
namespace onika
{
  namespace cuda
  {
    [[ deprecated ]]
    typedef ONIKA_GPU_DEVICE_EXECUTION_TYPE gpu_device_execution_t; // DO NOT use this one anymore, it breaks C++'s ODF

#   define gpu_device_execution() ONIKA_GPU_DEVICE_EXECUTION_TYPE{}
#   define gpu_frontend_compiler() ONIKA_GPU_FRONTEND_COMPILER{}
  }
}

