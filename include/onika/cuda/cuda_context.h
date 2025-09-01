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

// universal no-op function, takes any arguments. returns SUCCESS.
template<class... AnyArgs> static inline constexpr int _fake_cuda_api_noop(AnyArgs...){return 0;}


/***************************************************************/
/************************ Cuda API calls ***********************/
/***************************************************************/

#ifdef ONIKA_CUDA_VERSION

#ifdef ONIKA_HIP_VERSION

// HIP runtime API
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <roctracer/roctx.h>
#define ONIKA_CU_PROF_RANGE_PUSH(s)                    roctxRangePush(s)
#define ONIKA_CU_PROF_RANGE_POP()                      roctxRangePop()
#define ONIKA_CU_PROF_START()                          hipProfilerStart()
#define ONIKA_CU_PROF_STOP()                           hipProfilerStop()
#define ONIKA_CU_MEM_PREFETCH(ptr,sz,d,st)             hipMemPrefetchAsync((const void*)(ptr),sz,0,st)
#define ONIKA_CU_CREATE_STREAM_NON_BLOCKING(streamref) hipStreamCreateWithFlags( & streamref, hipStreamNonBlocking )
#define ONIKA_CU_STREAM_ADD_CALLBACK(stream,cb,udata)  hipStreamAddCallback(stream,cb,udata,0u)
#define ONIKA_CU_STREAM_SYNCHRONIZE(STREAM)            hipStreamSynchronize(STREAM)
#define ONIKA_CU_DESTROY_STREAM(streamref)             hipStreamDestroy(streamref)
#define ONIKA_CU_EVENT_QUERY(evt)                      hipEventQuery(evt)
#define ONIKA_CU_MALLOC(devPtrPtr,N)                   hipMalloc(devPtrPtr,N)
#define ONIKA_CU_MALLOC_MANAGED(devPtrPtr,N)           hipMallocManaged(devPtrPtr,N)
#define ONIKA_CU_FREE(devPtr)                          hipFree(devPtr)
#define ONIKA_CU_CREATE_EVENT(EVT)                     hipEventCreate(&EVT)
#define ONIKA_CU_DESTROY_EVENT(EVT)                    hipEventDestroy(EVT)
#define ONIKA_CU_STREAM_EVENT(EVT,STREAM)              hipEventRecord(EVT,STREAM)
#define ONIKA_CU_STREAM_HOST_FUNC(s,f,p)               hipLaunchHostFunc(s,f,p)
#define ONIKA_CU_EVENT_ELAPSED(T,EVT1,EVT2)            hipEventElapsedTime(&T,EVT1,EVT2)
#define ONIKA_CU_MEMSET(p,v,n,...)                     hipMemsetAsync(p,v,n OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_MEMCPY(d,s,n,...)                     hipMemcpyAsync(d,s,n,hipMemcpyDefault OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_MEMCPY_TO_SYMBOL(d,s,n,...)           hipMemcpyToSymbol(d,s,n,hipMemcpyHostToDevice OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_MEMCPY_KIND(d,s,n,k,...)              hipMemcpyAsync(d,s,n,k OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_GET_DEVICE_COUNT(iPtr)                hipGetDeviceCount(iPtr)
#define ONIKA_CU_SET_DEVICE(id)                        hipSetDevice(id)
#define ONIKA_CU_SET_SHARED_MEM_CONFIG(shmc)           hipDeviceSetSharedMemConfig(shmc)
#define ONIKA_CU_SET_LIMIT(l,v)                        hipDeviceSetLimit(l,v)
#define ONIKA_CU_GET_LIMIT(vptr,l)                     hipDeviceGetLimit(vptr,l)
#define ONIKA_CU_GET_DEVICE_PROPERTIES(propPtr,id)     hipGetDeviceProperties(propPtr,id)
#define ONIKA_CU_GET_DEVICE_ATTRIBUTE(valPtr,attr,id)  hipDeviceGetAttribute(valPtr,attr,id)
#define ONIKA_CU_DEVICE_SYNCHRONIZE()                  hipDeviceSynchronize()
#define ONIKA_CU_GET_ERROR_STRING(c)                   hipGetErrorString(code)
#define ONIKA_CU_NAME_STR                              "hip"
using onikaDeviceProp_t = hipDeviceProp_t;
using onikaStream_t     = hipStream_t;
using onikaEvent_t      = hipEvent_t;
using onikaError_t      = hipError_t;
using onikaLimit_t      = hipLimit_t;
using onikaDim3_t       = dim3;
static inline constexpr auto onikaSuccess                    = hipSuccess;
static inline constexpr auto onikaSharedMemBankSizeFourByte  = hipSharedMemBankSizeFourByte;
static inline constexpr auto onikaSharedMemBankSizeEightByte = hipSharedMemBankSizeEightByte;
static inline constexpr auto onikaSharedMemBankSizeDefault   = hipSharedMemBankSizeDefault;
static inline constexpr auto onikaLimitStackSize             = hipLimitStackSize;
static inline constexpr auto onikaLimitPrintfFifoSize        = hipLimitPrintfFifoSize;
static inline constexpr auto onikaLimitMallocHeapSize        = hipLimitMallocHeapSize;
static inline constexpr auto onikaMemcpyDeviceToHost         = hipMemcpyDeviceToHost;
static inline constexpr auto onikaMemcpyHostToDevice         = hipMemcpyHostToDevice;
static inline constexpr auto onikaDevAttrClockRate           = hipDevAttrClockRate;

#else

// Cuda runtime API
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#define ONIKA_CU_PROF_RANGE_PUSH(s)                    nvtxRangePush(s)
#define ONIKA_CU_PROF_RANGE_POP()                      nvtxRangePop()
#define ONIKA_CU_PROF_START()                          cudaProfilerStart()
#define ONIKA_CU_PROF_STOP()                           cudaProfilerStop()
#define ONIKA_CU_MEM_PREFETCH(ptr,sz,d,st)             cudaMemPrefetchAsync((const void*)(ptr),sz,0,st)
#define ONIKA_CU_CREATE_STREAM_NON_BLOCKING(streamref) cudaStreamCreateWithFlags( & streamref, cudaStreamNonBlocking )
#define ONIKA_CU_STREAM_ADD_CALLBACK(stream,cb,udata)  cudaStreamAddCallback(stream,cb,udata,0u)
#define ONIKA_CU_STREAM_SYNCHRONIZE(STREAM)            cudaStreamSynchronize(STREAM)
#define ONIKA_CU_DESTROY_STREAM(streamref)             cudaStreamDestroy(streamref)
#define ONIKA_CU_EVENT_QUERY(evt)                      cudaEventQuery(evt)
#define ONIKA_CU_MALLOC(devPtrPtr,N)                   cudaMalloc(devPtrPtr,N)
#define ONIKA_CU_MALLOC_MANAGED(devPtrPtr,N)           cudaMallocManaged(devPtrPtr,N)
#define ONIKA_CU_FREE(devPtr)                          cudaFree(devPtr)
#define ONIKA_CU_CREATE_EVENT(EVT)                     cudaEventCreate(&EVT)
#define ONIKA_CU_DESTROY_EVENT(EVT)                    cudaEventDestroy(EVT)
#define ONIKA_CU_STREAM_EVENT(EVT,STREAM)              cudaEventRecord(EVT,STREAM)
#define ONIKA_CU_STREAM_HOST_FUNC(s,f,p)               cudaLaunchHostFunc(s,f,p)
#define ONIKA_CU_EVENT_ELAPSED(T,EVT1,EVT2)            cudaEventElapsedTime(&T,EVT1,EVT2)
#define ONIKA_CU_MEMSET(p,v,n,...)                     cudaMemsetAsync(p,v,n OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_MEMCPY(d,s,n,...)                     cudaMemcpyAsync(d,s,n,cudaMemcpyDefault OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_MEMCPY_TO_SYMBOL(d,s,n,...)           cudaMemcpyToSymbol(d,s,n,cudaMemcpyHostToDevice OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_MEMCPY_KIND(d,s,n,k,...)              cudaMemcpyAsync(d,s,n,k OPT_COMMA_VA_ARGS(__VA_ARGS__) )
#define ONIKA_CU_GET_DEVICE_COUNT(iPtr)                cudaGetDeviceCount(iPtr)
#define ONIKA_CU_SET_DEVICE(id)                        cudaSetDevice(id)
#define ONIKA_CU_SET_SHARED_MEM_CONFIG(shmc)           cudaDeviceSetSharedMemConfig(shmc)
#define ONIKA_CU_SET_LIMIT(l,v)                        cudaDeviceSetLimit(l,v)
#define ONIKA_CU_GET_LIMIT(vptr,l)                     cudaDeviceGetLimit(vptr,l)
#define ONIKA_CU_GET_DEVICE_PROPERTIES(propPtr,id)     cudaGetDeviceProperties(propPtr,id)
#define ONIKA_CU_GET_DEVICE_ATTRIBUTE(valPtr,attr,id)  cudaDeviceGetAttribute(valPtr,attr,id)
#define ONIKA_CU_DEVICE_SYNCHRONIZE()                  cudaDeviceSynchronize()
#define ONIKA_CU_GET_ERROR_STRING(c)                   cudaGetErrorString(code)
#define ONIKA_CU_NAME_STR                              "cuda"
using onikaDeviceProp_t = cudaDeviceProp;
using onikaStream_t     = cudaStream_t;
using onikaEvent_t      = cudaEvent_t;
using onikaError_t      = cudaError_t;
using onikaLimit_t      = cudaLimit;
using onikaDim3_t       = dim3;
static inline constexpr auto onikaSuccess                    = cudaSuccess;
static inline constexpr auto onikaSharedMemBankSizeFourByte  = cudaSharedMemBankSizeFourByte;
static inline constexpr auto onikaSharedMemBankSizeEightByte = cudaSharedMemBankSizeEightByte;
static inline constexpr auto onikaSharedMemBankSizeDefault   = cudaSharedMemBankSizeDefault;
static inline constexpr auto onikaLimitStackSize             = cudaLimitStackSize;
static inline constexpr auto onikaLimitPrintfFifoSize        = cudaLimitPrintfFifoSize;
static inline constexpr auto onikaLimitMallocHeapSize        = cudaLimitMallocHeapSize;
static inline constexpr auto onikaMemcpyDeviceToHost         = cudaMemcpyDeviceToHost;
static inline constexpr auto onikaMemcpyHostToDevice         = cudaMemcpyHostToDevice;
static inline constexpr auto onikaDevAttrClockRate           = cudaDevAttrClockRate;

#endif

#else

struct onikaDeviceProp_t
{
  char name[256];
  int managedMemory = 0;
  int concurrentManagedAccess = 0;
  int totalGlobalMem = 0;
  int warpSize = 0;
  int multiProcessorCount = 0;
  int sharedMemPerBlock = 0;
};
struct onikaDim3_t
{
  unsigned int x = 0;
  unsigned int y = 0;
  unsigned int z = 0;
  inline constexpr onikaDim3_t operator + (const onikaDim3_t& rhs) const { return { x+rhs.x , y+rhs.y , z+rhs.z }; }
};

using onikaStream_t = int;
using onikaEvent_t = int*;
using onikaError_t = int;

static inline constexpr int onikaSuccess = 0;
static inline constexpr int onikaErrorNotReady = 0;

#define cudaEventQuery _fake_cuda_api_noop
#define cudaStreamAddCallback _fake_cuda_api_noop
#define cudaStreamCreate _fake_cuda_api_noop

#define ONIKA_CU_PROF_RANGE_PUSH             _fake_cuda_api_noop
#define ONIKA_CU_PROF_RANGE_POP              _fake_cuda_api_noop
#define ONIKA_CU_PROF_START                  _fake_cuda_api_noop
#define ONIKA_CU_PROF_STOP                   _fake_cuda_api_noop
#define ONIKA_CU_MEM_PREFETCH                _fake_cuda_api_noop
#define ONIKA_CU_CREATE_STREAM_NON_BLOCKING  _fake_cuda_api_noop
#define ONIKA_CU_STREAM_ADD_CALLBACK         _fake_cuda_api_noop
#define ONIKA_CU_CREATE_EVENT(EVT)           _fake_cuda_api_noop(EVT=nullptr)
#define ONIKA_CU_DESTROY_EVENT(EVT)          _fake_cuda_api_noop(EVT=nullptr)
#define ONIKA_CU_STREAM_EVENT(EVT,STREAM)    _fake_cuda_api_noop(EVT,STREAM)
#define ONIKA_CU_STREAM_HOST_FUNC(s,f,p)     _fake_cuda_api_noop(s,f,p)
#define ONIKA_CU_EVENT_ELAPSED(T,EVT1,EVT2)  _fake_cuda_api_noop(T=0.0f)
#define ONIKA_CU_DEVICE_SYNCHRONIZE()        _fake_cuda_api_noop()
#define ONIKA_CU_STREAM_SYNCHRONIZE(STREAM)  _fake_cuda_api_noop(STREAM)
#define ONIKA_CU_GET_DEVICE_PROPERTIES       _fake_cuda_api_noop
#define ONIKA_CU_EVENT_QUERY(EVT)            (onikaSuccess)
#define ONIKA_CU_MEMSET(p,v,n,...)           std::memset(p,v,n)
#define ONIKA_CU_MEMCPY(d,s,n,...)           std::memcpy(d,s,n)
#define ONIKA_CU_MEMCPY_TO_SYMBOL(d,s,n,...) std::memcpy(d,s,n)
#define ONIKA_CU_GET_DEVICE_COUNT(iPtr)      _fake_cuda_api_noop(*iPtr=0)
#define ONIKA_CU_MALLOC(devPtrPtr,N)         _fake_cuda_api_noop(*(void**)devPtrPtr=malloc(N))
#define ONIKA_CU_MALLOC_MANAGED(devPtrPtr,N) _fake_cuda_api_noop(*(void**)devPtrPtr=malloc(N))
#define ONIKA_CU_FREE(devPtr)                _fake_cuda_api_noop( (free(devPtr),0) )
#define ONIKA_CU_NAME_STR                    "vgpu"
#endif // ONIKA_CUDA_VERSION


#include <onika/memory/memory_usage.h>
#include <onika/cuda/cuda_error.h>
#include <onika/string_utils.h>
#include <functional>
#include <list>


struct onikaInt3_t
{
  ssize_t x = 0;
  ssize_t y = 0;
  ssize_t z = 0;
  ONIKA_HOST_DEVICE_FUNC
  inline constexpr onikaInt3_t operator + (const onikaDim3_t& rhs) const { return { x+rhs.x , y+rhs.y , z+rhs.z }; }
};

// specializations to avoid MemoryUsage template to dig into cuda aggregates
namespace onika
{

  namespace memory
  {
    template<> struct MemoryUsage<onikaDeviceProp_t>
    {
      static inline constexpr size_t memory_bytes(const onikaDeviceProp_t&) { return sizeof(onikaDeviceProp_t); }
    };
    template<> struct MemoryUsage<onikaStream_t>
    {
      static inline constexpr size_t memory_bytes(const onikaStream_t&) { return sizeof(onikaStream_t); }
    };
  }

  namespace cuda
  {

    inline constexpr long onika_dim3_size(const onikaDim3_t& d) { return d.x*d.y*d.z; }
    inline constexpr long onika_dim3_size(long d) { return d; }

    struct CudaDevice
    {
      onikaDeviceProp_t m_deviceProp;
      std::list< std::function<void()> > m_finalize_destructors;
      int device_id = 0;
      int m_clock_rate = 0;
    };

    struct CudaContext
    {
      std::vector<CudaDevice> m_devices;
      std::vector<onikaStream_t> m_threadStream;

      static bool s_global_gpu_enable;
      static std::shared_ptr<CudaContext> s_default_cuda_ctx;

      bool has_devices() const;
      unsigned int device_count() const;
      onikaStream_t getThreadStream(unsigned int tid);

      template<class StreamT>
      inline StreamT& to_stream(StreamT& out)
      {
        out << "=========== "<<ONIKA_CU_NAME_STR<<" ================"<<std::endl;
        if( ! m_devices.empty() )
        {
          const auto & dev = m_devices.front();
          out <<"GPUs : "<<m_devices.size()<< std::endl
              <<"Type : "<<dev.m_deviceProp.name << std::endl
              <<"SMs  : "<<dev.m_deviceProp.multiProcessorCount<<"x"<<dev.m_deviceProp.warpSize<<" threads @ "
                         << std::defaultfloat<< dev.m_clock_rate/1000000.0<<" Ghz" << std::endl
              <<"Mem  : "<< onika::memory_bytes_string(dev.m_deviceProp.totalGlobalMem)
                         <<" (shared "<<onika::memory_bytes_string(dev.m_deviceProp.sharedMemPerBlock,"%g %s")
                         <<" , L2 "<<onika::memory_bytes_string(dev.m_deviceProp.persistingL2CacheMaxSize,"%.2g %s")<<")" <<std::endl
              << "================================="<<std::endl<<std::endl;
       }
        else
        {
          out <<"No GPU found"<<std::endl;
        }
        return out;
      }

      static void set_global_gpu_enable(bool yn);
      static bool global_gpu_enable();
      static CudaContext* default_cuda_ctx();
    };

  }

}


