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
#include <onika/cuda/cuda_context.h>

// specializations to avoid MemoryUsage template to dig into cuda aggregates
namespace onika
{

  namespace cuda
  {

    bool CudaContext::s_global_gpu_enable = true;

    std::shared_ptr<CudaContext> CudaContext::s_default_cuda_ctx = nullptr;

    void CudaContext::set_global_gpu_enable(bool yn)
    {
      s_global_gpu_enable = yn;
    }

    bool CudaContext::global_gpu_enable()
    {
      return s_global_gpu_enable;
    }

    CudaContext* CudaContext::default_cuda_ctx()
    {
      if( ! global_gpu_enable() ) return nullptr;
      if( s_default_cuda_ctx == nullptr )
      {
        s_default_cuda_ctx = std::make_shared<CudaContext>();
        s_default_cuda_ctx->m_devices.resize(1);
        s_default_cuda_ctx->m_devices[0].device_id = 0;
        ONIKA_CU_CHECK_ERRORS( ONIKA_CU_GET_DEVICE_PROPERTIES( & s_default_cuda_ctx->m_devices[0].m_deviceProp , s_default_cuda_ctx->m_devices[0].device_id ) );
        ONIKA_CU_CHECK_ERRORS( ONIKA_CU_GET_DEVICE_ATTRIBUTE( & s_default_cuda_ctx->m_devices[0].m_clock_rate , onikaDevAttrClockRate, s_default_cuda_ctx->m_devices[0].device_id ) );
      }
      return s_default_cuda_ctx.get();
    }

    bool CudaContext::has_devices() const
    {
      return ! m_devices.empty();
    }

    unsigned int CudaContext::device_count() const
    {
      return m_devices.size();
    }

    onikaStream_t CudaContext::getThreadStream(unsigned int tid)
    {
      if( tid >= m_threadStream.size() )
      {
        unsigned int i = m_threadStream.size();
        m_threadStream.resize( tid+1 , 0 );
        for(;i<m_threadStream.size();i++)
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_CREATE_STREAM_NON_BLOCKING( m_threadStream[i] ) );
        }
      }
      return m_threadStream[tid];
    }

  }

}


