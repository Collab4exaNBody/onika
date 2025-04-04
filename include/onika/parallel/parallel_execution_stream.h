#pragma once

#include <onika/cuda/cuda_context.h>
#include <onika/parallel/stream_utils.h>
#include <mutex>
#include <atomic>
#include <functional>

namespace onika
{

  namespace parallel
  {

    // allows asynchronous sequential execution of parallel executions queued in the same stream
    // multiple kernel execution concurrency can be handled manually using several streams (same as Cuda stream)
    struct ParallelExecutionStream
    {
      // GPU device context, null if non device available for parallel execution
      // any parallel executiion enqueued to this stream must have either a null CudaContext or the same context as the stream
      onika::cuda::CudaContext* m_cuda_ctx = nullptr; 
      onikaStream_t m_cu_stream = 0;
      uint32_t m_stream_id = 0;
      std::atomic<uint32_t> m_omp_execution_count = 0;
      std::mutex m_mutex;
      
      inline void wait()
      {
        std::lock_guard lk( m_mutex );
        wait_nolock();
      }

      inline void wait_nolock()
      {
        // OpenMP wait
        if( m_omp_execution_count.load() > 0 )
        {
          auto * st = this;
#         pragma omp task default(none) firstprivate(st) depend(in:st[0]) if(0)
          {
            int n = st->m_omp_execution_count.load();
            if( n > 0 )
            {
              fatal_error()<<"Internal error : unterminated OpenMP tasks ("<<n<<") remain in queue"<<std::endl;
            }
          }
        }
        
        // Cuda wait
        if( m_cuda_ctx != nullptr )
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_SYNCHRONIZE( m_cu_stream ) );
        }
      }
    };
    
    using ParallelExecutionStreamAllocator = std::function<ParallelExecutionStream*(int)>;

  }

}

