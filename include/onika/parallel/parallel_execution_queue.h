#pragma once

#include <onika/cuda/cuda_context.h>
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/block_parallel_for_functor.h>
#include <mutex>
#include <atomic>

namespace onika
{

  namespace parallel
  {

    struct ParallelExecutionQueue
    {
      ParallelExecutionStream* m_stream = nullptr;        // execution stream to schedule paralel operations
      ParallelExecutionContext* m_exec_ctx = nullptr;     // head of parallel operations to schedule (following item chained through m_exec_ctx->m_next)
      ParallelExecutionQueue() = default;
      
      inline ParallelExecutionQueue(ParallelExecutionStream* st) : m_stream(st) , m_exec_ctx(nullptr) {}
      
      inline ParallelExecutionQueue(ParallelExecutionQueue && o)
        : m_stream( std::move(o.m_stream) )
        , m_exec_ctx( std::move(o.m_exec_ctx) )
      {
        o.m_stream = nullptr;
        o.m_exec_ctx = nullptr;
      }
      
      inline ParallelExecutionQueue& operator = (ParallelExecutionQueue && o)
      {
        wait();
        m_stream = std::move(o.m_stream);
        m_exec_ctx = std::move(o.m_exec_ctx);
        o.m_stream = nullptr;
        o.m_exec_ctx = nullptr;
        return *this;
      }
         
      inline ~ParallelExecutionQueue()
      {
        wait();
        m_stream = nullptr;
      }
      
      inline void wait()
      {
        if( m_stream != nullptr && m_exec_ctx != nullptr )
        {
          std::lock_guard lk( m_stream->m_mutex );

          // synchronize stream
          m_stream->wait_nolock();
          
          // collect execution times
          auto* pec = m_exec_ctx;
          while(pec!=nullptr)
          {
            float Tgpu = 0.0;
            if( pec->m_execution_target == ParallelExecutionContext::EXECUTION_TARGET_CUDA )
            {
              ONIKA_CU_CHECK_ERRORS( ONIKA_CU_EVENT_ELAPSED(Tgpu,pec->m_start_evt,pec->m_stop_evt) );
              pec->m_total_gpu_execution_time = Tgpu;
            }
            auto* next = pec->m_next;
            if( pec->m_finalize.m_func != nullptr )
            {
              // may account for elapsed time, and free pec allocated memory
              ( * pec->m_finalize.m_func ) ( pec , pec->m_finalize.m_data );
            }
            reinterpret_cast<BlockParallelForHostFunctor*>(pec->m_host_scratch.functor_data)-> ~BlockParallelForHostFunctor();
            pec = next;
          }
          
          m_exec_ctx = nullptr;
        }
      }
      
      inline bool query_status()    
      {
        if( m_stream == nullptr || m_exec_ctx == nullptr )
        {
          return true;
        }
        std::lock_guard lk( m_stream->m_mutex );
        if( m_stream->m_omp_execution_count.load() > 0 )
        {
          return false;
        }
        if( m_stream->m_cuda_ctx != nullptr && m_exec_ctx != nullptr )
        {
          assert( m_exec_ctx->m_stop_evt != nullptr );
          if( ONIKA_CU_EVENT_QUERY( m_exec_ctx->m_stop_evt ) != onikaSuccess )
          {
            return false;
          }
        }
        wait();
        return true;
      }
      
      inline bool empty() const     
      {
        if( m_stream == nullptr ) return true;
        std::lock_guard lk( m_stream->m_mutex );
        return m_exec_ctx == nullptr;
      }
      
      inline bool has_stream() const   
      {
        return m_stream != nullptr;
      }

    };
    
  }

}

