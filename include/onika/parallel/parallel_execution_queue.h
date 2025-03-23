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

    static inline constexpr int DEFAULT_EXECUTION_LANE = -1;
    static inline constexpr int MAX_EXECUTION_LANES = 256;

    struct ParallelExecutionStreamPool
    {
      ParallelExecutionStream* (*m_func) (void*,int) = nullptr;
      void *m_priv = nullptr;
      inline ParallelExecutionStream* operator () (int lane = DEFAULT_EXECUTION_LANE) { return (*m_func)(m_priv,lane); }
    };

    struct ParallelExecutionQueue
    {
      ParallelExecutionStreamPool m_stream_pool = {};     // execution stream to schedule paralel operations
      ParallelExecutionContext* m_exec_ctx = nullptr;  // head of parallel operations to schedule (following item chained through m_exec_ctx->m_next)
      std::mutex m_mutex;                              // for thread safe manipulation of queue
                     
      inline ~ParallelExecutionQueue()
      {
        wait();
        m_stream = nullptr;
      }
      
      inline void enqueue_and_schedule(ParallelExecutionContext* pec)
      {
        std::lock_guard lk_self( m_mutex );
        
        // if no automatic lane selection, may ask parallel operation if it has a preferred one
        auto exec_stream = m_stream_pool( pec->m_preferred_lane );
        std::lock_guard lk_stream( exec_stream->m_mutex );
      
        const auto & func = * reinterpret_cast<BlockParallelForHostFunctor*>( pec->m_host_scratch.functor_data );
        
        switch( pec->m_execution_target )
        {
          case ParallelExecutionContext::EXECUTION_TARGET_OPENMP :
          {
            if( pec->m_omp_num_tasks == 0 )
            {
              func.execute_omp_parallel_region( pec , exec_stream );
            }
            else
            {
              // preferred number of tasks : trade off between overhead (less is better) and load balancing (more is better)
              const unsigned int num_tasks = pec->m_omp_num_tasks * onika::parallel::ParallelExecutionContext::parallel_task_core_mult() + onika::parallel::ParallelExecutionContext::parallel_task_core_add() ;
              func.execute_omp_tasks( pec , exec_stream , num_tasks );
            }
          }
          break;
          
          case ParallelExecutionContext::EXECUTION_TARGET_CUDA :
          {
            if( exec_stream->m_cuda_ctx == nullptr || exec_stream->m_cuda_ctx != pec->m_cuda_ctx )
            {
              fatal_error() << "Mismatch Cuda context, cannot queue parallel execution to this stream" << std::endl;
            }
          
            // if device side scratch space hasn't be allocated yet, do it now
            pec->init_device_scratch();
            
            // insert start event for profiling
            assert( pec->m_start_evt != nullptr );
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_EVENT( pec->m_start_evt, exec_stream->m_cu_stream ) );

            // copy in return data intial value. mainly useful for reduction where you might want to start reduction with a given initial value
            if( pec->m_return_data_input != nullptr && pec->m_return_data_size > 0 )
            {
              ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( pec->m_cuda_scratch->return_data, pec->m_return_data_input , pec->m_return_data_size , exec_stream->m_cu_stream ) );
            }

            // sets all scratch counters to 0
            if( pec->m_reset_counters || pec->m_grid_size.x > 0 )
            {
              ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMSET( pec->m_cuda_scratch->counters, 0, GPUKernelExecutionScratch::MAX_COUNTERS * sizeof(unsigned long long int), exec_stream->m_cu_stream ) );
            }

            // Instantiaite device side functor : calls constructor with a placement new using scratch "functor_data" space
            // then call functor prolog if available
            func.stream_gpu_initialize( pec , exec_stream );
            func.stream_gpu_kernel( pec , exec_stream );
            func.stream_gpu_finalize( pec , exec_stream );
            
            // copy out return data to host space at given pointer
            if( pec->m_return_data_output != nullptr && pec->m_return_data_size > 0 )
            {
              ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( pec->m_return_data_output , pec->m_cuda_scratch->return_data , pec->m_return_data_size , exec_stream->m_cu_stream ) );
            }
            
            // inserts a callback to stream if user passed one in
            if( pec->m_execution_end_callback.m_func != nullptr )
            {
              ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_ADD_CALLBACK(exec_stream->m_cu_stream, ParallelExecutionContext::execution_end_callback , pec ) );
            }
            
            // inserts stop event to account for total execution time
            assert( pec->m_stop_evt != nullptr );
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_EVENT( pec->m_stop_evt, exec_stream->m_cu_stream ) );
          }
          break;          
          
          default:
          {
            fatal_error() << "Invalid execution target" << std::endl;
          }
          break;
        }
        
        // add parallel execution to queue
        pec->m_stream = exec_stream;
        pec->m_next = m_exec_ctx;
        m_exec_ctx = pec;
      }
      
      inline void wait()
      {
        std::lock_guard lk_self( m_mutex );
        if( m_exec_ctx != nullptr )
        {          
          // collect execution times
          auto* pec = m_exec_ctx;
          while(pec!=nullptr)
          {
            if( pec->m_stream == nullptr )
            {
              fatal_error() << "Executing operation has invalid stream" << std::endl;
            }
            std::lock_guard lk( pec->m_stream->m_mutex );

            // synchronize stream
            pec->m_stream->wait_nolock();

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
      
      // FIXME: way too conservative, requires all streams containing any of the executing tasks to be completed
      inline bool query_status()    
      {
        std::lock_guard lk_self( m_mutex );
        if( m_exec_ctx == nullptr )
        {
          return true;
        }
        auto* pec = m_exec_ctx;
        while(pec!=nullptr)
        {
          std::lock_guard lk( pec->m_stream->m_mutex );
          if( pec->m_stream->m_omp_execution_count.load() > 0 )
          {
            return false;
          }
          if( pec->m_stream->m_cuda_ctx != nullptr && m_exec_ctx->m_stop_evt != nullpt )
          {
            if( ONIKA_CU_EVENT_QUERY( m_exec_ctx->m_stop_evt ) != onikaSuccess )
            {
              return false;
            }
          }
        }               
        wait();
        return true;
      }
      
      inline bool empty() const     
      {
        std::lock_guard lk( m_stream->m_mutex );
        return m_exec_ctx == nullptr;
      }

    };
    
  }

}

