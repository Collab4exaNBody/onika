#pragma once

#include <onika/cuda/cuda_context.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for_functor.h>
#include <onika/parallel/stream_utils.h>
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/parallel_execution_queue.h>
#include <mutex>
#include <atomic>

namespace onika
{

  namespace parallel
  {

    // temporarily holds ParallelExecutionContext instance until it is either queued in a stream or graph execution flow,
    // or destroyed, in which case it inserts instance onto the default stream queue
    struct ParallelExecutionWrapper
    {
      ParallelExecutionContext* m_pec = nullptr;
      inline ~ParallelExecutionWrapper();      
    };

    // real implementation of how a parallel operation is pushed onto a stream queue
    inline ParallelExecutionQueue operator << ( ParallelExecutionQueue && pesq , ParallelExecutionWrapper && pew )
    {
      assert( pesq.m_stream != nullptr );
      std::lock_guard lk( pesq.m_stream->m_mutex );

      assert( pew.m_pec != nullptr );
      auto & pec = * pew.m_pec;
      pew.m_pec = nullptr;

      const auto & func = * reinterpret_cast<BlockParallelForHostFunctor*>( pec.m_host_scratch.functor_data );
      
      switch( pec.m_execution_target )
      {
        case ParallelExecutionContext::EXECUTION_TARGET_OPENMP :
        {
          if( pec.m_omp_num_tasks == 0 )
          {
            func.execute_omp_parallel_region( &pec , pesq.m_stream );
          }
          else
          {
            // preferred number of tasks : trade off between overhead (less is better) and load balancing (more is better)
            const unsigned int num_tasks = pec.m_omp_num_tasks * onika::parallel::ParallelExecutionContext::parallel_task_core_mult() + onika::parallel::ParallelExecutionContext::parallel_task_core_add() ;
            func.execute_omp_tasks( &pec , pesq.m_stream , num_tasks );
          }
        }
        break;
        
        case ParallelExecutionContext::EXECUTION_TARGET_CUDA :
        {
          if( pesq.m_stream->m_cuda_ctx == nullptr || pesq.m_stream->m_cuda_ctx != pec.m_cuda_ctx )
          {
            std::cerr << "Mismatch Cuda context, cannot queue parallel execution to this stream" << std::endl;
            std::abort();
          }
        
          // if device side scratch space hasn't be allocated yet, do it now
          pec.init_device_scratch();
          
          // insert start event for profiling
          assert( pec.m_start_evt != nullptr );
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_EVENT( pec.m_start_evt, pesq.m_stream->m_cu_stream ) );

          // copy in return data intial value. mainly useful for reduction where you might want to start reduction with a given initial value
          if( pec.m_return_data_input != nullptr && pec.m_return_data_size > 0 )
          {
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( pec.m_cuda_scratch->return_data, pec.m_return_data_input , pec.m_return_data_size , pesq.m_stream->m_cu_stream ) );
          }

          // sets all scratch counters to 0
          if( pec.m_reset_counters || pec.m_grid_size.x > 0 )
          {
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMSET( pec.m_cuda_scratch->counters, 0, GPUKernelExecutionScratch::MAX_COUNTERS * sizeof(unsigned long long int), pesq.m_stream->m_cu_stream ) );
          }

          // Instantiaite device side functor : calls constructor with a placement new using scratch "functor_data" space
          // then call functor prolog if available
          func.stream_gpu_initialize( &pec , pesq.m_stream );
          func.stream_gpu_kernel( &pec , pesq.m_stream );
          func.stream_gpu_finalize( &pec , pesq.m_stream );
          
          // copy out return data to host space at given pointer
          if( pec.m_return_data_output != nullptr && pec.m_return_data_size > 0 )
          {
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( pec.m_return_data_output , pec.m_cuda_scratch->return_data , pec.m_return_data_size , pesq.m_stream->m_cu_stream ) );
          }
          
          // inserts a callback to stream if user passed one in
          if( pec.m_execution_end_callback.m_func != nullptr )
          {
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_ADD_CALLBACK(pesq.m_stream->m_cu_stream, ParallelExecutionContext::execution_end_callback , &pec ) );
          }
          
          // inserts stop event to account for total execution time
          assert( pec.m_stop_evt != nullptr );
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_EVENT( pec.m_stop_evt, pesq.m_stream->m_cu_stream ) );
        }
        break;          
        
        default:
        {
          std::cerr << "Invalid execution target" << std::endl;
          std::abort();
        }
        break;
      }
      
      // add parallel execution to queue
      pec.m_next = pesq.m_exec_ctx;
      pesq.m_exec_ctx = &pec;
      
      return std::move(pesq);
    }

    inline ParallelExecutionWrapper::~ParallelExecutionWrapper()
    {
      if( m_pec != nullptr )
      {
        ParallelExecutionQueue{m_pec->m_default_stream} << ParallelExecutionWrapper{m_pec};
        m_pec = nullptr;
      }
    }

    
  }

}

