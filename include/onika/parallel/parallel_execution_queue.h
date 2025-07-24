#pragma once

#include <onika/cuda/cuda_context.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/constants.h>
#include <onika/parallel/block_parallel_for_functor.h>
#include <mutex>
#include <atomic>

namespace onika
{

  namespace parallel
  {

    struct ParallelExecutionStreamPool
    {
      ParallelExecutionStream* (*m_func) (void*,int) = nullptr;
      void *m_priv = nullptr;
      inline ParallelExecutionStream* operator () (int lane = DEFAULT_EXECUTION_LANE) { return (*m_func)(m_priv,lane); }
    };

    struct ParallelExecutionQueue
    {
      ParallelExecutionStreamPool m_stream_pool = {};     // execution stream to schedule paralel operations
      int m_preffered_lane = UNDEFINED_EXECUTION_LANE;
      ParallelExecutionContext* m_queue_list = nullptr;  // list of scheduled parallel operations
      ParallelExecutionContext* m_exec_list = nullptr;   // list of executing parallel operations
      std::mutex m_mutex;                                // for thread safe manipulation of queue

      ParallelExecutionQueue() = default;
      
      inline ParallelExecutionQueue(const ParallelExecutionStreamPool& pesp , int pl)
        : m_stream_pool( pesp )
        , m_preffered_lane( pl )
      {
      }

      inline ParallelExecutionQueue(const ParallelExecutionQueue& other)
        : m_stream_pool( other.m_stream_pool )
        , m_preffered_lane( other.m_preffered_lane )
      {
        assert( other.m_queue_list == nullptr );
        assert( other.m_exec_list == nullptr );
      }

      inline ParallelExecutionQueue(ParallelExecutionQueue && other)
        : m_stream_pool( other.m_stream_pool )
        , m_preffered_lane( other.m_preffered_lane )
        , m_queue_list( other.m_queue_list )
        , m_exec_list( other.m_exec_list )
      {
        const std::lock_guard lk_self( other.m_mutex );
        other.m_queue_list = nullptr;
        other.m_exec_list = nullptr;
      }
      
      inline ParallelExecutionQueue& operator = (const ParallelExecutionQueue & other)
      {
        assert( other.m_queue_list == nullptr );
        assert( other.m_exec_list == nullptr );
        const std::lock_guard lk_self( m_mutex );
        schedule_all();
        wait();
        m_stream_pool = other.m_stream_pool;
        m_preffered_lane = other.m_preffered_lane;
        return *this;
      }

      inline ParallelExecutionQueue& operator = (ParallelExecutionQueue && other)
      {
        const std::lock_guard lk_self( m_mutex );
        const std::lock_guard lk_other( other.m_mutex );
        m_stream_pool = other.m_stream_pool;
        m_preffered_lane = other.m_preffered_lane;
        m_queue_list = other.m_queue_list;
        m_exec_list = other.m_exec_list;
        other.m_queue_list = nullptr;
        other.m_exec_list = nullptr;
        return *this;
      }

      inline ~ParallelExecutionQueue()
      {
        schedule_all();
        wait();
        assert( m_queue_list == nullptr );
        assert( m_exec_list == nullptr );
      }

      inline void enqueue(ParallelExecutionContext* pec)
      {
        assert( pec->m_next == nullptr );
        assert( pec->m_stream == nullptr );
        const std::lock_guard lk_self( m_mutex );  
        if( m_queue_list == nullptr )
        {
          m_queue_list = pec;
        }
        else
        {
          ParallelExecutionContext* el = m_queue_list;
          while( el->m_next != nullptr ) el = el->m_next;
          assert( el->m_next == nullptr );
          el->m_next = pec;
        }
      }

      inline void schedule_all()
      {
        const std::lock_guard lk_self( m_mutex );        

        // fast forward to tail of executing operation list
        ParallelExecutionContext* el = m_exec_list;
        if( el != nullptr ) while( el->m_next != nullptr ) el = el->m_next;
        assert( el==nullptr || el->m_next == nullptr );
        
        while( m_queue_list != nullptr )
        {
          auto pec = m_queue_list;
          m_queue_list = m_queue_list->m_next;
          pec->m_next = nullptr;
          schedule( pec );
          if( el != nullptr )
          {
            el->m_next = pec;
            el = pec;
          }
          else
          {
            el = m_exec_list = pec;
          }
          assert( el->m_next == nullptr );
        }
      }

      inline void schedule(ParallelExecutionContext* pec)
      {
        assert( pec->m_stream == nullptr );
        assert( pec->m_next == nullptr );

        // query wich lane (i.e. stream id) to use for parallel operation scheduling
        int lane = m_preffered_lane;
        if( lane == UNDEFINED_EXECUTION_LANE ) lane = pec->m_preferred_lane;
        auto exec_stream = m_stream_pool( lane );
        std::lock_guard lk_stream( exec_stream->m_mutex );
        
        pec->m_stream = exec_stream;
      
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
              fatal_error() << "Cannot schedule GPU parallel operation onto stream with no GPU context" << std::endl;
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
        
      }
      
      inline void wait( int __unused_lane__ = UNDEFINED_EXECUTION_LANE )
      {
        std::lock_guard lk_self( m_mutex );
        while(m_exec_list!=nullptr)
        {
          assert( m_exec_list->m_stream != nullptr );
          std::lock_guard lk( m_exec_list->m_stream->m_mutex );

          // synchronize stream
          auto * pec = m_exec_list;
          m_exec_list = m_exec_list->m_next;
          pec->m_next = nullptr;
          
          pec->m_stream->wait_nolock();

          float Tgpu = 0.0;
          if( pec->m_execution_target == ParallelExecutionContext::EXECUTION_TARGET_CUDA )
          {
            ONIKA_CU_CHECK_ERRORS( ONIKA_CU_EVENT_ELAPSED(Tgpu,pec->m_start_evt,pec->m_stop_evt) );
            pec->m_total_gpu_execution_time = Tgpu;
          }
          if( pec->m_finalize.m_func != nullptr )
          {
            // may account for elapsed time, and free pec allocated memory
            ( * pec->m_finalize.m_func ) ( pec , pec->m_finalize.m_data );
          }
          pec->m_stream = nullptr;
          reinterpret_cast<BlockParallelForHostFunctor*>(pec->m_host_scratch.functor_data) -> ~BlockParallelForHostFunctor();
        }
        assert( m_exec_list == nullptr );
      }
      
      inline bool query_status( int __unused_lane__ = UNDEFINED_EXECUTION_LANE )    
      {
        const std::lock_guard lk_self( m_mutex );
        if( m_exec_list == nullptr && m_queue_list == nullptr )
        {
          return true;
        }
        if( m_queue_list != nullptr )
        {
          return false;
        }
          
        auto* pec = m_exec_list;
        while(pec!=nullptr)
        {
          std::lock_guard lk( pec->m_stream->m_mutex );
          if( pec->m_stream->m_omp_execution_count.load() > 0 )
          {
            return false;
          }
          if( pec->m_stream->m_cuda_ctx != nullptr && pec->m_stop_evt != nullptr )
          {
            if( ONIKA_CU_EVENT_QUERY( pec->m_stop_evt ) != onikaSuccess )
            {
              return false;
            }
          }
          pec = pec->m_next;
        } 

        wait();
        return true;
      }
      
      inline bool empty()
      {
        const std::lock_guard lk_self( m_mutex );
        return m_exec_list == nullptr && m_queue_list  == nullptr;
      }

    };
    
  }

}

