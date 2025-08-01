#pragma once

#include <onika/cuda/cuda_context.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/parallel_data_access.h>
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
      inline ParallelExecutionStream* operator () (int lane) { return (*m_func)(m_priv,lane); }
    };

    struct ParallelExecutionQueueBase
    {
      int m_lane = DEFAULT_EXECUTION_LANE;
      ParallelExecutionContext* m_queue_list = nullptr;  // head of "ready to be scheduled" parallel operations list
      ParallelDataAccessVector m_data_access;
      std::mutex m_mutex;                                // for thread safe manipulation of queue

      inline ~ParallelExecutionQueueBase()
      {
        assert( m_queue_list == nullptr );
      }

      inline void set_lane(int l)
      {
        m_lane = l;
      }

      inline void reset_data_access()
      {
        const std::lock_guard lk_self( m_mutex );
        m_data_access.clear();
      }

      inline void add_data_access(const ParallelDataAccess& pda)
      {
        m_data_access.push_back(pda);
      }
      inline void add_data_access(ParallelDataAccess && pda)
      {
        m_data_access.emplace_back( std::move(pda) );
      }

      inline void enqueue(ParallelExecutionContext* pec)
      {
        assert( pec->m_next == nullptr );
        assert( pec->m_stream == nullptr );
        assert( pec->m_lane == UNDEFINED_EXECUTION_LANE );
        assert( pec->m_data_access.empty() );
        const std::lock_guard lk_self( m_mutex );
        pec->m_lane = m_lane;
        // ensures allocated space is never lost
        std::swap( pec->m_data_access , m_data_access );
        m_data_access.clear();
        m_queue_list = pec_list_append( m_queue_list , pec );
      }

      inline void pre_process_queue( ParallelExecutionContext* head )
      {
        bool all_undefined_lane = true;
        bool all_have_explicit_data_access = true;
        ParallelExecutionContext* pec = head;
        while( pec != nullptr )
        {
          all_undefined_lane = all_undefined_lane && ( pec->m_lane == UNDEFINED_EXECUTION_LANE );
          all_have_explicit_data_access = all_have_explicit_data_access && ( ! pec->m_data_access.empty() );
          pec = pec->m_next;
        }
        if( ! ( all_undefined_lane && all_have_explicit_data_access ) ) return;  
        else
        {
          // do some lane assignment here
        }
      }

    };

    struct ParallelExecutionQueue : public ParallelExecutionQueueBase
    {
      ParallelExecutionStreamPool m_stream_pool = {};    // execution stream to schedule paralel operations
      ParallelExecutionContext* m_exec_list = nullptr;   // list of executing ( or at least scheduled for execution ) parallel operations

      inline ~ParallelExecutionQueue()
      {
        schedule_all();
        wait();
        assert( m_exec_list == nullptr );
      }

      inline std::pair<ParallelExecutionContext*,ParallelExecutionContext*> schedule_filter_list( ParallelExecutionContext* ql, int lane )
      {        
        if( ql == nullptr ) return { nullptr , nullptr };

        auto ql_next = ql->m_next;
        ql->m_next = nullptr;
        
        if( ql->m_lane == lane || lane < 0 )
        {
          schedule( ql );  
          auto [nql,nsl] = schedule_filter_list( ql_next , lane );
          ql->m_next = nsl;
          return { nql , ql };
        }
        else
        {
          auto [nql,nsl] = schedule_filter_list( ql_next , lane );
          ql->m_next = nql;
          return { ql , nsl };
        }
      }

      inline void schedule_all(int lane = UNDEFINED_EXECUTION_LANE )
      {
        const std::lock_guard lk_self( m_mutex );
        // queue preprocessing here ...

        auto [nql,nsl] = schedule_filter_list( m_queue_list , lane );
        m_queue_list = nql;
        m_exec_list = pec_list_append( m_exec_list , nsl );
      }

      inline void schedule(ParallelExecutionContext* pec)
      {
        assert( pec->m_stream == nullptr );
        assert( pec->m_next == nullptr );

        // query wich lane (i.e. stream id) to use for parallel operation scheduling
        int lane = pec->m_lane;

        // if lane is undefined, this means we are allowed to split operation and schedule onto several streams
        // following line will be replaced by transformations made on operation : split, co schedule, etc.
        // and each subsequent parallel operation will be assigned a scheduling lane (i.e. a stream)
        if( lane == UNDEFINED_EXECUTION_LANE ) lane = DEFAULT_EXECUTION_LANE;

        // arriving here, parallel operation has been assigned a specific lane (i.e. stream id)
        if( lane == DEFAULT_EXECUTION_LANE ) lane = 0;
        
        // we update lane on the parallel operation so that it remember on what lane it's executing
        assert( lane >= 0 );
        pec->m_lane = lane;
        
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

      inline ParallelExecutionContext* sync_and_remove(ParallelExecutionContext* pec, int lane = UNDEFINED_EXECUTION_LANE)
      {
        if( pec == nullptr )
        {
          return nullptr;
        }
        else if( pec->m_lane==lane || lane<0 )
        {
          assert( pec->m_stream != nullptr );
          ParallelExecutionContext* next = nullptr;
          { // ParallelExecutionContext's stream critical section 
            std::lock_guard lk( pec->m_stream->m_mutex );
            next = pec->m_next;
            pec->m_next = nullptr;
            
            // waits for both OpenMP tasks and Cuda kernels in the specified stream to terminate
            pec->m_stream->wait_nolock();

            float Tgpu = 0.0;
            if( pec->m_execution_target == ParallelExecutionContext::EXECUTION_TARGET_CUDA )
            {
              ONIKA_CU_CHECK_ERRORS( ONIKA_CU_EVENT_ELAPSED(Tgpu,pec->m_start_evt,pec->m_stop_evt) );
              pec->m_total_gpu_execution_time = Tgpu;
            }
            if( pec->m_finalize.m_func != nullptr )
            {
              ( * pec->m_finalize.m_func ) ( pec , pec->m_finalize.m_data );
            }
            pec->m_stream = nullptr;
            reinterpret_cast<BlockParallelForHostFunctor*>(pec->m_host_scratch.functor_data) -> ~BlockParallelForHostFunctor();
          }
          return sync_and_remove(next);
        }
        else
        {
          pec->m_next = sync_and_remove(pec->m_next,lane);
          return pec;
        }
      }
      
      inline void wait(int lane = UNDEFINED_EXECUTION_LANE)
      {
        std::lock_guard lk_self( m_mutex );
        m_exec_list = sync_and_remove( m_exec_list , lane );
      }
      
      inline bool query_status(int lane = UNDEFINED_EXECUTION_LANE)
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
          if( pec->m_stream->m_stream_id == uint32_t(lane) || lane < 0 )
          {
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
          }
          pec = pec->m_next;
        } 

        wait( lane );
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

