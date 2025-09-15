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

#include <onika/parallel/parallel_execution_queue.h>
#include <onika/log.h>

namespace onika
{

  namespace parallel
  {

    // ======== ParallelExecutionStreamAutoAllocator implementation ==============

    ParallelExecutionStream * ParallelExecutionStreamAutoAllocator::parallel_execution_stream(int lane)
    {
      if( lane == onika::parallel::DEFAULT_EXECUTION_LANE ) lane = 0;
      if( lane < 0 || lane >= onika::parallel::MAX_EXECUTION_LANES )
      {
        fatal_error() << "Invalid execution stream id ("<<lane<<")"<<std::endl;
      }
      std::lock_guard<std::mutex> lock(m_mutex);
      
      if( size_t(lane) >= m_pes.size() ) m_pes.resize( lane+1 , nullptr );
      if( m_pes[lane] == nullptr )
      {
        m_pes[lane] = std::make_shared<onika::parallel::ParallelExecutionStream>();
        if( m_cuda_ctx != nullptr )
        {
          m_pes[lane]->m_cuda_ctx = m_cuda_ctx;
          m_pes[lane]->m_cu_stream = m_cuda_ctx->getThreadStream(lane);
        }
         m_pes[lane]->m_stream_id = lane;
      }
      return m_pes[lane].get();
    }
    
    onika::parallel::ParallelExecutionStream* ParallelExecutionStreamAutoAllocator::parallel_execution_stream_cb(void* _self, int lane)
    {
      ParallelExecutionStreamAutoAllocator * self = ( ParallelExecutionStreamAutoAllocator* ) _self;
      return self->parallel_execution_stream(lane);
    }
    
    ParallelExecutionStreamPool ParallelExecutionStreamAutoAllocator::parallel_execution_stream_pool()
    {
      return { parallel_execution_stream_cb , this };
    }

    // ======== ParallelExecutionQueueBase implementation ==============
    
    ParallelExecutionQueueBase::~ParallelExecutionQueueBase()
    {
      assert( m_queue_list == nullptr );
    }

    void ParallelExecutionQueueBase::set_lane(int l)
    {
      m_lane = l;
    }

    void ParallelExecutionQueueBase::reset_data_access()
    {
      const std::lock_guard lk_self( m_mutex );
      m_data_access.clear();
    }

    void ParallelExecutionQueueBase::add_data_access(const ParallelDataAccess& pda)
    {
      m_data_access.push_back(pda);
    }
    void ParallelExecutionQueueBase::add_data_access(ParallelDataAccess && pda)
    {
      m_data_access.emplace_back( std::move(pda) );
    }

    void ParallelExecutionQueueBase::enqueue(ParallelExecutionContext* pec, bool from_other_queue)
    {
      assert( pec->m_next == nullptr );
      assert( pec->m_stream == nullptr );
      if( ! from_other_queue )
      {
        assert( pec->m_lane == UNDEFINED_EXECUTION_LANE );
        assert( pec->m_data_access.empty() );
      }
      const std::lock_guard lk_self( m_mutex );
      if( ! from_other_queue )
      {
        pec->m_lane = m_lane;
        std::swap( pec->m_data_access , m_data_access );
      }
      m_data_access.clear();
      m_queue_list = pec_list_append( m_queue_list , pec );
    }

    void ParallelExecutionQueueBase::pre_process_queue( ParallelExecutionContext* head )
    {
      int task_cnt = 0;
      int undefined_lane_cnt = 0;
      int explicit_data_access_cnt = 0;
      ParallelExecutionContext* pec = head;
      while( pec != nullptr )
      {
        if( pec->m_lane == UNDEFINED_EXECUTION_LANE ) ++ undefined_lane_cnt;
        if( ! pec->m_data_access.empty() ) ++ explicit_data_access_cnt;
        ++ task_cnt;
        pec = pec->m_next;
      }
      printf("pre_process_queue : task_cnt=%d, undefined_lane=%d, data_access=%d\n",task_cnt,undefined_lane_cnt,explicit_data_access_cnt);

      if( undefined_lane_cnt==task_cnt && explicit_data_access_cnt==task_cnt && task_cnt>0 )
      {
        // do some lane assignment here
        printf("optimizable sequence detected head=%p\n",head);
        pec = head;
        while( pec != nullptr )
        {
          const auto & pfor_adapter = get_block_parallel_functor( pec );
          const int ndims = pfor_adapter.execution_space_ndims();
          ssize_t crange[6];
          size_t n = pfor_adapter.execution_space_range(crange,6);
          printf(" %s/%s,%dD",pec->m_tag,pec->m_sub_tag,ndims);
          for(size_t i=0;i<n;i++) printf("%c%d", i==0 ? '[' : ( i==n/2 ? '-' : ',' ) , int(crange[i]));
          printf("]=%d",int(pfor_adapter.execution_space_nitems()));
          for(const auto & pda : pec->m_data_access)
          {
            printf(",%s@%p",pda.name(),pda.address());
            for(const auto & st : pda.m_stencil)
            {
              if(ndims>=3) printf(",%s@%d,%d,%d",st.mode_str(),st.ri(),st.rj(),st.rk());
              else if(ndims==2) printf(",%s@%d,%d",st.mode_str(),st.ri(),st.rj());
              else printf(",%s@%d",st.mode_str(),st.ri());
            }
          }
          printf("\n");
          pec = pec->m_next;
        }
      }
      
    }

  
  
    // ================= ParallelExecutionQueue implementation ====================
  
    std::shared_ptr<ParallelExecutionQueue> ParallelExecutionQueue::s_default_queue = nullptr;
    std::shared_ptr<ParallelExecutionStreamAutoAllocator> ParallelExecutionQueue::s_default_stream_allocator = nullptr;

    ParallelExecutionQueue& ParallelExecutionQueue::default_queue()
    {
      if( s_default_stream_allocator == nullptr )
      {
        s_default_stream_allocator = std::make_shared<ParallelExecutionStreamAutoAllocator>();
        s_default_stream_allocator->m_cuda_ctx = onika::cuda::CudaContext::default_cuda_ctx();
      }
      if( s_default_queue == nullptr )
      {
        s_default_queue = std::make_shared<ParallelExecutionQueue>();
        s_default_queue->m_stream_pool = s_default_stream_allocator->parallel_execution_stream_pool();
      }
      return * s_default_queue;
    }


    ParallelExecutionQueue::~ParallelExecutionQueue()
    {
      schedule_all();
      wait();
      assert( m_exec_list == nullptr );
    }

    std::pair<ParallelExecutionContext*,ParallelExecutionContext*> ParallelExecutionQueue::schedule_filter_list( ParallelExecutionContext* ql, int lane )
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

    void ParallelExecutionQueue::schedule_all(int lane)
    {
      const std::lock_guard lk_self( m_mutex );
      pre_process_queue( m_queue_list );
      auto [nql,nsl] = schedule_filter_list( m_queue_list , lane );
      m_queue_list = nql;
      m_exec_list = pec_list_append( m_exec_list , nsl );
    }

    void ParallelExecutionQueue::schedule(ParallelExecutionContext* pec)
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
      const auto & func = get_block_parallel_functor( pec );
      
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

    ParallelExecutionContext* ParallelExecutionQueue::sync_and_remove(ParallelExecutionContext* pec, int lane)
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
    
    void ParallelExecutionQueue::wait(int lane)
    {
      std::lock_guard lk_self( m_mutex );
      m_exec_list = sync_and_remove( m_exec_list , lane );
    }
    
    bool ParallelExecutionQueue::query_status(int lane)
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
    
    bool ParallelExecutionQueue::empty()
    {
      const std::lock_guard lk_self( m_mutex );
      return m_exec_list == nullptr && m_queue_list  == nullptr;
    }

  }

}

