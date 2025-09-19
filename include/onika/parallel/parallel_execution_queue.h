#pragma once

#include <onika/cuda/cuda_context.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/parallel_data_access.h>
#include <onika/parallel/constants.h>
#include <onika/parallel/block_parallel_for_functor.h>
#include <mutex>

#ifndef ONIKA_AUTO_LANE_CYCLE_HINT
#define ONIKA_AUTO_LANE_CYCLE_HINT 16
#endif

#ifndef ONIKA_PARALLEL_QUEUE_MAX_LANES
#define ONIKA_PARALLEL_QUEUE_MAX_LANES 32
#endif

namespace onika
{

  namespace parallel
  {
    // encapsulates a callback function whose purpose is to return
    // an execution stream associated with given execution lane number
    struct ParallelExecutionStreamPool
    {
      ParallelExecutionStream* (*m_func) (void*,int) = nullptr;
      void *m_priv = nullptr;
      inline ParallelExecutionStream* operator () (int lane) { return (*m_func)(m_priv,lane); }
    };

    // automatically and dynamically allocates execution streams associated with a lane number
    struct ParallelExecutionStreamAutoAllocator
    {
      onika::cuda::CudaContext * m_cuda_ctx = nullptr;
      std::vector< std::shared_ptr< onika::parallel::ParallelExecutionStream > > m_pes;
      std::mutex m_mutex;
      
      static inline onika::parallel::ParallelExecutionStream* parallel_execution_stream_cb(void* _self, int lane);
      onika::parallel::ParallelExecutionStream * parallel_execution_stream(int lane = DEFAULT_EXECUTION_LANE);
      ParallelExecutionStreamPool parallel_execution_stream_pool();
    };

    // Base class of execution queues. may be used as delay queue
    // which content is later pushed to a schedulable execution queue
    struct ParallelExecutionQueueBase
    {
      // where to place next enqueued parallel operation
      int m_lane = DEFAULT_EXECUTION_LANE;
      // preferred maximum number of lanes to use for auto lane assignment
      int m_auto_lane_cycle = ONIKA_AUTO_LANE_CYCLE_HINT;
      // head of "ready to be scheduled" parallel operations list
      ParallelExecutionContext* m_queue_list = nullptr;
      // data access(es) patterns
      ParallelDataAccessVector m_data_access;
      // for thread safe manipulation of queue
      std::mutex m_mutex;

      ~ParallelExecutionQueueBase();

      void set_lane(int l , int lane_cycle = ONIKA_AUTO_LANE_CYCLE_HINT);
      void reset_data_access();
      void add_data_access(const ParallelDataAccess& pda);
      void add_data_access(ParallelDataAccess && pda);
      void enqueue(ParallelExecutionContext* pec, bool from_other_queue = false);
      void pre_process_queue( ParallelExecutionContext* head );
    };

    // execution queue capable of scheduling parallel operations to CPU/GPU computation units
    // through execution streams
    struct ParallelExecutionQueue : public ParallelExecutionQueueBase
    {
      ParallelExecutionStreamPool m_stream_pool = {};    // execution stream to schedule paralel operations
      ParallelExecutionContext* m_exec_list = nullptr;   // list of executing ( or at least scheduled for execution ) parallel operations

      static std::shared_ptr<ParallelExecutionQueue> s_default_queue;
      static std::shared_ptr<ParallelExecutionStreamAutoAllocator> s_default_stream_allocator;

      static ParallelExecutionQueue& default_queue();

      ~ParallelExecutionQueue();

      std::pair<ParallelExecutionContext*,ParallelExecutionContext*> schedule_filter_list( ParallelExecutionContext* ql, int lane );
      void schedule_all(int lane = UNDEFINED_EXECUTION_LANE );
      void schedule(ParallelExecutionContext* pec);
      ParallelExecutionContext* sync_and_remove(ParallelExecutionContext* pec, int lane = UNDEFINED_EXECUTION_LANE);
      void wait(int lane = UNDEFINED_EXECUTION_LANE);
      bool query_status(int lane = UNDEFINED_EXECUTION_LANE);
      bool empty();
    };

  }

}

