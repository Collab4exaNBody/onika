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
      
      ParallelExecutionWrapper() = default;
      
      ParallelExecutionWrapper(const ParallelExecutionWrapper&) = delete;
      ParallelExecutionWrapper& operator = (const ParallelExecutionWrapper & other) = delete;

      inline ParallelExecutionWrapper(ParallelExecutionContext* pec) : m_pec(pec) {}
      inline ParallelExecutionWrapper(ParallelExecutionWrapper && other) : m_pec(other.m_pec) { other.m_pec=nullptr; }
      inline ParallelExecutionWrapper& operator = (ParallelExecutionWrapper && other) { m_pec = std::move(other.m_pec); other.m_pec = nullptr; return *this; }
      
      inline ~ParallelExecutionWrapper();
    };

    struct flush_t {};
    static inline constexpr flush_t flush = {};

    struct synchronize_t {};
    static inline constexpr synchronize_t synchronize = {};

    struct set_lane_t { int m_lane = UNDEFINED_EXECUTION_LANE; };
    static inline constexpr set_lane_t set_lane(int l) { return { l }; }
    static inline constexpr set_lane_t any_lane() { return {}; }

    // real implementation of how a parallel operation is pushed onto a stream queue
    inline ParallelExecutionQueue& operator << ( ParallelExecutionQueue& pesq , ParallelExecutionWrapper && pew )
    {
      auto * pec = pew.m_pec;
      pew.m_pec = nullptr;
      pesq.enqueue( pec );
      return pesq;
    }

    inline ParallelExecutionQueue& operator << ( ParallelExecutionQueue& pesq , flush_t )
    {
      pesq.schedule_all();
      return pesq;
    }

    inline ParallelExecutionQueue& operator << ( ParallelExecutionQueue& pesq , synchronize_t )
    {
      pesq.wait();
      return pesq;
    }

    inline ParallelExecutionQueue& operator << ( ParallelExecutionQueue& pesq , set_lane_t sl )
    {
      pesq.set_lane( sl.m_lane );
      return pesq;
    }

    inline ParallelExecutionWrapper::~ParallelExecutionWrapper()
    {
      if( m_pec != nullptr )
      {
        if( m_pec->m_default_queue == nullptr )
        {
          fatal_error() << "No default queue to schedule parallel operation" << std::endl;
        }
        auto * q = m_pec->m_default_queue;
        q->enqueue( m_pec );
        m_pec = nullptr;
        q->schedule_all();
        q->wait();
      }
    }

    
  }

}

