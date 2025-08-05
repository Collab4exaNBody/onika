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
#pragma once

#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_execution_queue.h>
#include <onika/log.h>

#include <mutex>
#include <vector>
#include <unordered_set>
#include <string_view>
#include <sstream>

namespace onika
{

  namespace parallel
  {
    
    struct ParallelExecutionContextAllocator
    {
      std::vector< ParallelExecutionContext* > m_free_parallel_execution_contexts;
      std::unordered_set< ParallelExecutionContext* > m_allocated_parallel_execution_contexts;
      std::mutex m_mutex;

      inline ~ParallelExecutionContextAllocator()
      {
        if( ! m_allocated_parallel_execution_contexts.empty() )
        {
          auto err_log = fatal_error();
          err_log << "m_allocated_parallel_execution_contexts is not empty :";
          for(auto pec : m_allocated_parallel_execution_contexts)
          {
            err_log << " @"<<(void*)pec<<" from "<<pec->m_tag;
          }
          err_log << std::endl;
        }
        for(auto pec : m_free_parallel_execution_contexts)
        {
          assert( pec != nullptr );
          delete pec;
        }
      }

      inline ParallelExecutionContext* create(
          std::string_view tag
        , std::string_view sub_tag = {""}
        , ParallelExecutionQueue * default_queue_ptr = nullptr
        , onika::cuda::CudaContext* default_cuda_ctx = cuda::CudaContext::default_cuda_ctx()
        , int omp_num_tasks = 0
        , ParallelExecutionFinalize && on_terminate_func = {nullptr,nullptr} )
      {
        if( default_queue_ptr == nullptr ) default_queue_ptr = & ParallelExecutionQueue::default_queue();
        
        std::lock_guard<std::mutex> lock(m_mutex);
        
        if( m_free_parallel_execution_contexts.empty() )
        {
          m_free_parallel_execution_contexts.push_back( new ParallelExecutionContext() );
        }
        auto pec = m_free_parallel_execution_contexts.back();
        m_free_parallel_execution_contexts.pop_back();
        m_allocated_parallel_execution_contexts.insert( pec );

        pec->reset();
        pec->m_tag = tag.data();
        pec->m_sub_tag = sub_tag.data();
        pec->m_cuda_ctx = default_cuda_ctx;
        pec->m_default_queue = default_queue_ptr;
        pec->m_omp_num_tasks = omp_num_tasks;
        pec->initialize_stream_events();
        if( on_terminate_func.m_func == nullptr ) pec->m_finalize = ParallelExecutionFinalize {destroy_cb,this};
        else pec->m_finalize = std::move(on_terminate_func);
        return pec;
      }

      inline void destroy(ParallelExecutionContext* pec)
      {
        assert( pec != nullptr );
        pec->reset();

        std::lock_guard<std::mutex> lock(m_mutex);

        auto it = m_allocated_parallel_execution_contexts.find( pec );
        assert( it != m_allocated_parallel_execution_contexts.end() );
        m_free_parallel_execution_contexts.push_back( *it );
        m_allocated_parallel_execution_contexts.erase( it );
      }

      static inline void destroy_cb(ParallelExecutionContext* pec, void * v_self)
      {
        ParallelExecutionContextAllocator* self = reinterpret_cast<ParallelExecutionContextAllocator*>( v_self );
        assert( self != nullptr );
        self->destroy(pec);
      }
    };
  }

}

