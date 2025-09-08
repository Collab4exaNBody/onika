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

#include <mutex>
#include <vector>
#include <unordered_set>
#include <string_view>

namespace onika
{

  namespace parallel
  {
    
    struct ParallelExecutionContextAllocator
    {
      std::vector< ParallelExecutionContext* > m_free_parallel_execution_contexts;
      std::unordered_set< ParallelExecutionContext* > m_allocated_parallel_execution_contexts;
      std::mutex m_mutex;

      ~ParallelExecutionContextAllocator();
      
      ParallelExecutionContext* create(
          std::string_view tag
        , std::string_view sub_tag = {""}
        , ParallelExecutionQueue * default_queue_ptr = nullptr
        , onika::cuda::CudaContext* default_cuda_ctx = cuda::CudaContext::default_cuda_ctx()
        , int omp_num_tasks = -1 // -1 means autodetectect, 0 => means fork-join parallelism (parallel/for construct)
        , ParallelExecutionFinalize && on_terminate_func = {nullptr,nullptr} );
        
      void free(ParallelExecutionContext* pec);
      static void free_cb(ParallelExecutionContext* pec, void * v_self);
    };
  }

}

