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

#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for_adapter.h>

namespace onika
{

  namespace parallel
  {

    const char* omp_scheduling_as_string(OMPScheduling sched)
    {
      switch(sched)
      {
        case OMP_SCHED_DYNAMIC: return "OMP_SCHED_DYNAMIC";
        case OMP_SCHED_GUIDED: return "OMP_SCHED_GUIDED";
        case OMP_SCHED_STATIC: return "OMP_SCHED_STATIC";
      }
      return "<unkown>";
    }

    void dmesg_gpu_start_kernel(void* userData)
    {
      ParallelExecutionContext * pec = (ParallelExecutionContext*) userData;
#     pragma omp critical(dbg_mesg)
      printf("Cuda start %s/%s : stream=%d\n",pec->m_tag,pec->m_sub_tag,int(pec->m_stream->m_stream_id));
    }
    
    void dmesg_gpu_end_kernel(void* userData)
    {
      ParallelExecutionContext * pec = (ParallelExecutionContext*) userData;
#     pragma omp critical(dbg_mesg)
      printf("Cuda end %s/%s\n",pec->m_tag,pec->m_sub_tag);
    }

    void dmesg_omp_start_kernel(ParallelExecutionContext * pec)
    {
#     pragma omp critical(dbg_mesg)
      printf("OpenMP start %s/%s : stream=%d, tasks=%d, sched=%s\n",pec->m_tag,pec->m_sub_tag,int(pec->m_stream->m_stream_id),int(pec->m_omp_num_tasks),omp_scheduling_as_string(pec->m_omp_sched));
    }
    
    void dmesg_omp_end_kernel(ParallelExecutionContext * pec)
    {
#     pragma omp critical(dbg_mesg)
      printf("OpenMP end %s/%s\n",pec->m_tag,pec->m_sub_tag);
    }

  }

}

