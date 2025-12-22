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

#include <onika/parallel/parallel_execution_debug.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_execution_stream.h>

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

    void dmesg_exec_gpu(void* userData)
    {
      ParallelExecutionContext * pec = (ParallelExecutionContext*) userData;
#     pragma omp critical(dbg_mesg)
      printf("exec.%s %s/%s : stream=%d\n",ONIKA_CU_NAME_STR,pec->m_tag,pec->m_sub_tag,int(pec->m_stream->m_stream_id));
    }

    void dmesg_end_gpu(void* userData)
    {
      ParallelExecutionContext * pec = (ParallelExecutionContext*) userData;
#     pragma omp critical(dbg_mesg)
      printf("end.%s %s/%s\n",ONIKA_CU_NAME_STR,pec->m_tag,pec->m_sub_tag);
    }

    void dmesg_exec_omp(ParallelExecutionContext * pec)
    {
#     pragma omp critical(dbg_mesg)
      printf("exec.omp %s/%s : stream=%d, tasks=%d, sched=%s, master=%08lX, team=%d\n",
             pec->m_tag,pec->m_sub_tag,int(pec->m_stream->m_stream_id),
             int(pec->m_omp_num_tasks),omp_scheduling_as_string(pec->m_omp_sched),
             std::hash<std::thread::id>{}(std::this_thread::get_id()),
             pec->m_omp_num_tasks==0 ? omp_get_max_threads() : omp_get_num_threads() );
    }

    void dmesg_sched_omp(ParallelExecutionContext * pec)
    {
#     pragma omp critical(dbg_mesg)
      printf("sched.omp %s/%s : stream=%d, tasks=%d, master=%08lX, team=%d\n",
             pec->m_tag,pec->m_sub_tag,int(pec->m_stream->m_stream_id),int(pec->m_omp_num_tasks),
             std::hash<std::thread::id>{}(std::this_thread::get_id()),
             pec->m_omp_num_tasks==0 ? omp_get_max_threads() : omp_get_num_threads() );
    }

    void dmesg_end_omp(ParallelExecutionContext * pec)
    {
#     pragma omp critical(dbg_mesg)
      printf("end.omp %s/%s\n",pec->m_tag,pec->m_sub_tag);
    }

  }

}

