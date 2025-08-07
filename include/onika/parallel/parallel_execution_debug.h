#pragma once

#include <onika/parallel/constants.h>
#include <omp.h>

namespace onika
{
  namespace parallel
  {
    class ParallelExecutionContext;

    const char* omp_scheduling_as_string(OMPScheduling sched);

    void dmesg_exec_gpu(void* userData);

    void dmesg_end_gpu(void* userData);

    void dmesg_sched_omp(ParallelExecutionContext * pec);

    void dmesg_exec_omp(ParallelExecutionContext * pec);

    void dmesg_end_omp(ParallelExecutionContext * pec);

  }
}

