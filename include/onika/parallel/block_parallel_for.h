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

#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/soatl/field_id.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/block_parallel_for_adapter.h>
#include <onika/lambda_tools.h>
#include <onika/stream_utils.h>

#ifndef ONIKA_BLKPARFOR_OMPSCHED_DEFAULT
#define ONIKA_BLKPARFOR_OMPSCHED_DEFAULT OMP_SCHED_GUIDED
#endif

namespace onika
{

  namespace parallel
  {  
    /*
     * BlockParallelForOptions holds options passed to block_parallel_for
     */
    struct BlockParallelForOptions
    {
      ParallelExecutionCallback user_cb = {};
      void * return_data = nullptr;
      size_t return_data_size = 0;
      unsigned int max_block_size = ONIKA_CU_MAX_THREADS_PER_BLOCK;
      bool enable_gpu = true;
      bool fixed_gpu_grid_size = false;
      bool n_div_blocksize = false; // if true, divide N by block_size, rounding to upper integer
      OMPScheduling omp_scheduling = ONIKA_BLKPARFOR_OMPSCHED_DEFAULT;
    };

    template< class FuncT , unsigned int ND, unsigned int ElemND>
    static inline
    ParallelExecutionWrapper
    block_parallel_for(
        ParallelExecutionSpace<ND,ElemND> par_space
      , const FuncT& func
      , ParallelExecutionContext * pec
      , const BlockParallelForOptions& opts = BlockParallelForOptions{} )
    {
      static_assert( lambda_is_compatible_with_v<FuncT,void,uint64_t> , "Functor in argument is incompatible with void(uint64_t) call signature" );
      using HostFunctorAdapter = BlockParallelForHostAdapter< FuncT , functor_gpu_support_v<FuncT> , ND,ElemND >;
      [[maybe_unused]] static constexpr AssertFunctorSizeFitIn< sizeof(HostFunctorAdapter) , HostKernelExecutionScratch::MAX_FUNCTOR_SIZE , FuncT > _check_cpu_functor_size = {};
      assert( pec != nullptr );

      // construct virtual functor adapter inplace, using reserved functor space
      static_assert( ( HostKernelExecutionScratch::MAX_FUNCTOR_ALIGNMENT % alignof(FuncT) ) == 0 , "functor_data alignment is not sufficient for user functor" );

      pec->m_execution_end_callback = opts.user_cb;
      pec->m_omp_sched = opts.omp_scheduling;
    
	    // printf("block_parallel_for: %s %s: cudacompat=%d\n", pec->m_tag != nullptr ? pec->m_tag : "<null>" , pec->m_sub_tag != nullptr ? pec->m_sub_tag : "" , int(  ) );
      bool allow_cuda_exec = false ;
      if constexpr ( functor_gpu_support_v<FuncT> )
      {
        allow_cuda_exec = opts.enable_gpu ;
        if( allow_cuda_exec ) allow_cuda_exec = ( pec->m_cuda_ctx != nullptr );
        if( allow_cuda_exec ) allow_cuda_exec = pec->m_cuda_ctx->has_devices();
        if( allow_cuda_exec )
        {
          pec->m_execution_target = ParallelExecutionContext::EXECUTION_TARGET_CUDA;
          pec->m_block_size = std::min( size_t(opts.max_block_size) , std::min( size_t(ONIKA_CU_MAX_THREADS_PER_BLOCK) , size_t(onika::parallel::ParallelExecutionContext::gpu_block_size()) ) );
          pec->m_grid_size = pec->m_cuda_ctx->m_devices[0].m_deviceProp.multiProcessorCount
                                      * onika::parallel::ParallelExecutionContext::gpu_sm_mult()
                                      + onika::parallel::ParallelExecutionContext::gpu_sm_add();

          if( opts.n_div_blocksize )
          {
            assert( ND == 1 &&  par_space.m_start[0]==0 );
            par_space.m_end[0] = ( par_space.m_end[0] + pec->m_block_size - 1 ) / pec->m_block_size;
          }

          if( ! opts.fixed_gpu_grid_size )
          { 
            pec->m_grid_size = 0;
          }

          pec->m_reset_counters = opts.fixed_gpu_grid_size;

          if( opts.return_data != nullptr && opts.return_data_size > 0 )
          {
            pec->set_return_data_input( opts.return_data , opts.return_data_size );
            pec->set_return_data_output( opts.return_data , opts.return_data_size );
          }
          else
          {
            pec->set_return_data_input( nullptr , 0 );
            pec->set_return_data_output( nullptr , 0 );
          }
        }
      }

      // ================== CPU / OpenMP execution path ====================
      if( ! allow_cuda_exec )
      {
        pec->m_execution_target = ParallelExecutionContext::EXECUTION_TARGET_OPENMP;  
      }
      
      new(pec->m_host_scratch.functor_data) HostFunctorAdapter( func , par_space );
      return {pec};
    }

    template<class FuncT>
    static inline
    ParallelExecutionWrapper
    block_parallel_for(
        size_t N
      , const FuncT& func
      , ParallelExecutionContext * pec
      , const BlockParallelForOptions& opts = BlockParallelForOptions{} )
    {
      return block_parallel_for( ParallelExecutionSpace<1>{ {0} , {ssize_t(N)} }, func, pec, opts );
    }

  }

}

