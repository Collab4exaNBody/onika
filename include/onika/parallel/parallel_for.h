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

#include <onika/parallel/block_parallel_for.h>

#ifndef ONIKA_PARFOR_OMPSCHED_DEFAULT
#define ONIKA_PARFOR_OMPSCHED_DEFAULT OMP_SCHED_GUIDED
#endif

namespace onika
{

  namespace parallel
  {  

    template<class FuncT> struct ParallelForFunctorTraits
    {      
      static inline constexpr bool CudaCompatible = false;
    };

    /*
     * ParallelForOptions holds options passed to block_parallel_for
     */
    struct ParallelForOptions
    {
      ParallelExecutionCallback user_cb = {};
      void * return_data = nullptr;
      size_t return_data_size = 0;
      bool enable_gpu = true;
      OMPScheduling omp_scheduling = ONIKA_PARFOR_OMPSCHED_DEFAULT;
    };

    template<class FuncT, class PES> struct ParallelForBlockAdapter;
    
    template<class FuncT, unsigned int ND>
    struct ParallelForBlockAdapter< FuncT, ParallelExecutionSpace<ND,0> >
    {
      const FuncT m_func;
      const onikaInt3_t m_offset;
      const onikaDim3_t m_dims;
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( ssize_t block_idx ) const
      {
        if constexpr ( ND == 1 )
        {
          ssize_t i = block_idx * ONIKA_CU_BLOCK_SIZE + ONIKA_CU_THREAD_IDX;
          if( i>=0 && i<m_dims.x ) m_func( m_offset.x + i );
        }
      }
      ONIKA_HOST_DEVICE_FUNC inline void operator () ( const onikaInt3_t& block_coord ) const
      {
        if constexpr ( ND > 1 )
        {
          const ssize_t i = block_coord.x * ONIKA_CU_BLOCK_DIMS.x + ONIKA_CU_THREAD_COORD.x;
          const ssize_t j = block_coord.y * ONIKA_CU_BLOCK_DIMS.y + ONIKA_CU_THREAD_COORD.y;
          const ssize_t k = block_coord.z * ONIKA_CU_BLOCK_DIMS.z + ONIKA_CU_THREAD_COORD.z;
          //printf("block %d,%d,%d : thread %d,%d,%d\n",int(block_coord.x),int(block_coord.y),int(block_coord.z), int(ONIKA_CU_THREAD_COORD.x), int(ONIKA_CU_THREAD_COORD.y), int(ONIKA_CU_THREAD_COORD.z) );
          if( i>=0 && i<m_dims.x
           && j>=0 && j<m_dims.y 
           && k>=0 && k<m_dims.z )
          {
            //printf("block %d,%d,%d : thread %d,%d,%d\n",int(block_coord.x),int(block_coord.y),int(block_coord.z), int(ONIKA_CU_THREAD_COORD.x), int(ONIKA_CU_THREAD_COORD.y), int(ONIKA_CU_THREAD_COORD.z) );
            m_func( onikaInt3_t{ m_offset.x + i , m_offset.y + j , m_offset.z + k } );
          }
        }
      }
    };

    template<class FuncT, class PES> struct BlockParallelForFunctorTraits< ParallelForBlockAdapter<FuncT,PES> >
    {      
      static inline constexpr bool CudaCompatible = ParallelForFunctorTraits<FuncT>::CudaCompatible;
    };

    template< class FuncT >
    static inline
    ParallelExecutionWrapper
    parallel_for(
        uint64_t N
      , const FuncT& func
      , ParallelExecutionContext * pec
      , const ParallelForOptions& opts = ParallelForOptions{} )
    {          
      BlockParallelForOptions bpfopts = {};
      bpfopts.user_cb = opts.user_cb;
      bpfopts.return_data = opts.return_data;
      bpfopts.return_data_size = opts.return_data_size;
      bpfopts.enable_gpu = opts.enable_gpu;
      bpfopts.omp_scheduling = opts.omp_scheduling;
      bpfopts.n_div_blocksize = true;
      
      using PES = ParallelExecutionSpace<1,0>;
      const onikaInt3_t offset = {0,0,0};
      const onikaDim3_t dims = {static_cast<unsigned int>(N),1,1};
      return block_parallel_for( PES{ {0} , {static_cast<ssize_t>(N)} } , ParallelForBlockAdapter<FuncT,PES>{func,offset,dims} , pec , bpfopts );
    }

    template<class FuncT, unsigned int ND, unsigned int ElemND>
    static inline
    ParallelExecutionWrapper
    parallel_for(
        ParallelExecutionSpace<ND,ElemND> par_space
      , const FuncT& func
      , ParallelExecutionContext * pec
      , const ParallelForOptions& opts = ParallelForOptions{} )
    {
      using PES = ParallelExecutionSpace<ND,ElemND>;
      static_assert( ElemND==0 , "Index list not supported yet" );
      static_assert( ND <= 3 );
             
      BlockParallelForOptions bpfopts = {};
      bpfopts.user_cb = opts.user_cb;
      bpfopts.return_data = opts.return_data;
      bpfopts.return_data_size = opts.return_data_size;
      bpfopts.enable_gpu = opts.enable_gpu;
      bpfopts.omp_scheduling = opts.omp_scheduling;
      bpfopts.n_div_blocksize = true;
      
      onikaInt3_t offset = { par_space.m_start[0] ,  0 , 0 };
      onikaDim3_t dims = { static_cast<unsigned int>( par_space.m_end[0] - par_space.m_start[0] ) , 1 , 1 };
      if constexpr ( ND >= 2 )
      {
        offset.y = par_space.m_start[1];
        dims.y = par_space.m_end[1] - par_space.m_start[1];
      }
      if constexpr ( ND >= 3 )
      {
        offset.z = par_space.m_start[2];
        dims.z = par_space.m_end[2] - par_space.m_start[2];
      }
      for(unsigned int i=0;i<ND;i++)
      {
        par_space.m_end[i] -= par_space.m_start[i];
        par_space.m_start[i] = 0;
        lout << "start["<<i<<"] = " <<par_space.m_start[i]<<" , end["<<i<<"] = "<<par_space.m_end[i]<<std::endl;
      }
      lout << "offset=("<<offset.x<<","<<offset.y<<","<<offset.z<<") dims=("<<dims.x<<","<<dims.y<<","<<dims.z<<")"<<std::endl;
      return block_parallel_for( par_space , ParallelForBlockAdapter<FuncT,PES>{func,offset,dims} , pec , bpfopts );
    }


  }

}

