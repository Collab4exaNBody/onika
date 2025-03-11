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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/cuda/cuda.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>

namespace onika { namespace scg
{

  struct GridBlock3DBenchmarkFunctor
  {
    double * const __restrict__ m_array = nullptr;
    const long m_size = 0;
  
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( onikaInt3_t coord ) const
    {
      //ONIKA_CU_SHARED sum;
      const ssize_t idx = ONIKA_CU_BLOCK_IDX;
      ONIKA_CU_ATOMIC_ADD( m_array[idx] , 1.0 );
    }
  };

  struct Grid3DBenchmarkFunctor
  {
    double * const __restrict__ m_array = nullptr;
    const long m_size = 0;
  
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( onikaInt3_t coord ) const
    {
      //printf("KERNEL %d,%d,%d\n",int(coord.x),int(coord.y),int(coord.z));
      const unsigned long N = m_size;
      const unsigned long i = coord.x;
      const unsigned long j = coord.y;
      const unsigned long k = coord.z;
      const ssize_t idx = (k*N+j)*N+i;
      ONIKA_CU_ATOMIC_ADD( m_array[idx] , 1.0 );
    }
  };

} }


namespace onika { namespace parallel
{
  template<> struct BlockParallelForFunctorTraits< onika::scg::Grid3DBenchmarkFunctor >
  {      
    static inline constexpr bool CudaCompatible = true;
  };
}}

namespace onika { namespace scg
{

  class ParallelFor3DBenchmark : public OperatorNode
  {
    using DoubleArray = onika::memory::CudaMMVector<double>;
  
    ADD_SLOT( long        , pfor3d_block_side , INPUT , 4 , DocString{"Thread teams (aka Cuda block) size"} );
    ADD_SLOT( long        , pfor3d_side , INPUT , 16 , DocString{"Number of terms to compute"} );
    ADD_SLOT( DoubleArray , scratch     , PRIVATE );

  public:

    inline void execute () override final
    {
      using onika::parallel::ParallelExecutionSpace;
      using onika::parallel::block_parallel_for;
      using onika::parallel::parallel_for;
      
      lout << "block_parallel_for 3D test" << std::endl;
      {
        const ssize_t N = *pfor3d_block_side;
        scratch->assign( N * N * N , 0.0 );
        Grid3DBenchmarkFunctor benchmark = { scratch->data() , N };
        ParallelExecutionSpace<3> parallel_range = { {0,0,0} , {N,N,N} };
        block_parallel_for( parallel_range , benchmark , parallel_execution_context() );
        for(int k=0;k<N;k++)
        {
          lout<<"K = "<<k<<std::endl;
          for(int j=0;j<N;j++)
          {
            lout<<"  J = "<<j<<std::endl<<"   ";
            for(int i=0;i<N;i++)
            {
              lout <<" "<< i<<":"<<scratch->at( (k*N+j)*N+i );
            }
            lout << std::endl;
          }
        }
      }

      lout << "parallel_for 3D test" << std::endl;
      {
        const ssize_t N = *pfor3d_side;
        scratch->assign( N * N * N , 0.0 );
        Grid3DBenchmarkFunctor benchmark = { scratch->data() , N };
        ParallelExecutionSpace<3> parallel_range = { {2,3,3} , {N-1,N-3,N-3} };
        
        parallel_for( parallel_range , benchmark , parallel_execution_context() );

        for(int k=0;k<N;k++)
        {
          lout<<"K="<<k<<std::endl;
          for(int j=0;j<N;j++)
          {
            lout<<"  J="<<std::setw(3)<<j<<" :";
            for(int i=0;i<N;i++)
            {
              lout <<" "<<int( scratch->at( (k*N+j)*N+i ) );
            }
            lout << std::endl;
          }
        }

      }
    }
  };
  
  // === register factories ===  
  ONIKA_AUTORUN_INIT(parallel_for_benchmark)
  {
   OperatorNodeFactory::instance()->register_factory( "parallel_for_3d_benchmark", make_compatible_operator< ParallelFor3DBenchmark > );
  }

} }


