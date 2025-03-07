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

namespace onika { namespace scg
{

  struct Grid3DBenchmarkFunctor
  {
    double * const __restrict__ m_array = nullptr;
    const long m_size = 0;
  
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( onikaInt3_t coord ) const
    {
      //ONIKA_CU_SHARED sum;
      const ssize_t idx = ONIKA_CU_BLOCK_IDX;
      const double x = m_array[idx];
      const double y = x*x - 2*x + 1;
      m_array[idx] = x + y;
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
  
    ADD_SLOT( long        , grid_size  , INPUT , 256 , DocString{"Number of terms to compute"} );
    ADD_SLOT( long        , block_size , INPUT , 256 , DocString{"Thread teams (aka Cuda block) size"} );
    ADD_SLOT( DoubleArray , scratch    , PRIVATE );

  public:

    inline void execute () override final
    {
      using onika::parallel::ParallelExecutionSpace;
    
      const ssize_t N = *grid_size;
      scratch->resize( N * N * N , 0.0 );
      Grid3DBenchmarkFunctor benchmark = { scratch->data() , N };
      ParallelExecutionSpace<3> grid = { {0,0,0} , {N,N,N} };
      block_parallel_for( grid , benchmark , parallel_execution_context() );
    }
  };
  
  // === register factories ===  
  ONIKA_AUTORUN_INIT(parallel_for_benchmark)
  {
   OperatorNodeFactory::instance()->register_factory( "parallel_for_3d_benchmark", make_compatible_operator< ParallelFor3DBenchmark > );
  }

} }


