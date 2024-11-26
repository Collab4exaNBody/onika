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
#include <onika/parallel/parallel_for.h>

namespace onika { namespace scg
{

  struct IterativeBenchmarkFunctor
  {
    double * const __restrict__ m_array = nullptr;
    const long m_iterations = 0;
  
    ONIKA_HOST_DEVICE_FUNC inline void operator () ( size_t idx ) const
    {
      double x = m_array[idx];
      double y = pow(x,sin(x));
      for(long i=0;i<m_iterations;i++)
      {
        x = x + y;
        y = pow(x,sin(x));
      }
      m_array[idx] = x * y;
    }
  };

} }


namespace onika { namespace parallel
{
  template<> struct ParallelForFunctorTraits< onika::scg::IterativeBenchmarkFunctor >
  {      
    static inline constexpr bool CudaCompatible = true;
  };
}}

namespace onika { namespace scg
{

  class ParallelForBenchmark : public OperatorNode
  {
    using DoubleArray = onika::memory::CudaMMVector<double>;
  
    ADD_SLOT( long        , samples    , INPUT , 4096 , DocString{"Number of terms to compute"} );
    ADD_SLOT( long        , iterations , INPUT , 4096 , DocString{"Number of iterations for each term"} );
    ADD_SLOT( long        , block_size , INPUT ,  256 , DocString{"Thread teams (aka Cuda block) size"} );
    ADD_SLOT( DoubleArray , scratch    , PRIVATE );

  public:

    inline void execute () override final
    {
      scratch->resize( *samples );
      IterativeBenchmarkFunctor benchmark = { scratch->data() , *iterations };
      parallel_for( *samples , benchmark , parallel_execution_context() );
    }
  };
  
  // === register factories ===  
  ONIKA_AUTORUN_INIT(parallel_for_benchmark)
  {
   OperatorNodeFactory::instance()->register_factory( "parallel_for_benchmark", make_compatible_operator< ParallelForBenchmark > );
  }

} }


