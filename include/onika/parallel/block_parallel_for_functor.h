#pragma once

#include <onika/cuda/cuda.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/lambda_tools.h>

namespace onika
{
  namespace parallel
  {
    // this template is here to know if compute buffer must be built or computation must be ran on the fly
    template<class FuncT> struct BlockParallelForFunctorTraits
    {      
      static inline constexpr bool CudaCompatible = false;
    };

    template<class FuncT> struct FunctorNeedsGPUSupport : public
#   ifdef ONIKA_CUDA_VERSION
      std::integral_constant<bool,BlockParallelForFunctorTraits<FuncT>::CudaCompatible>
#   else
      std::integral_constant<bool,false>
#   endif
    {};
    template<class FuncT> static inline constexpr bool functor_gpu_support_v = FunctorNeedsGPUSupport<FuncT>::value;

    // user can add an overloaded call operator taking one of this type as its only parameter
    // an overload with block_parallel_for_prolog_t will be used both as CPU and GPU launch prolog
    // while and overload with block_parallel_for_gpu_prolog_t will e called only in case of a GPU launch
    // and similarily with block_parallel_for_cpu_prolog_t
    struct block_parallel_for_prolog_t {};
    struct block_parallel_for_gpu_prolog_t : public block_parallel_for_prolog_t {};
    struct block_parallel_for_cpu_prolog_t : public block_parallel_for_prolog_t {};

    // same as block_parallel_for_prolog_t but for end of parallel for execution
    struct block_parallel_for_epilog_t {};
    struct block_parallel_for_gpu_epilog_t : public block_parallel_for_epilog_t {};
    struct block_parallel_for_cpu_epilog_t : public block_parallel_for_epilog_t {};

    // marker for single function call with a single task (no parallelism) event though parallel execution space span several elements
    struct block_parallel_for_single_task_t {};

    template<long long FunctorSize, long long MaxSize, class FuncT>
    struct AssertFunctorSizeFitIn
    {
      std::enable_if_t< (FunctorSize <= MaxSize) , int > x = 0;
    };

    class BlockParallelForHostFunctor
    {
    public:
      // Host batch execution interface
      virtual inline void execute_prolog( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const {}
      virtual inline void execute_omp_parallel_region( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const {}
      virtual inline void execute_omp_tasks( ParallelExecutionContext* pec, ParallelExecutionStream* pes, unsigned int num_tasks ) const {}
      virtual inline void execute_epilog( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const {}

      // Host individual task execution interface
      virtual inline void operator () (uint64_t i) const {}
      virtual inline void operator () (uint64_t i, uint64_t end) const { for(;i<end;i++) this->operator () (i); }
      virtual inline void operator () (const uint64_t * __restrict__ idx, uint64_t N) const { for(uint64_t i=0;i<N;i++) this->operator () (idx[i]); }

      virtual inline void operator () (const onikaInt3_t& c) const {}
      virtual inline void operator () (const onikaInt3_t& s, const onikaInt3_t& e) const
      {
        for(ssize_t k=s.z;k<e.z;k++) 
        for(ssize_t j=s.y;j<e.y;j++) 
        for(ssize_t i=s.x;i<e.x;i++)
        {
          this->operator () ( onikaInt3_t{i,j,k} );
        }
      }
      virtual inline void operator () (const onikaInt3_t * __restrict__ idx, uint64_t N) const
      {
        for(uint64_t i=0;i<N;i++) this->operator () ( idx[i] );
      }

      // GPU Kernel launch interface
      virtual inline void stream_gpu_initialize(ParallelExecutionContext*,ParallelExecutionStream*) const {}
      virtual inline void stream_gpu_kernel(ParallelExecutionContext*,ParallelExecutionStream*) const {}
      virtual inline void stream_gpu_finalize(ParallelExecutionContext*,ParallelExecutionStream*) const {}

      // destructor
      virtual inline ~BlockParallelForHostFunctor() {}
    };

  }
}

