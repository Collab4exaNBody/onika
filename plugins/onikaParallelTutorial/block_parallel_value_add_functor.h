#pragma once

#include <onika/parallel/block_parallel_for.h>
#include <onika/cuda/cuda.h>
#include "array2d.h"

// a specific namespace for our application space
namespace onika
{
  using namespace scg;

  namespace tutorial
  {
    
    // a functor = function applied in parallel
    // it is defined by a class (or struct) with the call operator ()
    template<bool AllowGpuExec = true>
    struct BlockParallelValueAddFunctor
    {
      Array2DReference m_array;
      double m_value_to_add = 0.0; // value to add

      ONIKA_HOST_DEVICE_FUNC            // works on CPU and GPU
      void operator () (size_t i) const // call operator with i in [0;n[
      {                                 // a whole block (all its threads) execute iteration i
        const size_t cols = m_array.columns();
        ONIKA_CU_BLOCK_SIMD_FOR(size_t, j, 0, cols)   // parallelization among the threads of the current block
        {                                             // for iterations on j in [0;columns[
          m_array[i][j] += m_value_to_add; // each thread executes 0, 1, or multiple iterations of j
        }
      }
    };

    template<bool AllowGpuExec = true>
    struct BlockParallelIterativeValueAddFunctor
    {
      Array2DReference m_array;
      double m_value_to_add = 0.0; // value to add
      const long m_iterations = 0;

      ONIKA_HOST_DEVICE_FUNC            // works on CPU and GPU
      void operator () (size_t i) const // call operator with i in [0;n[
      {                                 // a whole block (all its threads) execute iteration i
        const size_t cols = m_array.columns();
        ONIKA_CU_BLOCK_SIMD_FOR(size_t, j, 0, cols)   // parallelization among the threads of the current block
        {                                             // for iterations on j in [0;columns[
          double x = m_array[i][j];
          double y = pow(x,sin(x));
          for(long k=0;k<m_iterations;k++)
          {
            x = x + y + m_value_to_add;
            y = pow(x, sin(x) );
          }
          m_array[i][j] = x * y;
        }
      }
    };

    /*
    struct BlockParallelValueAddFunctor2D
    {
      Array2DReference m_array;
      double m_value_to_add = 0.0; // value to add

      ONIKA_HOST_DEVICE_FUNC            // works on CPU and GPU
      void operator () (size_t i) const // call operator with i in [0;n[
      {                                 // a whole block (all its threads) execute iteration i
        const size_t cols = m_array.columns();
        ONIKA_CU_BLOCK_SIMD_FOR(size_t, j, 0, cols)   // parallelization among the threads of the current block
        {                                             // for iterations on j in [0;columns[
          m_array[i][j] += m_value_to_add; // each thread executes 0, 1, or multiple iterations of j
        }
      }
    };
    */
    
  }

  namespace parallel
  {
    // specialization of BlockParallelForFunctorTraits, in the onika namespace,
    // allows to specify some compile time properties of our functor, like Cuda/HIP compatibility
    template<bool GpuEnable>
    struct BlockParallelForFunctorTraits<tutorial::BlockParallelValueAddFunctor<GpuEnable> >
    {
      static constexpr bool CudaCompatible = GpuEnable; // or false to prevent the code from being compiled with CUDA
    };

    template<bool GpuEnable>
    struct BlockParallelForFunctorTraits<tutorial::BlockParallelIterativeValueAddFunctor<GpuEnable> >
    {
      static constexpr bool CudaCompatible = GpuEnable; // or false to prevent the code from being compiled with CUDA
    };
  }

}


