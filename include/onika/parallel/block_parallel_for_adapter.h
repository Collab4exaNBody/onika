#pragma once

#include <onika/parallel/block_parallel_for_functor.h>
#include <onika/parallel/parallel_execution_stream.h>
#include <omp.h>

namespace onika
{
  namespace parallel
  {

    // GPU execution kernel for fixed size grid, using workstealing element assignment to blocks
    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    ONIKA_STATIC_INLINE_KERNEL
    void block_parallel_for_gpu_kernel_workstealing( uint64_t N, GPUKernelExecutionScratch* scratch, ONIKA_CU_GRID_CONSTANT const FuncT func )
    {
      // avoid use of compute buffer when possible
      ONIKA_CU_BLOCK_SHARED unsigned int i;
      do
      {
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          i = ONIKA_CU_ATOMIC_ADD( scratch->counters[0] , 1u );
        }
        ONIKA_CU_BLOCK_SYNC();
        if( i < N )
        {
          func( i );
        }
      }
      while( i < N );
    }

    // GPU execution kernel for adaptable size grid, a.k.a. conventional Cuda kernel execution on N element blocks
    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    ONIKA_STATIC_INLINE_KERNEL
    void block_parallel_for_gpu_kernel_regulargrid( ONIKA_CU_GRID_CONSTANT const FuncT func , ONIKA_CU_GRID_CONSTANT const unsigned int start )
    {
      func( start + ONIKA_CU_BLOCK_IDX );
    }

    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    ONIKA_STATIC_INLINE_KERNEL
    void block_parallel_for_gpu_kernel_regulargrid_3D( ONIKA_CU_GRID_CONSTANT const FuncT func , ONIKA_CU_GRID_CONSTANT const onikaInt3_t start_coord )
    {
      func( start_coord + ONIKA_CU_BLOCK_COORD );
    }

    template<class FuncT, bool GPUSupport , unsigned int ND=1, unsigned int ElemND=0>
    class BlockParallelForHostAdapter : public BlockParallelForHostFunctor
    {
      static_assert( !GPUSupport || gpu_frontend_compiler() );

      // what is he dimensionality of processed elements ?
      static inline constexpr unsigned int FuncParamDim = ( ElemND==0 ) ? ND : ElemND ;
      static_assert( FuncParamDim>=1 && FuncParamDim<=3 );      
      using FuncParamType = std::conditional_t< FuncParamDim==1 , ssize_t , onikaInt3_t >;
      static_assert( lambda_is_compatible_with_v<FuncT,void,FuncParamType> , "User defined functor is not compatible with execution space");

      static inline constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_prolog_t>;
      static inline constexpr bool functor_has_cpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_cpu_prolog_t>;
      static inline constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_epilog_t>;
      static inline constexpr bool functor_has_cpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_cpu_epilog_t>;
      
      alignas( alignof(FuncT) ) const FuncT m_func;
      const ParallelExecutionSpace<ND,ElemND> m_parallel_space;
      
    public:
      inline BlockParallelForHostAdapter( const FuncT& f , const ParallelExecutionSpace<ND,ElemND>& ps ) : m_func(f) , m_parallel_space(ps) {}

      // ================== GPU stream based execution interface =======================

      inline void stream_gpu_initialize(ParallelExecutionContext* pec , ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
          static constexpr bool functor_has_gpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_prolog_t,ParallelExecutionStream*>;
          static constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_prolog_t>;
          if constexpr ( functor_has_gpu_prolog ) { m_func( block_parallel_for_gpu_prolog_t{} , pes ); }
          else if constexpr ( functor_has_prolog ) { m_func( block_parallel_for_prolog_t{} ); }
        }
        else { fatal_error() << "called stream_gpu_initialize with no GPU support" << std::endl; }
      }
      
      inline void stream_gpu_kernel(ParallelExecutionContext* pec, ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
          assert( m_parallel_space.m_elements == nullptr );
          
          // launch compute kernel
          const size_t N = m_parallel_space.m_end[0] - m_parallel_space.m_start[0];
          const onikaInt3_t block_offset = { m_parallel_space.m_start[0], m_parallel_space.m_start[1], m_parallel_space.m_start[2] };
          if( pec->m_grid_size.x > 0 )
          {
            if constexpr ( FuncParamDim == 1 )
            {
              ONIKA_CU_LAUNCH_KERNEL(pec->m_grid_size.x,pec->m_block_size.x,0,pes->m_cu_stream, block_parallel_for_gpu_kernel_workstealing, N, pec->m_cuda_scratch.get(), m_func );
            }
            else
            {
              fatal_error() << "Work stealing GPU execution only supported for 1D kernels" << std::endl;
            }
          }
          else
          {
            if constexpr ( FuncParamDim == 1 )
            {            
              ONIKA_CU_LAUNCH_KERNEL(N,pec->m_block_size.x,0,pes->m_cu_stream, block_parallel_for_gpu_kernel_regulargrid, m_func, block_offset.x );
            }
            else if constexpr ( FuncParamDim > 1 )
            {
              onikaDim3_t exec_grid_size = { static_cast<unsigned int>( m_parallel_space.m_end[0] - m_parallel_space.m_start[0] )
                                           , static_cast<unsigned int>( m_parallel_space.m_end[1] - m_parallel_space.m_start[1] )
                                           , static_cast<unsigned int>( m_parallel_space.m_end[2] - m_parallel_space.m_start[2] ) };
              lout << "Kernel execute grid=("<<exec_grid_size.x<<","<<exec_grid_size.y<<","<<exec_grid_size.z<<") , block=("<<pec->m_block_size.x<<","<<pec->m_block_size.y<<","<<pec->m_block_size.z<<")" << std::endl;
              ONIKA_CU_LAUNCH_KERNEL(exec_grid_size , pec->m_block_size , 0 , pes->m_cu_stream , block_parallel_for_gpu_kernel_regulargrid_3D, m_func, block_offset );
            }
          }          
        }
        else
        {
          fatal_error() << "called stream_gpu_kernel with no GPU support" << std::endl;
        }
      }
      
      inline void stream_gpu_finalize(ParallelExecutionContext* pec, ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
          static constexpr bool functor_has_gpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_epilog_t,ParallelExecutionStream*>;
          static constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_epilog_t>;
          if constexpr ( functor_has_gpu_epilog ) { m_func( block_parallel_for_gpu_epilog_t{} , pes ); }
          else if constexpr ( functor_has_epilog ) { m_func( block_parallel_for_epilog_t{} ); }
        }
        else { fatal_error() << "called stream_gpu_finalize with no GPU support" << std::endl; }
      }


      // ================ CPU OpenMP execution interface ======================
      inline void execute_prolog( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {
        if constexpr (functor_has_cpu_prolog) { m_func(block_parallel_for_cpu_prolog_t{}); }
        else if constexpr (functor_has_prolog) { m_func(block_parallel_for_prolog_t{}); }
      }
      
      inline void execute_omp_parallel_region( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {      
        static_assert( FuncParamDim>=1 && FuncParamDim<=3 , "OpenMP backend only support 1D, 2D and 3D parallel execution spaces" );

        pes->m_omp_execution_count.fetch_add(1);
        assert( m_parallel_space.m_start[0] == 0 && m_parallel_space.m_elements == nullptr );
        const size_t N = m_parallel_space.m_end[0];

#       ifdef ONIKA_OMP_NUM_THREADS_WORKAROUND
        omp_set_num_threads( omp_get_max_threads() );
#       endif

        const auto T0 = std::chrono::high_resolution_clock::now();  
        execute_prolog( pec , pes );
        
        if constexpr ( FuncParamDim == 1 )
        {
#         pragma omp parallel
          {
            switch( pec->m_omp_sched )
            {
              case OMP_SCHED_DYNAMIC :
#               pragma omp for schedule(dynamic)
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++) { m_func( i ); }
                break;
              case OMP_SCHED_GUIDED :
#               pragma omp for schedule(guided)
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++) { m_func( i ); }
                break;
              case OMP_SCHED_STATIC :
#               pragma omp for schedule(static)
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++) { m_func( i ); }
                break;
            }
          }
        }
        else if constexpr ( FuncParamDim == 2 )
        {
#         pragma omp parallel
          {
            switch( pec->m_omp_sched )
            {
              case OMP_SCHED_DYNAMIC :
#               pragma omp for collapse(2) schedule(dynamic)
                for(ssize_t j=m_parallel_space.m_start[1];j<m_parallel_space.m_end[1];j++) 
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
                { m_func( onikaInt3_t{i,j,0} ); }
                break;
              case OMP_SCHED_GUIDED :
#               pragma omp for collapse(2) schedule(guided)
                for(ssize_t j=m_parallel_space.m_start[1];j<m_parallel_space.m_end[1];j++) 
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
                { m_func( onikaInt3_t{i,j,0} ); }
                break;
              case OMP_SCHED_STATIC :
#               pragma omp for collapse(2) schedule(static)
                for(ssize_t j=m_parallel_space.m_start[1];j<m_parallel_space.m_end[1];j++) 
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
                { m_func( onikaInt3_t{i,j,0} ); }
                break;
            }
          }
        }
        else if constexpr ( FuncParamDim == 3 )
        {
#         pragma omp parallel
          {
            switch( pec->m_omp_sched )
            {
              case OMP_SCHED_DYNAMIC :
#               pragma omp for collapse(3) schedule(dynamic)
                for(ssize_t k=m_parallel_space.m_start[2];k<m_parallel_space.m_end[2];k++) 
                for(ssize_t j=m_parallel_space.m_start[1];j<m_parallel_space.m_end[1];j++) 
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
                { m_func( onikaInt3_t{i,j,k} ); }
                break;
              case OMP_SCHED_GUIDED :
#               pragma omp for collapse(3) schedule(guided)
                for(ssize_t k=m_parallel_space.m_start[2];k<m_parallel_space.m_end[2];k++) 
                for(ssize_t j=m_parallel_space.m_start[1];j<m_parallel_space.m_end[1];j++) 
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
                { m_func( onikaInt3_t{i,j,k} ); }
                break;
              case OMP_SCHED_STATIC :
#               pragma omp for collapse(3) schedule(static)
                for(ssize_t k=m_parallel_space.m_start[2];k<m_parallel_space.m_end[2];k++) 
                for(ssize_t j=m_parallel_space.m_start[1];j<m_parallel_space.m_end[1];j++) 
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
                { m_func( onikaInt3_t{i,j,k} ); }
                break;
            }
          }
        }
        
        execute_epilog( pec , pes );
        pec->m_total_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
        if( pec->m_execution_end_callback.m_func != nullptr )
        {
          (* pec->m_execution_end_callback.m_func) ( pec->m_execution_end_callback.m_data );
        }
        pes->m_omp_execution_count.fetch_sub(1);
      }

      inline void execute_omp_tasks( ParallelExecutionContext* pec, ParallelExecutionStream* pes, unsigned int num_tasks ) const override final
      {
        pes->m_omp_execution_count.fetch_add(1);
        // encloses a taskgroup inside a task, so that we can wait for a single task which itself waits for the completion of the whole taskloop
        // refrenced variables must be privately copied, because the task may run after this function ends
#       pragma omp task default(none) firstprivate(pec,pes,num_tasks) depend(inout:pes[0])
        {
          assert( m_parallel_space.m_elements == nullptr );
          const size_t Ni = m_parallel_space.m_end[0] - m_parallel_space.m_start[0];
          const size_t Nj = m_parallel_space.m_end[1] - m_parallel_space.m_start[1];
          const size_t Nk = m_parallel_space.m_end[2] - m_parallel_space.m_start[2];
          const size_t N = Ni*Nj*Nk;
          const auto T0 = std::chrono::high_resolution_clock::now();
          execute_prolog( pec , pes );
          if( N > 0 )
          {
            // implicit taskgroup, ensures taskloop has completed before enclosing task ends
            // all refrenced variables can be shared because of implicit enclosing taskgroup
            const auto & func = m_func;
            const auto ps = m_parallel_space;
            if constexpr ( FuncParamDim==1 )
            {
#             pragma omp taskloop default(none) shared(pec,num_tasks,func,ps) num_tasks(num_tasks)
              for(ssize_t i=ps.m_start[0] ; i<ps.m_end[0] ; i++ )
              {
                func( i );
              }
            }
            else if constexpr ( FuncParamDim>1 )
            {
#             pragma omp taskloop collapse(3) default(none) shared(pec,num_tasks,func,ps) num_tasks(num_tasks)
              for(ssize_t k=ps.m_start[2];k<ps.m_end[2];k++) 
              for(ssize_t j=ps.m_start[1];j<ps.m_end[1];j++) 
              for(ssize_t i=ps.m_start[0];i<ps.m_end[0];i++)
              {
                func( onikaInt3_t{i,j,k} );
              }
            }
          }
          // here all tasks of taskloop have completed, since notaskgroup clause is not specified              
          execute_epilog( pec , pes );          
          pec->m_total_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
          if( pec->m_execution_end_callback.m_func != nullptr )
          {
            (* pec->m_execution_end_callback.m_func) ( pec->m_execution_end_callback.m_data );
          }
          pes->m_omp_execution_count.fetch_sub(1);
        }
      }

      inline void execute_epilog( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {
        if constexpr (functor_has_cpu_epilog) { m_func(block_parallel_for_cpu_epilog_t{}); }
        else if constexpr (functor_has_epilog) { m_func(block_parallel_for_epilog_t{}); }
      }

      // ================ CPU individual task execution interface ======================
      inline void operator () (uint64_t i) const override final
      {
        if constexpr (FuncParamDim==1) { m_func(i); }
      }
      inline void operator () (const onikaInt3_t& c) const override final
      {
        if constexpr (FuncParamDim>1) { m_func(c); }
      }
      inline void operator () (uint64_t i, uint64_t end) const override final
      {
        for(;i<end;i++) this->operator () (i);
      }
      inline void operator () (const uint64_t * __restrict__ idx, uint64_t N) const override final
      {
        for(uint64_t i=0;i<N;i++) this->operator () (idx[i]);
      }
      inline void operator () (const onikaInt3_t& s, const onikaInt3_t& e) const override final
      {
        for(ssize_t k=s.z;k<e.z;k++) 
        for(ssize_t j=s.y;j<e.y;j++) 
        for(ssize_t i=s.x;i<e.x;i++)
        {
          this->operator () ( onikaInt3_t{i,j,k} );
        }
      }
      inline void operator () (const onikaInt3_t * __restrict__ idx, uint64_t N) const override final
      {
        for(uint64_t i=0;i<N;i++) this->operator () ( idx[i] );
      }

      inline ~BlockParallelForHostAdapter() override {}
    };


  }
}

