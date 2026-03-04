#pragma once

#include <onika/parallel/block_parallel_for_functor.h>
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/parallel_execution_debug.h>
#include <omp.h>

#include <condition_variable>

namespace onika
{
  namespace parallel
  {
  
    // ========================== GPU execution kernels ==========================

    // GPU execution kernel for fixed size grid, using workstealing element assignment to blocks
    template<class ElementCoordRangeT, class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    ONIKA_STATIC_INLINE_KERNEL
    void block_parallel_for_gpu_kernel_workstealing( uint64_t N, GPUKernelExecutionScratch* scratch, const ElementCoordRangeT idx, ONIKA_CU_GRID_CONSTANT const FuncT func )
    {
      using ElementCoordT = std::remove_cv_t< std::remove_reference_t< decltype(idx[0]) > >;
      static constexpr unsigned int ElemND = element_coord_nd_v<ElementCoordT>;

      // avoid use of compute buffer when possible
      ONIKA_CU_BLOCK_SHARED unsigned int i;
      do
      {
        if( ONIKA_CU_THREAD_IDX == 0 )
        {
          i = ONIKA_CU_ATOMIC_ADD( scratch->counters[GPUKernelExecutionScratch::WORKSTEALING_COUNTER] , 1u );
        }
        ONIKA_CU_BLOCK_SYNC();
        if( i < N )
        {
          if constexpr (ElemND==0) func( i );
          if constexpr (ElemND>=1) func( idx[i] );
        }
      }
      while( i < N );
    }

    // GPU execution kernel for adaptable size grid, a.k.a. conventional Cuda kernel execution on N element blocks
    template<class ElementCoordRangeT, class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    ONIKA_STATIC_INLINE_KERNEL
    void block_parallel_for_gpu_kernel_regulargrid( const ElementCoordRangeT idx , ONIKA_CU_GRID_CONSTANT const FuncT func , ONIKA_CU_GRID_CONSTANT const unsigned int start )
    {
      using ElementCoordT = std::remove_cv_t< std::remove_reference_t< decltype(idx[0]) > >;
      static constexpr unsigned int ElemND = element_coord_nd_v<ElementCoordT>;
      if constexpr (ElemND==0) func( start + ONIKA_CU_BLOCK_IDX );
      if constexpr (ElemND>=1) func( idx[ start + ONIKA_CU_BLOCK_IDX ] );
    }

    template< class FuncT>
    ONIKA_DEVICE_KERNEL_FUNC
    ONIKA_DEVICE_KERNEL_BOUNDS(ONIKA_CU_MAX_THREADS_PER_BLOCK,ONIKA_CU_MIN_BLOCKS_PER_SM)
    ONIKA_STATIC_INLINE_KERNEL
    void block_parallel_for_gpu_kernel_regulargrid_3D( ONIKA_CU_GRID_CONSTANT const FuncT func , ONIKA_CU_GRID_CONSTANT const onikaInt3_t start_coord )
    {
      func( start_coord + ONIKA_CU_BLOCK_COORD );
    }



    // =============== Execution functor adapter for Cuda, OpenMP parallel and OpenMP tasks ==========================

    template<class FuncT, bool GPUSupport , unsigned int ND=1, unsigned int ElemND=0, class ElementListT = std::span< const element_coord_t<ElemND> > >
    class BlockParallelForHostAdapter : public BlockParallelForHostFunctor
    {
      static_assert( !GPUSupport || gpu_frontend_compiler() );

      // what is he dimensionality of processed elements ?
      static inline constexpr unsigned int FuncParamDim = ( ElemND==0 ) ? ND : ElemND ;
      static_assert( FuncParamDim>=1 && FuncParamDim<=3 );
      using FuncParamType = std::conditional_t< FuncParamDim==1 , ssize_t , onikaInt3_t >;
      using ParExecSpaceT = ParallelExecutionSpace<ND,ElemND,ElementListT>;

      static inline constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_prolog_t>;
      static inline constexpr bool functor_has_cpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_cpu_prolog_t>;
      static inline constexpr bool functor_has_gpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_prolog_t,ParallelExecutionStream*>;
      static inline constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_epilog_t>;
      static inline constexpr bool functor_has_cpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_cpu_epilog_t>;
      static inline constexpr bool functor_has_gpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_epilog_t,ParallelExecutionStream*>;
      static inline constexpr bool functor_is_single_task = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_single_task_t>;

      static_assert( lambda_is_compatible_with_v<FuncT,void,FuncParamType> || functor_is_single_task , "User defined functor is not compatible with execution space");
      static_assert( !GPUSupport || !functor_is_single_task );

      const FuncT m_func;
      const ParExecSpaceT m_parallel_space;

    public:
      inline BlockParallelForHostAdapter( const FuncT& f , const ParExecSpaceT& ps ) : m_func(f) , m_parallel_space(ps) {}

      // parallel execution space info
      inline unsigned int execution_space_ndims() const override final { return m_parallel_space.space_nd(); }
      inline size_t execution_space_nitems() const override final { return m_parallel_space.number_of_items(); }
      inline bool is_single_task() const override final { return functor_is_single_task; }
      inline size_t execution_space_range(ssize_t * buf, size_t bufsz) const override final
      {
        return coord_range_to_array(buf,bufsz,m_parallel_space.m_start,m_parallel_space.m_end);
      }

      // ================ GPU execution interface ======================

      // GPU execution prolog
      inline void stream_gpu_initialize(ParallelExecutionContext* pec , ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
#         ifdef ONIKA_ENABLE_KERNEL_DEBUG_INFO
          ONIKA_CU_STREAM_HOST_FUNC( pes->m_cu_stream , dmesg_exec_gpu , pec );
#         endif
          if constexpr ( functor_has_gpu_prolog ) { m_func( block_parallel_for_gpu_prolog_t{} , pes ); }
          else if constexpr ( functor_has_prolog ) { m_func( block_parallel_for_prolog_t{} ); }
        }
        else { fatal_error() << "called stream_gpu_initialize with no GPU support" << std::endl; }
      }

      // GPU execution kernel call
      inline void stream_gpu_kernel(ParallelExecutionContext* pec, ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
          // launch compute kernel
          const size_t N = m_parallel_space.m_end[0] - m_parallel_space.m_start[0];
          onikaInt3_t block_offset = { m_parallel_space.m_start[0], 0, 0 };
          if constexpr ( ND >= 2 ) block_offset.y = m_parallel_space.m_start[1];
          if constexpr ( ND >= 3 ) block_offset.z =  m_parallel_space.m_start[2];
          if( pec->m_grid_size.x > 0 )
          {
            if constexpr ( ND == 1 )
            {
              ONIKA_CU_LAUNCH_KERNEL(pec->m_grid_size.x,pec->m_block_size.x,0,pes->m_cu_stream, block_parallel_for_gpu_kernel_workstealing, N, pec->m_cuda_scratch.get(), m_parallel_space.m_elements, m_func );
            }
            else
            {
              fatal_error() << "Work stealing GPU execution only supported for 1D kernels" << std::endl;
            }
          }
          else
          {
            if constexpr ( ND == 1 )
            {
              ONIKA_CU_LAUNCH_KERNEL(N,pec->m_block_size.x,0,pes->m_cu_stream, block_parallel_for_gpu_kernel_regulargrid, m_parallel_space.m_elements, m_func, block_offset.x );
            }
            else if constexpr ( ND > 1 )
            {
              static_assert( ElemND == 0 , "element indices must be 1D" );
              onikaDim3_t exec_grid_size = { static_cast<unsigned int>( m_parallel_space.m_end[0] - m_parallel_space.m_start[0] )
                                           , static_cast<unsigned int>( m_parallel_space.m_end[1] - m_parallel_space.m_start[1] )
                                           , static_cast<unsigned int>( m_parallel_space.m_end[2] - m_parallel_space.m_start[2] ) };
              //lout << "Kernel execute grid=("<<exec_grid_size.x<<","<<exec_grid_size.y<<","<<exec_grid_size.z<<") , block=("<<pec->m_block_size.x<<","<<pec->m_block_size.y<<","<<pec->m_block_size.z<<")" << std::endl;
              ONIKA_CU_LAUNCH_KERNEL(exec_grid_size , pec->m_block_size , 0 , pes->m_cu_stream , block_parallel_for_gpu_kernel_regulargrid_3D, m_func, block_offset );
            }
          }
        }
        else
        {
          fatal_error() << "called stream_gpu_kernel with no GPU support" << std::endl;
        }
      }

      // GPU execution epilog
      inline void stream_gpu_finalize(ParallelExecutionContext* pec, ParallelExecutionStream* pes) const override final
      {
        if constexpr ( GPUSupport )
        {
          if constexpr ( functor_has_gpu_epilog ) { m_func( block_parallel_for_gpu_epilog_t{} , pes ); }
          else if constexpr ( functor_has_epilog ) { m_func( block_parallel_for_epilog_t{} ); }
#         ifdef ONIKA_ENABLE_KERNEL_DEBUG_INFO
          ONIKA_CU_STREAM_HOST_FUNC( pes->m_cu_stream , dmesg_end_gpu , pec );
#         endif
        }
        else { fatal_error() << "called stream_gpu_finalize with no GPU support" << std::endl; }
      }

      inline void execute_gpu(ParallelExecutionContext* pec, ParallelExecutionStream* exec_stream) const override final
      {
        if( exec_stream->m_cuda_ctx == nullptr || exec_stream->m_cuda_ctx != pec->m_cuda_ctx )
        {
          fatal_error() << "Cannot schedule GPU parallel operation onto stream with no GPU context" << std::endl;
        }

        // if device side scratch space hasn't be allocated yet, do it now
        pec->init_device_scratch();

        // insert start event for profiling
        assert( pec->m_start_evt != nullptr );
        ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_EVENT( pec->m_start_evt, exec_stream->m_cu_stream ) );

        // copy in return data intial value. mainly useful for reduction where you might want to start reduction with a given initial value
        if( pec->m_return_data_input != nullptr && pec->m_return_data_size > 0 )
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( pec->m_cuda_scratch->return_data, pec->m_return_data_input , pec->m_return_data_size , exec_stream->m_cu_stream ) );
        }

        // sets all scratch counters to 0
        if( pec->m_reset_counters || pec->m_grid_size.x > 0 )
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMSET( pec->m_cuda_scratch->counters, 0, GPUKernelExecutionScratch::MAX_COUNTERS * sizeof(unsigned long long int), exec_stream->m_cu_stream ) );
        }

        // Instantiaite device side functor : calls constructor with a placement new using scratch "functor_data" space
        // then call functor prolog if available
        this->stream_gpu_initialize( pec , exec_stream );
        this->stream_gpu_kernel( pec , exec_stream );
        this->stream_gpu_finalize( pec , exec_stream );

        // copy out return data to host space at given pointer
        if( pec->m_return_data_output != nullptr && pec->m_return_data_size > 0 )
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( pec->m_return_data_output , pec->m_cuda_scratch->return_data , pec->m_return_data_size , exec_stream->m_cu_stream ) );
        }

        // inserts a callback to stream if user passed one in
        if( pec->m_execution_end_callback.m_func != nullptr )
        {
          ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_ADD_CALLBACK(exec_stream->m_cu_stream, ParallelExecutionContext::execution_end_callback , pec ) );
        }

        // inserts stop event to account for total execution time
        assert( pec->m_stop_evt != nullptr );
        ONIKA_CU_CHECK_ERRORS( ONIKA_CU_STREAM_EVENT( pec->m_stop_evt, exec_stream->m_cu_stream ) );        
      }



      // ================ CPU OpenMP execution interface ======================
      inline void execute_prolog( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {
#       ifdef ONIKA_ENABLE_KERNEL_DEBUG_INFO
        dmesg_exec_omp(pec);
#       endif
        if constexpr (functor_has_cpu_prolog) { m_func( block_parallel_for_cpu_prolog_t{} ); }
        else if constexpr (functor_has_prolog) { m_func( block_parallel_for_prolog_t{} ); }
      }

      inline void execute_omp_parallel_region( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const
      {
        static_assert( FuncParamDim>=1 && FuncParamDim<=3 , "OpenMP backend only support 1D, 2D and 3D parallel execution spaces" );

        const auto * __restrict__ idx = m_parallel_space.m_elements.data();

#       ifdef ONIKA_OMP_NUM_THREADS_WORKAROUND
        omp_set_num_threads( omp_get_max_threads() );
#       endif

        const auto T0 = std::chrono::high_resolution_clock::now();
        execute_prolog( pec , pes );

        // store preferred OpenMP scheduling in a variable to avoid code duplicates

        if constexpr ( functor_is_single_task )
        {
          m_func( block_parallel_for_single_task_t{} );
        }
        else
        {
          switch(pec->m_omp_sched)
          {
            case OMP_SCHED_DYNAMIC : omp_set_schedule( omp_sched_dynamic, 0 ); break;
            case OMP_SCHED_GUIDED  : omp_set_schedule( omp_sched_guided , 0 ); break;
            case OMP_SCHED_STATIC  : omp_set_schedule( omp_sched_static , 0 ); break;
            default                : omp_set_schedule( omp_sched_auto   , 0 ); break;
          }
#         pragma omp parallel
          {
            if constexpr ( ND == 1 )
            {
#             pragma omp for schedule(runtime)
              for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
              {
                if constexpr ( ElemND==0 ) m_func( i );
                if constexpr ( ElemND>=1 ) m_func( idx[i] );
              }
            }
            else if constexpr ( ND == 2 )
            {
              static_assert( ElemND == 0 , "element indices must be 1D" );
#             pragma omp for collapse(2) schedule(runtime)
              for(ssize_t j=m_parallel_space.m_start[1];j<m_parallel_space.m_end[1];j++)
              for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
              { m_func( onikaInt3_t{i,j,0} ); }
            }
            else if constexpr ( ND == 3 )
            {
              static_assert( ElemND == 0 , "element indices must be 1D" );
#             pragma omp for collapse(3) schedule(runtime)
              for(ssize_t k=m_parallel_space.m_start[2];k<m_parallel_space.m_end[2];k++)
              for(ssize_t j=m_parallel_space.m_start[1];j<m_parallel_space.m_end[1];j++)
              for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
              { m_func( onikaInt3_t{i,j,k} ); }
            }
            else
            {
              fatal_error() << "Unsuported parallel execution space dimensionality "<<FuncParamDim<<std::endl;
            }
          }
        }

      }

      static inline void execute_omp_inner_taskloop( const BlockParallelForHostAdapter* self, ParallelExecutionContext* pec, ParallelExecutionStream* pes, unsigned int ntasks )
      {
        const auto & ps = self->m_parallel_space;
        const auto & func = self->m_func;

        unsigned long long N = ps.m_end[0] - ps.m_start[0];
        if constexpr (ND>=2) N *= ps.m_end[1] - ps.m_start[1];
        if constexpr (ND>=3) N *= ps.m_end[2] - ps.m_start[2];

        const auto T0 = std::chrono::high_resolution_clock::now();
        self->execute_prolog( pec , pes );

        if( N > 0 )
        {
          // implicit taskgroup ensures taskloop has completed before enclosing task ends
          // all referenced variables can be shared because of implicit enclosing taskgroup
          if constexpr ( functor_is_single_task )
          {
            func( block_parallel_for_single_task_t{} );
          }
          else if constexpr ( ND==1 )
          {
#           pragma omp taskloop default(none) shared(ps,func,ntasks) num_tasks(ntasks)
            for(ssize_t i=ps.m_start[0] ; i<ps.m_end[0] ; i++ )
            {
              if constexpr ( ElemND==0 ) func( i );
              if constexpr ( ElemND>=1 ) func( ps.m_elements[i] );
            }
          }
          else if constexpr ( ND==2 )
          {
            static_assert( ElemND == 0 , "element indices must be 1D" );
#           pragma omp taskloop collapse(2) default(none) shared(ps,func,ntasks) num_tasks(ntasks)
            for(ssize_t j=ps.m_start[1];j<ps.m_end[1];j++)
            for(ssize_t i=ps.m_start[0];i<ps.m_end[0];i++)
            {
              func( onikaInt3_t{i,j,0} );
            }
          }
          else if constexpr ( ND==3 )
          {
            static_assert( ElemND == 0 , "element indices must be 1D" );
#           pragma omp taskloop collapse(3) default(none) shared(ps,func,ntasks) num_tasks(ntasks)
            for(ssize_t k=ps.m_start[2];k<ps.m_end[2];k++)
            for(ssize_t j=ps.m_start[1];j<ps.m_end[1];j++)
            for(ssize_t i=ps.m_start[0];i<ps.m_end[0];i++)
            {
              func( onikaInt3_t{i,j,k} );
            }
          }
          else
          {
            fatal_error() << "Unsuported parallel execution space dimensionality "<<FuncParamDim<<std::endl;
          }
        }

      }

      inline void execute_epilog( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {
        if constexpr (functor_has_cpu_epilog) { m_func( block_parallel_for_cpu_epilog_t{} ); }
        else if constexpr (functor_has_epilog) { m_func( block_parallel_for_epilog_t{} ); }
#       ifdef ONIKA_ENABLE_KERNEL_DEBUG_INFO
        dmesg_end_omp(pec);
#       endif
      }

      inline auto execute_omp_start( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const
      {
        const auto T0 = std::chrono::high_resolution_clock::now();
        pes->m_omp_execution_count.fetch_add(1);
        this->execute_prolog( pec , pes );
        return T0; 
      }

      inline auto execute_omp_end( ParallelExecutionContext* pec, ParallelExecutionStream* pes , auto T0 ) const
      {
        execute_epilog( pec , pes );
        pec->m_total_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
        if( pec->m_execution_end_callback.m_func != nullptr )
        {
          (* pec->m_execution_end_callback.m_func) ( pec->m_execution_end_callback.m_data );
        }
        pes->m_omp_execution_count.fetch_sub(1);
      }

      inline void execute_omp_tasks( ParallelExecutionContext* pec, ParallelExecutionStream* pes, unsigned int ntasks, auto T0 ) const
      {
        const auto * self = this;

        // do i have in dependencies ? am i the only one to ask ?
        // if so, wait for dependency condition variable and process unlocked tasks
        if( /* pec->in_dep_count() > */ 0 )
        {
          // ...
          // scheduler . omp_task_scheduler_worker.compare_and_swap( nullptr , this )
          // if i am the chosen one (scheduler worker) , loop on condition variable to launch tasks until i am unlocked myself
        }

#       ifdef ONIKA_ENABLE_KERNEL_DEBUG_INFO
        dmesg_sched_omp(pec);
#       endif

        // if OpenMP only, parallel operation serialization feature is emulated
        // via an encapsulating task which declares an in/out dependency on the Onika stream object

        // encloses a taskgroup inside a task, so that we can wait for a single task which itself waits for the completion of the whole taskloop
        // referenced variables must be privately copied, because the task may run after this function ends
#       pragma omp task default(none) firstprivate(self,pec,pes,ntasks,T0) depend(inout:pes[0])
        {

          // while remaining unsatisfied external in dependencies
          //   | act as and ending task, i.e. check_if_tasks_ready_to_execute
          
          execute_omp_inner_taskloop(self,pec,pes,ntasks);
                    
          // act as and ending task, i.e. check_if_tasks_ready_to_execute
          
          self->execute_omp_end( pec , pes , T0 );
        } // end of encapsulating task

      }
 
      inline void execute_omp( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {
        auto T0 = execute_omp_start( pec , pes );
        if( pec->m_omp_num_tasks == 0 )
        {
          execute_omp_parallel_region( pec , pes );
          execute_omp_end( pec , pes , T0 );
        }
        else
        {
          // preferred number of tasks : trade off between overhead (less is better) and load balancing (more is better)
          const unsigned int num_tasks = pec->m_omp_num_tasks * onika::parallel::ParallelExecutionContext::parallel_task_core_mult() + onika::parallel::ParallelExecutionContext::parallel_task_core_add() ;
          execute_omp_tasks( pec , pes , num_tasks , T0 );
        }        
      }

      inline ~BlockParallelForHostAdapter() override {}
    };


  }
}

