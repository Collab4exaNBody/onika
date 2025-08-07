#pragma once

#include <onika/parallel/block_parallel_for_functor.h>
#include <onika/parallel/parallel_execution_stream.h>
#include <onika/parallel/parallel_execution_debug.h>
#include <omp.h>

namespace onika
{
  namespace parallel
  {

    struct ExecOpenMPTaskCallbackInfo
    {
      omp_event_handle_t cu_stream_sync_event;
      std::mutex next_operation_sync;
    };

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
      static_assert( lambda_is_compatible_with_v<FuncT,void,FuncParamType> , "User defined functor is not compatible with execution space");

      using ParExecSpaceT = ParallelExecutionSpace<ND,ElemND,ElementListT>;

      static inline constexpr bool functor_has_prolog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_prolog_t>;
      static inline constexpr bool functor_has_cpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_cpu_prolog_t>;
      static inline constexpr bool functor_has_gpu_prolog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_prolog_t,ParallelExecutionStream*>;
      static inline constexpr bool functor_has_epilog     = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_epilog_t>;
      static inline constexpr bool functor_has_cpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_cpu_epilog_t>;
      static inline constexpr bool functor_has_gpu_epilog = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_gpu_epilog_t,ParallelExecutionStream*>;
      static inline constexpr bool functor_is_single_task = lambda_is_compatible_with_v<FuncT,void,block_parallel_for_single_task_t>;

      static_assert( !GPUSupport || !functor_is_single_task );

      const FuncT m_func;
      const ParExecSpaceT m_parallel_space;

    public:
      inline BlockParallelForHostAdapter( const FuncT& f , const ParExecSpaceT& ps ) : m_func(f) , m_parallel_space(ps) {}


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


      // ================ CPU OpenMP execution interface ======================
      inline void execute_prolog( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {
#       ifdef ONIKA_ENABLE_KERNEL_DEBUG_INFO
        dmesg_exec_omp(pec);
#       endif
        if constexpr (functor_has_cpu_prolog) { m_func( block_parallel_for_cpu_prolog_t{} ); }
        else if constexpr (functor_has_prolog) { m_func( block_parallel_for_prolog_t{} ); }
      }

      inline void execute_omp_parallel_region( ParallelExecutionContext* pec, ParallelExecutionStream* pes ) const override final
      {
        static_assert( FuncParamDim>=1 && FuncParamDim<=3 , "OpenMP backend only support 1D, 2D and 3D parallel execution spaces" );

        pes->m_omp_execution_count.fetch_add(1);

        const auto * __restrict__ idx = m_parallel_space.m_elements.data();

#       ifdef ONIKA_OMP_NUM_THREADS_WORKAROUND
        omp_set_num_threads( omp_get_max_threads() );
#       endif

        const auto T0 = std::chrono::high_resolution_clock::now();
        execute_prolog( pec , pes );

        if constexpr ( functor_is_single_task )
        {
          m_func( block_parallel_for_single_task_t{} );
        }
        else if constexpr ( ND == 1 )
        {
#         pragma omp parallel
          {
            switch( pec->m_omp_sched )
            {
              case OMP_SCHED_DYNAMIC :
#               pragma omp for schedule(dynamic)
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
                {
                  if constexpr ( ElemND==0 ) m_func( i );
                  if constexpr ( ElemND>=1 ) m_func( idx[i] );
                }
                break;
              case OMP_SCHED_GUIDED :
#               pragma omp for schedule(guided)
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
                {
                  if constexpr ( ElemND==0 ) m_func( i );
                  if constexpr ( ElemND>=1 ) m_func( idx[i] );
                }
                break;
              case OMP_SCHED_STATIC :
#               pragma omp for schedule(static)
                for(ssize_t i=m_parallel_space.m_start[0];i<m_parallel_space.m_end[0];i++)
                {
                  if constexpr ( ElemND==0 ) m_func( i );
                  if constexpr ( ElemND>=1 ) m_func( idx[i] );
                }
                break;
            }
          }
        }
        else if constexpr ( ND == 2 )
        {
          static_assert( ElemND == 0 , "element indices must be 1D" );
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
        else if constexpr ( ND == 3 )
        {
          static_assert( ElemND == 0 , "element indices must be 1D" );
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
        else
        {
          fatal_error() << "Unsuported parallel execution space dimensionality "<<FuncParamDim<<std::endl;
        }

        execute_epilog( pec , pes );
        pec->m_total_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
        if( pec->m_execution_end_callback.m_func != nullptr )
        {
          (* pec->m_execution_end_callback.m_func) ( pec->m_execution_end_callback.m_data );
        }
        pes->m_omp_execution_count.fetch_sub(1);
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

        // here all tasks of taskloop have completed, since notaskgroup clause is _NOT_ specified
        self->execute_epilog( pec , pes );
        pec->m_total_cpu_execution_time = ( std::chrono::high_resolution_clock::now() - T0 ).count() / 1000000.0;
        if( pec->m_execution_end_callback.m_func != nullptr )
        {
          (* pec->m_execution_end_callback.m_func) ( pec->m_execution_end_callback.m_data );
        }

        // finally, notify that one less OpenMP task is executing
        pes->m_omp_execution_count.fetch_sub(1);
      }

      static inline void execute_omp_inner_taskloop_cb( void* userData )
      {
        // first we trigger execution of OpenMP task
        ExecOpenMPTaskCallbackInfo * cb_info = reinterpret_cast<ExecOpenMPTaskCallbackInfo*>(userData);
        omp_fulfill_event( cb_info->cu_stream_sync_event );

        // then we wait until it finishes and unlock this
        cb_info->next_operation_sync.lock();
      }

      inline void execute_omp_tasks( ParallelExecutionContext* pec, ParallelExecutionStream* pes, unsigned int ntasks ) const override final
      {
        pes->m_omp_execution_count.fetch_add(1);

        const auto * self = this;

#       ifdef ONIKA_ENABLE_KERNEL_DEBUG_INFO
        dmesg_sched_omp(pec);
#       endif

        ExecOpenMPTaskCallbackInfo * cb_info = nullptr;
        // if a Cuda stream is available, we'll use it to serialize OpenMP tasks based parallel operations,
        // such that they will be also serialized with Cuda kernel executions throughout the stream
        if( pes->m_cuda_ctx != nullptr )
        {
          if( pec->m_host_scratch.available_data_bytes() < sizeof(ExecOpenMPTaskCallbackInfo) )
          {
            fatal_error() << "Internal error: not enaough functor scratch space left to chain OpenMP task to CU stream" << std::endl;
          }
          // we schedule an empty task wich uses depend(inout:pes[0]) like all other parallel tasks,
          // and we also add a detach clause to be able to trigger its completion event from an other thread.
          // finally, we enqueue a host function (execute_omp_inner_taskloop_cb) in cuda stream
          // which execution calls omp_fulfill_event so that it triggers completion of previously scheduled empty task,
          // which in turn unlock real parallel OpenMP taskloop through depend clause
          cb_info = new(pec->m_host_scratch.alloc_functor_data(sizeof(ExecOpenMPTaskCallbackInfo))) ExecOpenMPTaskCallbackInfo{};
          omp_event_handle_t sync_event;
          std::memset( & sync_event , 0 , sizeof(omp_event_handle_t) );
#         pragma omp task default(none) firstprivate(self,pec,pes,ntasks) depend(inout:pes[0]) detach(sync_event)
          {
            if(int(ntasks)<0) { printf("ERROR: ntasks=%d\n",int(ntasks)); }
          }
          cb_info->cu_stream_sync_event = sync_event;
          cb_info->next_operation_sync.lock();
          ONIKA_CU_STREAM_HOST_FUNC( pes->m_cu_stream , execute_omp_inner_taskloop_cb , cb_info );
        }

        // if OpenMP only, parallel operation serialization feature is emulated
        // via an encapsulating task which declares an in/out dependency on the Onika stream object

        // encloses a taskgroup inside a task, so that we can wait for a single task which itself waits for the completion of the whole taskloop
        // referenced variables must be privately copied, because the task may run after this function ends
#       pragma omp task default(none) firstprivate(self,pec,pes,ntasks,cb_info) depend(inout:pes[0])
        {
          execute_omp_inner_taskloop(self,pec,pes,ntasks);
          if( cb_info != nullptr ) cb_info->next_operation_sync.unlock();
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

