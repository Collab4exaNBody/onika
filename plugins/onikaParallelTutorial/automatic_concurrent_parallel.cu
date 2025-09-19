#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/parallel/block_parallel_for.h>

#include <onika/extras/array2d.h>
#include <onika/extras/block_parallel_value_add_functor.h>

namespace onika
{
  using namespace scg;
  using namespace extras;

  namespace tutorial
  {

    class AutomaticConcurrentParallelTest : public OperatorNode
    {
      ADD_SLOT(Array2D, array1, INPUT_OUTPUT, Array2D{} );
      ADD_SLOT(Array2D, array2, INPUT_OUTPUT, Array2D{} );
      ADD_SLOT(double, value, INPUT, 1.0);
      ADD_SLOT(long, iter_count, INPUT, 1024 );
      ADD_SLOT(long, columns, INPUT, 1024 );
      ADD_SLOT(long, rows, INPUT, 1024 );

      public:
      
      inline bool is_sink() const override final { return true; }
      
      inline void execute() override final
      {
        using onika::parallel::block_parallel_for;
        using onika::parallel::any_lane;
        using onika::parallel::flush;
        using onika::parallel::AccessStencilElement;
        using onika::parallel::local_access;

        // if array1 is empty, allocate it
        if( array1->rows() == 0 || array1->columns() == 0 )
        {
          array1->resize( *rows , *columns );
        }

        // if array2 is empty, allocate it
        if( array2->rows() == 0 || array2->columns() == 0 )
        {
          array2->resize( *rows , *columns );
        }

        BlockParallelIterativeValueAddFunctor<true> array1_kernel1 = { *array1, *value, *iter_count };
        BlockParallelValueAddFunctor<true>          array1_kernel2 = { *array1, *value };
        BlockParallelIterativeValueAddFunctor<true> array2_kernel1 = { *array2, *value, *iter_count };
        BlockParallelValueAddFunctor<true>          array2_kernel2 = { *array2, *value };

        // describes an access to array1, which is 2D, for read and write access to elements @ location of block_parallel_for iterator
        const auto array1_rw_access = local_access(array1->m_data.data(),2,AccessStencilElement::RW,"a1_rw");
        
        // describes an access to array2, which is 2D, for read and write access to elements @ location of block_parallel_for iterator
        const auto array2_rw_access = local_access(array2->m_data.data(),2,AccessStencilElement::RW,"a2_rw");
                               
        // Launching the parallel operation, which can execute on GPU if the execution context allows
        // result of parallel operation construct is captured into variable 'my_addition',
        // thus it can be scheduled in a stream queue for asynchronous execution rather than being executed right away
        auto array1_par_op1 = block_parallel_for( array1->rows(), array1_kernel1, parallel_execution_context("a1_k1") );

        // we create a second parallel operation we want to execute sequentially after the first addition
        auto array1_par_op2 = block_parallel_for( array1->rows(), array1_kernel2, parallel_execution_context("a1_k2") );

        // we finally create a third parallel operation we want to execute concurrently with the two others
        auto array2_par_op1 = block_parallel_for( array2->rows(), array2_kernel1, parallel_execution_context("a2_k1") );

        // we finally create a third parallel operation we want to execute concurrently with the two others
        auto array2_par_op2 = block_parallel_for( array2->rows(), array2_kernel2, parallel_execution_context("a2_k2") );

        // enqueue operations in two distinct custom queues with different default stream ids
        lout << "Enqueue parallel operations ..." << std::endl;

        // any_lane()  allows for parallel operation overlapping, auto lane selection and/or kernel split into several kernels
        // data access pattern is reset each time an operation is issued, so we must re-activate it for the next parallel operation, even though it has the same access pattern
        parallel_execution_queue() << any_lane()
                                   << array1_rw_access << std::move(array1_par_op1)
                                   << array1_rw_access << std::move(array1_par_op2)
                                   << array2_rw_access << std::move(array2_par_op1) 
                                   << array2_rw_access << std::move(array2_par_op2);
                                           
        lout << "schedule for execution ..." << std::endl;
        parallel_execution_queue() << flush;
        // it also retores preselected queue's lane to DEFAULT_EXECUTION_LANE, restoring de default behavior
        // it's then needed to stream any_lane() again to restart another set of automatic overlapping parallel operations

        // parallel_execution_queue().wait(); // wait for all opeartions in all streams to complete
        parallel_execution_queue() << onika::parallel::synchronize ; // the same as above
        
        lout << "All operations have terminated !" << std::endl;
      }
    };

    // === register factories ===
    ONIKA_AUTORUN_INIT(automatic_concurrent_parallel)
    {
     OperatorNodeFactory::instance()->register_factory( "automatic_concurrent_parallel", make_simple_operator< AutomaticConcurrentParallelTest > );
    }

  }
}
