#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/parallel/block_parallel_for.h>

#include "array2d.h"
#include "block_parallel_value_add_functor.h"

namespace onika
{
  using namespace scg;

  namespace tutorial
  {

    class ConcurrentBlockParallelTest : public OperatorNode
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

        // we create 2 custom queues with different default execution lane
        auto stream_0_control = parallel_execution_custom_queue(0); // in this queue, when stream id is not specified it will default to stream #0
        auto stream_1_control = parallel_execution_custom_queue(1); // in this queue, when stream id is not specified it will default to stream #1

        // enqueue operations in two distinct custom queues with different default stream ids        
        stream_0_control << std::move(array1_par_op1) << std::move(array1_par_op2) << onika::parallel::flush ;
        stream_1_control << std::move(array2_par_op1) << std::move(array2_par_op2) << onika::parallel::flush ;
        
        lout << "Parallel operations are executing..." << std::endl;
        stream_1_control.wait(); // wait for all operations in stream queue #1 to complete
        stream_0_control.wait(); // wait for all operations in stream queue #0 to complete
        lout << "All parallel operations have terminated !" << std::endl;
      }
    };

    // === register factories ===
    ONIKA_AUTORUN_INIT(concurrent_block_parallel)
    {
     OperatorNodeFactory::instance()->register_factory( "concurrent_block_parallel", make_simple_operator< ConcurrentBlockParallelTest > );
    }

  }
}
