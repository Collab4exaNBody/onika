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

#include <memory>

#include <onika/parallel/parallel_execution_queue.h>
#include <onika/parallel/parallel_execution_operators.h>
#include <onika/app/api.h>
#include <onika/extras/array2d.h>
#include <onika/extras/block_parallel_value_add_functor.h>

#include <chrono>

#include <omp.h>



// parallel test core
void run_test(auto & pq, const auto & parallel_execution_context, std::string_view test_mode, bool kernel_1d = true )
{
  using namespace onika::extras;
  using onika::parallel::block_parallel_for;
  using onika::parallel::ParallelExecutionQueueBase;
  using onika::parallel::AccessStencilElement;
  using onika::parallel::ParallelExecutionSpace;
  using onika::parallel::local_access;
  using onika::parallel::make_single_task_block_parallel_functor;

  static constexpr ssize_t array_rows = 1024;
  static constexpr ssize_t array_cols = 1024;
  const ParallelExecutionSpace<2> array_range_2d = { { 0 , 0 } , { array_rows , array_cols } };

  if( kernel_1d ) std::cout<<"1D kernel parallelization space"<<std::endl;
  else std::cout<<"2D kernel parallelization space"<<std::endl;

  Array2D array1;
  Array2D array2;
  array1.resize( array_rows , array_cols );
  array2.resize( array_rows , array_cols );

  BlockParallelIterativeValueAddFunctor<true> array1_kernel1 = { array1, 1.1, 50 };
  BlockParallelValueAddFunctor<false>         array1_kernel2 = { array1, 1.2 };
  BlockParallelIterativeValueAddFunctor<true> array2_kernel1 = { array2, 1.3, 50 };
  BlockParallelValueAddFunctor<false>         array2_kernel2 = { array2, 1.4 };

  // Launching the parallel operation, which can execute on GPU if the execution context allows
  // result of parallel operation construct is captured into variable 'my_addition',
  // thus it can be scheduled in a stream queue for asynchronous execution rather than being executed right away
  auto array1_par_op1 = kernel_1d ?
                        block_parallel_for( array1.rows() , array1_kernel1, parallel_execution_context("Array1","Kernel1") )
                      : block_parallel_for( array_range_2d, array1_kernel1, parallel_execution_context("Array1","Kernel1") );

  // we create a second parallel operation we want to execute sequentially after the first addition
  auto array1_par_op2 = kernel_1d ?
                        block_parallel_for( array1.rows() , array1_kernel2, parallel_execution_context("Array1","Kernel2") )
                      : block_parallel_for( array_range_2d, array1_kernel2, parallel_execution_context("Array1","Kernel2") );

  // we finally create a third parallel operation we want to execute concurrently with the two others
  auto array2_par_op1 = kernel_1d ?
                        block_parallel_for( array2.rows() , array2_kernel1, parallel_execution_context("Array2","Kernel1") )
                      : block_parallel_for( array_range_2d, array2_kernel1, parallel_execution_context("Array2","Kernel1") );

  // we finally create a third parallel operation we want to execute concurrently with the two others
  auto array2_par_op2 = kernel_1d ?
                        block_parallel_for( array2.rows() , array2_kernel2, parallel_execution_context("Array2","Kernel2") )
                      : block_parallel_for( array_range_2d, array2_kernel2, parallel_execution_context("Array2","Kernel2") );

  std::shared_ptr<std::mutex> user_task_start_sync;

  if( test_mode == "hybrid-delayed-lane" )
  {
    std::cout << "Delay parallel operations ..." << std::endl;
    ParallelExecutionQueueBase delay_queue_a;
    ParallelExecutionQueueBase delay_queue_b;
    delay_queue_a << onika::parallel::set_lane(0) << std::move(array1_par_op1) << onika::parallel::set_lane(1) << std::move(array2_par_op1);
    delay_queue_b << onika::parallel::set_lane(0) << std::move(array1_par_op2) << onika::parallel::set_lane(1) << std::move(array2_par_op2);

    std::cout << "Enqueue operations ..." << std::endl;
    pq << std::move(delay_queue_a) << std::move(delay_queue_b) ;
  }
  else if( test_mode == "hybrid-lane" )
  {
    // would be the same as following
    std::cout << "Enqueue operations ..." << std::endl;
    pq << onika::parallel::set_lane(0) << std::move(array1_par_op1) << onika::parallel::set_lane(1) << std::move(array2_par_op1)
       << onika::parallel::set_lane(0) << std::move(array1_par_op2) << onika::parallel::set_lane(1) << std::move(array2_par_op2);
  }
  else if( test_mode == "hybrid-sequence" )
  {
    // simple hybrid execution test
    pq << std::move(array1_par_op1) << std::move(array1_par_op2) << std::move(array2_par_op1) << std::move(array2_par_op2);
  }
  else if( test_mode == "singletask-dependency" )
  {
    // describes an access to array1, which is 2D, for read and write access to elements @ location of block_parallel_for iterator
    const auto array1_rw_access = local_access(array1.m_data.data(),2,AccessStencilElement::RW,"a1_rw");
    const ParallelExecutionSpace<2> single_task_data_space = { {64,64} , {980,980} };

    user_task_start_sync = std::make_shared<std::mutex> ();
    user_task_start_sync->lock();

    auto single_task = make_single_task_block_parallel_functor( [user_task_start_sync]()
      {
        std::cout<<"User single task waiting for start ready mutex"<<std::endl<< std::flush;
        user_task_start_sync->lock();
        std::cout<<"User single task code execution"<<std::endl<< std::flush;
      }
    );
    
    //std::cout << "Enqueue single task ..." << std::endl << std::flush;
    pq  << onika::parallel::any_lane()
        << array1_rw_access 
        << block_parallel_for( single_task_data_space, single_task, parallel_execution_context("Array1","UnlockTask1") )

    //std::cout << "Enqueue 1st parallel task ..." << std::endl << std::flush;
        << onika::parallel::any_lane()
        << array1_rw_access 
        << std::move(array1_par_op1)

    //std::cout << "Enqueue 2nd parallel task ..." << std::endl << std::flush;
        << onika::parallel::any_lane()
        << array1_rw_access 
        << std::move(array1_par_op2);
    
    // describes an access to array2, which is 2D, for read and write access to elements @ location of block_parallel_for iterator
    //const auto array2_rw_access = local_access(array2.m_data.data(),2,AccessStencilElement::RW,"a2_rw");
  }
  else
  {
    std::cerr<<"unknown test mode "<<test_mode<<std::endl;
    std::cerr<<"valid test modes are : hybrid-delayed-lane , hybrid-lane and hybrid-sequence"<<std::endl;
    std::abort();
  }

  if( user_task_start_sync != nullptr )
  {
    std::cout << "User unlock of parallel task" << std::endl << std::flush;
    user_task_start_sync->unlock();
  }

  std::cout << "Schedule parallel operations ..." << std::endl << std::flush;
  pq << onika::parallel::flush;

  std::cout << "Synchronize parallel operations ..." << std::endl << std::flush;
  pq <<  onika::parallel::synchronize ;
  std::cout << "done" << std::endl << std::flush;
}


int main(int argc,char*argv[])
{
  using onika::parallel::ParallelExecutionQueue;
  using onika::parallel::ParallelExecutionContext;
  using onika::parallel::ParallelExecutionContextAllocator;
  using onika::cuda::CudaContext;

  std::string test_mode = "hybrid-lane";
  std::string omp_mode = "task";
  std::string parallel_dims = "1d";
  if( argc > 1 ) test_mode = argv[1];
  if( argc > 2 ) omp_mode = argv[2];
  if( argc > 3 ) parallel_dims = argv[3];

  // application initialization
  onika::app::ApplicationConfiguration config = {};
  onika::app::intialize_openmp( config );
  onika::app::initialize_gpu( config );

  std::cout << "ParallelExecutionContext = "<< sizeof(ParallelExecutionContext)<<" bytes"<<std::endl;

  if( CudaContext::default_cuda_ctx() != nullptr )
  {
    CudaContext::default_cuda_ctx()->to_stream(std::cout);
  }

  int omp_num_tasks = -1;
  if( omp_mode == "task" ) omp_num_tasks = omp_get_max_threads();
  else if( omp_mode == "pfor" ) omp_num_tasks = 0;
  else
  {
    std::cout<<"unkown OpenMP mode "<<omp_mode<<". valid options are task and pfor"<<std::endl;
    std::abort();
  }

  const bool kernel_1d = ( parallel_dims=="1d" || parallel_dims=="1D" );

  // convinience function to allocate parallel task contexts
  auto & pq = ParallelExecutionQueue::default_queue();
  ParallelExecutionContextAllocator peca;
  auto parallel_execution_context = [pq_ptr=&pq,cu_ctx=CudaContext::default_cuda_ctx(),&peca,omp_num_tasks](auto tag, auto sub_tag)
  -> ParallelExecutionContext *
  {
    return peca.create(tag,sub_tag,pq_ptr,cu_ctx,omp_num_tasks);
  };


  if( omp_mode == "task" )
  {
#   pragma omp parallel
    {
#     pragma omp master
      {
#       pragma omp taskgroup
        {
          run_test(pq,parallel_execution_context,test_mode,kernel_1d);
        }
      }
    }
  }
  else
  {
    run_test(pq,parallel_execution_context,test_mode,kernel_1d);
  }

  return 0;
}

