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
void run_test(auto & pq, const auto & parallel_execution_context, std::string_view test_mode )
{
  using namespace onika::extras;
  using onika::parallel::block_parallel_for;
  using onika::parallel::ParallelExecutionQueueBase;

  Array2D array1;
  Array2D array2;
  array1.resize( 1024 , 1024 );
  array2.resize( 1024 , 1024 );

  BlockParallelIterativeValueAddFunctor<true> array1_kernel1 = { array1, 1.1, 50 };
  BlockParallelValueAddFunctor<false>         array1_kernel2 = { array1, 1.2 };
  BlockParallelIterativeValueAddFunctor<true> array2_kernel1 = { array2, 1.3, 50 };
  BlockParallelValueAddFunctor<false>         array2_kernel2 = { array2, 1.4 };

  // Launching the parallel operation, which can execute on GPU if the execution context allows
  // result of parallel operation construct is captured into variable 'my_addition',
  // thus it can be scheduled in a stream queue for asynchronous execution rather than being executed right away
  auto array1_par_op1 = block_parallel_for( array1.rows(), array1_kernel1, parallel_execution_context("Array1","Kernel1") );

  // we create a second parallel operation we want to execute sequentially after the first addition
  auto array1_par_op2 = block_parallel_for( array1.rows(), array1_kernel2, parallel_execution_context("Array1","Kernel2") );

  // we finally create a third parallel operation we want to execute concurrently with the two others
  auto array2_par_op1 = block_parallel_for( array2.rows(), array2_kernel1, parallel_execution_context("Array2","Kernel1") );

  // we finally create a third parallel operation we want to execute concurrently with the two others
  auto array2_par_op2 = block_parallel_for( array2.rows(), array2_kernel2, parallel_execution_context("Array2","Kernel2") );

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
  else
  {
    std::cerr<<"unknown test mode "<<test_mode<<std::endl;
    std::cerr<<"valid test modes are : hybrid-delayed-lane , hybrid-lane and hybrid-sequence"<<std::endl;
    std::abort();
  }

  std::cout << "Schedule parallel operations ..." << std::endl;
  pq << onika::parallel::flush;

  std::cout << "Synchronize parallel operations ..." << std::endl;
  pq <<  onika::parallel::synchronize ;
  std::cout << "done" << std::endl;
}


int main(int argc,char*argv[])
{
  using onika::parallel::ParallelExecutionQueue;
  using onika::parallel::ParallelExecutionContext;
  using onika::parallel::ParallelExecutionContextAllocator;
  using onika::cuda::CudaContext;

  std::string test_mode = "hybrid-lane";
  std::string omp_mode = "task";
  if( argc > 1 ) test_mode = argv[1];
  if( argc > 2 ) omp_mode = argv[2];

  // application initialization
  onika::app::ApplicationConfiguration config = {};
  onika::app::intialize_openmp( config );
  onika::app::initialize_gpu( config );

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
          run_test(pq,parallel_execution_context,test_mode);
        }
      }
    }
  }
  else
  {
    run_test(pq,parallel_execution_context,test_mode);
  }

  return 0;
}

