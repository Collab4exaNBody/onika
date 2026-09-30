
// sur hfgpu
// nvcc -arch=compute_80 -code=sm_80 --compiler-options -fopenmp benchmark.cu -o benchmark_hfgpu
// execavec memoire unifiee
// OMP_NUM_THREADS=64 ccc_mprun -n1 -c128 -phfgpu -T3600 ./benchmark_hfgpu <<< "0 1 1037"
// avec memoire device et copie host/device
// OMP_NUM_THREADS=64 ccc_mprun -n1 -c128 -phfgpu -T3600 ./benchmark_hfgpu <<< "0 0 1037"

// sur HE
// nvcc -arch=compute_80 -code=sm_80 --compiler-options -fopenmp benchmark.cu -o benchmark_he
// nvcc -arch=compute_90 -code=sm_90 --compiler-options -fopenmp benchmark.cu -o benchmark_he
// execavec memoire unifiee
// OMP_NUM_THREADS=64 ./benchmark_he <<< "0 1 1037"
// avec memoire device et copie host/device
// OMP_NUM_THREADS=64 ./benchmark_he <<< "0 0 1037"

#include <iostream>
#include <chrono>
#include <omp.h>

//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_context.h>
#include <onika/memory/allocator.h>
#include <onika/parallel/parallel_execution_context.h>

#define N (1024*1024)
#define M (10000000)

void cpu_init( double * __restrict__ data )
{
# pragma omp parallel
  {
#   pragma omp single
    {
      std::cout << "using "<<omp_get_num_threads()<<" CPU threads"<<std::endl;
    }
#   pragma omp for schedule(static)
    for(int i = 0; i < N; i++)
    {
      data[i] = i * 1.0 / N;
    }
  }
}

void cpu_compute( double * __restrict__ data )
{
# pragma omp parallel for schedule(static)
  for(int i = 0; i < N; i++)
  {
    for(int j = 0; j < M; j++)
    {
       data[i] = data[i] * data[i] - 0.25;
    }
  }
}

ONIKA_DEVICE_KERNEL_FUNC void gpu_compute( double * __restrict__ data )
{
# ifdef ONIKA_GPU_DEVICE_COMPILE
  const int i = ONIKA_CU_THREAD_IDX + ONIKA_CU_BLOCK_IDX * ONIKA_CU_BLOCK_SIZE;
  for(int j = 0; j < M; j++)
  {
    data[i] = data[i] * data[i] - 0.25;
  }
# endif
}

int main()
{
  int n_gpus = 0;
  onika::cuda::CudaContext::set_global_gpu_enable( true );
  auto cu_dev_count_rc = ONIKA_CU_GET_DEVICE_COUNT(&n_gpus);
  if( n_gpus > 0 )
  {
    ONIKA_CU_CHECK_ERRORS( cu_dev_count_rc );
  }
  else
  {
    std::cout<<"no GPU found, aborting" << std::endl;
    return 1;
  }
  std::cout << "found "<<n_gpus<<" GPU(s)"<<std::endl;
  onika::memory::GenericHostAllocator::set_cuda_enabled( true );
  onika::parallel::ParallelExecutionContext::s_gpu_sm_mult    = 2;
  onika::parallel::ParallelExecutionContext::s_gpu_sm_add     = 0;
  onika::parallel::ParallelExecutionContext::s_gpu_block_size = 256;

  double *h_data = nullptr;
  double *d_data = nullptr;

  int run_host=0, uvm=0, idx=0;
  std::cin >> run_host >> uvm >> idx;
  std::cout << "run_host="<<run_host<<", uvm="<<uvm<<" , idx="<<idx<<std::endl;

  if( uvm )
  {
    ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MALLOC_MANAGED( & h_data , N * sizeof(double) ) );
    d_data = h_data;
  }
  else
  {
    h_data = new double[N];
  }

  cpu_init( h_data );

  if( ! uvm )
  {
    ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MALLOC( & d_data, N * sizeof(double)) );
    ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( d_data, h_data, N * sizeof(double) /*, onikaMemcpyHostToDevice */ ) );
  }

  const auto T0 = std::chrono::high_resolution_clock::now();

  if(run_host) cpu_compute( h_data );
  const double vhost = h_data[idx];

  const auto T1 = std::chrono::high_resolution_clock::now();
  ONIKA_CU_LAUNCH_KERNEL(N/256,256,0,0,gpu_compute,d_data);
  const auto T2 = std::chrono::high_resolution_clock::now();

  if( ! uvm ) { ONIKA_CU_CHECK_ERRORS( ONIKA_CU_MEMCPY( h_data, d_data, N * sizeof(double) /*, onikaMemcpyDeviceToHost */ ) ); }
  ONIKA_CU_CHECK_ERRORS( ONIKA_CU_DEVICE_SYNCHRONIZE() );
  const double vcuda = h_data[idx];
  const auto T3 = std::chrono::high_resolution_clock::now();

  std::cout << "result["<<idx<<"] = "<< vhost<<" / "<<vcuda<<std::endl;
  if(run_host) std::cout << "host time = "<< (T1-T0).count() / 1000000.0 << std::endl;
  std::cout << "cuda time = "<< (T2-T1).count() / 1000000.0 << " + "<< (T3-T2).count() / 1000000.0 << " = "<< (T3-T1).count() / 1000000.0 <<std::endl;
  if(run_host) std::cout << "ratio = "<< (T1-T0).count() * 1.0 / (T3-T1).count()  << std::endl;

  if( ! uvm ) { ONIKA_CU_CHECK_ERRORS( ONIKA_CU_FREE(d_data) ); }

  return 0;
}

