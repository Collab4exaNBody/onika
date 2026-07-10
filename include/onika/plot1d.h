#pragma once

#include <map>
#include <vector>
#include <string>
#include <onika/memory/allocator.h>
#include <onika/cuda/stl_adaptors.h>


namespace onika
{
  using PlotSample = onika::cuda::pair<double,double>;
  using Plot1D = onika::memory::CudaMMVector< PlotSample >;

  struct Plot1DSet
  {
    std::map< std::string , Plot1D > m_plots;
    std::map< std::string , std::string > m_captions;
  };

}

