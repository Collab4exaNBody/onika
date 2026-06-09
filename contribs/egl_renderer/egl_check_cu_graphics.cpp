
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

#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/log.h>

#include <EGLRender/egl_render_manager.h>
#include <EGLRender/cu_graphics_gl.h>

namespace OnikaEGLRender
{
  using namespace onika;
  using namespace onika::scg;
  using namespace EGLRender;

  class EGLRenderCheckCuGraphics : public OperatorNode
  {
    ADD_SLOT( std::string , surface        , INPUT , "window" );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      // check if a Compatible Cuda device is associated with the current OpenGL context
      constexpr unsigned int MAX_CUDA_DEVICES = 16;
      int cudaDeviceCount = 0;
      int cudaDevices[MAX_CUDA_DEVICES];

      egl_render_manager->surface(*surface).make_current();
      lout << "Checking Compute API / Graphics connection :"<<std::endl;
      lout << "OpenGL :\n\t"<<glGetString(GL_VENDOR)<<"\n\t"<<glGetString(GL_RENDERER)<<"\n\t"<<glGetString(GL_VERSION)<<std::endl;
      ONIKA_CU_GET_DEVICE_COUNT(&cudaDeviceCount);
      lout << "Compute devices : "<<cudaDeviceCount<<std::endl;
      egl_gpu_compute_gl_get_devices(&cudaDeviceCount, cudaDevices, MAX_CUDA_DEVICES);
      if( cudaDeviceCount == 0 )
      {
        fatal_error()<<"No compatible Cuda device is attached to current OpenGL context"<<std::endl;
      }
      lout<<"Found "<<cudaDeviceCount<<" compatible compute devices : [";
      for(int i=0;i<cudaDeviceCount;i++) lout<<((i>0)?",":"")<<cudaDevices[i];
      lout<<"]"<<std::endl;
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_check_cu_graphics)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_check_cu_graphics", make_compatible_operator< EGLRenderCheckCuGraphics > );
  }

}

