
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
#include <onika/math/basic_types.h>

#include <EGLRender/egl_render_manager.h>

namespace OnikaEGLRender
{

  using namespace onika;
  using namespace onika::math;
  using namespace onika::scg;
  using namespace EGLRender;

  class EGLRenderCameraCreate : public OperatorNode
  {
    ADD_SLOT( std::string , camera , INPUT_OUTPUT , "camera" );
    ADD_SLOT( Vec3d , position , INPUT , Vec3d{-5,5,0} );
    ADD_SLOT( Vec3d , lookat , INPUT , Vec3d{0,0,0} );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline void execute() override final
    {
      auto cam_id = egl_render_manager->create_camera( *camera );
      auto & cam = egl_render_manager->camera(cam_id);
      cam.lookAt( position->x, position->y, position->z, lookat->x, lookat->y, lookat->z );
      ldbg << "EGL : create camera " << *camera <<" id="<<cam_id
           << ", left=("<<cam.m_left[0]<<","<<cam.m_left[0]<<","<<cam.m_left[0]<<")"
           << ", up=("<<cam.m_up[0]<<","<<cam.m_up[0]<<","<<cam.m_up[0]<<")"
           << ", front=("<<cam.m_front[0]<<","<<cam.m_front[0]<<","<<cam.m_front[0]<<")"
           << std::endl;
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_create_camera)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_create_camera", make_compatible_operator< EGLRenderCameraCreate > );
  }

}

