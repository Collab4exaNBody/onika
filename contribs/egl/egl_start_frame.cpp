
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

namespace OnikaEGLRender
{

  using namespace onika;
  using namespace onika::scg;
  using namespace EGLRender;

  class EGLRenderSurfaceStartFrame : public OperatorNode
  {
    ADD_SLOT( std::string , surface        , INPUT , "window" );
    ADD_SLOT( std::string , camera         , INPUT , "camera" );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      auto & render_surface = egl_render_manager->surface(*surface);
      ldbg << "EGL : make_current surface="<< *surface << std::endl;
      render_surface.make_current();
      glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

      const auto cam_id = egl_render_manager->camera_id(*camera);
      if( cam_id != -1 )
      {
        ldbg << "updating camera "<< *camera << std::endl;
        egl_render_manager->camera(cam_id).update_uniform();
      }
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_start_frame)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_start_frame", make_compatible_operator< EGLRenderSurfaceStartFrame > );
  }

}

