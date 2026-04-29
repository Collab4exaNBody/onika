
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
    ADD_SLOT( std::string , shader , INPUT_OUTPUT , "shader" );
    ADD_SLOT( Vec3d       , position , INPUT , Vec3d{-5,5,0} );
    ADD_SLOT( Vec3d       , look_at , INPUT , Vec3d{0,0,0} );
    ADD_SLOT( double      , fov , INPUT , 60.0 );
    ADD_SLOT( double      , aspect , INPUT , 16.0/9.0 );
    ADD_SLOT( double      , near , INPUT , 0.1 );
    ADD_SLOT( double      , far , INPUT , 100.0 );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline void execute() override final
    {      
      auto shader_prog_id = egl_render_manager->shader_program_id(*shader);
      if( shader_prog_id < 0 )
      {
        fatal_error()<<"shader '"<< *shader <<"' not found"<<std::endl;
      }
      
      auto cam_id = egl_render_manager->create_camera( *camera );
      auto & cam = egl_render_manager->camera(cam_id);
      cam.look_at( { static_cast<GLfloat>(position->x), static_cast<GLfloat>(position->y), static_cast<GLfloat>(position->z) }
                 , { static_cast<GLfloat>(look_at->x), static_cast<GLfloat>(look_at->y), static_cast<GLfloat>(look_at->z) } );
      cam.perspective(*fov,*aspect,*near,*far);

      cam.attach_to_shader( egl_render_manager->shader_program_ptr(shader_prog_id), "camera" , "modelview", "projection" );
      cam.update_uniform();

      ldbg << "EGL : create camera " << *camera <<" id="<<cam_id<< " shader="<< *shader << std::endl;
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_create_camera)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_create_camera", make_compatible_operator< EGLRenderCameraCreate > );
  }

}

