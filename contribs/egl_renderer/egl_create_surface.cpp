
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

  class EGLRenderSurfaceCreate : public OperatorNode
  {
    ADD_SLOT( std::string , surface , INPUT_OUTPUT , "onika-egl-window" );
    ADD_SLOT( std::string , surface_type , INPUT , "window" );
    ADD_SLOT( long , width     , INPUT , 800 );
    ADD_SLOT( long , height     , INPUT , 800 );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline void execute() override final
    {
      EGLRenderSurfaceClass surf_type = ( (*surface_type)=="window" || (*surface_type)=="WINDOW" ) ? EGLRenderSurfaceClass::WINDOW : EGLRenderSurfaceClass::PBUFFER ;
      const auto surf_id = egl_render_manager->create_surface( *surface, surf_type, *width , *height );
      //auto & surf = egl_render_manager->surface( surf_id );
      ldbg << "EGL : create surface " << *surface << " , type="<< render_surface_type_as_string(surf_type) <<", size="<< *width <<"x"<< *height <<" , id="<<surf_id<< std::endl;
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_create_surface)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_create_surface", make_compatible_operator< EGLRenderSurfaceCreate > );
  }

}

