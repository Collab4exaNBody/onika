
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

  class EGLRenderInit : public OperatorNode
  {
    ADD_SLOT( bool , enable_display , INPUT , true );
    ADD_SLOT( bool , use_gles       , INPUT , false );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:

    inline void execute() override final
    {
      ldbg << "init EGL : onscreen=" << *enable_display << " , GLES="<< *use_gles <<std::endl;
      egl_render_manager->init_platform( *enable_display, *use_gles );
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_init)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_init", make_compatible_operator< EGLRenderInit > );
  }

}

