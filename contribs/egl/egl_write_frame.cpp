
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
#include <fstream>

namespace OnikaEGLRender
{

  using namespace onika;
  using namespace onika::scg;
  using namespace EGLRender;

  class EGLRenderSurfaceWriteFrame : public OperatorNode
  {
    ADD_SLOT( std::string , surface        , INPUT , "window" );
    ADD_SLOT( std::string , filename       , INPUT , "eglframe" );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      auto & surf = egl_render_manager->surface(*surface);
      long width = surf.width();
      long height = surf.height();

      long pixel_data_sz = width * height * 4;
      auto pixel_data = std::make_unique_for_overwrite<char[]>(pixel_data_sz);
      glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixel_data.get() );

      const std::string& formattedname = *filename;
      std::ofstream data_out(formattedname+".raw");
      data_out.write( pixel_data.get() , pixel_data_sz );
      data_out.close();

      std::ofstream meta_out(formattedname+".to-png");
      meta_out << std::format("convert -depth 8 -size {}x{}+0 rgba:{}.raw {}.png",width,height,formattedname,formattedname) << std::endl;
      meta_out.close();
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_write_frame)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_write_frame", make_compatible_operator< EGLRenderSurfaceWriteFrame > );
  }

}

