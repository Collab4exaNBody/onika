
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

  class EGLRenderDrawVertexBuffer : public OperatorNode
  {
    using IntVector = std::vector<GLint>;

    ADD_SLOT( std::string , vertex_buffer  , INPUT , "vertices" );
    ADD_SLOT( std::string , elements       , INPUT , "all" );
    ADD_SLOT( std::string , shader_program , INPUT , "shader" );
    ADD_SLOT( long , element_start , INPUT , 0 );
    ADD_SLOT( long , element_count , INPUT , -1 );
    ADD_SLOT( std::string , primitive , INPUT , "GL_POINTS" , DocString{"May be one of the following : GL_POINTS, GL_LINE_STRIP or GL_TRIANGLE_STRIP"} );

//    ADD_SLOT( bool , sim_continue , INPUT_OUTPUT, true );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      GLenum prim_type = gl_enum_from_string( *primitive );
      if( prim_type!=GL_POINTS && prim_type!=GL_LINE_STRIP && prim_type!=GL_TRIANGLE_STRIP )
      {
        onika::fatal_error() << "primitive must be one of the following : GL_POINTS, GL_LINE_STRIP or GL_TRIANGLE_STRIP. Found "<< (*primitive) << " instead"<< std::endl;
      }

      auto & vbo = egl_render_manager->vertex_buffers(*vertex_buffer);
      auto & shader = egl_render_manager->shader_program(*shader_program);
      std::shared_ptr<EGLRender::GLElementBuffer> elbuf = egl_render_manager->element_buffer_ptr( *elements );

      long vstart = *element_start;
      long vcount = *element_count;
      ldbg << "EGL : vbo="<< *vertex_buffer << " shader="<< *shader_program << "elements="<< *elements << " vstart="<<vstart<<" vcount="<<vcount << std::endl;

      shader.use();
      vbo.use();

      if( elbuf != nullptr )
      {
        elbuf->draw(prim_type, vstart, vcount);
      }
      else
      {
        if(vcount==-1) vcount = vbo.number_of_vertices() - vstart;
        glDrawArrays(prim_type, vstart, vcount);
      }
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_draw)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_draw", make_compatible_operator< EGLRenderDrawVertexBuffer > );
  }

}

