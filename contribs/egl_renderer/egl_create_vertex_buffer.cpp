
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

  class EGLRenderVertexBufferCreate : public OperatorNode
  {
    using AttribVector = std::vector<std::string>;

    ADD_SLOT( std::string , vertex_buffer , INPUT_OUTPUT , "vertices" );
    ADD_SLOT( long , number_of_vertices , INPUT_OUTPUT , 0 );
    ADD_SLOT( AttribVector , vertex_attribs , INPUT , AttribVector({"GL_FLOAT","3"}) );

    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline void execute() override final
    {
      const size_t n_attribs = vertex_attribs->size() / 2;
      assert( vertex_attribs->size() == n_attribs*2 );

      std::vector<GLint> num_attribs;
      for(size_t i=0;i<n_attribs;i++)
      {
        num_attribs.push_back( gl_enum_from_string( vertex_attribs->at(i*2) ) );
        num_attribs.push_back( std::atoi( vertex_attribs->at(i*2+1).data() ) );
      }

      const auto buf_id = egl_render_manager->create_vertex_buffers( *vertex_buffer, *number_of_vertices, num_attribs );
      ldbg << "EGL : create vertex buffer " << *vertex_buffer << " , nv="<< *number_of_vertices <<", attribs=";
      for(size_t i=0;i<n_attribs;i++)
      {
        ldbg <<" "<<gl_enum_to_string(num_attribs.at(i*2))<<"x"<<num_attribs.at(i*2+1);
      }
      ldbg << " id="<<buf_id<<std::endl;
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_create_vertex_buffer)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_create_vertex_buffer", make_compatible_operator< EGLRenderVertexBufferCreate > );
  }

}

