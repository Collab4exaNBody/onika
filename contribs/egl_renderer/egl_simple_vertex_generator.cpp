
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

  class EGLRenderSimpleVertexGenerator : public OperatorNode
  {
    ADD_SLOT( std::string , vertex_buffer , INPUT_OUTPUT , "vertices" );
    ADD_SLOT( long , timestep , INPUT_OUTPUT , 0 );
    ADD_SLOT( long , number_of_vertices , INPUT_OUTPUT , 3 );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline void execute() override final
    {
      int buf_id = egl_render_manager->vertex_buffers_id( *vertex_buffer );
      if( buf_id == -1 )
      {
        ldbg << "EGL : create vertex buffer " << *vertex_buffer <<std::endl;
        const GLint attrib_formats[] = { GL_FLOAT,3 , GL_FLOAT,1 };
        buf_id = egl_render_manager->create_vertex_buffers( *vertex_buffer , *number_of_vertices , attrib_formats );
      }
      auto & glvbos = egl_render_manager->vertex_buffers(buf_id);
      ldbg << "EGL : update vertex buffer " << *vertex_buffer << " , nv="<< *number_of_vertices << " , id="<<buf_id<<std::endl;

      if( glvbos.number_of_attribs() != 2 || glvbos.attrib_type(0)!=GL_FLOAT || glvbos.attrib_type(1)!=GL_FLOAT || glvbos.attrib_components(0)!=3 || glvbos.attrib_components(1)!=1 )
      {
        onika::fatal_error() << "Works only with 2 vertex attributes : attribute 0 with format GLfloat x3 and attribute 1 with format GLfloat x1" << std::endl;
      }

      const long n_points = *number_of_vertices;
      glvbos.set_number_of_vertices( n_points );

      const long t = *timestep;
      GLfloat phi_base = t*0.003f;

      GLfloat* v = (GLfloat*) glvbos.host_map_write_only(0);
      GLfloat* a = (GLfloat*) glvbos.host_map_write_only(1);
      for(int j=0;j<n_points;j++)
      {
        GLfloat phi = phi_base + (2*M_PI*j/n_points);
        v[j*3+0]=std::cos(phi)*0.5f;
        v[j*3+1]=std::sin(phi)*0.5f;
        v[j*3+2]=0.0f;
        a[j] = -phi_base*10.0f;
      }
      glvbos.host_unmap(0);
      glvbos.host_unmap(1);
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_simple_vertex_generator)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_simple_vertex_generator", make_compatible_operator< EGLRenderSimpleVertexGenerator > );
  }

}

