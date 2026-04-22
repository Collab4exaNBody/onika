
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

namespace YAML
{
  template <> struct convert<EGLRender::GLPipelineConfig>
  {
    static inline bool decode(const Node &node, EGLRender::GLPipelineConfig &pc)
    {
      using EGLRender::gl_enum_from_string;
      if( !node.IsMap() ) return false;

      pc = EGLRender::GLPipelineConfig{};
      if( node["gl_enable"] )
      {
        if( ! node["gl_enable"].IsSequence() ) return false;
        for( const auto& es: node["gl_enable"].as< std::vector<std::string> >() )
        {
          pc.m_enable_flags.push_back( gl_enum_from_string(es) );
        }
      }
      if( node["gl_disable"] )
      {
        if( ! node["gl_disable"].IsSequence() ) return false;
        for( const auto& es: node["gl_disable"].as< std::vector<std::string> >() )
        {
          pc.m_disable_flags.push_back( gl_enum_from_string(es) );
        }
      }
      if( node["alpha_func"        ] ) pc.m_alpha_func         = gl_enum_from_string( node["alpha_func"        ].as<std::string>() );
      if( node["alpha_func_ref"    ] ) pc.m_alpha_func_ref     =                      node["alpha_func_ref"    ].as<double>();
      if( node["blend_src"         ] ) pc.m_blend_src          = gl_enum_from_string( node["blend_src"         ].as<std::string>() );
      if( node["blend_dst"         ] ) pc.m_blend_dst          = gl_enum_from_string( node["blend_dst"         ].as<std::string>() );
      if( node["stencil_mask"      ] ) pc.m_stencil_mask       =                      node["stencil_mask"      ].as<int>();
      if( node["stencil_func"      ] ) pc.m_stencil_func       = gl_enum_from_string( node["stencil_func"      ].as<std::string>() );
      if( node["stencil_func_ref"  ] ) pc.m_stencil_func_ref   =                      node["stencil_func_ref"  ].as<int>();
      if( node["stencil_func_mask" ] ) pc.m_stencil_func_mask  =                      node["stencil_func_mask" ].as<int>();
      if( node["stencil_op_sfail"  ] ) pc.m_stencil_op_sfail   = gl_enum_from_string( node["stencil_op_sfail"  ].as<std::string>() );
      if( node["stencil_op_dpfail" ] ) pc.m_stencil_op_dpfail  = gl_enum_from_string( node["stencil_op_dpfail" ].as<std::string>() );
      if( node["stencil_op_dppass" ] ) pc.m_stencil_op_dppass  = gl_enum_from_string( node["stencil_op_dppass" ].as<std::string>() );
      return true;
    }
  };
}

namespace OnikaEGLRender
{

  using namespace onika;
  using namespace onika::scg;
  using namespace EGLRender;

  class EGLRenderShaderProgramCreate : public OperatorNode
  {
    ADD_SLOT( std::string , shader_program , INPUT , "shader" );
    ADD_SLOT( std::string  , vertex_shader , INPUT , "" );
    ADD_SLOT( std::string  , geometry_shader , INPUT , "" );
    ADD_SLOT( std::string  , fragment_shader , INPUT , "" );
    ADD_SLOT( GLPipelineConfig , pipeline_config , INPUT , GLPipelineConfig{} );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );

  public:
    inline void execute() override final
    {
      const auto prog_id = egl_render_manager->create_shader_program( *shader_program, *vertex_shader, *geometry_shader, *fragment_shader, *pipeline_config );
      ldbg << "EGL : create shader program " << *shader_program << " id="<<prog_id<<std::endl;
      ldbg << "Vertex shader:" << std::endl << *vertex_shader << std::endl;
      ldbg << "Geometry shader:" << std::endl << *geometry_shader << std::endl;
      ldbg << "Fragment shader:" << std::endl << *fragment_shader << std::endl;
      ldbg << "Pipeline config:" << std::endl;
      egl_render_manager->shader_program(*shader_program).m_pipeline_config.to_stream( ldbg );
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_create_shader_program)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_create_shader_program", make_compatible_operator< EGLRenderShaderProgramCreate > );
  }

}

