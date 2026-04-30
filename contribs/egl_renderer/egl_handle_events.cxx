
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

  struct UserInteractionState
  {
    GLfloat cam_dist = 5.0f;
    GLfloat angle_h = 0.0f;
    GLfloat angle_v = 0.0f;
    int mouse_last_x = -1;
    int mouse_last_y = -1;
    int should_exit = false;
    int left_drag = false;
    int right_drag = false;
  };

  class EGLRenderEventHandler : public OperatorNode
  {
    ADD_SLOT( std::string , surface        , INPUT , "window" );
    ADD_SLOT( bool , sim_continue , INPUT_OUTPUT , true );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );
    ADD_SLOT( UserInteractionState , egl_interaction_state , INPUT_OUTPUT );

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      auto & ren_surf = egl_render_manager->surface(*surface);
      ldbg << "EGL : make_current surface="<< *surface << std::endl;

      if( surf_type == EGLRenderSurfaceClass::WINDOW )
      {
        auto & uistate = *egl_interaction_state;
        ren_surf.m_event_handler.on_button_press = [&uistate,f=ren_surf.m_event_handler.on_button_press](int state, int b, int x,int y)
        {
          f(state,b,x,y);
          switch( b )
          {
            case 1 : uistate.left_drag=true; break;
            case 2 : uistate.right_drag=true; break;
            case 3 : uistate.should_exit=true; break;
          }
        };
        ren_surf.m_event_handler.on_button_release = [&uistate,f=ren_surf.m_event_handler.on_button_release](int state, int b, int x,int y)
        {
          f(state,b,x,y);
          switch( b )
          {
            case 1 : uistate.left_drag=false; break;
            case 2 : uistate.right_drag=false; break;
          }
        };
        ren_surf.m_event_handler.on_mouse_move = [&uistate,&camera,f=ren_surf.m_event_handler.on_mouse_move](int x,int y)
        {
          f(x,y);
          int dx = x - uistate.mouse_last_x;
          int dy = y - uistate.mouse_last_y;
          bool update_cam = false;
          if( uistate.left_drag )
          {
            uistate.angle_h += dx * 0.01f;
            uistate.angle_v += dy * 0.01f;
            update_cam = true;        
          }
          else if( uistate.right_drag )
          {
            uistate.cam_dist += dy * 0.01f;
            update_cam = true;
          }
          if(update_cam)
          {
            const auto d = uistate.cam_dist;
            const auto h = uistate.angle_h;
            const auto v = uistate.angle_v;
            GLfloat eyeZ = d * cos(v);
            GLfloat eyeY = d * sin(v);
            GLfloat eyeX = eyeZ * sin(h);
            eyeZ = eyeZ * cos(v);
            camera.look_at( {eyeX,eyeY,eyeZ} , {0,0,0} );
          }
          uistate.mouse_last_x = x;
          uistate.mouse_last_y = y;
        };
      }

      ren_surf.process_events();
      if( egl_interaction_state->should_exit )
      {
        ldbg<<"user requested stop"<<std::endl;
        *sim_continue = false;
      }
      ren_surf.make_current();
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_handle_events)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_handle_events", make_compatible_operator< EGLRenderEventHandler > );
  }

}

