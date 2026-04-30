
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
    int mouse_coord[2] = { 0, 0 };
    int mouse_button[3] = { 0, 0, 0 };
    int tilt[2] = { 0, 0 }; 
    int translate[2] = { 0, 0 };
    int forward = 0;
    int key_pressed = 0;
    int should_exit = false;
  };

  struct UserInteractionHandler
  {
    UserInteractionState m_uistate = {};
    EGLRender::NativeWindowEventHandler m_callbacks = {};
  };

  class EGLRenderEventHandler : public OperatorNode
  {
    ADD_SLOT( std::string , surface        , INPUT , "window" );
    ADD_SLOT( std::string , camera        , INPUT , "camera" );
    ADD_SLOT( bool , sim_continue , INPUT_OUTPUT , true );
    ADD_SLOT( EGLRenderManager , egl_render_manager , INPUT_OUTPUT );
    ADD_SLOT( UserInteractionHandler , egl_interaction_handler , PRIVATE );

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      auto & ren_surf = egl_render_manager->surface(*surface);
      ldbg << "EGL : egl_handle_events surface="<< *surface << " , camera="<< *camera <<std::endl;

      auto & uistate = egl_interaction_handler->m_uistate;
      if( ! egl_interaction_handler->m_callbacks.on_button_press )
      {
        egl_interaction_handler->m_callbacks.on_button_press = [&uistate](int state, int b, int x,int y)
        {
          uistate.mouse_button[b-1] = 1;
          uistate.mouse_coord[0] = x;
          uistate.mouse_coord[1] = y;
        };
        ren_surf.m_event_handler.on_button_press = egl_interaction_handler->m_callbacks.on_button_press;
      }
      if( ! egl_interaction_handler->m_callbacks.on_button_release )
      {
        egl_interaction_handler->m_callbacks.on_button_release = [&uistate](int state, int b, int x,int y)
        {
          uistate.mouse_button[b-1] = 0;
          uistate.mouse_coord[0] = x;
          uistate.mouse_coord[1] = y;
        };
        ren_surf.m_event_handler.on_button_release = egl_interaction_handler->m_callbacks.on_button_release;
      }
      if( ! egl_interaction_handler->m_callbacks.on_mouse_move )
      {
        egl_interaction_handler->m_callbacks.on_mouse_move = [&uistate](int x,int y)
        {
          int dx = x - uistate.mouse_coord[0];
          int dy = y - uistate.mouse_coord[1];
          uistate.mouse_coord[0] = x;
          uistate.mouse_coord[1] = y;
          if( uistate.mouse_button[0] )
          {
            uistate.tilt[0] += dx;
            uistate.tilt[1] += dy;
          }
          if( uistate.mouse_button[1] )
          {
            uistate.move[0] += dx;
            uistate.move[1] += dy;
          }
          if( uistate.mouse_button[2] )
          {
            uistate.move[2] += dy;
          }
        };
        ren_surf.m_event_handler.on_mouse_move = egl_interaction_handler->m_callbacks.on_mouse_move;
      }
      if( ! egl_interaction_handler->m_callbacks.on_key_release )
      {
        egl_interaction_handler->m_callbacks.on_key_release = [&uistate](int key)
        {
          if( key == 65307 ) uistate.should_exit = 1;
        };
        ren_surf.m_event_handler.on_key_release = egl_interaction_handler->m_callbacks.on_key_release;
      }

      ren_surf.process_events();
      
      if( uistate.should_exit )
      {
        ldbg<<"user requested stop"<<std::endl;
        *sim_continue = false;
      }
      
      const auto cam_id = egl_render_manager->camera_id( *camera );
      if( cam_id >= 0 )
      {
        egl_render_manager->camera(cam_id).tilt( uistate.tilt[0]*0.1f , uistate.tilt[1]*0.1f );
        egl_render_manager->camera(cam_id).move( uistate.translate[0]*0.1f , uistate.translate[1]*0.1f , uistate.translate[2]*0.1f );
        uistate.tilt[0] = 0;
        uistate.tilt[1] = 0;
        uistate.move[0] = 0;
        uistate.move[1] = 0;
        uistate.move[2] = 0;
      }

    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(egl_handle_events)
  {
    OperatorNodeFactory::instance()->register_factory( "egl_handle_events", make_compatible_operator< EGLRenderEventHandler > );
  }

}

