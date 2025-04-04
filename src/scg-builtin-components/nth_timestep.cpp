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
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>

#include <memory>


namespace onika
{
  namespace scg_builtin
  {

    using namespace scg;
    // =====================================================================
    // ========================== NthTimeStepNode ========================
    // =====================================================================

    struct NthTimeStepCache
    {
      int64_t m_last_time_step = 0;
      bool m_first_time = true;
    };

    class NthTimeStepNode : public OperatorNode
    {
      ADD_SLOT( long , timestep , INPUT);

      ADD_SLOT( bool , first , INPUT, true );
      ADD_SLOT( long , freq , INPUT , 1 );
      ADD_SLOT( bool , delayed , INPUT , false );
      ADD_SLOT( long , at_timestep , INPUT , -1 ); // exact timestep match

      ADD_SLOT( bool , result , INPUT_OUTPUT);
      ADD_SLOT( NthTimeStepCache , nth_timestep_cache , PRIVATE);

      public:
      void execute() override final
      {
        if( *at_timestep != -1 )
        {
          *result = ( (*at_timestep) == (*timestep) );
        }
        else if( nth_timestep_cache->m_first_time )
        {
          *result = (*first);
          nth_timestep_cache->m_first_time = false;
        }
        else if( (*freq) > 0 )
        {
          *result = 
            ( ( (*timestep) % (*freq) ) == 0 )
            || ( (*delayed) && ( ( (*timestep) - (*freq) ) >= nth_timestep_cache->m_last_time_step ) );
        }
        else
        {
          *result = false;
        }

        if( *result )
        {
          nth_timestep_cache->m_last_time_step = (*timestep);
        }
      }

    };

    // === register factories ===  
    ONIKA_AUTORUN_INIT(nth_timestep)
    {
      OperatorNodeFactory::instance()->register_factory( "nth_timestep", make_compatible_operator< NthTimeStepNode > );
    }
  }
}
