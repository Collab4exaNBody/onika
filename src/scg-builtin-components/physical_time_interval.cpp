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

namespace onika
{
  namespace scg_builtin
  {

    using namespace scg;

    class PhysicalTimeInterval : public OperatorNode
    {
      static inline constexpr double default_end_time = std::numeric_limits<double>::infinity();
      static inline constexpr double default_prev_time = - std::numeric_limits<double>::infinity();
    
      ADD_SLOT( double , physical_time , INPUT_OUTPUT );

      ADD_SLOT( double , start_time , INPUT , 0.0 );
      ADD_SLOT( double , end_time , INPUT , default_end_time );
      ADD_SLOT( double , time_interval , INPUT , 0.0 );

      ADD_SLOT( double , previous_physical_time , INPUT_OUTPUT , default_prev_time );
      ADD_SLOT( bool , result , INPUT_OUTPUT);

      public:
      void execute() override final
      {
        if( (*time_interval) > 0.0 && (*physical_time) >= (*start_time) && (*physical_time) <= (*end_time) && ( (*physical_time) - (*previous_physical_time) ) >= (*time_interval) )
        {
          *previous_physical_time = *physical_time;
          *result = true;
        }
        else
        {
          *result = false;
        }
      }

    };

    // === register factories ===  
    ONIKA_AUTORUN_INIT(physical_time_interval)
    {
      OperatorNodeFactory::instance()->register_factory( "physical_time_interval", make_compatible_operator< PhysicalTimeInterval > );
    }
  }
}
