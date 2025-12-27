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

//#pragma xstamp_cuda_enable // DO NOT REMOVE THIS LINE

#include <utility>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/physics/units.h>
#include <onika/string_utils.h>

namespace onika { namespace scg_builtin
{

  using namespace scg;

  class ApplicationUnitSystem : public onika::scg::OperatorNode
  {
    using UnitSystem = onika::physics::UnitSystem;
    ADD_SLOT( UnitSystem , unit_system , INPUT , OPTIONAL , DocString{"Defines default internal unit system used for quantity conversions whe no unit system specified."} );
    ADD_SLOT( bool       , verbose     , INPUT , false    , DocString{"If true prints a report of defined internal units used by default."} );

  public:

    inline void execute () override final
    {
      if( unit_system.has_value() )
      {
        onika::physics::set_internal_unit_system( *unit_system );
      }
      const auto& ius = onika::physics::internal_unit_system();
      if( *verbose )
      {
        lout << FormattedText{TEXT_FORMAT_ANSI,"Internal unit system for default conversions"} << std::endl;
        lout << FormattedText{TEXT_FORMAT_ANSI,"+-------------+-------------------+--------+"} << std::endl;
        lout << FormattedText{TEXT_FORMAT_ANSI,"| \033[31mType\033[0m        | Unit              | Symbol |"} << std::endl;
        lout << FormattedText{TEXT_FORMAT_ANSI,"+-------------+-------------------+--------+"} << std::endl;
        for(int i=0;i<=onika::physics::NUMBER_OF_UNIT_CLASSES;i++)
        {
          lout << format_string("| %-12s| %-18s| %-7s|",
                  std::string(onika::physics::g_unit_class_str[i]) ,
                  std::string(ius.m_units[i].m_name) ,
                  std::string(ius.m_units[i].m_short_name) ) << std::endl;
          lout << "+-------------+-------------------+--------+" << std::endl;
        }
      }
    }

    FormattedText formatted_documentation() const override final
    {
      return {TEXT_FORMAT_ANSI,
"Initializes the default unit system.\
These units are used for conversion when user indicates some units in input file and no specific unit system is specified for conversion.\
Default unit system is \033[1mSI international unit system\033[0m."};
    }

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(application_unit_system)
  {
   onika::scg::OperatorNodeFactory::instance()->register_factory( "unit_system", onika::scg::make_compatible_operator< ApplicationUnitSystem > );
  }

} }


