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

#include <fstream>

namespace onika { namespace scg_builtin
{

  using namespace scg;

  // =====================================================================
  // ========================== NthTimeStepNode ========================
  // =====================================================================

  class FileExists : public OperatorNode
  {
  public:
  
    ADD_SLOT( std::string , filename , INPUT , REQUIRED );
    ADD_SLOT( bool , result , INPUT_OUTPUT );
    
    void execute() override final
    {
      std::ifstream fin(*filename);
      *result = fin.good();
    }
  };

  // === register factories ===  
  ONIKA_AUTORUN_INIT(file_exists)
  {
    OperatorNodeFactory::instance()->register_factory( "file_exists", make_compatible_operator< FileExists > );
  }

} }

