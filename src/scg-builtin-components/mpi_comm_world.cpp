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

#include <mpi.h>

namespace onika { namespace scg_builtin
{

  using namespace scg;

  struct MpiCommWorld : public OperatorNode
  {
    ADD_SLOT( MPI_Comm , mpi , INPUT_OUTPUT , MPI_COMM_WORLD );
    inline void execute () override final
    {
      int size = 1;      
      int rank = 0;
      MPI_Comm_rank(*mpi,&rank);
      MPI_Comm_size(*mpi,&size);
      ldbg << "MPI processes in communicator : "<<size<<std::endl;
      ldbg << "MPI rank = rank"<<std::endl;
    }
  };

  // === register factory ===
  ONIKA_AUTORUN_INIT(mpi_comm_world)
  {
    OperatorNodeFactory::instance()->register_factory( "mpi_comm_world", make_compatible_operator<MpiCommWorld> );
  }

} }

