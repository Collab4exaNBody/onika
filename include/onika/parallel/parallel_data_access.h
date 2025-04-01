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

#pragma once

#include <onika/oarray.h>
#include <onika/flat_tuple.h>
#include <type_traits>

namespace onika
{

  namespace parallel
  {

    struct ParallelDataAccess
    {
      const void * const m_data_ptr = nullptr; // acts as a unique id for data. can be object's address, array start, etc.
      const ssize_t * const m_access_stencil = nullptr; // number of elements mus be m_ndim * m_stencil_size
      const char * const m_name = nullptr; // optional, may hold a string constant with the name of 
      const unsigned int m_ndim = 0; // 0 means scalar, or, not parallel execution space correlated data access
      const unsigned int m_stencil_size = 0;
    };

  }

}

