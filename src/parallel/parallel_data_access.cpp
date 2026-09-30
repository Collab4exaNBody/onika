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

#include <onika/parallel/parallel_data_access.h>

namespace onika
{

  namespace parallel
  {

    bool concurrent_data_access(const ParallelDataAccessVector& dav1 , const ParallelDataAccessVector& dav2 )
    {
      for(const auto & da1 : dav1)
      {
        unsigned int am1 = 0;
        for(auto ae:da1.m_stencil) am1 |= ae.mode();
        for(const auto & da2 : dav2)
        {
          unsigned int am2 = 0;
          for(auto ae:da2.m_stencil) am2 |= ae.mode();
          if( da1.m_data_ptr == da2.m_data_ptr && ( am1 & am2 ) != 0 ) return true;
        }
      }
      return false;
    }

    void concurrent_data_access_conflict_map(const ParallelDataAccessVector& dav1 , const ParallelDataAccessVector& dav2 , AccessConflictMap & conflict_map )
    {
      conflict_map.clear();
      for(const auto & da1 : dav1)
      {
        for(const auto & da2 : dav2)
        {
          if( da1.m_data_ptr == da2.m_data_ptr )
          {
            for(auto ae1 : da1.m_stencil)
            {        
              for(auto ae2 : da2.m_stencil)
              {
                if( ae1.mode()!=0 && ae2.mode()!=0 && ( ( ae1.mode() | ae2.mode() ) & AccessStencilElement::WO ) != 0 )
                {
                  const int ri = ae1.ri() - ae2.ri();
                  const int rj = ae1.rj() - ae2.rj();
                  const int rk = ae1.rk() - ae2.rk();
                  conflict_map.push_back( {{ri,rj,rk}} );
                }
              }
            }
          }
        }
      }
    }


  }

}

