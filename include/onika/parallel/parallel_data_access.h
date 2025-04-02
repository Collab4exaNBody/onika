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
      static inline constexpr int8_t RO = 0x01;
      static inline constexpr int8_t WO = 0x02;
      static inline constexpr int8_t RW = 0x03;
      static inline constexpr int8_t END = 0xFF;

      const void * const m_data_ptr = nullptr; // acts as a unique id for data. can be object's address, array start, etc.
      const unsigned int m_ndim = 0; // 0 means scalar, or, not parallel execution space correlated data access
      const unsigned int m_stencil_size = 0;
      const int8_t m_access_stencil[4*28] ; // number of elements mus be (m_ndim+1) * m_stencil_size
      const char m_name[16] = {'\0',}; // optional, may hold a string constant with the name of accessed data
      
      void set_ndim(unsigned int nd);
      void add_stencil_access( int8_t mode, int8_t ri, int8_t rj=0, int8_t rk=0 );
      void set_name(const std::string& s);
      void set_address(const void * p);
      
      const char* name() const;
    };

    /*
      // exemple usage. note that stencil mus tbe static and const, it's lifespan must be longer than it's usage for any parallel kernel.
      ParallelDataAccess pda = { my_array_2d , 2 , 4 , { RW,0,0 , RO,-1,0 , RO,1,0 , RO,0,-1 , RO,0,1 , END } , "velocity" };
    */

  }

}

