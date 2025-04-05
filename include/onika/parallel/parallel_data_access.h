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
#include <cmath>

namespace onika
{

  namespace parallel
  {

    struct AccessStencilElement
    {
      static inline constexpr int8_t RO = 0x01;
      static inline constexpr int8_t WO = 0x02;
      static inline constexpr int8_t RW = 0x03;

      int8_t m_mode = RW;
      int8_t ri = 0;
      int8_t rj = 0;
      int8_t rk = 0;
    };

    struct ParallelDataAccess
    {      
      static inline constexpr size_t MAX_NAME_LENGTH = 16;
      static inline constexpr size_t MAX_STENCIL_SIZE = 27;

      const void * m_data_ptr = nullptr; // acts as a unique id for data. can be object's address, array start, etc.
      unsigned int m_ndim = 0; // 0 means scalar, or, not parallel execution space correlated data access
      unsigned int m_stencil_size = 0;
      AccessStencilElement m_stencil[MAX_STENCIL_SIZE];
      char m_name[MAX_NAME_LENGTH] = {'\0',}; // optional, may hold a string constant with the name of accessed data
      
      inline void set_ndim(unsigned int nd) { assert( nd <= 3 ); m_ndim = nd; }
      inline unsigned int ndim() const { return m_ndim; }
      inline void reset() { m_stencil_size = 0; }
      inline void set_name(const std::string& s) { strncpy( m_name , s.c_str() , MAX_NAME_LENGTH ); m_name[MAX_NAME_LENGTH-1]='\0'; }
      inline void set_address(const void * p) { assert( p != nullptr ); m_data_ptr = p; }
      inline const char* name() const { return m_name; }
      inline void add_stencil_access( const AccessStencilElement& ase  ) 
      {
        assert( m_stencil_size < MAX_STENCIL_SIZE );
        m_stencil[ m_stencil_size ++ ] = ase; 
      }
    };

    // exemple usage. note that stencil mus tbe static and const, it's lifespan must be longer than it's usage for any parallel kernel.
    inline ParallelDataAccess access_cross_2d(const void * p , int8_t center_mode = AccessStencilElement::RW, int8_t side_mode = AccessStencilElement::RO , const char* name_tag = nullptr)
    {
      ParallelDataAccess pda = { p , 2 , 5 ,
        { { center_mode, 0, 0 }
        , { side_mode  ,-1, 0 } , { side_mode,1,0 }
        , { side_mode  , 0,-1 } , { side_mode,0,1 } }
        , {'*','\0',} };
      if( name_tag != nullptr ) pda.set_name( name_tag );
      return pda;
    }

    inline ParallelDataAccess access_cross_3d(const void * p , int8_t center_mode = AccessStencilElement::RW, int8_t side_mode = AccessStencilElement::RO , const char* name_tag = nullptr)
    {
      ParallelDataAccess pda = { p , 3 , 7 ,
      { { center_mode, 0, 0, 0 }
      , { side_mode  ,-1, 0, 0 } , { side_mode,1,0,0 }
      , { side_mode  , 0,-1, 0 } , { side_mode,0,1,0 }
      , { side_mode  , 0, 0,-1 } , { side_mode,0,0,1 } }
      , {'*','\0',} };
      if( name_tag != nullptr ) pda.set_name( name_tag );
      return pda;
    }

    inline ParallelDataAccess access_nbh_3d(const void * p , int8_t center_mode = AccessStencilElement::RW, int8_t side_mode = AccessStencilElement::RO , const char* name_tag = nullptr)
    {
      ParallelDataAccess pda = { p , 3 , 0 , {} , {'*','\0',} };
      for(int8_t k=-1;k<=1;k++)
      for(int8_t j=-1;j<=1;j++)
      for(int8_t i=-1;i<=1;i++)
      {
        pda.add_stencil_access( { (std::abs(i)+std::abs(j)+std::abs(k))==0 ? center_mode : side_mode , i , j , k } );
      }
      if( name_tag != nullptr ) pda.set_name( name_tag );
      return pda;
    }

    inline ParallelDataAccess local_access(const void * p , unsigned int nd=3, int8_t mode = AccessStencilElement::RW , const char* name_tag = nullptr)
    {
      ParallelDataAccess pda = { p , nd , 1 , { {mode,0,0,0} } , {'*','\0',} };
      if( name_tag != nullptr ) pda.set_name( name_tag );
      return pda;
    }

  }

}

