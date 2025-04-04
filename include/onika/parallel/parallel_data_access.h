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

    namespace ParallelDataAccessConstants
    {
      static inline constexpr int8_t RO = 0x01;
      static inline constexpr int8_t WO = 0x02;
      static inline constexpr int8_t RW = 0x03;
      static inline constexpr int8_t END = 0xFF;
    }

    struct ParallelDataAccess
    {      
      static inline constexpr size_t MAX_NAME_LENGTH = 16;

      const void * m_data_ptr = nullptr; // acts as a unique id for data. can be object's address, array start, etc.
      unsigned int m_ndim = 0; // 0 means scalar, or, not parallel execution space correlated data access
      unsigned int m_stencil_size = 0;
      int8_t m_access_stencil[4*28] = { ParallelDataAccessConstants::END, }; // number of elements mus be (m_ndim+1) * m_stencil_size
      char m_name[MAX_NAME_LENGTH] = {'\0',}; // optional, may hold a string constant with the name of accessed data
      
      inline void set_ndim(unsigned int nd) { assert( nd <= 3 ); m_ndim = nd; }
      inline unsigned int ndim() const { return m_ndim; }
      inline void reset() { m_stencil_size=0; m_access_stencil[0]=ParallelDataAccessConstants::END; }
      inline void set_name(const std::string& s) { strncpy( m_name , s.c_str() , MAX_NAME_LENGTH ); m_name[MAX_NAME_LENGTH-1]='\0'; }
      inline void set_address(const void * p) { asser( p != nullptr ); m_data_ptr = p; }
      inline const char* name() const { return m_name; }
      inline void add_stencil_access( int8_t mode, int8_t ri, int8_t rj=0, int8_t rk=0 )
      {
        using namespace ParallelDataAccessConstants;
        m_access_stencil[ m_stencil_size*(ndim()+1) + 0 ] = mode;
        m_access_stencil[ m_stencil_size*(ndim()+1) + 1 ] = ri;
        if( ndim() >= 2 ) m_access_stencil[ m_stencil_size*(ndim()+1)+2] = rj;
        if( ndim() >= 3 ) m_access_stencil[ m_stencil_size*(ndim()+1)+3] = rk;
        ++ m_stencil_size;
        m_access_stencil[ m_stencil_size*(ndim()+1) ] = END;
      }
    };



    // exemple usage. note that stencil mus tbe static and const, it's lifespan must be longer than it's usage for any parallel kernel.
    inline ParallelDataAccess access_cross_2d(const void * p , int8_t center_mode = ParallelDataAccessConstants::RW, int8_t side_mode = ParallelDataAccessConstants::RO , const char* name_tag = nullptr)
    {
      using namespace ParallelDataAccessConstants;
      ParallelDataAccess pda = { p , 2 , 5 ,
        { center_mode, 0, 0 
        , side_mode  ,-1, 0 , side_mode,1,0 
        , side_mode  , 0,-1 , side_mode,0,1 
        , END } , {'*','\0',} };
      if( name_tag != nullptr ) pda.set_name( name_tag );
      return pda;
    }

    inline ParallelDataAccess access_cross_3d(const void * p , int8_t center_mode = ParallelDataAccessConstants::RW, int8_t side_mode = ParallelDataAccessConstants::RO , const char* name_tag = nullptr)
    {
      using namespace ParallelDataAccessConstants;
      ParallelDataAccess pda = { p , 3 , 7 ,
      { center_mode, 0, 0, 0 
      , side_mode  ,-1, 0, 0 , side_mode,1,0,0 
      , side_mode  , 0,-1, 0 , side_mode,0,1,0 
      , side_mode  , 0, 0,-1 , side_mode,0,0,1
      , END } , {'*','\0',} };
      if( name_tag != nullptr ) pda.set_name( name_tag );
      return pda;
    }

    inline ParallelDataAccess access_nbh_3d(const void * p , int8_t center_mode = ParallelDataAccessConstants::RW, int8_t side_mode = ParallelDataAccessConstants::RO , const char* name_tag = nullptr)
    {
      using namespace ParallelDataAccessConstants;
      ParallelDataAccess pda = { p , 3 , 27 , { END, } , {'*','\0',} };
      int idx = 0;
      for(int k=-1;k<=1;k++)
      for(int j=-1;j<=1;j++)
      for(int i=-1;i<=1;i++)
      {
        pda.m_access_stencil[ idx ++ ] = (std::abs(i)+std::abs(j)+std::abs(k))==0 ? center_mode : side_mode;
        pda.m_access_stencil[ idx ++ ] = i;
        pda.m_access_stencil[ idx ++ ] = j;
        pda.m_access_stencil[ idx ++ ] = k;
      }
      pda.m_access_stencil[ idx ++ ] = END;
      assert( idx == (3+1)*27 + 1 );
      if( name_tag != nullptr ) pda.set_name( name_tag );
      return pda;
    }

    inline ParallelDataAccess local_access(const void * p , unsigned int nd=3, int8_t mode = ParallelDataAccessConstants::RW , const char* name_tag = nullptr)
    {
      using namespace ParallelDataAccessConstants;
      ParallelDataAccess pda = { p , nd , 1 , { mode,0,0,0, } , {'*','\0',} };
      pda.m_access_stencil [ nd + 1 ] = END;
      if( name_tag != nullptr ) pda.set_name( name_tag );
      return pda;
    }

  }

}

