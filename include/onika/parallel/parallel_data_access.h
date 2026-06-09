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
#include <cstdint>
#include <cassert>

namespace onika
{

  namespace parallel
  {

    struct AccessStencilElement
    {
      static inline constexpr uint8_t NA = 0x00; // No access, should'nt be used at all
      static inline constexpr uint8_t RO = 0x01;
      static inline constexpr uint8_t WO = 0x02;
      static inline constexpr uint8_t RW = 0x03;
      static inline constexpr const char * ACC_STR[4] = { "NA","RO","WO","RW" };
      uint8_t m_relative_access = 0;
      inline constexpr AccessStencilElement( unsigned int mode = RO , int ri=0 , int rj=0 , int rk=0 )
      {
        assert( mode<=3 && ri>=-1 && ri<=1 && rj>=-1 && rj<=1 && rk>=-1 && rk<=1 );
        m_relative_access = mode | uint8_t(ri+1)<<2 | uint8_t(rj+1)<<4 | uint8_t(rk+1)<<6 ;
      }
      inline constexpr unsigned int mode() const { return m_relative_access & 0x03; }
      inline constexpr const char* mode_str() const { return ACC_STR[mode()]; }
      inline constexpr int ri() const { return int( ( m_relative_access>>2 ) & 0x3 ) - 1; }
      inline constexpr int rj() const { return int( ( m_relative_access>>4 ) & 0x3 ) - 1; }
      inline constexpr int rk() const { return int( ( m_relative_access>>6 ) & 0x3 ) - 1; }
    };

    struct ParallelDataAccess
    {      
      static inline constexpr size_t MAX_NAME_LENGTH = 16;
      static inline constexpr size_t MAX_STENCIL_SIZE = 28;

      const void * m_data_ptr = nullptr; // acts as a unique id for data. can be object's address, array start, etc.
      uint32_t m_ndim = 0; // 0 means scalar, or, no parallel execution space correlated data access
      onika::inplace_vector<AccessStencilElement,MAX_STENCIL_SIZE> m_stencil;
      char m_name[MAX_NAME_LENGTH] = {'\0',}; // optional, may hold a string constant with the name of accessed data
      
      inline void set_ndim(unsigned int nd) { assert( nd <= 3 ); m_ndim = nd; }
      inline unsigned int ndim() const { return m_ndim; }
      inline void reset()
      {
        m_data_ptr=nullptr;
        m_ndim=0;
        m_stencil.clear();
        m_name[0]='\0';
      }
      inline void set_name(std::convertible_to<std::string_view> auto && s)
      {
        std::string_view sv = s;
        if( ! sv.empty() ) strncpy( m_name , sv.data() , MAX_NAME_LENGTH );
        m_name[ std::min( sv.size() , MAX_NAME_LENGTH-1 ) ] = '\0';
      }
      inline void set_address(const void * p) { assert( p != nullptr ); m_data_ptr = p; }
      inline const void* address() const { return m_data_ptr; }
      inline const char* name() const { return m_name; }
    };

    static_assert( sizeof(ParallelDataAccess) == 64 );

    static inline constexpr size_t MAX_PARALLEL_DATA_ACCESSES = 2;
    using ParallelDataAccessVector = onika::inplace_vector<ParallelDataAccess,MAX_PARALLEL_DATA_ACCESSES>;

    // return true if two access data sets may involve concurrent data access conflicts (read/write or write/write conflict on the same data pointer)
    bool concurrent_data_access(const ParallelDataAccessVector& dav1 , const ParallelDataAccessVector& dav2 );

    static inline constexpr size_t MAX_CONFLICT_MAP_SIZE = 128;
    using AccessConflictMap = onika::inplace_vector< oarray_t<int,3> , MAX_CONFLICT_MAP_SIZE >;
    void concurrent_data_access_conflict_map(const ParallelDataAccessVector& dav1 , const ParallelDataAccessVector& dav2 , AccessConflictMap & conflict_map );


    // exemple usage. note that stencil mus tbe static and const, it's lifespan must be longer than it's usage for any parallel kernel.
    inline ParallelDataAccess access_cross_2d(const void * p , uint8_t center_mode = AccessStencilElement::RW, uint8_t side_mode = AccessStencilElement::RO , std::convertible_to<std::string_view> auto && name_tag = std::string_view() )
    {
      ParallelDataAccess pda = { p , 2 ,
        { { center_mode, 0, 0 }
        , { side_mode  ,-1, 0 } , { side_mode,1,0 }
        , { side_mode  , 0,-1 } , { side_mode,0,1 } } };
      pda.set_name( std::string_view(name_tag) );
      return pda;
    }

    inline ParallelDataAccess access_cross_3d(const void * p , uint8_t center_mode = AccessStencilElement::RW, uint8_t side_mode = AccessStencilElement::RO , std::convertible_to<std::string_view> auto && name_tag = std::string_view() )
    {
      ParallelDataAccess pda = { p , 3 ,
        { { center_mode, 0, 0, 0 }
        , { side_mode  ,-1, 0, 0 } , { side_mode,1,0,0 }
        , { side_mode  , 0,-1, 0 } , { side_mode,0,1,0 }
        , { side_mode  , 0, 0,-1 } , { side_mode,0,0,1 } } };
      pda.set_name( std::string_view(name_tag) );
      return pda;
    }

    inline ParallelDataAccess access_nbh_3d(const void * p , uint8_t center_mode = AccessStencilElement::RW, uint8_t side_mode = AccessStencilElement::RO , std::convertible_to<std::string_view> auto && name_tag = std::string_view() )
    {
      ParallelDataAccess pda = { p , 3 };
      for(int8_t k=-1;k<=1;k++)
      for(int8_t j=-1;j<=1;j++)
      for(int8_t i=-1;i<=1;i++)
      {
        pda.m_stencil.push_back( { (std::abs(i)+std::abs(j)+std::abs(k))==0 ? center_mode : side_mode , i , j , k } );
      }
      pda.set_name( std::string_view(name_tag) );
      return pda;
    }

    inline ParallelDataAccess local_access(const void * p , unsigned int nd=3, uint8_t mode = AccessStencilElement::RW , std::convertible_to<std::string_view> auto && name_tag = std::string_view())
    {
      ParallelDataAccess pda = { p , uint32_t(nd) , { {mode} } };
      pda.set_name( std::string_view(name_tag) );
      return pda;
    }

  }

}

