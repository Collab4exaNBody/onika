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
    template<bool _RelativeCoord, ssize_t ... _CoordVec>
    struct StaticElementCoord
    {
      static inline constexpr bool IsParallelSpaceRelative = _RelativeCoord;
      static inline constexpr unsigned int NDim = sizeof...(_CoordVec);
      static inline constexpr onika::oarray_t<ssize_t,NDim> m_value = { _CoordVec ... };
    };

    template<bool _RelativeCoord, unsigned int _NDim>
    struct DynamicElementCoord
    {
      static inline constexpr bool IsParallelSpaceRelative = _RelativeCoord;
      static inline constexpr unsigned int NDim = _NDim;
      const onika::oarray_t<ssize_t,NDim> m_value = {};
    };

    template<class AccessElementCoordT>                  struct IsAccessElementCoord                                                    : public std::false_type {};
    template<bool _RelativeCoord, ssize_t ... _CoordVec> struct IsAccessElementCoord< StaticElementCoord<_RelativeCoord,_CoordVec...> > : public std::true_type {};
    template<bool _RelativeCoord, unsigned int _NDim>    struct IsAccessElementCoord< DynamicElementCoord<_RelativeCoord,_NDim> >       : public std::true_type {};
    template<class T> static inline constexpr bool is_access_element_coord_v = IsAccessElementCoord<T>::value ;

    template<unsigned int _NDim , class... _AccessStencilCoords>
    struct ParallelDataAccess
    {
      static_assert( ( ... && is_access_element_coord_v<_AccessStencilCoords> ) );
      static_assert( ( ... && (_AccessStencilCoords::NDim==_NDim) ) );
      static inline constexpr unsigned int NDim = _NDim;
      static inline constexpr unsigned int StencilSize = sizeof...(_AccessStencilCoords);
      using access_stencil_t = FlatTuple< _AccessStencilCoords ... >;
      const void * const m_data_id = nullptr;
      const access_stencil_t m_access_stencil = {};
    };

  }

}

