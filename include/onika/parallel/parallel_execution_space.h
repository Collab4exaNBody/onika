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
#include <span>

namespace onika
{

  namespace parallel
  {
    template<unsigned int ND=1> struct ElementCoordT { using type = onika::oarray_t<ssize_t,ND>; };
    template<> struct ElementCoordT<1> { using type = ssize_t; };
    template<unsigned int ND> using element_coord_t = typename ElementCoordT<ND>::type;
    
    template<class T, bool = std::is_integral_v<T> > struct ElementCoordND { static inline constexpr unsigned int value = 1; };
    template<class T> struct ElementCoordND<T,false> { static inline constexpr unsigned int value = T::array_size; };
    template<class T> static inline constexpr unsigned int element_coord_nd_v = ElementCoordND<T>::value;

    template<unsigned int _NDim=1, unsigned int _ElementListNDim=0, class _ElementListT = std::span< const element_coord_t<_ElementListNDim> > >
    struct ParallelExecutionSpace
    {
      static_assert( _NDim>=1 && _NDim<=3 && _ElementListNDim>=0 && _ElementListNDim<=3 );
      static_assert( _ElementListNDim==0 || _NDim==1 , "Element lists are only supported for 1D parallel execution spaces" );
      static inline constexpr unsigned int NDim = _NDim;
      static inline constexpr unsigned int ElementListNDim = _ElementListNDim;
      using coord_t = onika::oarray_t<ssize_t,NDim>;
      using element_list_t = _ElementListT;
      using element_t = std::remove_cv_t< std::remove_reference_t< decltype( _ElementListT{}[0] ) > >;
      coord_t m_start;
      coord_t m_end;
      element_list_t m_elements = {};
      
    };
  }

}

