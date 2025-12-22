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
    
    template<class T, bool = std::is_integral_v<T>, bool = is_orray_v<T> > struct ElementCoordND { static inline constexpr unsigned int value = 0; };
    template<class T> struct ElementCoordND<T,true,false> { static inline constexpr unsigned int value = 1; };
    template<class T> struct ElementCoordND<T,false,true> { static inline constexpr unsigned int value = T::array_size; };
    template<class T> static inline constexpr unsigned int element_coord_nd_v = ElementCoordND<T>::value;

    template<some_integral_orray_t ArrayT>
    inline constexpr size_t coord_range_size( const ArrayT& range_start , const ArrayT& range_end )
    {
      constexpr unsigned int ND = ArrayT::array_size;
      if( ND == 0 ) return 0;
      size_t n = 1;
      for(unsigned int i=0;i<ND;i++) n *= ( range_end[i] > range_start[i] ) ? ( range_end[i] - range_start[i] ) : 0;
      return n;
    }
    
    inline constexpr size_t coord_range_size(ssize_t range_start, ssize_t range_end)
    {
      return ( range_end > range_start ) ? ( range_end - range_start ) : 0 ;
    }

    template<some_orray_t ArrayT>
    inline constexpr bool coord_range_iterator_increment(ArrayT& range_it , const ArrayT& range_start , const ArrayT& range_end )
    {
      constexpr unsigned int ND = ArrayT::array_size;
      bool carry = true;
      for(int d=0;d<ND;d++)
      {
        if( carry ) ++ range_it[d];
        if( range_it[d] >= range_end[d] )
        {
          range_it[d] = range_start[d];
          carry = true;
        }
        else
        {
          carry = false;
        }
      }
      if( carry ) range_it = range_end;
      return ! carry;
    }

    inline bool coord_range_iterator_increment(ssize_t& range_it , ssize_t range_start , ssize_t range_end )
    {
      ++ range_it;
      if( range_it > range_end ) { range_it = range_end; return false; }
      else return true;
    }

    template<class T>
    concept some_space_coord = some_integral_orray_t<T> || std::is_integral_v<T>;

    template<some_space_coord CoordRangeT>
    inline auto coord_range_to_list( const CoordRangeT& range_start , const CoordRangeT& range_end )
    {
      using SpaceCoord = element_coord_t< element_coord_nd_v<CoordRangeT> >;
      const size_t n = coord_range_size( range_start, range_end );
      memory::CudaMMArray<SpaceCoord> elements;
      if( n > 0 )
      {
        elements.resize(n);
        auto it = range_start;
        size_t el_idx = 0;
        do
        {
          assert( el_idx < n );
          elements[el_idx++] = it;
        } while( coord_range_iterator_increment(it,range_start,range_end) );
      }
      assert( std::is_sorted( elements.begin() , elements.end() ) );
      return elements;
    }

    template<some_space_coord CoordRangeT>
    inline size_t coord_range_to_array( ssize_t * array, size_t arraysize, const CoordRangeT& range_start , const CoordRangeT& range_end )
    {
      if( element_coord_nd_v<CoordRangeT>*2 > arraysize ) return 0;
      if constexpr (std::is_integral_v<CoordRangeT> )
      {
        array[0] = range_start;
        array[1] = range_start;
      }
      else
      {
        for(auto x:range_start) *(array++) = x;
        for(auto x:range_end) *(array++) = x;
      }
      return element_coord_nd_v<CoordRangeT> * 2;
    }

    template<unsigned int _NDim=1, unsigned int _ElementListNDim=0, class _ElementListT = std::span< const element_coord_t<_ElementListNDim> > >
    struct ParallelExecutionSpace
    {
      static_assert( _NDim>=1 && _NDim<=3 && _ElementListNDim>=0 && _ElementListNDim<=3 );
      static_assert( _ElementListNDim==0 || _NDim==1 , "Element lists are only supported for 1D parallel execution spaces" );
      static inline constexpr unsigned int NDim = _NDim;
      static inline constexpr unsigned int ElementListNDim = _ElementListNDim;
      static inline constexpr unsigned int SpaceNDim = ( ElementListNDim==0 ) ? NDim : ElementListNDim ;
      using space_coord_t = onika::oarray_t<ssize_t,SpaceNDim>;
      using coord_t = onika::oarray_t<ssize_t,NDim>;
      using element_list_t = _ElementListT;
      using element_t = std::remove_cv_t< std::remove_reference_t< decltype( _ElementListT{}[0] ) > >;
      coord_t m_start;
      coord_t m_end;
      element_list_t m_elements = {};
      
      static inline consteval unsigned int space_nd() { return SpaceNDim; }
      inline constexpr size_t number_of_items() const { return coord_range_size( m_start , m_end ); }
    };

    template<typename T> struct is_parallel_execution_space_t : public std::false_type {};
    template<unsigned int Ndim,unsigned int ElNdim,class ElLst> struct is_parallel_execution_space_t< ParallelExecutionSpace<Ndim,ElNdim,ElLst> > : public std::true_type {};

    template<typename T> static inline constexpr bool is_parallel_execution_space_v = is_parallel_execution_space_t<T>::value;

    template<typename T> concept SortOfParallelExecutionSpace = is_parallel_execution_space_v<T>;
    template<typename T, typename PrevT> concept CompatibleParallelExecutionSpace = is_parallel_execution_space_v<T> && is_parallel_execution_space_v<PrevT> && T::SpaceNDim == PrevT::SpaceNDim;

    template<SortOfParallelExecutionSpace PES1, CompatibleParallelExecutionSpace<PES1> PES2 >
    inline void concurrent_execution_space_elements( const PES1& pes1, const PES2& pes2
                                                   , std::span< element_coord_t<PES1::SpaceNDim> > conflict_stencil
                                                   , std::vector< element_coord_t<PES1::SpaceNDim> > & pes2_dependent_elements
                                                   , std::vector< element_coord_t<PES1::SpaceNDim> > & pes2_remaining_elements )
    {
      
    }
  }
}

