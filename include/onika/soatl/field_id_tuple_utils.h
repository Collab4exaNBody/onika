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

#include <onika/cuda/cuda.h>
#include <onika/flat_tuple.h>
#include <span>

namespace onika
{
  
  namespace soatl
  {
        
    template<class FieldT>
    ONIKA_HOST_DEVICE_FUNC
    inline size_t field_id_size_bytes( const onika::cuda::span<FieldT>& fa )
    {
      using ValueType = typename FieldT::value_type ;
      return fa.size() * sizeof(ValueType);
    }

    template<class FieldT>
    ONIKA_HOST_DEVICE_FUNC
    inline size_t field_id_size_bytes( const FieldT& fa )
    {
      using ValueType = typename FieldT::value_type ;
      return sizeof(ValueType);
    }

    template<class FieldTupleT, size_t ... FieldIndex>
    ONIKA_HOST_DEVICE_FUNC
    inline size_t field_id_tuple_size_bytes( const FieldTupleT& ft , std::index_sequence<FieldIndex...> )
    {
      size_t n = 0;
      if constexpr ( sizeof...(FieldIndex) > 0 )
      {
        ( ... , ( n += field_id_size_bytes( ft.get(onika::tuple_index_t<FieldIndex>{}) ) ) );
      }
      return n;
    }

    template<class... FieldT>
    ONIKA_HOST_DEVICE_FUNC
    inline size_t field_id_tuple_size_bytes( const onika::FlatTuple<FieldT...> & ft )
    {
      return field_id_tuple_size_bytes( ft , std::make_index_sequence< ft.size() >{} );
    }

  }
}

