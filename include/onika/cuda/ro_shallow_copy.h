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

#include <cstdlib>
#include <cassert>
#include <onika/cuda/cuda.h>
#include <vector>
#include <span>
#include <onika/memory/mm_vector.h>
#include <onika/cuda/stl_adaptors.h>

namespace onika
{
  namespace cuda
  {
    template<class T> struct ReadOnlyShallowCopyType { using type = T; };
    template<class T, class A> struct ReadOnlyShallowCopyType< std::vector<T,A> > { using type = onika::cuda::span<T>; };
    template<class T> struct ReadOnlyShallowCopyType< onika::memory::CudaMMVector<T> > { using type = onika::cuda::span<T>; };

    template<class T> using ro_shallow_copy_t = typename ReadOnlyShallowCopyType<T>::type;
  }

}

