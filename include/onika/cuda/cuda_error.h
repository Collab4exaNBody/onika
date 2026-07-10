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
#include <iostream>
#include <onika/cuda/cuda_context.h>

#if defined(ONIKA_CUDA_VERSION) || defined(ONIKA_HIP_VERSION)

#ifndef ONIKA_CU_CHECK_ERRORS
#define ONIKA_CU_CHECK_ERRORS( _expr_ ) ::onika::cuda::assertSuccess((_expr_), __FILE__, __LINE__)
#endif

namespace onika
{
  namespace cuda
  {
    ONIKA_HOST_DEVICE_FUNC
    inline void assertSuccess(onikaError_t code, const char *file, int line, bool abort_on_failure=true)
    {
       if( code != onikaSuccess ) 
       {
          printf( "%s error at %s:%d\n%s\n",ONIKA_CU_NAME_STR,file,int(line),ONIKA_CU_GET_ERROR_STRING(code));
          if( abort_on_failure ) { ONIKA_CU_ABORT(); }
       }
    }
  } 
}

#else

#ifndef ONIKA_CU_CHECK_ERRORS
#define ONIKA_CU_CHECK_ERRORS( _expr_ ) do{ auto x=(_expr_); auto y=x; x=y; }while(0)
#endif

#endif


