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

#include <string>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <ranges>

#include <onika/debug.h>
#include <onika/deprecated.h>

namespace onika
{

  template<typename FormatArg>
  inline FormatArg convert_format_arg(const FormatArg& a) { return a; }

  inline const char* convert_format_arg(std::convertible_to<std::string_view> auto && s) { return std::string_view{s}.data(); }

  inline int format_string_buffer( std::convertible_to< std::span<char> > auto && dest , std::convertible_to<std::string_view> auto && format, auto && ... args)
  {
    std::span<char> buffer = dest;
    std::fill( buffer.begin() , buffer.end() , '\0' );
    std::string_view format_view = format;
    return std::snprintf( buffer.data() , buffer.size() , format_view.data(), convert_format_arg(args)... );
  }

  // for backward compatibility
  ONIKA_FUTURE_DEPRECATED
  inline int format_string_buffer( char* buf_ptr, size_t buf_size , std::convertible_to<std::string_view> auto && format, auto && ... args)
  {
    return format_string_buffer( std::span<char>{buf_ptr,buf_size} , format , args... );
  }

  inline std::string format_string(std::convertible_to<std::string_view> auto && format, auto && ... args)
  {
    std::string_view format_view = format;
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic warning "-Wformat-truncation=0"
#endif
    const int len = std::snprintf( nullptr, 0, format_view.data(), convert_format_arg(args)... );
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    assert(len>=0);
    std::string s(len+1,' ');
    ONIKA_DEBUG_ONLY( int len2 = ) std::snprintf( s.data(), len+1, format_view.data(), convert_format_arg(args)... );
    assert(len2==len);
    s[len] = '\0';
    s.resize(len);
    return s;
  }
  
  std::vector<std::string> split_string(const std::string& s, char delim=' ');

  void function_name_and_args(const std::string& proto, std::string& name, std::vector<std::string>& args );

  std::string str_tolower(std::string s);

  std::string str_indent(const std::string& s, int width=4, char indent_char=' ', const std::string& indent_suffix="" );
  
  std::string large_integer_to_string(size_t n);

  std::string plurial_suffix(double n, const std::string& suffix="s");

  std::string memory_bytes_string( ssize_t n , const char* fmt = "%.2f %s" );

}
