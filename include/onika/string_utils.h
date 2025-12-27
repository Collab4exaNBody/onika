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
#include <span>

#include <onika/debug.h>
#include <onika/deprecated.h>

namespace onika
{
  enum TextFormat { TEXT_FORMAT_RAW, TEXT_FORMAT_ANSI, TEXT_FORMAT_MARKDOWN };

  // direct access to user console (or graphic system)
  struct FormattedText
  {
    TextFormat m_format = TEXT_FORMAT_RAW;
    std::string m_text;

    inline TextFormat format() const { return m_format; }
    inline std::string_view formatted_text() const { return m_text; }
    std::string to_raw() const;
    std::string to_ansi() const;
  };

  template<typename FormatArg>
  inline FormatArg convert_format_arg(const FormatArg& a) { return a; }

  inline const char* convert_format_arg(const std::string& a) { return a.c_str(); }

  inline int format_string_buffer( std::convertible_to< std::span<char> > auto && buf_arg, std::convertible_to<std::string_view> auto && format, auto && ... args)
  {
    std::span<char> buf = buf_arg;
    std::fill( buf.begin() , buf.end() , '\0' );
    return std::snprintf( buf.data(), buf.size(), format.data(), convert_format_arg(args)... );
  }

  ONIKA_FUTURE_DEPRECATED
  inline int format_string_buffer(char* buf, size_t bufsize, const std::string& format, auto && ... args)
  {
    return format_string_buffer( std::span<char>{buf,bufsize} , format , args ... );
  }

  template<typename... Args>
  inline std::string format_string(const std::string& format, const Args & ... args)
  {
    int len = std::snprintf( nullptr, 0, format.c_str(), convert_format_arg(args)... );
    assert(len>=0);
    std::string s(len+1,' ');
    ONIKA_DEBUG_ONLY( int len2 = ) std::snprintf( & s[0], len+1, format.c_str(), convert_format_arg(args)... );
    assert(len2==len);
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

  std::string remove_ansi_codes(std::string_view str_in);

  std::string markdown_to_raw(std::string_view str_in);

  std::string markdown_to_ansi(std::string_view str_in);

}
