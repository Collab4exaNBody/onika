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
#include <set>
#include <map>
#include <cstdlib>

namespace onika
{

  void generate_plugin_db( const std::string& filename );
  void plugin_db_register( const std::string& itemCategory, const std::string& itemName );

  using PluginDBMap = std::map< std::string , std::map< std::string,std::string> >;
  void read_plugin_db( const std::string& filename );
  const PluginDBMap * get_plugin_db();
  const std::string& suggest_plugin_for( const std::string& itemCategory, const std::string& itemName );

  void set_plugin_search_dirs(const std::string& default_dir);
  const std::vector<std::string>& plugin_search_dirs();
  std::string plugin_path_env(); // concatenated version with paths separated by ':' , as given through ONIKA_PLUGIN_PATH env

  void set_quiet_plugin_register(bool b);
  bool quiet_plugin_register();

  // load a set of pluings, return the number of successfuly loaded
  std::vector<std::string> plugin_files_from_search_directories(const std::vector<std::string> & plugin_files_or_directories);
  size_t load_plugins( const std::vector<std::string> & plugin_files = plugin_files_from_search_directories(plugin_search_dirs()) );

  // list of successfuly loaded plugins
  const std::set<std::string>& loaded_plugins();
}

