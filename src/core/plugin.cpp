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

#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <regex>
#include <dlfcn.h>

#include <onika/plugin.h>
#include <onika/log.h>
#include <onika/string_utils.h>

#ifndef PLUGIN_DYNLIB_FORMAT
#define PLUGIN_DYNLIB_FORMAT "%s/lib%s.so"
#endif

namespace onika
{
  // format used to convert a plugin name to a dynamic library file name
  static std::string g_plugin_to_dynlib_format = PLUGIN_DYNLIB_FORMAT ;
  
  // list of directory where to load plugins from
  static std::vector<std::string> g_plugin_search_dirs;
  
  // associates (optionnaly) a namespace name with a plugin path.
  // every operator loaded from a plugin in one of this directory
  // is both named with its original name (i.e. xxx) and with its namespace compbounded
  // name (i.e. yyy@xxxx) where xxx is the operator name and yyy is the namespace.
  static std::map<std::string,std::string> g_plugin_dir_namespace;
  
  // similar to g_plugin_dir_namespace but at the file scope
  static std::map<std::string,std::string> g_plugin_file_namespace;
  
  // current namespace to associate with dynamic libraries to load
  static std::string g_plugin_namespace = "";
  
  // set of already loaded dynamix libraries
  static std::set<std::string> g_loaded_dynlibs;
  
  // name of plugin currently beeing loaded and installed
  static std::string g_loading_plugin;
  
  // name of database file containing all known plugins information
  static std::string g_plugin_db_filename;
  
  // loaded plugin database
  static PluginDBMap g_plugin_db;
  
  // keep track of dynlib files loaded
  static std::set<std::string> g_plugin_db_files;
  
  // if true, no console output is generated during plugin loads
  static bool g_quiet_plugin_register = true;

  // ======================= plugin API implementation =======================

  void set_quiet_plugin_register(bool b) { g_quiet_plugin_register = b; }
  bool quiet_plugin_register() { return g_quiet_plugin_register; }

  const std::string& get_plugin_namespace()
  {
    return g_plugin_namespace;
  }

  void set_plugin_search_dirs(const std::string& str)
  {
    g_plugin_search_dirs.clear();
    std::string::size_type s = 0;
    std::string::size_type e = str.find(':');
    auto add_plugin_search_dir = [](std::string pdir)
    {
      auto namespace_separator_pos = pdir.find('@');
      if( namespace_separator_pos != std::string::npos )
      {
        auto nspace = pdir.substr(0,namespace_separator_pos);
        auto dname = pdir.substr(namespace_separator_pos+1);
        g_plugin_search_dirs.push_back( dname );
        g_plugin_dir_namespace[dname] = nspace;
      }
      else
      {
        g_plugin_search_dirs.push_back( pdir );
        g_plugin_dir_namespace[pdir] = "";
      }
    };
    while( e != std::string::npos )
    {
      if( e != s ) add_plugin_search_dir( str.substr(s,e-s) );
      s = e + 1;
      e = str.find(':',s);
    } 
    e = str.length();
    if( e != s ) add_plugin_search_dir( str.substr(s,e-s) );

    ldbg << "g_plugin_search_dirs =";
    for(const auto& p:g_plugin_search_dirs) ldbg <<" "<<p;
    ldbg<<std::endl;
  }

  const std::vector<std::string>& plugin_search_dirs()
  {
    return g_plugin_search_dirs;
  }

  std::string plugin_path_env()
  {
    const char * pluginpath = std::getenv("ONIKA_PLUGIN_PATH"); // where to look for component plugins to load
    return ( pluginpath != nullptr ) ? pluginpath : "";
  }

  void generate_plugin_db( const std::string& filename )
  {
    g_plugin_db_filename = filename;
    //std::cout << "empty "<<g_plugin_db_filename<<std::endl;
    std::ofstream fout(g_plugin_db_filename); // zero file
  }

  void plugin_db_register( const std::string& itemCategory, const std::string& itemName )
  {
    if( !g_plugin_db_filename.empty() && !g_loading_plugin.empty() )
    {
      // std::cout << "plugin append : "<< g_loading_plugin << " " << itemCategory << " " << itemName << std::endl;
      std::ofstream fout(g_plugin_db_filename, std::ios::app);
      fout << g_loading_plugin << " " << itemCategory << " " << itemName << std::endl;
    }
  }

  void read_plugin_db( const std::string& filename )
  {
//    std::cout<< "read plugin.db @ "<<filename<<std::endl;
    g_plugin_db_files.clear();
    g_plugin_db.clear();
    std::ifstream fin(filename);
    while( fin )
    {
      std::string p, c, i;
      fin >> p >> c >> i;
      if( !p.empty() && !c.empty() && !i.empty() )
      {
        g_plugin_db[c][i].push_back(p);
        g_plugin_db_files.insert( p );
//        std::cout<< c << " : " << i << " -> "<<p<<std::endl;
      }
    }
  }

  const std::set<std::string>& get_plugin_db_files()
  {
    return g_plugin_db_files;
  }

  const PluginDBMap * get_plugin_db()
  {
    return & g_plugin_db;
  }

  
  void check_load_plugins_for( const std::string& itemCategory, const std::string& itemName )
  {
    //std::cout<<"check_load_plugins_for("<<itemCategory<<","<<itemName<<") =>";
    //for(const auto& x:g_plugin_db[itemCategory][itemName]) std::cout<<" "<<x;
    //std::cout<<std::endl;
    load_plugins( g_plugin_db[itemCategory][itemName] );
  }

  const std::set<std::string>& loaded_plugins()
  {
    return g_loaded_dynlibs;
  }

  static bool load_plugin_priv( const std::string& filePath)
  {
//    std::cout << "load plugin lib "<<filePath<<std::endl;
  	void* handle = dlopen(filePath.c_str(), RTLD_NOW|RTLD_GLOBAL);
  	if( handle == nullptr )
  	{
	    lerr << "plugin '"<<filePath<<"' not loaded ! " << dlerror() << std::endl << std::flush ;
	  }
	  else
	  {
	    g_loaded_dynlibs.insert( filePath );
	  }
    return handle != nullptr;
  }

  std::vector<std::string> plugin_files_from_search_directories(const std::vector<std::string> & plugin_files_or_directories)
  {
    std::vector<std::string> plugin_files;
    for( const std::string& p : plugin_files_or_directories )
    {
      //lout<<"scan plugin path "<< p << std::endl;
      if( std::filesystem::status(p).type() == std::filesystem::file_type::directory )
      {
        const auto dirnamespace = g_plugin_dir_namespace[p];
        for (auto it{std::filesystem::directory_iterator(p)}; it != std::filesystem::directory_iterator(); ++it)
        {
          if( std::filesystem::status(it->path()).type() == std::filesystem::file_type::regular )
          {
            const std::string filepath = it->path().string();
            if( std::regex_match(filepath,std::regex(format_string(g_plugin_to_dynlib_format,p,".*"))) )
            {
              plugin_files.push_back( filepath );
              //std::cout << "plugin " << filepath <<" has namesapce '"<<dirnamespace<<"'"<<std::endl;
              g_plugin_file_namespace[filepath] = dirnamespace;
            }
          }
        }
      }
      else
      {
        plugin_files.push_back(p);
      }
    }
    return plugin_files;
  }

  size_t load_plugins( const std::vector<std::string> & plugin_files )
  {
    size_t n_loaded = 0;

    std::string loading_plugin_backup = g_loading_plugin;
    for( const std::string& p : plugin_files )
    {
      g_loading_plugin = p;
      std::string fp = p;
      auto it = plugin_search_dirs().begin();
      while( it != plugin_search_dirs().end() && std::filesystem::status(fp).type() == std::filesystem::file_type::not_found )
      {
        fp = format_string( g_plugin_to_dynlib_format , *it , p );
        ++ it;
      }
      
      g_plugin_namespace = g_plugin_file_namespace[p];
      //std::cout << "load shared lib "<<p<<" with namesapce '"<< g_plugin_namespace << "'" << std::endl;
      if( g_loaded_dynlibs.find(fp) == g_loaded_dynlibs.end() )
      {
        if( ! quiet_plugin_register() ) lout<<"+ "<<fp<<std::endl;
        if( ! load_plugin_priv( fp ) ) { lerr<<"Warning, could not load plugin "<<p<<std::endl; }
        else { ++ n_loaded; }
      }
      g_plugin_namespace = "";
    }
    g_loading_plugin = loading_plugin_backup;
    return n_loaded;
  }


}

