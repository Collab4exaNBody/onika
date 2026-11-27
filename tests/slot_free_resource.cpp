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

// Freeing the resource of a slot (OperatorNode::free_all_resources(), called by onika::app::end())
// must also forget the slot's cached value pointer: afterwards the slot has no value, instead of
// pointing to released memory, and a later initialization builds the value again.

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/app/api.h>

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace onika::scg;

static int g_failures = 0;

#define CHECK(cond) do { if( !(cond) ) { std::printf("FAILED line %d: %s\n", __LINE__, #cond); ++ g_failures; } } while(0)

struct OwnDefaults : public OperatorNode
{
  ADD_SLOT( std::vector<double> , data    , INPUT  , std::make_tuple(size_t(4),1.5) );
  ADD_SLOT( long                , counter , OUTPUT , 7 );
  ADD_SLOT( std::vector<double> , cache   , PRIVATE );
  inline void execute() override final {}
};

struct Producer : public OperatorNode
{
  ADD_SLOT( long , value , OUTPUT , 42 );
  inline void execute() override final {}
};

struct Consumer : public OperatorNode
{
  ADD_SLOT( long , value , INPUT , REQUIRED );
  inline void execute() override final {}
};

// 1. slots holding their own values (defaults, output, private)
static void test_own_values()
{
  OwnDefaults op;
  op.compile();
  op.initialize_slots_resource();
  CHECK( op.data.has_value() && op.data->size() == 4 && op.data->at(0) == 1.5 );
  CHECK( op.counter.has_value() && *op.counter == 7 );
  CHECK( op.cache.has_value() );

  op.free_all_resources();
  CHECK( ! op.data.has_value() );
  CHECK( ! op.counter.has_value() );
  CHECK( ! op.cache.has_value() );
  CHECK( op.data.get_pointer() == nullptr );

  // values are built again by a later initialization
  op.initialize_slots_resource();
  CHECK( op.data.has_value() && op.data->size() == 4 && op.data->at(3) == 1.5 );
  CHECK( op.counter.has_value() && *op.counter == 7 );
  op.free_all_resources();
}

// 2. an output connected to an input: both slots share one resource
static void test_shared_resource()
{
  Producer p;
  Consumer c;
  p.compile();
  c.compile();
  OperatorSlotBase::connect( &p.value , &c.value );
  p.initialize_slots_resource();
  c.initialize_slots_resource();
  CHECK( p.value.has_value() && c.value.has_value() );
  CHECK( p.value.get_pointer() == c.value.get_pointer() && *c.value == 42 );

  // freeing both (as end() does for every operator) resets both, the memory is released once
  p.free_all_resources();
  c.free_all_resources();
  CHECK( ! p.value.has_value() );
  CHECK( ! c.value.has_value() );
}

// 3. the C++ application API: a graph that has run, then end()
static void test_app_end( const char* prog )
{
  const char* input = "slot_free_resource_test.msp";
  {
    std::ofstream f( input );
    f << "global: { dt: 1.0, nsteps: 2, timestep: 0, compute_loop_continue: true }\n"
         "compute_loop_stop: { rebind: { end_at: nsteps, result: compute_loop_continue }, body: [ sim_continue ] }\n"
         "compute_loop: { loop: true, condition: compute_loop_continue, body: [ next_time_step, compute_loop_stop ] }\n"
         "simulation: [ global, compute_loop ]\n";
  }
  std::vector<std::string> args = { prog, input };
  auto app = onika::app::init( args );
  CHECK( app->get_error_code() < 0 );
  if( app->get_error_code() >= 0 ) return;

  onika::app::run( app );
  std::shared_ptr<OperatorNode> graph = app->m_simulation_graph;   // keeps the nodes alive after end()
  OperatorSlotBase* timestep = app->node("simulation.global")->out_slot("timestep");
  CHECK( timestep != nullptr && timestep->has_value() );
  CHECK( timestep != nullptr && *timestep->output_value_pointer<long>() == 3 );

  onika::app::end( app );
  CHECK( timestep != nullptr && ! timestep->has_value() );
  std::remove( input );
}

int main(int argc, char* argv[])
{
  test_own_values();
  test_shared_resource();
  test_app_end( argv[0] );
  std::printf( "slot_free_resource: %s (%d failure%s)\n", g_failures==0 ? "OK" : "FAILED", g_failures, g_failures==1 ? "" : "s" );
  return g_failures == 0 ? 0 : 1;
}
